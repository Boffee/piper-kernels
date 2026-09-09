"""Partitions preserve an already quantized full weight, including scale bytes."""

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

from piper_kernels.linear.convrot import convrot_int8_compile_options
from piper_kernels.linear.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor, convrot_nvfp4_compile_options
from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor, nvfp4_compile_options
from piper_kernels.linear.nvfp4._layout import swap_packed_pairs
from piper_kernels.linear.sharding import shard_quantized_weight

_WRAPPERS = (ConvRotInt8Tensor, PiperNVFP4Tensor, ConvRotNVFP4Tensor)
_FORMATS = [(ConvRotInt8Tensor, False, False)] + [
    (cls, swizzled, high_first)
    for cls in (PiperNVFP4Tensor, ConvRotNVFP4Tensor)
    for swizzled in (False, True)
    for high_first in (False, True)
]


def _weight(cls, *, rows=32, features=128, swizzled=False, high_first=False, group_size=64):
    torch.manual_seed(113)
    source = torch.randn(rows, features, dtype=torch.bfloat16)
    options = {"group_size": group_size} if cls is not PiperNVFP4Tensor else {}
    if cls is not ConvRotInt8Tensor:
        options.update(
            compute_per_tensor_scale=True,
            is_swizzled_scales=swizzled,
            act_per_tensor_scale=torch.tensor(0.002),
            act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(
                block_size=16,
                is_swizzled_scales=True,
                use_triton_kernel=False,
                use_dynamic_per_tensor_scale=False,
            ),
        )
    weight = cls.from_hp(source, **options)
    if high_first:
        weight.qdata = swap_packed_pairs(weight.qdata)
        weight.high_first = True
    return weight


def _assert_metadata_and_ownership(shard, full):
    assert type(shard) is type(full)
    assert shard.dtype == full.dtype
    assert shard.device == full.device
    assert shard.__tensor_flatten__() == full.__tensor_flatten__()
    for name in full.__tensor_flatten__()[0]:
        original, selected = getattr(full, name), getattr(shard, name)
        assert original.dtype == selected.dtype
        assert selected.is_contiguous()
        assert not torch._C._is_alias_of(original, selected)
        if name not in ("qdata", "scale"):
            torch.testing.assert_close(selected, original, rtol=0, atol=0)


@pytest.mark.parametrize(("cls", "swizzled", "high_first"), _FORMATS)
@pytest.mark.parametrize("rows", [32, 258])
@pytest.mark.parametrize("flat_scales", [False, True])
def test_both_halves_preserve_full_weight(cls, swizzled, high_first, rows, flat_scales):
    full = _weight(cls, rows=rows, swizzled=swizzled, high_first=high_first)
    if cls is not ConvRotInt8Tensor and flat_scales:
        full.scale = full.scale.flatten()
    reference = full.dequantize(torch.float32)
    for dim in (0, 1):
        length = full.shape[dim] // 2
        for start in (0, length):
            with (
                patch.object(cls, "dequantize", side_effect=AssertionError("must not dequantize")),
                patch.object(cls, "from_hp", side_effect=AssertionError("must not requantize")),
            ):
                shard = shard_quantized_weight(full, dim=dim, start=start, length=length)
            _assert_metadata_and_ownership(shard, full)
            torch.testing.assert_close(
                shard.dequantize(torch.float32),
                reference.narrow(dim, start, length),
                rtol=0,
                atol=0,
            )
            packed_divisor = 2 if cls is not ConvRotInt8Tensor and dim == 1 else 1
            torch.testing.assert_close(
                shard.qdata,
                full.qdata.narrow(dim, start // packed_divisor, length // packed_divisor),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("cls", _WRAPPERS)
@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_complete_rotation_groups_and_nested_shards(cls, group_size):
    full = _weight(cls, features=1024, group_size=group_size, swizzled=True)
    first = shard_quantized_weight(full, dim=-1, start=512, length=512)
    second = shard_quantized_weight(first, dim=-1, start=group_size, length=group_size)
    shard = shard_quantized_weight(second, dim=-2, start=3, length=1)
    # Changing the GEMM batch shape can change FP32 Hadamard accumulation order.
    torch.testing.assert_close(
        shard.dequantize(torch.float32),
        full.dequantize(torch.float32)[3:4, 512 + group_size : 512 + 2 * group_size],
        rtol=1e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize("cls", _WRAPPERS)
def test_full_extent_is_an_independent_copy(cls):
    full = _weight(cls, swizzled=True)
    reference = full.dequantize(torch.float32)
    shard = shard_quantized_weight(full, dim=0, start=0, length=full.shape[0])
    _assert_metadata_and_ownership(shard, full)
    for name in shard.__tensor_flatten__()[0]:
        getattr(shard, name).zero_()
    torch.testing.assert_close(full.dequantize(torch.float32), reference, rtol=0, atol=0)


@pytest.mark.parametrize(("cls", "swizzled", "high_first"), _FORMATS)
def test_dequantized_linear_composition(cls, swizzled, high_first):
    full = _weight(cls, swizzled=swizzled, high_first=high_first)
    activation = torch.randn(7, 128)
    bias = torch.randn(32)
    expected = F.linear(activation, full.dequantize(torch.float32), bias)
    for dim in (0, 1):
        length = full.shape[dim] // 2
        outputs = []
        for start in (0, length):
            shard = shard_quantized_weight(full, dim=dim, start=start, length=length)
            x = activation if dim == 0 else activation[:, start : start + length]
            b = bias[start : start + length] if dim == 0 else None
            outputs.append(F.linear(x, shard.dequantize(torch.float32), b))
        actual = torch.cat(outputs, dim=-1) if dim == 0 else outputs[0] + outputs[1] + bias
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("cls", _WRAPPERS)
@pytest.mark.parametrize(
    ("dim", "start", "length", "error", "message"),
    [
        (2, 0, 16, IndexError, "dimension"),
        (-3, 0, 16, IndexError, "dimension"),
        (0, -1, 16, ValueError, "within"),
        (0, 0, 0, ValueError, "nonempty"),
        (0, 0, -1, ValueError, "nonempty"),
        (0, 17, 16, ValueError, "within"),
        (0, 0.0, 16, TypeError, "integers"),
        (True, 0, 16, TypeError, "integers"),
        (1, 1, 64, ValueError, "align"),
        (1, 0, 63, ValueError, "align"),
    ],
)
def test_invalid_partition_is_rejected(cls, dim, start, length, error, message):
    with pytest.raises(error, match=message):
        shard_quantized_weight(_weight(cls), dim=dim, start=start, length=length)


@pytest.mark.parametrize("cls", [ConvRotInt8Tensor, ConvRotNVFP4Tensor])
def test_cut_inside_rotation_group_is_rejected(cls):
    with pytest.raises(ValueError, match="rotation group size 64"):
        shard_quantized_weight(_weight(cls), dim=1, start=16, length=64)


@pytest.mark.parametrize("cls", _WRAPPERS)
def test_transposed_partition_is_rejected(cls):
    with pytest.raises(NotImplementedError, match="transposed"):
        shard_quantized_weight(_weight(cls).t(), dim=0, start=0, length=16)


@pytest.mark.parametrize(
    "bad_layout", ["strided", "scale_shape", "per_expert", "block_size", "rank"]
)
def test_unsupported_nvfp4_layout_is_rejected(bad_layout):
    full = _weight(PiperNVFP4Tensor, swizzled=True)
    if bad_layout == "strided":
        full.qdata = full.qdata[:, ::2]
    elif bad_layout == "scale_shape":
        full.scale = full.scale.flatten()[:-1]
    elif bad_layout == "per_expert":
        full.per_tensor_scale = torch.ones(1, 1, 1)
    elif bad_layout == "block_size":
        full.block_size = 32
    else:
        full = PiperNVFP4Tensor.from_hp(torch.randn(2, 32, 128, dtype=torch.bfloat16))
    with pytest.raises(NotImplementedError):
        shard_quantized_weight(full, dim=0, start=0, length=16)


def test_dense_tensor_is_rejected():
    with pytest.raises(TypeError, match="Piper quantized weight"):
        shard_quantized_weight(torch.randn(32, 128), dim=0, start=0, length=16)


@pytest.mark.parametrize("cls", [PiperNVFP4Tensor, ConvRotNVFP4Tensor])
def test_nvfp4_without_global_scales(cls):
    full = cls.from_hp(
        torch.randn(32, 128, dtype=torch.bfloat16),
        **({"group_size": 64} if cls is ConvRotNVFP4Tensor else {}),
    )
    for dim in (0, 1):
        length = full.shape[dim] // 2
        shard = shard_quantized_weight(full, dim=dim, start=length, length=length)
        assert shard.per_tensor_scale is None
        assert shard.act_per_tensor_scale is None
        assert shard.act_quant_kwargs is None
        torch.testing.assert_close(
            shard.dequantize(torch.float32),
            full.dequantize(torch.float32).narrow(dim, length, length),
            rtol=0,
            atol=0,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("cls", "high_first"),
    [(ConvRotInt8Tensor, False)]
    + [(cls, high) for cls in (PiperNVFP4Tensor, ConvRotNVFP4Tensor) for high in (False, True)],
)
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("creation_device", ["cpu", "cuda"])
def test_cuda_quantized_linear_composition(cls, high_first, compiled, creation_device):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires an SM120 CUDA device")
    full = _weight(cls, rows=384, features=512, swizzled=True, high_first=high_first).to(
        creation_device
    )
    activation = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(384, device="cuda", dtype=torch.bfloat16)
    linear = F.linear
    if compiled:
        torch.compiler.reset()
        compile_options = {
            ConvRotInt8Tensor: convrot_int8_compile_options,
            PiperNVFP4Tensor: nvfp4_compile_options,
            ConvRotNVFP4Tensor: convrot_nvfp4_compile_options,
        }[cls]
        linear = torch.compile(
            F.linear, fullgraph=True, options=compile_options({"triton.cudagraphs": False})
        )
    with torch.no_grad():
        expected = F.linear(activation, full.to("cuda"), bias).float()
        for dim in (0, 1):
            length = full.shape[dim] // 2
            outputs = []
            for start in (0, length):
                shard = shard_quantized_weight(full, dim=dim, start=start, length=length).to("cuda")
                x = activation if dim == 0 else activation[:, start : start + length].contiguous()
                b = bias[start : start + length] if dim == 0 else None
                actual = linear(x, shard, b)
                if compiled:
                    torch.testing.assert_close(actual, F.linear(x, shard, b), rtol=0, atol=0)
                outputs.append(actual.float())
            combined = torch.cat(outputs, dim=-1) if dim == 0 else outputs[0] + outputs[1] + bias
            # INT8 dynamically quantizes each local activation row. NVFP4 uses
            # a fixed activation scale here; both formats round local outputs.
            relative_error = (combined - expected).norm() / expected.norm()
            assert relative_error < 0.015
