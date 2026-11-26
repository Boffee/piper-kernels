"""Partitions preserve an already quantized full weight, including scale bytes."""

from unittest.mock import patch

import pytest
import torch
from _accelerators import exact_sm120_available
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F  # noqa: N812

from piper_kernels import compile_options
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor
from piper_kernels.weights.sharding import (
    ByteRange,
    shard_quantized_weight,
    trellis_row_shard_byte_range,
)

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
            act_quant="static",
        )
    weight = cls.from_hp(source, **options)
    if high_first:
        weight.qdata = (((weight.qdata & 0x0F) << 4) | (weight.qdata >> 4)).to(torch.uint8)
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
        if name in ("per_tensor_scale", "act_per_tensor_scale"):
            torch.testing.assert_close(selected, original, rtol=0, atol=0)


def _trellis_weight(
    *,
    rows=32,
    features=256,
    block_size=64,
    group_size=64,
    variable_rows=False,
    widths=(16, 20, 24, 32, 40, 48),
    extra_tail_bytes=0,
    dtype=torch.bfloat16,
    device="cpu",
    seed=119,
):
    generator = torch.Generator().manual_seed(seed)
    blocks_per_row = features // block_size
    block_scales = (torch.rand(rows, blocks_per_row, generator=generator) * 0.5 + 0.5).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(rows, 1, generator=generator) * 0.02 + 0.005
    if variable_rows:
        row_widths = torch.tensor(
            [widths[idx % len(widths)] for idx in range(rows)],
            dtype=torch.uint8,
        )
        total_bytes = blocks_per_row * int(row_widths.sum().item()) + extra_tail_bytes
        bitstreams = torch.randint(0, 256, (total_bytes,), dtype=torch.uint8, generator=generator)
    else:
        row_widths = None
        bytes_per_block = widths[0]
        bitstreams = torch.randint(
            0, 256, (rows, blocks_per_row, bytes_per_block), dtype=torch.uint8, generator=generator
        )
    weight = ConvRotTrellisTensor.from_quantized(
        bitstreams,
        block_scales,
        scale,
        block_size=block_size,
        group_size=group_size,
        logical_dtype=dtype,
        row_widths=row_widths,
    )
    return weight if device == "cpu" else weight.to(device)


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
        assert shard.act_quant is None
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
    if not exact_sm120_available():
        pytest.skip("requires an SM120 CUDA device")
    full = _weight(cls, rows=384, features=512, swizzled=True, high_first=high_first).to(
        creation_device
    )
    activation = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(384, device="cuda", dtype=torch.bfloat16)
    linear = F.linear
    if compiled:
        torch.compiler.reset()
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


@pytest.mark.parametrize(
    ("variable_rows", "widths", "extra_tail_bytes"),
    [
        (False, (24,), 0),
        (True, (24,), 0),
        (True, (16, 20, 24, 32, 40, 48), 0),
        (True, (16, 23, 31, 48), 19),
    ],
)
def test_trellis_sharding_preserves_full_weight_and_linear(variable_rows, widths, extra_tail_bytes):
    full = _trellis_weight(
        rows=18,
        features=256,
        block_size=64,
        group_size=64,
        variable_rows=variable_rows,
        widths=widths,
        extra_tail_bytes=extra_tail_bytes,
    )
    reference = full.dequantize(torch.float32)
    activation = torch.randn(5, 256)
    bias = torch.randn(18)
    expected_linear = F.linear(activation, reference, bias)

    for dim in (0, 1):
        splits = ((0, 9), (9, 9), (5, 1)) if dim == 0 else ((0, 128), (128, 128), (64, 64))
        for start, length in splits:
            with (
                patch.object(
                    ConvRotTrellisTensor,
                    "dequantize",
                    side_effect=AssertionError("must not decode"),
                ),
                patch.object(
                    ConvRotTrellisTensor,
                    "from_hp",
                    side_effect=AssertionError("must not re-encode"),
                ),
            ):
                shard = shard_quantized_weight(full, dim=dim, start=start, length=length)
            _assert_metadata_and_ownership(shard, full)
            torch.testing.assert_close(
                shard.dequantize(torch.float32),
                reference.narrow(dim, start, length),
                rtol=0,
                atol=0,
            )

        half = full.shape[dim] // 2
        halves = [shard_quantized_weight(full, dim=dim, start=s, length=half) for s in (0, half)]
        outs = [
            F.linear(
                activation if dim == 0 else activation[:, s : s + half],
                shard.dequantize(torch.float32),
                bias[s : s + half] if dim == 0 else None,
            )
            for shard, s in zip(halves, (0, half), strict=True)
        ]
        combined = torch.cat(outs, dim=-1) if dim == 0 else outs[0] + outs[1] + bias
        torch.testing.assert_close(combined, expected_linear, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("variable_rows", [False, True])
def test_trellis_full_extent_nested_and_lcm_alignment(variable_rows):
    full = _trellis_weight(
        rows=12,
        features=2688,
        block_size=224,
        group_size=64,
        variable_rows=variable_rows,
        widths=(56, 79, 112, 140, 168) if variable_rows else (112,),
    )
    reference = full.dequantize(torch.float32)

    copy_full = shard_quantized_weight(full, dim=0, start=0, length=full.shape[0])
    _assert_metadata_and_ownership(copy_full, full)
    for name in copy_full.__tensor_flatten__()[0]:
        getattr(copy_full, name).zero_()
    torch.testing.assert_close(full.dequantize(torch.float32), reference, rtol=0, atol=0)

    dim1_full = shard_quantized_weight(full, dim=1, start=0, length=full.shape[1])
    _assert_metadata_and_ownership(dim1_full, full)
    torch.testing.assert_close(dim1_full.dequantize(torch.float32), reference, rtol=0, atol=0)

    first = shard_quantized_weight(full, dim=-1, start=448, length=1344)
    second = shard_quantized_weight(first, dim=-1, start=448, length=448)
    nested = shard_quantized_weight(second, dim=-2, start=3, length=4)
    torch.testing.assert_close(
        nested.dequantize(torch.float32),
        reference[3:7, 896:1344],
        rtol=0,
        atol=0,
    )

    with pytest.raises(ValueError, match="rotation group size 64"):
        shard_quantized_weight(full, dim=1, start=224, length=224)
    with pytest.raises(ValueError, match="block size 224"):
        shard_quantized_weight(full, dim=1, start=64, length=64)

    pow2_trellis = _trellis_weight(
        rows=8, features=512, block_size=256, group_size=64, widths=(96,)
    )
    with pytest.raises(ValueError, match="block size 256"):
        shard_quantized_weight(pow2_trellis, dim=1, start=64, length=64)


def test_trellis_row_shard_byte_range_and_validation():
    full = _trellis_weight(
        rows=10,
        features=256,
        block_size=64,
        group_size=64,
        variable_rows=True,
        widths=(16, 20, 27, 36, 48),
        extra_tail_bytes=11,
    )
    assert full.row_widths is not None
    blocks_per_row = full.shape[1] // full.block_size
    expected_full = full.dequantize(torch.float32)

    span = trellis_row_shard_byte_range(
        full.row_widths,
        blocks_per_row=blocks_per_row,
        start=2,
        length=5,
        block_size=full.block_size,
    )
    assert isinstance(span, ByteRange)
    offset, byte_len = span
    assert (offset, byte_len) == (span.offset, span.length)

    _, metadata = full.as_storage()
    # Reconstruct the row shard from only the sliced byte span (simulated FileSlice read).
    file_slice_weight = ConvRotTrellisTensor.from_storage(
        {
            "bitstreams": full.bitstreams[offset : offset + byte_len].clone(),
            "block_scales": full.block_scales[2:7].clone(),
            "scale": full.scale[2:7].clone(),
            "row_widths": full.row_widths[2:7].clone(),
        },
        metadata,
    )
    torch.testing.assert_close(file_slice_weight.dequantize(torch.float32), expected_full[2:7])
    assert trellis_row_shard_byte_range(
        full.row_widths, blocks_per_row=blocks_per_row, start=0, length=10
    ) == ByteRange(0, blocks_per_row * int(full.row_widths.sum().item()))
    strided_widths = torch.stack((full.row_widths, full.row_widths), dim=1)[:, 0]
    assert trellis_row_shard_byte_range(
        strided_widths,
        blocks_per_row=blocks_per_row,
        start=2,
        length=5,
        block_size=full.block_size,
    ) == ByteRange(offset, byte_len)

    for bad_widths, err, msg in (
        ([16, 20], TypeError, "row_widths must be a torch.Tensor"),
        (torch.tensor([16, 20], dtype=torch.int32), ValueError, "1-D uint8 tensor"),
        (torch.tensor([[16, 20]], dtype=torch.uint8), ValueError, "1-D uint8 tensor"),
        (torch.tensor([15, 20], dtype=torch.uint8), ValueError, "variable-row widths"),
        (torch.tensor([16, 49], dtype=torch.uint8), ValueError, "variable-row widths"),
        (
            torch.empty((4,), dtype=torch.uint8, device="meta"),
            NotImplementedError,
            "materialized",
        ),
    ):
        with pytest.raises(err, match=msg):
            trellis_row_shard_byte_range(
                bad_widths,  # type: ignore[arg-type]
                blocks_per_row=blocks_per_row,
                start=0,
                length=1,
                block_size=64,
            )
    with pytest.raises(TypeError, match="concrete integers"):
        trellis_row_shard_byte_range(
            full.row_widths,
            blocks_per_row=True,
            start=0,
            length=1,  # type: ignore[arg-type]
        )
    for kw, msg in (
        ({"blocks_per_row": 0, "start": 0, "length": 1}, "blocks_per_row must be positive"),
        ({"blocks_per_row": blocks_per_row, "start": -1, "length": 2}, "within row_widths"),
        ({"blocks_per_row": blocks_per_row, "start": 0, "length": 0}, "within row_widths"),
        ({"blocks_per_row": blocks_per_row, "start": 8, "length": 4}, "within row_widths"),
        (
            {"blocks_per_row": blocks_per_row, "start": 0, "length": 2, "block_size": 512},
            "16 to 256",
        ),
    ):
        with pytest.raises(ValueError, match=msg):
            trellis_row_shard_byte_range(full.row_widths, **kw)


def test_trellis_sharding_boundary_rejections_and_meta():
    full = _trellis_weight(
        rows=8,
        features=256,
        block_size=64,
        group_size=64,
        variable_rows=True,
        widths=(16, 24, 32, 48),
    )
    _, metadata = full.as_storage()

    with FakeTensorMode():
        fake_widths = torch.ones((4,), dtype=torch.uint8)
        with pytest.raises(NotImplementedError, match="materialized"):
            trellis_row_shard_byte_range(fake_widths, blocks_per_row=4, start=0, length=2)

    truncated = ConvRotTrellisTensor.from_quantized(
        full.bitstreams[:-15].clone(),
        full.block_scales,
        full.scale,
        block_size=full.block_size,
        group_size=full.group_size,
        row_widths=full.row_widths,
    )
    with pytest.raises(ValueError, match="shorter than row_widths requires"):
        shard_quantized_weight(truncated, dim=0, start=6, length=2)
    with pytest.raises(ValueError, match="shorter than row_widths requires"):
        shard_quantized_weight(truncated, dim=1, start=0, length=128)
    with pytest.raises(NotImplementedError, match="transposed"):
        shard_quantized_weight(full.t(), dim=0, start=0, length=2)

    out_of_range = _trellis_weight(
        rows=4,
        features=256,
        block_size=64,
        group_size=64,
        variable_rows=True,
        widths=(1, 2, 1, 2),
    )
    assert shard_quantized_weight(out_of_range, dim=0, start=1, length=2).shape == (2, 256)
    assert shard_quantized_weight(out_of_range, dim=1, start=64, length=128).shape == (4, 128)
    assert out_of_range.row_widths is not None
    assert trellis_row_shard_byte_range(
        out_of_range.row_widths, blocks_per_row=4, start=1, length=2
    ) == ByteRange(4, 12)
    with pytest.raises(ValueError, match="variable-row widths"):
        trellis_row_shard_byte_range(
            out_of_range.row_widths, blocks_per_row=4, start=1, length=2, block_size=64
        )

    noncontig_var = _trellis_weight(
        rows=8,
        features=256,
        block_size=64,
        group_size=64,
        variable_rows=True,
        widths=(16, 24, 32, 48),
    )
    noncontig_var.bitstreams = torch.empty(noncontig_var.bitstreams.numel() * 2, dtype=torch.uint8)[
        ::2
    ]
    with pytest.raises(NotImplementedError, match="contiguous"):
        shard_quantized_weight(noncontig_var, dim=0, start=0, length=2)

    meta_var = ConvRotTrellisTensor.from_storage(
        {k: v.to("meta") for k, v in full.as_storage()[0].items()},
        metadata,
    )
    for dim in (0, 1):
        with pytest.raises(NotImplementedError, match="materialized"):
            shard_quantized_weight(meta_var, dim=dim, start=0, length=64 if dim == 1 else 2)

    meta_uni = ConvRotTrellisTensor.from_storage(
        {k: v.to("meta") for k, v in _trellis_weight(rows=8, features=256).as_storage()[0].items()},
        metadata,
    )
    assert shard_quantized_weight(meta_uni, dim=0, start=2, length=4).shape == (4, 256)
    assert shard_quantized_weight(meta_uni, dim=1, start=64, length=128).shape == (8, 128)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["uniform_hp", "uniform", "variable"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("creation_device", ["cpu", "cuda"])
def test_cuda_trellis_linear_composition(mode, compiled, creation_device):
    if not exact_sm120_available():
        pytest.skip("requires an SM120 CUDA device")
    if mode == "uniform_hp":
        torch.manual_seed(131)
        source = torch.randn(384, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        full = ConvRotTrellisTensor.from_hp(source, group_size=64, bits_per_weight=4.5).to(
            creation_device
        )
    else:
        full = _trellis_weight(
            rows=384,
            features=512,
            block_size=64,
            group_size=64,
            variable_rows=mode == "variable",
            widths=(16, 24, 32, 48) if mode == "variable" else (32,),
            device=creation_device,
        )
    bias = torch.randn(384, device="cuda", dtype=torch.bfloat16)
    linear = F.linear
    if compiled:
        torch.compiler.reset()
        linear = torch.compile(
            F.linear, fullgraph=True, options=compile_options({"triton.cudagraphs": False})
        )
    with torch.no_grad():
        full_cuda = full.to("cuda")
        for batch_rows in (2, 32):
            activation = torch.randn(batch_rows, 512, device="cuda", dtype=torch.bfloat16)
            expected = F.linear(activation, full_cuda, bias).float()
            for dim in (0, 1):
                length = full.shape[dim] // 2
                outputs = []
                for start in (0, length):
                    shard = shard_quantized_weight(full, dim=dim, start=start, length=length).to(
                        "cuda"
                    )
                    x = (
                        activation
                        if dim == 0
                        else activation[:, start : start + length].contiguous()
                    )
                    b = bias[start : start + length] if dim == 0 else None
                    actual = linear(x, shard, b)
                    if compiled:
                        torch.testing.assert_close(actual, F.linear(x, shard, b), rtol=0, atol=0)
                    outputs.append(actual.float())
                combined = (
                    torch.cat(outputs, dim=-1) if dim == 0 else outputs[0] + outputs[1] + bias
                )
                relative_error = (combined - expected).norm() / expected.norm()
                assert relative_error < 0.015


@pytest.mark.gpu
def test_cuda_variable_row_trellis_multichunk_dim1_shard():
    if not exact_sm120_available():
        pytest.skip("requires an SM120 CUDA device")
    # 1024 rows * 64 blocks (halved from 128) * avg 36 bytes = 2.36 MiB > 2 MiB gather chunk.
    full = _trellis_weight(
        rows=1024,
        features=8192,
        block_size=64,
        group_size=64,
        variable_rows=True,
        widths=(24, 32, 40, 48),
        extra_tail_bytes=37,
        device="cuda",
        seed=211,
    )
    assert full.row_widths is not None
    span = trellis_row_shard_byte_range(
        full.row_widths, blocks_per_row=128, start=256, length=512, block_size=64
    )
    assert span == trellis_row_shard_byte_range(
        full.row_widths.cpu(), blocks_per_row=128, start=256, length=512, block_size=64
    )
    row_shard = shard_quantized_weight(full, dim=0, start=256, length=512)
    assert torch.equal(
        row_shard.bitstreams, full.bitstreams[span.offset : span.offset + span.length]
    )

    reference = full.dequantize(torch.float32)
    shard = shard_quantized_weight(full, dim=1, start=4096, length=4096)
    _assert_metadata_and_ownership(shard, full)
    torch.testing.assert_close(shard.dequantize(torch.float32), reference[:, 4096:], rtol=0, atol=0)

    x = torch.randn(3, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(
            F.linear(x, shard),
            F.linear(x.float(), reference[:, 4096:]).bfloat16(),
            rtol=2.5e-2,
            atol=2.5e-2,
        )
