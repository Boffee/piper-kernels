"""Contract tests for the QuantizedWeight storage boundary and 3-D view dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from _accelerators import (
    exact_sm120_available,
    trellis_encoder_available,
    w4a16_nvfp4_available,
)
from torch import nn
from torch.nn import functional as F  # noqa: N812

from piper_kernels.trellis import TrellisStorage
from piper_kernels.weights import QuantizedWeight
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

_NON_EMPTY_META_SWIZZLED_SCALE_SHAPE = (32, 64)


def _swap_nibbles(qdata: torch.Tensor) -> torch.Tensor:
    return (((qdata & 0x0F) << 4) | (qdata >> 4)).to(torch.uint8)


def _make_int8_2d(*, device: str = "cpu", static_act: bool = False) -> ConvRotInt8Tensor:
    torch.manual_seed(11)
    act_scale = torch.tensor(0.25, device=device, dtype=torch.float32) if static_act else None
    dense = torch.randn(32, 64, device=device, dtype=torch.bfloat16) * 0.2
    return ConvRotInt8Tensor.from_hp(dense, group_size=16, act_per_tensor_scale=act_scale)


def _make_int8_conv3d(*, device: str = "cpu") -> ConvRotInt8Tensor:
    qdata = torch.randint(-60, 60, (16, 3, 3, 3, 64), device=device, dtype=torch.int8)
    scale = torch.full((16, 1), 0.05, device=device, dtype=torch.float32)
    act_scale = torch.tensor(0.25, device=device, dtype=torch.float32)
    return ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=16,
        logical_dtype=torch.float16,
        act_per_tensor_scale=act_scale,
    )


def _make_nvfp4_2d(
    *,
    convrot: bool = False,
    device: str = "cpu",
    act_quant: str | None = None,
    high_first: bool = False,
    swizzled: bool = True,
    flat_scale: bool = False,
) -> PiperNVFP4Tensor:
    torch.manual_seed(23)
    dense = torch.randn(128, 256, device=device, dtype=torch.bfloat16) * 0.25
    act_scale = (
        torch.tensor(0.125, device=device, dtype=torch.float32) if act_quant == "static" else None
    )
    options: dict[str, Any] = {
        "compute_per_tensor_scale": True,
        "act_per_tensor_scale": act_scale,
        "is_swizzled_scales": swizzled,
        "act_quant": act_quant,
    }
    if convrot:
        base: PiperNVFP4Tensor = ConvRotNVFP4Tensor.from_hp(dense, group_size=64, **options)
    else:
        base = PiperNVFP4Tensor.from_hp(dense, **options)
    if not high_first and not flat_scale:
        return base
    qdata = _swap_nibbles(base.qdata) if high_first else base.qdata
    scale = base.scale.reshape(-1) if flat_scale else base.scale
    if isinstance(base, ConvRotNVFP4Tensor):
        return ConvRotNVFP4Tensor.from_quantized(
            qdata,
            scale,
            group_size=base.group_size,
            logical_dtype=base.dtype,
            block_size=base.block_size,
            per_tensor_scale=base.per_tensor_scale,
            act_per_tensor_scale=base.act_per_tensor_scale,
            is_swizzled_scales=base.is_swizzled_scales,
            act_quant=base.act_quant,
            high_first=high_first,
        )
    return PiperNVFP4Tensor.from_quantized(
        qdata,
        scale,
        logical_dtype=base.dtype,
        block_size=base.block_size,
        per_tensor_scale=base.per_tensor_scale,
        act_per_tensor_scale=base.act_per_tensor_scale,
        is_swizzled_scales=base.is_swizzled_scales,
        act_quant=base.act_quant,
        high_first=high_first,
    )


def _make_nvfp4_3d_moe() -> PiperNVFP4Tensor:
    experts, rows, features = 2, 128, 256
    qdata = torch.randint(0, 256, (experts, rows, features // 2), dtype=torch.uint8)
    scale = torch.ones((experts, *_NON_EMPTY_META_SWIZZLED_SCALE_SHAPE), dtype=torch.float8_e4m3fn)
    per_tensor_scale = torch.full((experts, 1, 1), 0.125, dtype=torch.float32)
    return PiperNVFP4Tensor.from_quantized(
        qdata,
        scale,
        logical_dtype=torch.bfloat16,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
    )


def _make_trellis_uniform() -> ConvRotTrellisTensor:
    rows, blocks, block_size = 8, 4, 16
    bitstreams = torch.randint(0, 256, (rows, blocks, 8), dtype=torch.uint8)
    block_scales = torch.ones((rows, blocks), dtype=torch.float8_e4m3fn)
    scale = torch.full((rows, 1), 0.25, dtype=torch.float32)
    return ConvRotTrellisTensor.from_quantized(
        bitstreams,
        block_scales,
        scale,
        block_size=block_size,
        group_size=16,
        logical_dtype=torch.bfloat16,
    )


def _make_trellis_variable_row() -> ConvRotTrellisTensor:
    rows, blocks, block_size = 8, 4, 16
    row_widths = torch.tensor([4, 6, 8, 10, 12, 4, 6, 8], dtype=torch.uint8)
    total_bytes = int(row_widths.sum().item()) * blocks
    bitstreams = torch.randint(0, 256, (total_bytes,), dtype=torch.uint8)
    block_scales = torch.ones((rows, blocks), dtype=torch.float8_e4m3fn)
    scale = torch.full((rows, 1), 0.25, dtype=torch.float32)
    return ConvRotTrellisTensor.from_quantized(
        bitstreams,
        block_scales,
        scale,
        block_size=block_size,
        group_size=16,
        logical_dtype=torch.bfloat16,
        row_widths=row_widths,
    )


def _assert_zero_copy_storage_round_trip(weight: QuantizedWeight) -> QuantizedWeight:
    assert isinstance(weight, QuantizedWeight)
    tensors, metadata = weight.as_storage()
    assert isinstance(tensors, dict)
    assert isinstance(metadata, dict)
    assert "transposed" not in metadata
    assert all(isinstance(name, str) for name in tensors)
    assert all(
        isinstance(value, torch.Tensor) and not isinstance(value, QuantizedWeight)
        for value in tensors.values()
    )
    assert all(not isinstance(value, torch.Tensor) for value in metadata.values())

    rebuilt = type(weight).from_storage(tensors, metadata)
    assert type(rebuilt) is type(weight)
    assert tuple(rebuilt.shape) == tuple(weight.shape)
    assert rebuilt.stride() == weight.stride()
    assert rebuilt.dtype is weight.dtype
    for name, buf in tensors.items():
        orig = getattr(weight, name)
        assert isinstance(orig, torch.Tensor)
        assert buf is orig
        assert buf.data_ptr() == orig.data_ptr()
        restored = getattr(rebuilt, name)
        assert restored is buf
        assert restored.data_ptr() == orig.data_ptr()

    rebuilt_tensors, rebuilt_metadata = rebuilt.as_storage()
    assert list(rebuilt_tensors) == list(tensors)
    assert rebuilt_metadata == metadata
    return rebuilt


@pytest.mark.parametrize(
    "builder",
    [
        lambda: _make_int8_2d(static_act=False),
        lambda: _make_int8_2d(static_act=True),
        _make_int8_conv3d,
        lambda: _make_nvfp4_2d(act_quant=None),
        lambda: _make_nvfp4_2d(act_quant="static"),
        lambda: _make_nvfp4_2d(act_quant="dynamic"),
        lambda: _make_nvfp4_2d(act_quant=None, high_first=True),
        lambda: _make_nvfp4_2d(act_quant=None, swizzled=False),
        lambda: _make_nvfp4_2d(act_quant=None, swizzled=True, flat_scale=True),
        _make_nvfp4_3d_moe,
        lambda: _make_nvfp4_2d(convrot=True, act_quant=None),
        lambda: _make_nvfp4_2d(convrot=True, act_quant="static"),
        lambda: _make_nvfp4_2d(convrot=True, act_quant="dynamic"),
        lambda: _make_nvfp4_2d(convrot=True, act_quant=None, high_first=True),
        _make_trellis_uniform,
        _make_trellis_variable_row,
    ],
)
def test_storage_boundary_round_trip_and_offload_cpu(
    builder: Callable[[], QuantizedWeight],
) -> None:
    weight = builder()
    rebuilt = _assert_zero_copy_storage_round_trip(weight)

    if weight.ndim == 2 and not isinstance(weight, ConvRotTrellisTensor):
        torch.testing.assert_close(rebuilt.dequantize(), weight.dequantize(), rtol=0, atol=0)  # type: ignore[attr-defined]

    # Simulate offloader: export CPU buffers -> allocate independent buffers -> copy -> reconstruct.
    tensors, metadata = weight.as_storage()
    copied_buffers = {name: torch.empty_like(buf).copy_(buf) for name, buf in tensors.items()}
    offloaded = type(weight).from_storage(copied_buffers, metadata)
    assert all(
        getattr(offloaded, name).data_ptr() == copied_buffers[name].data_ptr()
        for name in copied_buffers
    )
    if isinstance(weight, ConvRotInt8Tensor) and weight.ndim == 2:
        x = torch.randn(3, weight.shape[1], dtype=weight.dtype)
        torch.testing.assert_close(F.linear(x, offloaded), F.linear(x, weight), rtol=0, atol=0)
    elif isinstance(weight, PiperNVFP4Tensor) and weight.ndim == 2 and weight.act_quant is None:
        if weight.is_swizzled_scales and weight.scale.ndim == 2:
            x = torch.randn(3, weight.shape[1], dtype=weight.dtype)
            torch.testing.assert_close(F.linear(x, offloaded), F.linear(x, weight), rtol=0, atol=0)


def test_storage_boundary_meta_template_round_trip() -> None:
    source = _make_nvfp4_2d(convrot=True, act_quant=None, high_first=True)
    cpu_tensors, metadata = source.as_storage()
    meta_tensors = {name: torch.empty_like(buf, device="meta") for name, buf in cpu_tensors.items()}
    meta_weight = ConvRotNVFP4Tensor.from_storage(meta_tensors, metadata)
    assert meta_weight.device.type == "meta"

    dest_tensors = {name: torch.empty_like(buf, device="cpu") for name, buf in meta_tensors.items()}
    for name, dst in dest_tensors.items():
        dst.copy_(cpu_tensors[name])
    materialized = ConvRotNVFP4Tensor.from_storage(dest_tensors, metadata)
    x = torch.randn(2, source.shape[1], dtype=source.dtype)
    torch.testing.assert_close(F.linear(x, materialized), F.linear(x, source), rtol=0, atol=0)


def test_storage_boundary_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="concrete subclass"):
        QuantizedWeight.from_storage({}, {})

    weight = _make_nvfp4_2d(act_quant="static")
    with pytest.raises(NotImplementedError, match="transposed"):
        weight.t().as_storage()

    tensors, metadata = weight.as_storage()

    with pytest.raises(TypeError, match="mapping"):
        PiperNVFP4Tensor.from_storage(list(tensors.items()), metadata)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        PiperNVFP4Tensor.from_storage(tensors, list(metadata.items()))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="transposed"):
        PiperNVFP4Tensor.from_storage(tensors, {**metadata, "transposed": False})
    with pytest.raises(ValueError, match="missing required tensors"):
        PiperNVFP4Tensor.from_storage({"qdata": tensors["qdata"]}, metadata)
    with pytest.raises(ValueError, match="unexpected tensors"):
        PiperNVFP4Tensor.from_storage({**tensors, "extra": torch.empty(1)}, metadata)
    with pytest.raises(TypeError, match=r"plain torch\.Tensor"):
        PiperNVFP4Tensor.from_storage({**tensors, "qdata": weight}, metadata)
    with pytest.raises(TypeError, match=r"plain torch\.Tensor"):
        PiperNVFP4Tensor.from_storage({**tensors, "per_tensor_scale": "bad"}, metadata)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="missing required metadata"):
        PiperNVFP4Tensor.from_storage(tensors, {"dtype": metadata["dtype"]})
    with pytest.raises(ValueError, match="unexpected metadata"):
        PiperNVFP4Tensor.from_storage(tensors, {**metadata, "unknown": 1})
    with pytest.raises(TypeError, match="must not contain tensors"):
        PiperNVFP4Tensor.from_storage(tensors, {**metadata, "dtype": torch.empty(1)})
    with pytest.raises(ValueError, match="contiguous"):
        PiperNVFP4Tensor.from_storage(
            {**tensors, "qdata": torch.empty(128, 256, dtype=torch.uint8)[:, ::2]},
            metadata,
        )
    with pytest.raises(ValueError, match="device"):
        PiperNVFP4Tensor.from_storage(
            {**tensors, "scale": tensors["scale"].to("meta")},
            metadata,
        )
    with pytest.raises(ValueError, match="uint8"):
        PiperNVFP4Tensor.from_storage(
            {**tensors, "qdata": tensors["qdata"].to(torch.int8)},
            metadata,
        )
    with pytest.raises(ValueError, match="scale shape"):
        PiperNVFP4Tensor.from_storage(
            {**tensors, "scale": torch.ones(16, 16, dtype=torch.float8_e4m3fn)},
            metadata,
        )

    convrot_w = _make_nvfp4_2d(convrot=True, act_quant="static")
    _, conv_meta = convrot_w.as_storage()
    with pytest.raises(ValueError, match="two-dimensional"):
        ConvRotNVFP4Tensor.from_storage(
            {
                "qdata": torch.randint(0, 256, (2, 128, 128), dtype=torch.uint8),
                "scale": torch.ones((2, 32, 64), dtype=torch.float8_e4m3fn),
            },
            {**conv_meta, "act_quant": None},
        )
    with pytest.raises(ValueError, match="divisible"):
        ConvRotNVFP4Tensor.from_storage(
            {
                "qdata": torch.randint(0, 256, (128, 96), dtype=torch.uint8),
                "scale": torch.ones((32, 48), dtype=torch.float8_e4m3fn),
            },
            {**conv_meta, "group_size": 256, "act_quant": None},
        )

    trellis_var = _make_trellis_variable_row()
    t_tensors, t_meta = trellis_var.as_storage()
    strided_bits = torch.empty(t_tensors["bitstreams"].numel() * 2, dtype=torch.uint8)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        ConvRotTrellisTensor.from_storage({**t_tensors, "bitstreams": strided_bits}, t_meta)
    with pytest.raises(ValueError, match="row_widths"):
        ConvRotTrellisTensor.from_storage(
            {**t_tensors, "row_widths": torch.ones(8, dtype=torch.int32)},
            t_meta,
        )


def test_nvfp4_from_hp_scale_and_w4a16_linear_boundaries() -> None:
    with pytest.raises(ValueError, match="high-precision weight"):
        PiperNVFP4Tensor.from_hp(torch.randn(16, dtype=torch.bfloat16))

    dense = torch.randn(32, 64, dtype=torch.bfloat16)
    for cls, extra in ((PiperNVFP4Tensor, {}), (ConvRotNVFP4Tensor, {"group_size": 16})):
        with pytest.raises(ValueError, match="act_per_tensor_scale"):
            cls.from_hp(
                dense, act_per_tensor_scale=torch.tensor(0.5, dtype=torch.bfloat16), **extra
            )
        with pytest.raises(ValueError, match="act_per_tensor_scale"):
            cls.from_hp(dense, act_per_tensor_scale=torch.empty((), device="meta"), **extra)
        with pytest.raises(ValueError, match="per_tensor_scale"):
            cls.from_hp(dense, per_tensor_scale=torch.ones(2, dtype=torch.float32), **extra)
        with pytest.raises(ValueError, match="both compute and receive"):
            cls.from_hp(
                dense,
                per_tensor_scale=torch.tensor(1.0),
                compute_per_tensor_scale=True,
                **extra,
            )

    dummy_gguf = torch.zeros(32, 32, dtype=torch.uint8)
    for bad_act_scale in (
        torch.tensor(0.5, dtype=torch.bfloat16),
        torch.empty((), device="meta"),
        torch.ones(1, dtype=torch.float32),
    ):
        with pytest.raises(ValueError, match="act_per_tensor_scale"):
            ConvRotNVFP4Tensor.from_gguf(
                dummy_gguf, group_size=16, act_per_tensor_scale=bad_act_scale
            )
    with pytest.raises(ValueError, match="act_quant"):
        ConvRotNVFP4Tensor.from_gguf(
            dummy_gguf,
            group_size=16,
            act_quant="invalid",  # type: ignore[arg-type]
        )

    w_with_act = PiperNVFP4Tensor.from_hp(
        dense,
        compute_per_tensor_scale=True,
        act_per_tensor_scale=torch.tensor(0.5),
        is_swizzled_scales=True,
        act_quant=None,
    )
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="weight-only execution does not accept activation scales"):
        F.linear(x, w_with_act)

    moe = _make_nvfp4_3d_moe()
    with pytest.raises(ValueError, match="2-D"):
        F.linear(torch.randn(2, 256, dtype=torch.bfloat16), moe)

    unswizzled = PiperNVFP4Tensor.from_hp(dense, compute_per_tensor_scale=True, act_quant=None)
    torch.testing.assert_close(
        F.linear(x, unswizzled), F.linear(x, unswizzled.dequantize()), rtol=0, atol=0
    )


def test_flat_1d_swizzled_scale_linear_and_copy_cpu() -> None:
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    for convrot in (False, True):
        base = _make_nvfp4_2d(convrot=convrot, act_quant=None)
        flat = type(base).from_storage(
            {**base.as_storage()[0], "scale": base.scale.reshape(-1)},
            base.as_storage()[1],
        )
        torch.testing.assert_close(F.linear(x, flat), F.linear(x, base), rtol=0, atol=0)

        dst_2d = base.clone()
        dst_flat = flat.clone()
        dst_flat.copy_(base)
        dst_2d.copy_(flat)
        torch.testing.assert_close(F.linear(x, dst_flat), F.linear(x, base), rtol=0, atol=0)
        torch.testing.assert_close(F.linear(x, dst_2d), F.linear(x, base), rtol=0, atol=0)


def test_nvfp4_from_quantized_contract() -> None:
    qdata = torch.randint(0, 256, (128, 128), dtype=torch.uint8)
    scale = torch.ones(_NON_EMPTY_META_SWIZZLED_SCALE_SHAPE, dtype=torch.float8_e4m3fn)
    plain = PiperNVFP4Tensor.from_quantized(
        qdata,
        scale,
        is_swizzled_scales=True,
        high_first=True,
        act_quant="dynamic",
    )
    assert plain.qdata is qdata
    assert plain.scale is scale
    assert plain.high_first is True
    assert plain.act_quant == "dynamic"

    strided = torch.arange(128 * 256, dtype=torch.uint8).reshape(128, 256)[:, ::2]
    assert not strided.is_contiguous()
    canonical = ConvRotNVFP4Tensor.from_quantized(
        strided,
        scale,
        group_size=64,
        is_swizzled_scales=True,
        act_quant=None,
    )
    assert canonical.qdata.is_contiguous()
    assert torch.equal(canonical.qdata, strided.contiguous())
    assert canonical.scale is scale
    assert canonical.act_quant is None

    for bad_act_quant in ("invalid", True):
        with pytest.raises(ValueError, match="act_quant"):
            PiperNVFP4Tensor.from_quantized(
                qdata,
                scale,
                is_swizzled_scales=True,
                act_quant=bad_act_quant,  # type: ignore[arg-type]
            )

    with pytest.raises(ValueError, match="group size"):
        ConvRotNVFP4Tensor.from_quantized(
            torch.randint(0, 256, (128, 96), dtype=torch.uint8),
            torch.ones((32, 48), dtype=torch.float8_e4m3fn),
            group_size=256,
            is_swizzled_scales=True,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm GPU")
def test_storage_boundary_round_trip_and_offload_cuda() -> None:
    torch.manual_seed(29)
    linear_weights: list[QuantizedWeight] = [
        _make_int8_2d(device="cuda", static_act=True),
        _make_trellis_uniform().to("cuda"),
        _make_trellis_variable_row().to("cuda"),
    ]
    if trellis_encoder_available():
        dense_trellis = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        trellis_from_hp = ConvRotTrellisTensor.from_hp(
            dense_trellis, group_size=64, bits_per_weight=4.0
        )
        trellis_storage = TrellisStorage.from_hp(dense_trellis, group_size=64, bits_per_weight=4.0)
        trellis_from_store = ConvRotTrellisTensor.from_trellis_storage(
            trellis_storage, group_size=64
        )
        torch.testing.assert_close(
            trellis_from_hp.dequantize(), trellis_from_store.dequantize(), rtol=0, atol=0
        )
        linear_weights.append(trellis_from_hp)
    if w4a16_nvfp4_available():
        linear_weights.extend(
            [
                _make_nvfp4_2d(device="cuda", act_quant=None, high_first=True),
                _make_nvfp4_2d(convrot=True, device="cuda", act_quant=None, high_first=True),
            ]
        )
    if exact_sm120_available():
        linear_weights.extend(
            [
                _make_nvfp4_2d(device="cuda", act_quant="static", high_first=True),
                _make_nvfp4_2d(device="cuda", act_quant="dynamic"),
                _make_nvfp4_2d(convrot=True, device="cuda", act_quant="dynamic"),
            ]
        )
    for weight in linear_weights:
        rebuilt = _assert_zero_copy_storage_round_trip(weight)
        tensors, metadata = weight.as_storage()
        copied = {name: torch.empty_like(buf).copy_(buf) for name, buf in tensors.items()}
        offloaded = type(weight).from_storage(copied, metadata)
        x = torch.randn(4, weight.shape[1], device="cuda", dtype=weight.dtype)
        bias = torch.randn(weight.shape[0], device="cuda", dtype=weight.dtype)
        expected = F.linear(x, weight, bias)
        torch.testing.assert_close(F.linear(x, rebuilt, bias), expected, rtol=0, atol=0)
        torch.testing.assert_close(F.linear(x, offloaded, bias), expected, rtol=0, atol=0)

    conv3d_weight = _make_int8_conv3d(device="cuda")
    rebuilt_conv = _assert_zero_copy_storage_round_trip(conv3d_weight)
    conv_tensors, conv_meta = conv3d_weight.as_storage()
    offloaded_conv = ConvRotInt8Tensor.from_storage(
        {name: torch.empty_like(buf).copy_(buf) for name, buf in conv_tensors.items()},
        conv_meta,
    )
    vol = torch.randn(1, 64, 5, 5, 5, device="cuda", dtype=torch.float16)
    expected_conv = F.conv3d(vol, conv3d_weight)
    torch.testing.assert_close(F.conv3d(vol, rebuilt_conv), expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(F.conv3d(vol, offloaded_conv), expected_conv, rtol=0, atol=0)


def _assert_nonfoldable_3d_views_match(
    weight: QuantizedWeight,
    *,
    compiled: bool,
) -> None:
    device = weight.device
    dtype = weight.dtype
    in_features = weight.shape[1]
    bias = torch.randn(weight.shape[0], device=device, dtype=dtype)
    layer = nn.Linear(in_features, weight.shape[0], bias=True, device="meta").eval()
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.bias = nn.Parameter(bias, requires_grad=False)

    views = (
        torch.randn(2, 10, in_features, device=device, dtype=dtype)[:, 1:6, :],
        torch.randn(2, in_features, 5, device=device, dtype=dtype).transpose(1, 2),
    )
    for x in views:
        assert not x.is_contiguous()
        with pytest.raises(RuntimeError):
            x.view(-1, in_features)
        expected = F.linear(x.contiguous(), weight, bias)
        for mode in (torch.no_grad, torch.inference_mode):
            with mode():
                torch.testing.assert_close(F.linear(x, weight, bias), expected, rtol=0, atol=0)
                torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    x @ weight.t(), F.linear(x.contiguous(), weight), rtol=0, atol=0
                )
                if compiled:
                    torch._dynamo.reset()
                    compiled_out = torch.compile(F.linear, fullgraph=True)(x, weight, bias)
                    torch.testing.assert_close(compiled_out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: _make_int8_2d(device="cpu", static_act=False),
        lambda: _make_int8_2d(device="cpu", static_act=True),
        lambda: _make_nvfp4_2d(device="cpu", act_quant=None, high_first=False),
        lambda: _make_nvfp4_2d(device="cpu", act_quant=None, high_first=True),
        lambda: _make_nvfp4_2d(convrot=True, device="cpu", act_quant=None, high_first=True),
    ],
)
def test_nonfoldable_3d_activation_views_cpu(builder: Callable[[], QuantizedWeight]) -> None:
    _assert_nonfoldable_3d_views_match(builder(), compiled=False)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm GPU")
@pytest.mark.parametrize(
    ("requirement", "builder"),
    [
        ("cuda", lambda: _make_int8_2d(device="cuda", static_act=False)),
        ("cuda", lambda: _make_trellis_uniform().to("cuda")),
        ("cuda", lambda: _make_trellis_variable_row().to("cuda")),
        ("w4a16", lambda: _make_nvfp4_2d(device="cuda", act_quant=None, high_first=True)),
        ("sm120", lambda: _make_nvfp4_2d(device="cuda", act_quant="static", high_first=True)),
        ("sm120", lambda: _make_nvfp4_2d(device="cuda", act_quant="dynamic")),
        (
            "w4a16",
            lambda: _make_nvfp4_2d(convrot=True, device="cuda", act_quant=None, high_first=True),
        ),
        ("sm120", lambda: _make_nvfp4_2d(convrot=True, device="cuda", act_quant="dynamic")),
        (
            "trellis_encode",
            lambda: ConvRotTrellisTensor.from_hp(
                torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.2,
                group_size=64,
                bits_per_weight=4.0,
            ),
        ),
    ],
)
def test_nonfoldable_3d_activation_views_cuda(
    requirement: str, builder: Callable[[], QuantizedWeight]
) -> None:
    if requirement == "sm120" and not exact_sm120_available():
        pytest.skip("requires exact NVIDIA SM120")
    if requirement == "w4a16" and not w4a16_nvfp4_available():
        pytest.skip("requires W4A16 NVFP4 support")
    if requirement == "trellis_encode" and not trellis_encoder_available():
        pytest.skip("requires NVIDIA SM80+ Trellis encoding support")
    torch.manual_seed(31)
    _assert_nonfoldable_3d_views_match(builder(), compiled=True)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_nvfp4_flat_scales_and_w4a16_boundaries_cuda() -> None:
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    for convrot in (False, True):
        for act_quant in (None, "static", "dynamic"):
            base = _make_nvfp4_2d(convrot=convrot, device="cuda", act_quant=act_quant)
            tensors, metadata = base.as_storage()
            flat = type(base).from_storage(
                {**tensors, "scale": tensors["scale"].reshape(-1)}, metadata
            )
            torch.testing.assert_close(F.linear(x, flat), F.linear(x, base), rtol=0, atol=0)

    w_with_act = PiperNVFP4Tensor.from_hp(
        torch.randn(32, 64, device="cuda", dtype=torch.bfloat16),
        compute_per_tensor_scale=True,
        act_per_tensor_scale=torch.tensor(0.5, device="cuda"),
        is_swizzled_scales=True,
        act_quant=None,
    )
    with pytest.raises(ValueError, match="weight-only execution does not accept activation scales"):
        F.linear(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16), w_with_act)
