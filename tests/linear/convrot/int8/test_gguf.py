"""Tests for direct GGUF-to-ConvRot-INT8 conversion."""

from unittest.mock import Mock

import pytest
import torch
from gguf_format._fixtures import dequantize_reference, finite_packed

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.gguf import GGUFQuantizationType
from piper_kernels.linear.convrot.int8 import ConvRotInt8Tensor, _backend, _gguf
from piper_kernels.linear.convrot.int8 import triton as legacy_triton
from piper_kernels.linear.convrot.int8._generic import triton as generic_triton
from piper_kernels.linear.convrot.int8._kernels import triton as kernels

_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.version.hip is not None or not torch.cuda.is_available(), reason="requires NVIDIA CUDA"
)
@pytest.mark.parametrize("quant_type", list(GGUFQuantizationType))
def test_nvidia_conversion_preserves_exact_bf16_reference(quant_type):
    torch.manual_seed(820 + int(quant_type))
    packed = finite_packed(quant_type)
    dense = dequantize_reference(packed, quant_type).cuda()
    expected = ConvRotInt8Tensor.from_hp(dense, group_size=64)
    actual = ConvRotInt8Tensor.from_gguf(packed.cuda(), quant_type=quant_type, group_size=64)
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("quant_type", list(GGUFQuantizationType))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("tiled", [False, True])
def test_from_gguf_matches_materialized_reference(
    monkeypatch, quant_type, dtype, group_size, tiled
):
    if tiled:
        monkeypatch.setattr(generic_triton, "select_conversion_chunks", lambda target, width: None)
    torch.manual_seed(820 + int(quant_type))
    packed = finite_packed(quant_type, rows=3, features=1280)
    dense = dequantize_reference(packed, quant_type, dtype=dtype).cuda()

    expected = ConvRotInt8Tensor.from_hp(dense, group_size=group_size)
    actual = ConvRotInt8Tensor.from_gguf(
        packed.cuda(),
        quant_type=quant_type,
        group_size=group_size,
        logical_dtype=dtype,
    )

    assert (actual.qdata.short() - expected.qdata.short()).abs().max().item() <= 1
    torch.testing.assert_close(actual.scale, expected.scale, rtol=2e-6, atol=1e-30)
    torch.testing.assert_close(
        actual.dequantize(torch.float32),
        expected.dequantize(torch.float32),
        rtol=0,
        atol=float(expected.scale.max()) * group_size**0.5 * 1.01,
    )


@pytest.mark.gpu
@_gpu
def test_copy_from_gguf_refills_existing_storage() -> None:
    first = finite_packed(GGUFQuantizationType.Q4_K)
    second = finite_packed(GGUFQuantizationType.Q4_K)
    weight = ConvRotInt8Tensor.from_gguf(
        first.cuda(),
        quant_type=GGUFQuantizationType.Q4_K,
        group_size=64,
    )
    expected = ConvRotInt8Tensor.from_gguf(
        second.cuda(),
        quant_type=GGUFQuantizationType.Q4_K,
        group_size=64,
    )
    qdata = weight.qdata
    scale = weight.scale

    result = weight.copy_from_gguf_(second.cuda(), quant_type=GGUFQuantizationType.Q4_K)

    assert result is weight
    assert weight.qdata is qdata
    assert weight.scale is scale
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale, expected.scale)


def test_from_gguf_rejects_cpu_storage() -> None:
    packed = finite_packed(GGUFQuantizationType.Q5_1)
    attributed = packed.as_subclass(torch.Tensor)
    attributed.quant_type = int(GGUFQuantizationType.Q5_1)  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="requires a supported Triton accelerator"):
        ConvRotInt8Tensor.from_gguf(attributed, group_size=64)


@pytest.mark.gpu
@_gpu
def test_from_gguf_reads_quant_type_attribute() -> None:
    packed = finite_packed(GGUFQuantizationType.Q5_1).cuda()
    packed.quant_type = int(GGUFQuantizationType.Q5_1)  # type: ignore[attr-defined]

    actual = ConvRotInt8Tensor.from_gguf(packed, group_size=64)
    expected = ConvRotInt8Tensor.from_gguf(
        packed,
        quant_type=GGUFQuantizationType.Q5_1,
        group_size=64,
    )

    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)


@pytest.mark.parametrize("architecture", ["gfx1036", "gfx942", "gfx1201", "gfx9999"])
def test_generic_conversion_does_not_require_a_tuned_amd_target(monkeypatch, architecture):
    monkeypatch.setattr(
        AcceleratorTarget, "from_device", lambda device: AcceleratorTarget("hip", architecture)
    )
    monkeypatch.setattr(generic_triton, "supports_device", lambda device: True)
    assert _backend.select_gguf_converter(torch.empty(1)) is generic_triton.convert_gguf_out
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    assert _backend.select_gguf_converter(torch.empty(1)) is generic_triton.convert_gguf_out


def test_conversion_requires_a_matching_triton_driver(monkeypatch):
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    monkeypatch.setattr(generic_triton, "supports_device", lambda device: False)
    assert _backend.select_gguf_converter(torch.empty(1)) is None
    monkeypatch.setattr(_backend, "_generic_backend", None)
    assert _backend.select_gguf_converter(torch.empty(1)) is None


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("quant_type", "width"),
    [
        (GGUFQuantizationType.F32, 16),
        (GGUFQuantizationType.Q4_0, 32),
        (GGUFQuantizationType.Q4_K, 768),
        (GGUFQuantizationType.Q5_K, 5376),
        (GGUFQuantizationType.Q6_K, 32768),
        (GGUFQuantizationType.IQ4_XS, 65792),
    ],
)
def test_generic_conversion_widths_and_bounded_batches(monkeypatch, quant_type, width, dtype):
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    monkeypatch.setattr(_backend, "_amd_backend", None)
    monkeypatch.setattr(generic_triton, "select_conversion_chunks", lambda target, width: None)
    # Force repeated batches, including a one-row tail and maxima larger than the budget.
    monkeypatch.setattr(generic_triton, "_GGUF_MAXIMA_BYTES", 48)
    torch.manual_seed(75)
    packed = finite_packed(quant_type, rows=5, features=width)
    dense = dequantize_reference(packed, quant_type, dtype=dtype).cuda()
    expected = ConvRotInt8Tensor.from_hp(dense, group_size=16)
    device_packed = packed.cuda()
    # Packed inputs may be noncontiguous; output views must keep their surrounding canaries.
    strided = torch.zeros(5, device_packed.shape[1] * 2, device="cuda", dtype=torch.uint8)
    strided[:, ::2] = device_packed
    q_storage = torch.full((5 * width + 8,), 99, device="cuda", dtype=torch.int8)
    s_storage = torch.full((13,), -99.0, device="cuda")
    outputs = (q_storage[4:-4].reshape(5, width), s_storage[4:-4].reshape(5, 1))
    actual = _gguf.convert(
        strided[:, ::2], quant_type=quant_type, group_size=16, logical_dtype=dtype, out=outputs
    )
    assert actual[0] is outputs[0]
    assert actual[1] is outputs[1]
    assert (actual[0].short() - expected.qdata.short()).abs().max().item() <= 1
    torch.testing.assert_close(actual[1], expected.scale, rtol=2e-6, atol=1e-30)
    assert (q_storage[:4] == 99).all()
    assert (q_storage[-4:] == 99).all()
    assert (s_storage[:4] == -99).all()
    assert (s_storage[-4:] == -99).all()


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("shape", [(0, 256), (3, 0)])
def test_empty_conversion_does_not_launch(monkeypatch, shape):
    def unexpected(*args):
        raise AssertionError("empty GGUF conversion launched a kernel")

    monkeypatch.setattr(_backend, "select_gguf_converter", lambda value: unexpected)
    packed = torch.empty(shape, device="cuda", dtype=torch.float32)
    weight = ConvRotInt8Tensor.from_gguf(packed, quant_type=GGUFQuantizationType.F32, group_size=16)
    assert weight.qdata.shape == shape
    assert (weight.scale == 1e-30).all()
    assert weight.copy_from_gguf_(packed, quant_type=GGUFQuantizationType.F32) is weight


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("magnitude", [0.0, 1e-6, 1e-31])
@pytest.mark.parametrize("tiled", [False, True])
def test_generic_conversion_zero_and_tiny_scales(monkeypatch, dtype, magnitude, tiled):
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    if tiled:
        monkeypatch.setattr(generic_triton, "select_conversion_chunks", lambda target, width: None)
    dense = torch.zeros(3, 1280, device="cuda", dtype=dtype)
    dense[:, 0] = magnitude
    expected = ConvRotInt8Tensor.from_hp(dense, group_size=256)
    actual = ConvRotInt8Tensor.from_gguf(
        dense.float(), quant_type=GGUFQuantizationType.F32, group_size=256, logical_dtype=dtype
    )
    assert torch.equal(actual.qdata, expected.qdata)
    torch.testing.assert_close(actual.scale, expected.scale, rtol=1e-6, atol=0)
    assert actual.scale.isfinite().all()


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("tiled", [False, True])
def test_generic_conversion_does_not_materialize_dense_weights(monkeypatch, tiled):
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    if tiled:
        monkeypatch.setattr(generic_triton, "select_conversion_chunks", lambda target, width: None)
    packed = finite_packed(GGUFQuantizationType.Q4_K, rows=1, features=8192)
    packed = packed.expand(1024, -1).contiguous().cuda()
    weight = ConvRotInt8Tensor.from_gguf(
        packed, quant_type=GGUFQuantizationType.Q4_K, group_size=64
    )
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    weight.copy_from_gguf_(packed, quant_type=GGUFQuantizationType.Q4_K)
    torch.cuda.synchronize()
    # A dense BF16 weight is 16 MiB. Only the tiled fallback retains a maxima buffer.
    assert torch.cuda.max_memory_allocated() - baseline < 2 * 1024 * 1024


def test_legacy_conversion_exports_point_to_shared_implementations():
    assert legacy_triton._convert_gguf_out is generic_triton.convert_gguf_out
    assert legacy_triton.rotate_quantize_rows_kernel is kernels.rotate_quantize_rows_kernel


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("width", [5376, 6144, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("quant_type", list(GGUFQuantizationType))
def test_shared_fused_conversion_matches_tiled_multichunk_rows(
    monkeypatch, width, dtype, quant_type
):
    torch.manual_seed(740)
    packed = finite_packed(quant_type, rows=3, features=width).cuda()
    # Both paths are generic; disable tuned linear implementations to verify independence.
    monkeypatch.setattr(_backend, "_nvidia_backend", None)
    monkeypatch.setattr(_backend, "_amd_backend", None)
    fused = Mock(wraps=generic_triton.rotate_quantize_rows_kernel)
    # JIT kernels launch through __getitem__, so record the selected grid explicitly.
    launch = Mock(wraps=generic_triton.rotate_quantize_rows_kernel.__getitem__)
    fused.__getitem__ = launch
    monkeypatch.setattr(generic_triton, "rotate_quantize_rows_kernel", fused)
    actual = ConvRotInt8Tensor.from_gguf(
        packed, quant_type=quant_type, group_size=256, logical_dtype=dtype
    )
    launch.assert_called_once_with((3,))
    monkeypatch.setattr(generic_triton, "select_conversion_chunks", lambda target, width: None)
    expected = ConvRotInt8Tensor.from_gguf(
        packed, quant_type=quant_type, group_size=256, logical_dtype=dtype
    )
    assert (actual.qdata.short() - expected.qdata.short()).abs().max().item() <= 1
    torch.testing.assert_close(actual.scale, expected.scale, rtol=2e-6, atol=1e-30)


@pytest.mark.gpu
@_gpu
def test_shared_conversion_keeps_wide_row_fallback(monkeypatch):
    monkeypatch.setattr(
        generic_triton, "rotate_quantize_rows_kernel", Mock(side_effect=AssertionError("fused"))
    )
    width = 65792 if torch.version.hip is None else 16384
    packed = finite_packed(GGUFQuantizationType.Q4_K, features=width).cuda()
    weight = ConvRotInt8Tensor.from_gguf(
        packed, quant_type=GGUFQuantizationType.Q4_K, group_size=256
    )
    assert weight.qdata.shape == (2, width)
    assert weight.scale.isfinite().all()


@pytest.mark.parametrize("invalid", ["shape", "dtype", "noncontiguous", "scale"])
def test_conversion_validates_output_storage_before_execution(monkeypatch, invalid):
    def unexpected(*args):
        raise AssertionError("invalid output reached the converter")

    monkeypatch.setattr(_backend, "select_gguf_converter", lambda value: unexpected)
    packed = finite_packed(GGUFQuantizationType.Q4_K)
    qdata = torch.empty(2, 256, dtype=torch.int8)
    scale = torch.empty(2, 1)
    if invalid == "shape":
        qdata = qdata[:1]
    elif invalid == "dtype":
        qdata = qdata.float()
    elif invalid == "noncontiguous":
        qdata = torch.empty(2, 512, dtype=torch.int8)[:, ::2]
    else:
        scale = scale.double()
    with pytest.raises(ValueError, match="output storage is incompatible"):
        _gguf.convert(
            packed,
            quant_type=GGUFQuantizationType.Q4_K,
            group_size=64,
            logical_dtype=torch.bfloat16,
            out=(qdata, scale),
        )
