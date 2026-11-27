"""Tests for ``ConvRotTrellisTensor`` and hybrid ``F.linear`` execution."""

from __future__ import annotations

import io
import re
import sys

import pytest
import torch
import triton
from _accelerators import convrot_int8_w8a8_available, rdna4_available
from _compile_capture import TargetCapturePass
from _nvidia_resources import capture_cuda_spill_stores
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F  # noqa: N812
from triton.backends.compiler import GPUTarget

from piper_kernels import compile_options
from piper_kernels.linear.convrot import convrot_int8_linear
from piper_kernels.linear.convrot.int8 import reference as convrot_reference
from piper_kernels.linear.convrot.int8.trellis_triton import compile_trellis_kernels, linear_a16
from piper_kernels.linear.convrot.int8.trellis_triton import linear as triton_trellis_linear
from piper_kernels.trellis import TrellisHessian, TrellisLayout, TrellisStorage, resolve_layout
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor

_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="trellis encoding requires an NVIDIA GPU",
)
_multi_gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >= 2 GPUs")


def _hip_target_param(arch: str, warp_size: int) -> object:
    return pytest.param(
        GPUTarget("hip", arch, warp_size),
        marks=pytest.mark.skipif(sys.platform != "linux", reason="HIP compiler is Linux-only"),
        id=f"hip-{arch}",
    )


_PRIMARY_ISA_TARGETS = [
    pytest.param(GPUTarget("cuda", 75, 32), id="cuda-75"),
    pytest.param(GPUTarget("cuda", 80, 32), id="cuda-80"),
    pytest.param(GPUTarget("cuda", 120, 32), id="cuda-120"),
    _hip_target_param("gfx942", 64),
    _hip_target_param("gfx1201", 32),
]

_SECONDARY_ISA_TARGETS = [
    pytest.param(GPUTarget("cuda", 86, 32), id="cuda-86"),
    pytest.param(GPUTarget("cuda", 89, 32), id="cuda-89"),
    _hip_target_param("gfx1100", 32),
    _hip_target_param("gfx1151", 32),
    _hip_target_param("gfx1200", 32),
]


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float32:
        return (2e-4, 2e-4)
    if dtype is torch.float16:
        return (6e-3, 6e-3)
    return (2.5e-2, 2.5e-2)


def _finite_e4m3(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    codes = torch.randint(0x18, 0x48, shape, dtype=torch.uint8, generator=generator)
    return codes.view(torch.float8_e4m3fn)


def _make_trellis_inputs(
    out_features: int,
    in_features: int,
    *,
    block_size: int = 256,
    bits_per_weight: float = 4.5,
    bytes_per_block: int | None = None,
    row_widths: list[int] | None = None,
    seed: int = 101,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert in_features % block_size == 0
    blocks_per_row = in_features // block_size
    generator = torch.Generator().manual_seed(seed)
    block_scales = _finite_e4m3((out_features, blocks_per_row), generator)
    scale = (torch.rand(out_features, 1, generator=generator) * 0.03 + 0.005) / max(
        1.0, (in_features / 256.0) ** 0.5
    )
    if row_widths is not None:
        assert len(row_widths) == out_features
        widths = torch.tensor(row_widths, dtype=torch.uint8)
        for width in row_widths:
            TrellisLayout.from_storage(block_size, width)
        bitstreams = torch.randint(
            0, 256, (blocks_per_row * sum(row_widths),), dtype=torch.uint8, generator=generator
        )
        return bitstreams, block_scales, scale, widths

    if bytes_per_block is None:
        bytes_per_block = resolve_layout(block_size, bits_per_weight).bytes_per_block
    TrellisLayout.from_storage(block_size, bytes_per_block)
    bitstreams = torch.randint(
        0,
        256,
        (out_features, blocks_per_row, bytes_per_block),
        dtype=torch.uint8,
        generator=generator,
    )
    return bitstreams, block_scales, scale, None


def _make_weight(
    out_features: int,
    in_features: int,
    *,
    block_size: int = 256,
    group_size: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    bits_per_weight: float = 4.5,
    bytes_per_block: int | None = None,
    row_widths: list[int] | None = None,
    seed: int = 101,
) -> ConvRotTrellisTensor:
    bitstreams, block_scales, scale, widths = _make_trellis_inputs(
        out_features,
        in_features,
        block_size=block_size,
        bits_per_weight=bits_per_weight,
        bytes_per_block=bytes_per_block,
        row_widths=row_widths,
        seed=seed,
    )
    return ConvRotTrellisTensor.from_quantized(
        bitstreams.to(device),
        block_scales.to(device),
        scale.to(device),
        block_size=block_size,
        group_size=group_size,
        logical_dtype=dtype,
        row_widths=None if widths is None else widths.to(device),
    )


def _variable_row_widths(out_features: int, block_size: int) -> list[int]:
    lo = (2 * block_size + 7) // 8 + 2
    hi = (6 * block_size) // 8 + 2
    mid = (lo + hi) // 2
    choices = (lo, mid, hi)
    return [choices[idx % len(choices)] for idx in range(out_features)]


# ---------------------------------------------------------------------------
# Subclass, validation, views, meta/FakeTensor, and portable CPU contracts
# ---------------------------------------------------------------------------


def test_from_quantized_reuses_contiguous_storage_and_canonicalizes_strided() -> None:
    bitstreams, block_scales, scale, _ = _make_trellis_inputs(8, 256, block_size=128)
    flat_scale = scale.view(-1)
    reused = ConvRotTrellisTensor.from_quantized(
        bitstreams,
        block_scales,
        flat_scale,
        block_size=128,
        group_size=64,
        logical_dtype=torch.float16,
    )
    assert reused.shape == (8, 256)
    assert reused.dtype is torch.float16
    assert reused.scale.shape == (8, 1)
    assert reused.bitstreams.data_ptr() == bitstreams.data_ptr()
    assert reused.block_scales.data_ptr() == block_scales.data_ptr()
    assert reused.scale.data_ptr() == flat_scale.data_ptr()

    widths = torch.tensor(_variable_row_widths(8, 128), dtype=torch.uint8)
    var_bs = torch.randint(0, 256, (2 * int(widths.sum().item()),), dtype=torch.uint8)
    var_reused = ConvRotTrellisTensor.from_quantized(
        var_bs, block_scales, flat_scale, block_size=128, group_size=64, row_widths=widths
    )
    assert var_reused.bitstreams.data_ptr() == var_bs.data_ptr()
    assert var_reused.row_widths is not None
    assert var_reused.row_widths.data_ptr() == widths.data_ptr()

    strided_bs = torch.stack((bitstreams, bitstreams), dim=-1)[..., 0]
    strided_sc = block_scales.t().contiguous().t()
    strided_scale = torch.stack((flat_scale, flat_scale), dim=1)[:, 0]
    assert not strided_bs.is_contiguous()
    assert not strided_scale.is_contiguous()
    canon = ConvRotTrellisTensor.from_quantized(
        strided_bs, strided_sc, strided_scale, block_size=128, group_size=64
    )
    assert canon.bitstreams.is_contiguous()
    assert canon.block_scales.is_contiguous()
    assert canon.scale.is_contiguous()
    assert torch.equal(canon.dequantize(torch.float32), reused.dequantize(torch.float32))


def test_from_storage_and_from_hp_cpu_rejection() -> None:
    bitstreams, block_scales, scale, _ = _make_trellis_inputs(4, 256, block_size=256)
    storage = TrellisStorage(bitstreams, block_scales, scale, block_size=256)
    wrapped = ConvRotTrellisTensor.from_trellis_storage(
        storage, group_size=64, logical_dtype=torch.float32
    )
    assert wrapped.shape == (4, 256)
    assert wrapped.dtype is torch.float32
    assert wrapped.bitstreams.data_ptr() == bitstreams.data_ptr()

    with pytest.raises(ValueError, match="NVIDIA CUDA device"):
        ConvRotTrellisTensor.from_hp(
            torch.randn(4, 256, dtype=torch.bfloat16), group_size=64, bits_per_weight=4.5
        )
    with pytest.raises(ValueError, match="hessian_damping requires hessian"):
        ConvRotTrellisTensor.from_hp(
            torch.randn(4, 256, dtype=torch.bfloat16),
            group_size=64,
            bits_per_weight=4.5,
            hessian_damping=0.01,
        )


@pytest.mark.gpu
@_nvidia
def test_gpu_from_hp_forwards_hessian() -> None:
    torch.manual_seed(995)
    weight = (torch.randn(16, 512, dtype=torch.bfloat16, device="cuda") * 0.02).contiguous()
    hessian = torch.eye(512, device="cuda") + 0.25
    factor = TrellisHessian.from_hessian(hessian, group_size=256)
    direct = ConvRotTrellisTensor.from_hp(
        weight, group_size=256, bits_per_weight=3.0, hessian=factor
    )
    storage = TrellisStorage.from_hp(weight, group_size=256, bits_per_weight=3.0, hessian=factor)
    assert torch.equal(direct.bitstreams, storage.bitstreams)
    assert torch.equal(direct.block_scales, storage.block_scales)
    assert torch.equal(direct.scale, storage.scale)
    x = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda") * 0.1
    rtol, atol = _tolerance(torch.bfloat16)
    torch.testing.assert_close(
        F.linear(x, direct), F.linear(x, direct.dequantize()), rtol=rtol, atol=atol
    )


def test_views_transpose_conversion_serialization_and_matmul_contracts() -> None:
    weight = _make_weight(
        7,
        256,
        block_size=128,
        group_size=64,
        dtype=torch.bfloat16,
        row_widths=_variable_row_widths(7, 128),
    )
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    bias = torch.randn(7, dtype=torch.float32)

    assert weight.t().shape == (256, 7)
    assert weight.transpose(0, 1).transposed
    assert weight.mT.transposed
    assert weight.permute(1, 0).transposed
    assert torch._C._is_alias_of(weight.t(), weight)
    assert torch.equal(weight.t().dequantize(torch.float32), weight.dequantize(torch.float32).t())
    assert torch.equal(weight.view(weight.shape).dequantize(), weight.dequantize())
    assert torch.equal(
        weight.as_strided(weight.shape, weight.stride()).dequantize(), weight.dequantize()
    )
    assert torch.equal(weight.clone().dequantize(), weight.dequantize())
    assert torch.equal(weight.detach().dequantize(), weight.dequantize())

    fp16 = weight.to(torch.float16)
    assert fp16.dtype is torch.float16
    assert fp16.bitstreams is weight.bitstreams
    copied = weight.to(dtype=torch.float32, copy=True)
    assert copied.dtype is torch.float32
    assert not torch._C._is_alias_of(copied, weight)

    buf = io.BytesIO()
    torch.save(weight.t(), buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    assert type(loaded) is ConvRotTrellisTensor
    assert loaded.transposed
    assert torch.equal(loaded.dequantize(torch.float32), weight.t().dequantize(torch.float32))

    expected = F.linear(x, weight)
    assert torch.equal(x @ weight.t(), expected)
    assert torch.equal(torch.mm(x, weight.t()), expected)
    assert torch.equal(torch.addmm(bias, x, weight.t()), F.linear(x, weight, bias))


def test_unsupported_block1_operations_raise_not_implemented() -> None:
    weight = _make_weight(4, 256, block_size=256, group_size=64)
    x = torch.randn(2, 256, dtype=torch.bfloat16)

    for call in (
        lambda: F.embedding(torch.tensor([0, 2]), weight),
        lambda: weight.add_(torch.zeros(4, 256, dtype=torch.bfloat16)),
        lambda: weight.addmm_(torch.zeros(4, 2), torch.zeros(2, 256, dtype=torch.bfloat16)),
        lambda: torch.ops.aten.add_.Tensor(weight, torch.zeros(4, 256, dtype=torch.bfloat16)),
        lambda: torch.ops.aten.addmm_.default(
            weight,
            torch.zeros(4, 2, dtype=torch.bfloat16),
            torch.zeros(2, 256, dtype=torch.bfloat16),
        ),
        lambda: weight.fsdp_pre_all_gather(None),  # type: ignore[arg-type]
        lambda: weight.fsdp_post_all_gather((), None, torch.bfloat16),
        lambda: weight[:2],
        lambda: F.linear(torch.randn(2, 4, dtype=torch.bfloat16), weight.t()),
        lambda: torch.addmm(torch.zeros(4), x, weight.t(), alpha=0.5, beta=1),
        lambda: convrot_int8_linear(
            torch.randn(2, 512, dtype=torch.bfloat16), weight, activation_fn="swiglu"
        ),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_boundary_validation_and_autograd_rejection() -> None:
    bitstreams, block_scales, scale, _ = _make_trellis_inputs(4, 256, block_size=128)

    with pytest.raises(ValueError, match="group size"):
        ConvRotTrellisTensor.from_quantized(
            bitstreams, block_scales, scale, block_size=128, group_size=32
        )
    with pytest.raises(ValueError, match="divisible by group size"):
        ConvRotTrellisTensor.from_quantized(
            bitstreams[:, :1], block_scales[:, :1], scale, block_size=128, group_size=256
        )
    with pytest.raises(ValueError, match="logical dtype"):
        ConvRotTrellisTensor.from_quantized(
            bitstreams, block_scales, scale, block_size=128, group_size=64, logical_dtype=torch.int8
        )
    with pytest.raises(ValueError, match="share a device"):
        ConvRotTrellisTensor.from_quantized(
            bitstreams.to("meta"), block_scales, scale, block_size=128, group_size=64
        )
    with pytest.raises(ValueError, match="must be contiguous"):
        ConvRotTrellisTensor(
            torch.stack((bitstreams, bitstreams), dim=-1)[..., 0],
            block_scales,
            scale,
            block_size=128,
            group_size=64,
        )

    weight = ConvRotTrellisTensor.from_quantized(
        bitstreams, block_scales, scale, block_size=128, group_size=64, logical_dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="expected 256"):
        F.linear(torch.randn(2, 128, dtype=torch.bfloat16), weight)
    with pytest.raises(ValueError, match="logical dtype"):
        F.linear(torch.randn(2, 256, dtype=torch.float32), weight)
    with pytest.raises(ValueError, match="bias must have shape"):
        F.linear(torch.randn(2, 256, dtype=torch.bfloat16), weight, torch.zeros(3))
    with pytest.raises(ValueError, match="bias dtype"):
        F.linear(
            torch.randn(2, 256, dtype=torch.bfloat16),
            weight,
            torch.zeros(4, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="dequantization output dtype"):
        weight.dequantize(torch.int8)

    grad_input = torch.randn(2, 256, dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        F.linear(grad_input, weight)

    grad_scale = scale.clone().requires_grad_(True)
    w_grad_scale = ConvRotTrellisTensor.from_quantized(
        bitstreams, block_scales, grad_scale, block_size=128, group_size=64
    )
    with pytest.raises(RuntimeError, match="does not support autograd"):
        F.linear(torch.randn(2, 256, dtype=torch.bfloat16), w_grad_scale)

    grad_bias = torch.zeros(4, dtype=torch.float32, requires_grad=True)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        F.linear(torch.randn(2, 256, dtype=torch.bfloat16), weight, grad_bias)


@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable"])
def test_meta_and_fake_tensor_shape_inference(variable_rows: bool) -> None:
    row_widths = _variable_row_widths(6, 128) if variable_rows else None
    meta_weight = _make_weight(
        6,
        256,
        block_size=128,
        group_size=64,
        dtype=torch.bfloat16,
        device="meta",
        row_widths=row_widths,
    )
    meta_x = torch.empty(2, 3, 256, dtype=torch.bfloat16, device="meta")
    meta_bias = torch.empty(6, dtype=torch.float32, device="meta")

    assert meta_weight.dequantize(torch.float16).shape == (6, 256)
    assert meta_weight.dequantize(torch.float16).dtype is torch.float16
    assert meta_weight.t().dequantize().shape == (256, 6)
    meta_out = F.linear(meta_x, meta_weight, meta_bias)
    assert meta_out.device.type == "meta"
    assert meta_out.dtype is torch.bfloat16
    assert meta_out.shape == (2, 3, 6)

    with FakeTensorMode():
        fake_w = _make_weight(
            6,
            256,
            block_size=128,
            group_size=64,
            dtype=torch.float16,
            device="cpu",
            row_widths=row_widths,
        )
        fake_x = torch.empty(5, 256, dtype=torch.float16)
        fake_out = F.linear(fake_x, fake_w)
        assert fake_out.shape == (5, 6)
        assert fake_out.dtype is torch.float16
        assert fake_w.dequantize().shape == (6, 256)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable"])
@pytest.mark.parametrize(
    ("block_size", "in_features", "group_size"),
    [(256, 256, 64), (224, 448, 16), (16, 64, 16)],
)
def test_cpu_linear_matches_dequantized_reference(
    dtype: torch.dtype,
    variable_rows: bool,
    block_size: int,
    in_features: int,
    group_size: int,
) -> None:
    out_features = 9
    row_widths = _variable_row_widths(out_features, block_size) if variable_rows else None
    weight = _make_weight(
        out_features,
        in_features,
        block_size=block_size,
        group_size=group_size,
        dtype=dtype,
        row_widths=row_widths,
        seed=out_features + block_size,
    )
    bias = torch.randn(out_features, dtype=torch.float32)

    for input_shape in ((in_features,), (2, 3, in_features)):
        x = torch.randn(input_shape, dtype=dtype)
        actual = convrot_int8_linear(x, weight, bias)
        expected = F.linear(x.float(), weight.dequantize(torch.float32), bias).to(dtype)
        rtol, atol = _tolerance(dtype)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    # Empty dimensions
    assert F.linear(torch.empty(0, in_features, dtype=dtype), weight).shape == (0, out_features)
    empty_w = _make_weight(
        out_features, 0, block_size=block_size, group_size=group_size, dtype=dtype
    )
    assert torch.equal(
        F.linear(torch.randn(2, 0, dtype=dtype), empty_w, bias),
        bias.to(dtype).expand(2, out_features),
    )


def test_cpu_autocast_and_aot_eager_compile() -> None:
    weight = _make_weight(6, 256, block_size=128, group_size=64, dtype=torch.float32)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        auto_out = F.linear(x, weight)
    assert auto_out.dtype is torch.bfloat16
    torch.testing.assert_close(
        auto_out,
        F.linear(x, weight.to(torch.bfloat16)),
        rtol=0,
        atol=0,
    )

    torch.compiler.reset()
    bf16_weight = weight.to(torch.bfloat16)
    compiled = torch.compile(
        lambda inp: F.linear(inp, bf16_weight),
        backend="aot_eager",
        fullgraph=True,
        dynamic=True,
    )
    for rows in (3, 12):
        inp = torch.randn(rows, 256, dtype=torch.bfloat16)
        torch.testing.assert_close(compiled(inp), F.linear(inp, bf16_weight), rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, _gpu])],
)
def test_trellis_swiglu_ffn_compile_skips_int8_fx_passes(device: str) -> None:
    w_up = _make_weight(
        64, 128, block_size=128, group_size=64, dtype=torch.bfloat16, device=device, seed=141
    )
    w_gate = _make_weight(
        64, 128, block_size=128, group_size=64, dtype=torch.bfloat16, device=device, seed=142
    )
    w_down = _make_weight(
        128, 64, block_size=64, group_size=64, dtype=torch.bfloat16, device=device, seed=143
    )

    def trellis_ffn(inp: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(inp, w_gate)) * F.linear(inp, w_up), w_down)

    torch.manual_seed(144)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device=device) * 0.1
    eager = trellis_ffn(x)

    torch.compiler.reset()
    capture = TargetCapturePass()
    compiled = torch.compile(
        trellis_ffn,
        fullgraph=True,
        options=capture.wrap_options(compile_options(ffn=True, attention=True)),
    )
    actual = compiled(x)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, eager, rtol=1e-2, atol=1e-2)
    piper_targets = [
        target
        for target in capture.targets
        if getattr(target, "namespace", None) == "piper_kernels"
    ]
    assert piper_targets == [torch.ops.piper_kernels.convrot_int8_linear.default] * 3


def test_copy_and_pin_contracts() -> None:
    dst = _make_weight(
        7,
        256,
        block_size=128,
        group_size=64,
        dtype=torch.bfloat16,
        bytes_per_block=72,
        seed=151,
    )
    src_fp16 = _make_weight(
        7,
        256,
        block_size=128,
        group_size=64,
        dtype=torch.float16,
        bytes_per_block=72,
        seed=152,
    )
    ret = dst.copy_(src_fp16)
    assert ret is dst
    assert dst.dtype is torch.bfloat16
    assert not torch._C._is_alias_of(dst.bitstreams, src_fp16.bitstreams)
    assert torch.equal(dst.dequantize(torch.float32), src_fp16.dequantize(torch.float32))

    # Variable-row copy_ copies row_widths as well as packed payloads.
    widths_dst = [48, 72, 96, 60, 72, 84, 72]
    widths_src = list(reversed(widths_dst))
    dst_var = _make_weight(7, 256, block_size=128, group_size=64, row_widths=widths_dst, seed=153)
    src_var = _make_weight(7, 256, block_size=128, group_size=64, row_widths=widths_src, seed=154)
    dst_var.copy_(src_var)
    assert dst_var.row_widths is not None
    assert src_var.row_widths is not None
    assert torch.equal(dst_var.row_widths, src_var.row_widths)
    assert torch.equal(dst_var.dequantize(torch.float32), src_var.dequantize(torch.float32))

    assert dst.is_pinned() is False
    assert dst_var.is_pinned() is False

    # Unsupported copies
    dense = torch.zeros(7, 256, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        dst.copy_(dense)
    with pytest.raises(NotImplementedError):
        dense.copy_(dst)

    snap_bs = dst.bitstreams.clone()
    snap_sc = dst.block_scales.view(torch.uint8).clone()
    snap_scale = dst.scale.clone()
    snap_var_rw = dst_var.row_widths.clone()
    snap_var_scale = dst_var.scale.clone()

    for bad_src, err in (
        (
            _make_weight(7, 256, block_size=128, group_size=16, bytes_per_block=72),
            NotImplementedError,
        ),
        (
            _make_weight(7, 256, block_size=64, group_size=64, bytes_per_block=36),
            NotImplementedError,
        ),
        (src_fp16.t(), NotImplementedError),
        (_make_weight(7, 256, block_size=128, group_size=64, bytes_per_block=56), ValueError),
        (dst_var, ValueError),
    ):
        with pytest.raises(err):
            dst.copy_(bad_src)

    with pytest.raises(ValueError, match="storage tensor"):
        dst_var.copy_(
            _make_weight(
                7,
                256,
                block_size=128,
                group_size=64,
                row_widths=[48] * 7,
                seed=155,
            )
        )
    assert torch.equal(dst.bitstreams, snap_bs)
    assert torch.equal(dst.block_scales.view(torch.uint8), snap_sc)
    assert torch.equal(dst.scale, snap_scale)
    assert torch.equal(dst_var.row_widths, snap_var_rw)
    assert torch.equal(dst_var.scale, snap_var_scale)


# ---------------------------------------------------------------------------
# Offline multi-target compile and register-spill checks
# ---------------------------------------------------------------------------


def _compile_and_assert_zero_spills(
    target: GPUTarget,
    *,
    expect_splitk: bool,
    **kwargs: object,
) -> dict[str, triton.compiler.CompiledKernel]:
    spills_by_hash, compile_fn = capture_cuda_spill_stores()
    kernels = compile_trellis_kernels(target, compile_fn=compile_fn, **kwargs)  # type: ignore[arg-type]
    assert "gemv" in kernels
    assert ("reduce" in kernels) is expect_splitk
    for name, compiled in kernels.items():
        asm = compiled.asm
        if target.backend == "hip":
            hsaco = asm["hsaco"]
            amdgcn = asm["amdgcn"]
            assert hsaco
            match = re.search(r"\.vgpr_spill_count:\s*(\d+)", amdgcn)
            assert match is not None, f"missing .vgpr_spill_count in {target.arch} {name}"
            assert int(match.group(1)) == 0, f"{target.arch} {name} spilled {match.group(1)} VGPRs"
            priv = re.search(r"\.private_segment_fixed_size:\s*(\d+)", amdgcn)
            assert priv is not None, f"missing .private_segment_fixed_size in {target.arch} {name}"
            assert int(priv.group(1)) == 0, f"{target.arch} {name} used private memory"
            if name == "gemv":
                assert ("v_sad_u8" in amdgcn) == (target.arch in ("gfx1200", "gfx1201"))
        else:
            assert asm["ptx"]
            spills = spills_by_hash[compiled.hash]
            assert spills == 0, f"sm{target.arch} {name} spilled {spills} bytes"
    return kernels


@pytest.mark.parametrize("target", _PRIMARY_ISA_TARGETS)
@pytest.mark.parametrize(
    (
        "rows",
        "out_features",
        "input_features",
        "block_size",
        "variable_rows",
        "input_dtype",
        "bias_dtype",
        "expect_splitk",
    ),
    [
        (1, 4096, 4096, 256, False, torch.bfloat16, None, False),
        (4, 4096, 2688, 224, True, torch.bfloat16, torch.bfloat16, False),
        (8, 4096, 4096, 256, False, torch.float16, torch.float16, False),
        (1, 64, 65536, 256, True, torch.bfloat16, None, True),
    ],
    ids=["m1-pow2-uniform", "m4-nonpow2-var", "m8-fp16-bias", "m1-splitk-var-nobias"],
)
def test_trellis_gemv_compiles_without_spills_primary_targets(
    target: GPUTarget,
    rows: int,
    out_features: int,
    input_features: int,
    block_size: int,
    variable_rows: bool,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
    expect_splitk: bool,
) -> None:
    _compile_and_assert_zero_spills(
        target,
        expect_splitk=expect_splitk,
        rows=rows,
        out_features=out_features,
        input_features=input_features,
        block_size=block_size,
        input_dtype=input_dtype,
        bias_dtype=bias_dtype,
        variable_rows=variable_rows,
        num_sms=64,
    )


@pytest.mark.parametrize("target", _SECONDARY_ISA_TARGETS)
def test_trellis_gemv_compiles_without_spills_secondary_targets(target: GPUTarget) -> None:
    _compile_and_assert_zero_spills(
        target,
        expect_splitk=True,
        rows=8,
        out_features=64,
        input_features=65408,  # 292 * 224
        block_size=224,
        input_dtype=torch.bfloat16,
        bias_dtype=torch.bfloat16,
        variable_rows=True,
        num_sms=64,
    )


@pytest.mark.parametrize(
    (
        "rows",
        "out_features",
        "input_features",
        "block_size",
        "bytes_per_block",
        "variable_rows",
        "expect_splitk",
    ),
    [
        (1, 14336, 4096, 256, 144, False, False),
        (1, 1024, 4096, 256, 144, False, False),
        (2, 4096, 4096, 256, 144, False, True),
        (1, 4096, 4096, 256, 194, False, True),
        (4, 4096, 2688, 224, None, True, False),
        (8, 4096, 14336, 256, 144, False, True),
        (8, 64, 65536, 256, 143, False, True),
        (1, 4096, 4096, 128, 72, False, True),
        (4, 1024, 512, 64, 36, False, False),
        (17, 64, 256, 128, None, True, False),
        (1, 77, 512, 256, 143, False, False),
        (1, 256, 256, 16, 9, False, False),
        (1, 64, 400, 200, None, False, False),
        (1, 64, 400, 200, None, True, False),
    ],
    ids=[
        "m1-14336x4096-u2",
        "m1-1024x4096-no-splitk",
        "m2-4096x4096-u2-splitk",
        "m1-4096x4096-6bpw-3word-splitk",
        "m4-bs224-var-3word",
        "m8-4096x14336-splitk",
        "m8-64x65536-splitk",
        "m1-bs128-splitk",
        "m4-bs64",
        "m17-direct-a16",
        "m1-unaligned-bytes",
        "m1-bs16",
        "m1-bs200-default-layout",
        "m1-bs200-var-default-layout",
    ],
)
def test_trellis_sm120_regimes_compile_without_spills(
    rows: int,
    out_features: int,
    input_features: int,
    block_size: int,
    bytes_per_block: int | None,
    variable_rows: bool,
    expect_splitk: bool,
) -> None:
    kernels = _compile_and_assert_zero_spills(
        GPUTarget("cuda", 120, 32),
        expect_splitk=expect_splitk,
        rows=rows,
        out_features=out_features,
        input_features=input_features,
        block_size=block_size,
        input_dtype=torch.bfloat16,
        bias_dtype=torch.bfloat16,
        variable_rows=variable_rows,
        bytes_per_block=bytes_per_block,
        num_sms=170,
    )
    assert "dp4a.s32.u32" in kernels["gemv"].asm["ptx"]
    if (
        block_size == triton.next_power_of_2(block_size)
        and block_size % 4 == 0
        and input_features % 4 == 0
    ):
        assert "ld.global.v4.b32" in kernels["gemv"].asm["ptx"]


@pytest.mark.skipif(sys.platform != "linux", reason="HIP compiler is Linux-only")
@pytest.mark.parametrize(
    (
        "rows",
        "out_features",
        "input_features",
        "block_size",
        "bytes_per_block",
        "variable_rows",
        "expect_splitk",
    ),
    [
        (1, 14336, 4096, 256, 143, False, False),
        (2, 4096, 4096, 256, 144, False, False),
        (4, 4096, 2688, 224, None, True, False),
        (8, 4096, 14336, 256, 143, False, False),
        (8, 64, 65408, 224, None, True, True),
        (4, 77, 400, 200, None, False, False),
        (4, 77, 400, 200, None, True, False),
        (4, 77, 51, 17, 9, False, False),
    ],
    ids=[
        "m1-14336x4096",
        "m2-4096x4096-bn2",
        "m4-bs224-bn2-var-w2",
        "m8-4096x14336-bn2",
        "m8-64x65408-bn2-splitk",
        "m4-odd-n-bs200-uniform",
        "m4-odd-n-bs200-var",
        "m4-odd-n-bs17-tail-xmul4-false",
    ],
)
def test_trellis_rdna4_regimes_compile_without_spills(
    rows: int,
    out_features: int,
    input_features: int,
    block_size: int,
    bytes_per_block: int | None,
    variable_rows: bool,
    expect_splitk: bool,
) -> None:
    kernels = _compile_and_assert_zero_spills(
        GPUTarget("hip", "gfx1201", 32),
        expect_splitk=expect_splitk,
        rows=rows,
        out_features=out_features,
        input_features=input_features,
        block_size=block_size,
        input_dtype=torch.bfloat16,
        bias_dtype=torch.bfloat16,
        variable_rows=variable_rows,
        bytes_per_block=bytes_per_block,
        num_sms=32,
    )
    amdgcn = kernels["gemv"].asm["amdgcn"]
    assert "v_sad_u8" in amdgcn
    lds = re.search(r"\.group_segment_fixed_size:\s*(\d+)", amdgcn)
    assert lds is not None
    assert int(lds.group(1)) == 0


# ---------------------------------------------------------------------------
# Hardware GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable"])
@pytest.mark.parametrize("block_size", [256, 224], ids=["bs256", "bs224"])
def test_gpu_dequantize_matches_cpu_reference(variable_rows: bool, block_size: int) -> None:
    out_features, in_features = 16, block_size * 4
    row_widths = _variable_row_widths(out_features, block_size) if variable_rows else None
    bitstreams, block_scales, scale, widths = _make_trellis_inputs(
        out_features, in_features, block_size=block_size, row_widths=row_widths, seed=303
    )
    gpu_bs = bitstreams.cuda()
    gpu_sc = block_scales.cuda()
    gpu_scale = scale.cuda()
    gpu_widths = None if widths is None else widths.cuda()

    w_trellis = ConvRotTrellisTensor.from_quantized(
        gpu_bs, gpu_sc, gpu_scale, block_size=block_size, group_size=16, row_widths=gpu_widths
    )
    w_cpu = ConvRotTrellisTensor.from_quantized(
        bitstreams, block_scales, scale, block_size=block_size, group_size=16, row_widths=widths
    )
    torch.testing.assert_close(
        w_trellis.dequantize(torch.float32).cpu(),
        w_cpu.dequantize(torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        w_trellis.t().dequantize(torch.float32).cpu(),
        w_cpu.t().dequantize(torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    w_int8 = ConvRotInt8Tensor.from_trellis(
        gpu_bs, gpu_sc, gpu_scale, block_size=block_size, group_size=16, row_widths=gpu_widths
    )
    diff = (w_trellis.dequantize(torch.float32) - w_int8.dequantize(torch.float32)).abs()
    norm_rms = (diff / gpu_scale.reshape(-1, 1)).pow(2).mean(dim=1).sqrt()
    assert float(diff.max().item()) > 0.0
    assert float(norm_rms.max().item()) <= 0.5 + 1e-4


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("use_bias", [False, True], ids=["nobias", "bias"])
def test_gpu_small_batch_a16_gemv_matches_reference(
    dtype: torch.dtype, rows: int, use_bias: bool
) -> None:
    out_features, in_features = 77, 512  # unaligned N tail
    weight = _make_weight(
        out_features,
        in_features,
        block_size=256,
        group_size=256,
        dtype=dtype,
        device="cuda",
        bytes_per_block=143,
        seed=400 + rows,
    )
    torch.manual_seed(400 + rows)
    x = torch.randn(rows, in_features, dtype=dtype, device="cuda") * 0.1
    bias = torch.randn(out_features, dtype=torch.float32, device="cuda") if use_bias else None

    actual = F.linear(x, weight, bias)
    expected = convrot_reference.linear_trellis_a16(
        x.cpu(),
        weight.bitstreams.cpu(),
        weight.block_scales.cpu(),
        weight.scale.cpu(),
        None,
        None if bias is None else bias.cpu(),
        weight.block_size,
        weight.group_size,
    )
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)

    if rows == 4 and rdna4_available(x.device):
        for extra_rows in (5, 8):
            torch.manual_seed(400 + extra_rows)
            x_extra = torch.randn(extra_rows, in_features, dtype=dtype, device="cuda") * 0.1
            actual_extra = F.linear(x_extra, weight, bias)
            expected_extra = convrot_reference.linear_trellis_a16(
                x_extra.cpu(),
                weight.bitstreams.cpu(),
                weight.block_scales.cpu(),
                weight.scale.cpu(),
                None,
                None if bias is None else bias.cpu(),
                weight.block_size,
                weight.group_size,
            )
            torch.testing.assert_close(actual_extra.cpu(), expected_extra, rtol=rtol, atol=atol)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_gpu_sm120_aligned_a16_gemv_matches_reference(dtype: torch.dtype, rows: int) -> None:
    out_features, in_features = 128, 512
    weight = _make_weight(
        out_features,
        in_features,
        block_size=256,
        group_size=256,
        dtype=dtype,
        device="cuda",
        bytes_per_block=144,
        seed=420 + rows,
    )
    torch.manual_seed(420 + rows)
    x = torch.randn(rows, in_features, dtype=dtype, device="cuda") * 0.1
    bias = torch.randn(out_features, dtype=dtype, device="cuda")

    actual = F.linear(x, weight, bias)
    expected = convrot_reference.linear_trellis_a16(
        x.cpu(),
        weight.bitstreams.cpu(),
        weight.block_scales.cpu(),
        weight.scale.cpu(),
        None,
        bias.cpu(),
        weight.block_size,
        weight.group_size,
    )
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize(
    (
        "out_features",
        "in_features",
        "block_size",
        "group_size",
        "bytes_per_block",
        "variable_rows",
        "rows",
    ),
    [
        (256, 512, 256, 256, 67, False, 1),
        (256, 512, 256, 256, 67, False, 4),
        (256, 512, 256, 256, 194, False, 1),
        (256, 512, 256, 256, 194, False, 4),
        (192, 2688, 224, 16, None, True, 4),
        (64, 65536, 256, 256, 143, False, 4),
    ],
    ids=[
        "m1-2.1bpw",
        "m4-2.1bpw",
        "m1-6.0bpw",
        "m4-6.0bpw",
        "m4-bs224-var-aligned",
        "m4-splitk-n64-k65536",
    ],
)
def test_gpu_sm120_aligned_a16_rates_var_and_splitk(
    out_features: int,
    in_features: int,
    block_size: int,
    group_size: int,
    bytes_per_block: int | None,
    variable_rows: bool,
    rows: int,
) -> None:
    row_widths = _variable_row_widths(out_features, block_size) if variable_rows else None
    weight = _make_weight(
        out_features,
        in_features,
        block_size=block_size,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        bytes_per_block=bytes_per_block,
        row_widths=row_widths,
        seed=520 + rows,
    )
    assert weight.bitstreams.numel() % 4 == 0
    torch.manual_seed(520 + rows)
    x = torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda") * 0.05
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")

    actual = F.linear(x, weight, bias)
    expected = convrot_reference.linear_trellis_a16(
        x.cpu(),
        weight.bitstreams.cpu(),
        weight.block_scales.cpu(),
        weight.scale.cpu(),
        None if weight.row_widths is None else weight.row_widths.cpu(),
        bias.cpu(),
        block_size,
        group_size,
    )
    rtol, atol = _tolerance(torch.bfloat16)
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize(
    ("out_features", "in_features", "block_size", "group_size", "bytes_per_block", "variable_rows"),
    [
        (128, 512, 256, 256, 66, False),  # 2.0 code bpw
        (301, 512, 256, 64, 111, False),  # 3.5 bpw + N tail
        (128, 512, 256, 256, 176, False),  # 5.5 bpw
        (96, 2688, 224, 16, None, True),  # non-pow2 block size + variable-row
        (64, 64, 16, 16, 6, False),  # minimal block_size=16 at 2.0 code bpw
        (64, 64, 16, 16, 9, False),  # minimal block_size=16
        (64, 65536, 256, 256, 143, False),  # triggers Split-K > 1 (> 2 MiB, low N)
    ],
    ids=[
        "2.0bpw",
        "3.5bpw-tail",
        "5.5bpw",
        "bs224-variable",
        "bs16-2bpw",
        "bs16",
        "splitk-n64-k65536",
    ],
)
def test_gpu_gemv_variants_rates_nonpow2_and_splitk(
    out_features: int,
    in_features: int,
    block_size: int,
    group_size: int,
    bytes_per_block: int | None,
    variable_rows: bool,
) -> None:
    row_widths = _variable_row_widths(out_features, block_size) if variable_rows else None
    weight = _make_weight(
        out_features,
        in_features,
        block_size=block_size,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        bytes_per_block=bytes_per_block,
        row_widths=row_widths,
        seed=501,
    )
    torch.manual_seed(502)
    x_full = torch.randn(4, 2 * in_features, dtype=torch.bfloat16, device="cuda")[..., ::2] * 0.05
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")

    biases = (bias, None) if in_features >= 65536 else (bias,)
    for active_bias in biases:
        for x in (x_full[0], x_full.view(2, 2, in_features)):
            actual = F.linear(x, weight, active_bias)
            expected = convrot_reference.linear_trellis_a16(
                x.cpu(),
                weight.bitstreams.cpu(),
                weight.block_scales.cpu(),
                weight.scale.cpu(),
                None if weight.row_widths is None else weight.row_widths.cpu(),
                None if active_bias is None else active_bias.cpu(),
                block_size,
                group_size,
            )
            rtol, atol = _tolerance(torch.bfloat16)
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


@pytest.mark.gpu
@_gpu
def test_gpu_a16_unaligned_stream_xmul4_false_tail_and_splitk_odd_n() -> None:
    rtol, atol = _tolerance(torch.bfloat16)

    # x_mul4=False (T=17) + unaligned bitstream pointer (1-byte offset) + odd out_features=77.
    bs_x, sc_x, scale_x, _ = _make_trellis_inputs(
        77, 272, block_size=17, bytes_per_block=9, seed=540
    )
    padded = torch.empty(bs_x.numel() + 4, dtype=torch.uint8, device="cuda")
    offset = 1 if padded.data_ptr() % 4 == 0 else 0
    bs_unaligned = padded[offset : offset + bs_x.numel()].view(bs_x.shape)
    bs_unaligned.copy_(bs_x.cuda())
    w_unaligned = ConvRotTrellisTensor.from_quantized(
        bs_unaligned,
        sc_x.cuda(),
        scale_x.cuda(),
        block_size=17,
        group_size=16,
        logical_dtype=torch.bfloat16,
    )
    bias_x = torch.randn(77, dtype=torch.bfloat16, device="cuda")
    for rows in (1, 4):
        torch.manual_seed(541 + rows)
        x = torch.randn(rows, 272, dtype=torch.bfloat16, device="cuda") * 0.1
        actual = F.linear(x, w_unaligned, bias_x)
        expected = convrot_reference.linear_trellis_a16(
            x.cpu(), bs_x, sc_x, scale_x, None, bias_x.cpu(), 17, 16
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)

    # Real partial trailing word (77 * 1 * 9 = 693 bytes; 693 % 4 == 1) across small-batch cutoffs.
    w_tail = _make_weight(
        77,
        16,
        block_size=16,
        group_size=16,
        bytes_per_block=9,
        dtype=torch.bfloat16,
        device="cuda",
        seed=545,
    )
    assert w_tail.bitstreams.numel() % 4 == 1
    bias_tail = torch.randn(77, dtype=torch.bfloat16, device="cuda")
    a16_cutoff = 8 if rdna4_available(w_tail.device) else 4
    for rows in (1, 4, 8):
        torch.manual_seed(546 + rows)
        x = torch.randn(rows, 16, dtype=torch.bfloat16, device="cuda") * 0.1
        actual = F.linear(x, w_tail, bias_tail)
        ref_fn = (
            convrot_reference.linear_trellis_a16
            if rows <= a16_cutoff or not convrot_int8_w8a8_available(x.device)
            else convrot_reference.linear_trellis_w8a8
        )
        expected = ref_fn(
            x.cpu(),
            w_tail.bitstreams.cpu(),
            w_tail.block_scales.cpu(),
            w_tail.scale.cpu(),
            None,
            bias_tail.cpu(),
            16,
            16,
        )
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=max(rtol, 2e-2), atol=max(atol, 5e-3)
        )

    # M=8, odd out_features=65, deep K=65536 (exercises block_m=8, block_n=2, split_k > 1).
    w_sk = _make_weight(
        65,
        65536,
        block_size=256,
        group_size=256,
        bytes_per_block=143,
        dtype=torch.bfloat16,
        device="cuda",
        seed=550,
    )
    torch.manual_seed(551)
    x_sk = torch.randn(8, 65536, dtype=torch.bfloat16, device="cuda") * 0.05
    bias_sk = torch.randn(65, dtype=torch.bfloat16, device="cuda")
    actual_a16 = linear_a16(
        x_sk, w_sk.bitstreams, w_sk.block_scales, w_sk.scale, None, bias_sk, 256, 256
    )
    expected_sk = convrot_reference.linear_trellis_a16(
        x_sk.cpu(),
        w_sk.bitstreams.cpu(),
        w_sk.block_scales.cpu(),
        w_sk.scale.cpu(),
        None,
        bias_sk.cpu(),
        256,
        256,
    )
    torch.testing.assert_close(actual_a16.cpu(), expected_sk, rtol=rtol, atol=atol)
    if rdna4_available(x_sk.device):
        torch.testing.assert_close(
            F.linear(x_sk, w_sk, bias_sk).cpu(), expected_sk, rtol=rtol, atol=atol
        )


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("rows", [5, 8, 9, 16, 32, 64])
@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)
def test_gpu_large_batch_w8a8_prefill_matches_int8(
    rows: int, variable_rows: bool, dtype: torch.dtype
) -> None:
    block_size, out_features, in_features, group_size = (
        (224, 160, 448, 16) if variable_rows else (256, 320, 512, 256)
    )
    row_widths = _variable_row_widths(out_features, block_size) if variable_rows else None
    bitstreams, block_scales, scale, widths = _make_trellis_inputs(
        out_features, in_features, block_size=block_size, row_widths=row_widths, seed=600 + rows
    )
    gpu_bs = bitstreams.cuda()
    gpu_sc = block_scales.cuda()
    gpu_scale = scale.cuda()
    gpu_widths = None if widths is None else widths.cuda()

    w_trellis = ConvRotTrellisTensor.from_quantized(
        gpu_bs,
        gpu_sc,
        gpu_scale,
        block_size=block_size,
        group_size=group_size,
        logical_dtype=dtype,
        row_widths=gpu_widths,
    )
    w_int8 = ConvRotInt8Tensor.from_trellis(
        gpu_bs,
        gpu_sc,
        gpu_scale,
        block_size=block_size,
        group_size=group_size,
        logical_dtype=dtype,
        row_widths=gpu_widths,
    )
    torch.manual_seed(600 + rows)
    x = torch.randn(rows, in_features, dtype=dtype, device="cuda") * 0.1
    bias_tensor = torch.randn(out_features, dtype=torch.float32, device="cuda")

    rtol, atol = _tolerance(dtype)
    a16_cutoff = 8 if rdna4_available(x.device) else 4
    use_a16 = rows <= a16_cutoff or not convrot_int8_w8a8_available(x.device)
    for active_bias in (None, bias_tensor):
        actual = F.linear(x, w_trellis, active_bias)
        if use_a16:
            expected_a16 = convrot_reference.linear_trellis_a16(
                x.cpu(),
                bitstreams,
                block_scales,
                scale,
                widths,
                None if active_bias is None else active_bias.cpu(),
                block_size,
                group_size,
            )
            torch.testing.assert_close(actual.cpu(), expected_a16, rtol=rtol, atol=atol)
            continue

        expected_int8 = F.linear(x, w_int8, active_bias)
        assert torch.equal(actual, expected_int8)

        expected_ref = convrot_reference.linear_trellis_w8a8(
            x.cpu(),
            bitstreams,
            block_scales,
            scale,
            widths,
            None if active_bias is None else active_bias.cpu(),
            block_size,
            group_size,
        )
        # Dynamic INT8 activation quantization can flip +/-0.5 ties between GPU and CPU FWHT.
        torch.testing.assert_close(
            actual.cpu(), expected_ref, rtol=max(rtol, 2e-2), atol=max(atol, 5e-3)
        )


@pytest.mark.gpu
@_gpu
@pytest.mark.skipif(not convrot_int8_w8a8_available(), reason="requires ConvRot INT8 W8A8 backend")
@pytest.mark.parametrize(
    ("rows", "out_features", "in_features"),
    [
        (16, 9000, 4096),
        (1024, 9000, 4096),
        (256, 4096, 8448),
    ],
    ids=["m16-32mib", "m1024-64mib", "m256-widek"],
)
@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable"])
def test_gpu_large_batch_w8a8_multi_chunk_matches_int8(
    rows: int, out_features: int, in_features: int, variable_rows: bool
) -> None:
    # Exercises both 32 MiB multi-chunk and 64 MiB single-chunk scratch budgets across targets.
    block_size, group_size = 256, 256
    row_widths = _variable_row_widths(out_features, block_size) if variable_rows else None
    bitstreams, block_scales, scale, widths = _make_trellis_inputs(
        out_features, in_features, block_size=block_size, row_widths=row_widths, seed=707 + rows
    )
    gpu_bs = bitstreams.cuda()
    gpu_sc = block_scales.cuda()
    gpu_scale = scale.cuda()
    gpu_widths = None if widths is None else widths.cuda()

    w_trellis = ConvRotTrellisTensor.from_quantized(
        gpu_bs,
        gpu_sc,
        gpu_scale,
        block_size=block_size,
        group_size=group_size,
        row_widths=gpu_widths,
    )
    w_int8 = ConvRotInt8Tensor.from_trellis(
        gpu_bs,
        gpu_sc,
        gpu_scale,
        block_size=block_size,
        group_size=group_size,
        row_widths=gpu_widths,
    )
    torch.manual_seed(708 + rows)
    x = torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda") * 0.05
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")

    actual = F.linear(x, w_trellis, bias)
    expected = F.linear(x, w_int8, bias)
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@_gpu
def test_gpu_direct_a16_large_batch_fallback_and_truncated_variable_bounds() -> None:
    weight = _make_weight(
        64,
        256,
        block_size=128,
        group_size=64,
        dtype=torch.bfloat16,
        device="cuda",
        row_widths=_variable_row_widths(64, 128),
        seed=801,
    )
    for rows in (8, 17):
        torch.manual_seed(802 + rows)
        x = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda") * 0.1
        bias = torch.randn(64, dtype=torch.bfloat16, device="cuda")
        actual = linear_a16(
            x,
            weight.bitstreams,
            weight.block_scales,
            weight.scale,
            weight.row_widths,
            bias,
            weight.block_size,
            weight.group_size,
        )
        expected = convrot_reference.linear_trellis_a16(
            x.cpu(),
            weight.bitstreams.cpu(),
            weight.block_scales.cpu(),
            weight.scale.cpu(),
            weight.row_widths.cpu(),  # type: ignore[union-attr]
            bias.cpu(),
            weight.block_size,
            weight.group_size,
        )
        rtol, atol = _tolerance(torch.bfloat16)
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)

    # Truncated variable-row bitstreams must clamp safely without device fault in both regimes.
    for num_bytes in (0, 1):
        trunc = ConvRotTrellisTensor.from_quantized(
            torch.zeros(num_bytes, dtype=torch.uint8, device="cuda"),
            torch.zeros(1, 1, dtype=torch.float8_e4m3fn, device="cuda"),
            torch.ones(1, 1, dtype=torch.float32, device="cuda"),
            block_size=16,
            group_size=16,
            row_widths=torch.tensor([6], dtype=torch.uint8, device="cuda"),
        )
        for rows in (1, 16):
            out = F.linear(torch.ones(rows, 16, dtype=torch.bfloat16, device="cuda"), trunc)
            assert out.shape == (rows, 1)
            torch.cuda.synchronize()


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable"])
def test_gpu_pin_memory_h2d_copy_and_direct_zero_k(variable_rows: bool) -> None:
    row_widths = _variable_row_widths(16, 128) if variable_rows else None
    cpu_w = _make_weight(
        16,
        256,
        block_size=128,
        group_size=64,
        dtype=torch.bfloat16,
        row_widths=row_widths,
        seed=820,
    )
    pinned = cpu_w.pin_memory()
    assert pinned.is_pinned() is True
    gpu_dst = _make_weight(
        16,
        256,
        block_size=128,
        group_size=64,
        dtype=torch.bfloat16,
        device="cuda",
        row_widths=list(reversed(row_widths)) if row_widths is not None else None,
        seed=821,
    )
    gpu_dst.copy_(pinned, non_blocking=True)
    torch.cuda.synchronize()
    assert gpu_dst.is_pinned() is False
    assert torch.equal(gpu_dst.bitstreams.cpu(), pinned.bitstreams)
    assert torch.equal(
        gpu_dst.block_scales.view(torch.uint8).cpu(), pinned.block_scales.view(torch.uint8)
    )
    assert torch.equal(gpu_dst.scale.cpu(), pinned.scale)
    if variable_rows:
        assert gpu_dst.row_widths is not None
        assert pinned.row_widths is not None
        assert torch.equal(gpu_dst.row_widths.cpu(), pinned.row_widths)
    torch.testing.assert_close(
        gpu_dst.dequantize(torch.float32).cpu(),
        pinned.dequantize(torch.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    # Direct Triton entrypoints must handle K == 0 cleanly without ZeroDivisionError.
    if not variable_rows:
        empty_w = _make_weight(8, 0, block_size=128, group_size=64, device="cuda", seed=822)
        bias = torch.randn(8, dtype=torch.float32, device="cuda")
        for rows in (1, 16):
            x = torch.empty(rows, 0, dtype=torch.bfloat16, device="cuda")
            for fn in (linear_a16, triton_trellis_linear):
                assert torch.equal(
                    fn(
                        x,
                        empty_w.bitstreams,
                        empty_w.block_scales,
                        empty_w.scale,
                        None,
                        None,
                        empty_w.block_size,
                        empty_w.group_size,
                    ),
                    torch.zeros(rows, 8, dtype=torch.bfloat16, device="cuda"),
                )
                assert torch.equal(
                    fn(
                        x,
                        empty_w.bitstreams,
                        empty_w.block_scales,
                        empty_w.scale,
                        None,
                        bias,
                        empty_w.block_size,
                        empty_w.group_size,
                    ),
                    bias.to(torch.bfloat16).expand(rows, 8),
                )


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable"])
def test_gpu_torch_compile_dynamic_single_graph(variable_rows: bool) -> None:
    out_features, in_features, block_size, group_size = 128, 512, 256, 256
    row_widths = _variable_row_widths(out_features, block_size) if variable_rows else None
    weight = _make_weight(
        out_features,
        in_features,
        block_size=block_size,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        row_widths=row_widths,
        seed=901,
    )
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")

    torch.compiler.reset()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(
        lambda inp: F.linear(inp, weight, bias),
        backend=counter,
        fullgraph=True,
        dynamic=True,
    )
    for rows in (3, 4, 5, 8, 9, 16, 32):
        x = torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda") * 0.1
        torch._dynamo.mark_dynamic(x, 0)
        actual = compiled(x)
        expected = F.linear(x, weight, bias)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert counter.frame_count == 1

    # Also verify M=1 under dynamic compilation.
    x1 = torch.randn(1, in_features, dtype=torch.bfloat16, device="cuda") * 0.1
    torch.testing.assert_close(compiled(x1), F.linear(x1, weight, bias), rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("variable_rows", [False, True], ids=["uniform", "variable"])
@pytest.mark.parametrize("rows", [1, 32], ids=["gemv-m1", "prefill-m32"])
def test_gpu_cuda_graph_capture_and_replay(variable_rows: bool, rows: int) -> None:
    out_features, in_features, block_size, group_size = 128, 512, 256, 256
    row_widths = _variable_row_widths(out_features, block_size) if variable_rows else None
    weight = _make_weight(
        out_features,
        in_features,
        block_size=block_size,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        row_widths=row_widths,
        seed=950 + rows,
    )
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")
    static_x = torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda") * 0.1
    static_out = torch.empty(rows, out_features, dtype=torch.bfloat16, device="cuda")

    # Warmup before capture
    F.linear(static_x, weight, bias)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out.copy_(F.linear(static_x, weight, bias))

    for step in range(2):
        torch.manual_seed(960 + step)
        static_x.copy_(torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda") * 0.1)
        graph.replay()
        expected = F.linear(static_x, weight, bias)
        torch.testing.assert_close(static_out, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@_multi_gpu
def test_gpu_executes_cleanly_on_non_current_device() -> None:
    torch.cuda.set_device(0)
    weight = _make_weight(
        96,
        512,
        block_size=256,
        group_size=256,
        dtype=torch.bfloat16,
        device="cuda:1",
        row_widths=_variable_row_widths(96, 256),
        seed=990,
    )
    bias = torch.randn(96, dtype=torch.bfloat16, device="cuda:1")
    deq = weight.dequantize()
    assert deq.device == torch.device("cuda:1")
    assert torch.cuda.current_device() == 0

    a16_cutoff = 8 if rdna4_available(weight.device) else 4
    has_w8a8 = convrot_int8_w8a8_available(weight.device)
    for rows in (1, 16):
        x = torch.randn(rows, 512, dtype=torch.bfloat16, device="cuda:1") * 0.1
        out = F.linear(x, weight, bias)
        assert out.device == torch.device("cuda:1")
        assert torch.cuda.current_device() == 0
        expected = (
            convrot_reference.linear_trellis_w8a8(
                x.cpu(),
                weight.bitstreams.cpu(),
                weight.block_scales.cpu(),
                weight.scale.cpu(),
                weight.row_widths.cpu(),  # type: ignore[union-attr]
                bias.cpu(),
                weight.block_size,
                weight.group_size,
            )
            if rows > a16_cutoff and has_w8a8
            else convrot_reference.linear_trellis_a16(
                x.cpu(),
                weight.bitstreams.cpu(),
                weight.block_scales.cpu(),
                weight.scale.cpu(),
                weight.row_widths.cpu(),  # type: ignore[union-attr]
                bias.cpu(),
                weight.block_size,
                weight.group_size,
            )
        )
        rtol, atol = _tolerance(torch.bfloat16)
        torch.testing.assert_close(out.cpu(), expected, rtol=rtol, atol=atol)
