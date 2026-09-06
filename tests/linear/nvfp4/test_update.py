"""Numerics, mutation, and workspace coverage for fused NVFP4 weight updates."""

import pytest
import torch
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.linear.convrot._rotation import rotate_groups
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor, _layout
from piper_kernels.linear.nvfp4._backend import triton_backend
from piper_kernels.linear.nvfp4.tensor import _MIN_PER_TENSOR_SCALE

_CUDA_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
gpu = [pytest.mark.gpu, pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires Blackwell CUDA")]


def _weight(
    group_size=0,
    *,
    dtype=torch.bfloat16,
    two_level=True,
    swizzled=True,
    high_first=False,
    rows=33,
    features=768,
    device="cuda",
):
    generator = torch.Generator().manual_seed(511)
    source = torch.randn(rows, features, dtype=torch.float32, generator=generator)
    source = source.to(torch.bfloat16 if dtype is torch.float16 else dtype)
    cls = ConvRotNVFP4Tensor if group_size else PiperNVFP4Tensor
    weight = cls.from_hp(
        source,
        compute_per_tensor_scale=two_level,
        is_swizzled_scales=swizzled,
        **({"group_size": group_size} if group_size else {}),
    ).to(device=device, dtype=dtype)
    if high_first:
        weight.qdata.copy_(_layout.swap_packed_pairs(weight.qdata))
        weight.high_first = True
    return weight


def _operands(weight, operation, *, rank=19, strided=False):
    generator = torch.Generator(device=weight.device).manual_seed(901)

    def operand(rows, columns):
        value = torch.randn(
            (columns, rows) if strided else (rows, columns),
            device=weight.device,
            dtype=weight.dtype,
            generator=generator,
        )
        return value.t() if strided else value

    rows, features = weight.shape
    if operation == "add":
        return (operand(rows, features),)
    return operand(rows, rank), operand(rank, features)


def _merged(weight, operands, operation, alpha, beta):
    base = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    group_size = getattr(weight, "group_size", 0)
    update = rotate_groups(operands[-1].float(), group_size) if group_size else operands[-1].float()
    if operation == "addmm" and group_size and weight.device.type == "cuda":
        # The optimized dot deliberately retains native FP16/BF16 operands.
        # Model that hardware boundary, but keep dequantization and merging FP32.
        update = update.to(operands[-1].dtype).float()
    if operation == "add":
        return torch.add(base, update, alpha=alpha)
    # Use FP32 accumulation as the oracle instead of inheriting cuBLAS's
    # reduced-precision BF16 GEMM choices before the NVFP4 quantization boundary.
    return torch.addmm(
        base.float(),
        operands[0].float(),
        update.float(),
        alpha=alpha,
        beta=beta,
    )


def _apply(weight, operands, operation, *, alpha=0.375, beta=0.25, seed=None):
    kwargs = {"alpha": alpha}
    if seed is not None:
        kwargs["rounding_seed"] = seed
    if operation == "addmm":
        kwargs["beta"] = beta
    return getattr(weight, f"{operation}_")(*operands, **kwargs)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
def test_merge_retains_update_smaller_than_bfloat16_ulp(device):
    weight = PiperNVFP4Tensor.from_hp(
        torch.ones(1, 16, device=device, dtype=torch.bfloat16),
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
    )
    base = weight.dequantize(torch.float32)
    previous_scale = weight.per_tensor_scale.clone()
    update = torch.ones_like(base, dtype=torch.bfloat16)
    expected_scale = per_tensor_amax_to_scale((base + 2**-10).abs().amax())

    weight.add_(update, alpha=2**-10)

    assert weight.per_tensor_scale > previous_scale
    torch.testing.assert_close(weight.per_tensor_scale, expected_scale, rtol=2e-7, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
@pytest.mark.parametrize("operation", ["add", "addmm"])
@pytest.mark.parametrize(
    ("group_size", "dtype"),
    [(0, torch.float32)]
    + [(group, dtype) for group in (0, 16, 64, 256) for dtype in (torch.bfloat16, torch.float16)],
)
def test_update_matches_quantized_dense_reference(device, operation, group_size, dtype):
    weight = _weight(group_size, dtype=dtype, device=device)
    operands = _operands(weight, operation, strided=True)
    merged = _merged(weight, operands, operation, 0.375, 0.25)
    global_scale = per_tensor_amax_to_scale(merged.float().abs().amax()).clamp_min(
        _MIN_PER_TENSOR_SCALE
    )
    expected = TorchAONVFP4Tensor.to_nvfp4(
        merged.float(),
        per_tensor_scale=global_scale,
        is_swizzled_scales=True,
    )
    storages = (weight.qdata, weight.scale, weight.per_tensor_scale)
    versions = tuple(value._version for value in storages)

    assert _apply(weight, operands, operation) is weight

    assert all(
        actual is old
        for actual, old in zip(
            (weight.qdata, weight.scale, weight.per_tensor_scale),
            storages,
            strict=True,
        )
    )
    assert all(value._version > version for value, version in zip(storages, versions, strict=True))
    torch.testing.assert_close(weight.per_tensor_scale, global_scale, rtol=0.008, atol=1e-7)
    actual = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    expected_dense = expected.dequantize(torch.float32)
    # Different GEMM/rotation reduction orders can move a value across a code boundary.
    relative_rmse = (
        actual - expected_dense
    ).square().mean().sqrt() / expected_dense.square().mean().sqrt()
    assert relative_rmse < 0.035
    assert (weight.qdata != expected.qdata).float().mean() < 0.015


@pytest.mark.parametrize("operation", ["add", "addmm"])
@pytest.mark.parametrize("group_size", [0, 64])
@pytest.mark.parametrize("two_level", [False, True])
@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.parametrize("high_first", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires Blackwell CUDA")
def test_seeded_updates_preserve_layout_scales_and_rng(
    operation,
    group_size,
    two_level,
    swizzled,
    high_first,
):
    weight = _weight(group_size, two_level=two_level, swizzled=swizzled, high_first=high_first)
    operands = _operands(weight, operation, rank=65, strided=True)
    replay, other, deterministic = (weight.clone() for _ in range(3))
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    seed = (1 << 64) - 1

    _apply(weight, operands, operation, seed=seed)
    _apply(replay, operands, operation, seed=seed)
    _apply(other, operands, operation, seed=seed - 1)
    _apply(deterministic, operands, operation)

    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert torch.equal(weight.qdata, replay.qdata)
    assert not torch.equal(weight.qdata, other.qdata)
    assert not torch.equal(weight.qdata, deterministic.qdata)
    for result in (replay, other, deterministic):
        assert torch.equal(weight.scale.view(torch.uint8), result.scale.view(torch.uint8))
        if two_level:
            assert torch.equal(weight.per_tensor_scale, result.per_tensor_scale)
        else:
            assert result.per_tensor_scale is None
    merged = _merged(
        _weight(group_size, two_level=two_level, swizzled=swizzled, high_first=high_first),
        operands,
        operation,
        0.375,
        0.25,
    )
    actual = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    assert (actual - merged).square().mean().sqrt() / merged.float().square().mean().sqrt() < 0.2


@pytest.mark.parametrize("group_size", [0, 256])
@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires Blackwell CUDA")
def test_zero_coefficients_skip_nonfinite_inputs_and_no_op_preserves_versions(group_size):
    weight = _weight(group_size)
    operands = tuple(value.fill_(float("nan")) for value in _operands(weight, "addmm"))
    versions = (weight.qdata._version, weight.scale._version, weight.per_tensor_scale._version)
    weight.addmm_(*operands, alpha=0, beta=1, rounding_seed=1)
    assert versions == (
        weight.qdata._version,
        weight.scale._version,
        weight.per_tensor_scale._version,
    )
    weight.addmm_(*operands, alpha=0, beta=0, rounding_seed=1)
    assert torch.equal(weight.qdata, torch.zeros_like(weight.qdata))
    assert torch.isfinite(weight.per_tensor_scale).all()
    assert (weight.per_tensor_scale > 0).all()
    assert not weight.dequantize().any()


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires Blackwell CUDA")
def test_stochastic_rounding_is_unbiased_and_preserves_exact_codes():
    weight = _weight(two_level=False, swizzled=False, dtype=torch.float32, rows=8192, features=16)
    weight.qdata.zero_()
    levels = torch.tensor(
        [
            0.125,
            0.375,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            -0.125,
            -0.375,
            -0.75,
            -1.25,
            -2.5,
            -5.0,
            0.0,
            6.0,
        ],
        device="cuda",
    )
    weight.add_(levels.expand(weight.shape), rounding_seed=765)
    actual = weight.dequantize()
    torch.testing.assert_close(actual.mean(dim=0), levels, atol=0.04, rtol=0)
    assert (actual[:, -2] == 0).all()
    assert (actual[:, -1] == 6).all()
    assert (weight.scale.float() == 1).all()


@pytest.mark.parametrize("operation", ["add", "addmm"])
@pytest.mark.parametrize(
    ("group_size", "rows", "features"), [(0, 1, 16), (16, 17, 48), (64, 129, 192)]
)
@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires Blackwell CUDA")
def test_partial_tiles_and_negative_coefficients(operation, group_size, rows, features, swizzled):
    weight = _weight(group_size, rows=rows, features=features, swizzled=swizzled)
    operands = _operands(weight, operation, rank=3, strided=True)
    merged = _merged(weight, operands, operation, -0.5, -1)
    expected = TorchAONVFP4Tensor.to_nvfp4(
        merged.float(),
        per_tensor_scale=per_tensor_amax_to_scale(merged.abs().amax()),
        is_swizzled_scales=swizzled,
    )

    _apply(weight, operands, operation, alpha=-0.5, beta=-1)

    actual = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    error = (actual - merged.float()).square().mean().sqrt()
    expected_error = (expected.dequantize(torch.float32) - merged.float()).square().mean().sqrt()
    assert error <= expected_error * 1.05 + 1e-6
    if swizzled:
        # Both valid and padded lanes retain canonical hardware scale storage.
        assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))


@pytest.mark.parametrize("group_size", [0, 64])
@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires Blackwell CUDA")
def test_addmm_zero_rank_and_zero_beta(group_size):
    weight = _weight(group_size)
    empty = _operands(weight, "addmm", rank=0)
    before = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    weight.addmm_(*empty, beta=-1)
    after = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    torch.testing.assert_close(after, -before, atol=0.04, rtol=0.02)

    operands = _operands(weight, "addmm")
    weight.scale.fill_(float("nan"))
    weight.per_tensor_scale.fill_(float("nan"))
    weight.addmm_(*operands, beta=0, rounding_seed=3)
    assert torch.isfinite(weight.dequantize()).all()
    assert weight.dequantize().any()


@pytest.mark.parametrize("operation", ["add", "addmm"])
@pytest.mark.parametrize("group_size", [0, 256])
@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires Blackwell CUDA")
def test_stochastic_workspace_does_not_materialize_dense_weights(operation, group_size):
    weight = _weight(group_size, rows=2048, features=2048)
    operands = _operands(weight, operation, rank=16)
    _apply(weight, operands, operation, seed=123)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    _apply(weight, operands, operation, seed=123)
    torch.cuda.synchronize()

    peak_scratch = torch.cuda.max_memory_allocated() - baseline
    # A single BF16 weight is 8 MiB; the fused path only allocates tile maxima.
    assert peak_scratch < 256 * 1024


@pytest.mark.parametrize("operation", ["add", "addmm"])
@pytest.mark.parametrize("group_size", [0, 64])
@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires Blackwell CUDA")
def test_compiled_update_preserves_mutation(operation, group_size):
    weight = _weight(group_size)
    expected = weight.clone()
    operands = _operands(weight, operation)
    _apply(expected, operands, operation)

    def update(value, *inputs):
        return _apply(value, inputs, operation)

    result = torch.compile(update, fullgraph=True)(weight, *operands)

    assert result is weight
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))


@pytest.mark.parametrize("operation", ["add", "addmm"])
@pytest.mark.parametrize("group_size", [0, 64])
@pytest.mark.parametrize("two_level", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires Blackwell CUDA")
def test_update_custom_op_fake_and_aot_dispatch(operation, group_size, two_level):
    weight = _weight(group_size, two_level=two_level)
    operands = _operands(weight, operation, strided=True)
    coefficients = (0.25, 0.375) if operation == "addmm" else (0.375,)
    assert triton_backend is not None

    result = torch.library.opcheck(
        getattr(triton_backend, f"{operation}_"),
        (
            weight.qdata,
            weight.scale,
            weight.per_tensor_scale,
            *operands,
            group_size,
            *coefficients,
            weight.is_swizzled_scales,
            weight.high_first,
            123,
        ),
        # PyTorch's schema checker uses allclose, which lacks FP8 CUDA support.
        # The end-to-end update tests check storage mutation directly.
        test_utils=("test_faketensor", "test_aot_dispatch_dynamic"),
    )

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.parametrize("operation", ["add", "addmm"])
@pytest.mark.parametrize("group_size", [0, 64])
def test_aten_dispatch_and_invalid_inputs(operation, group_size):
    weight = _weight(group_size, device="cpu")
    expected = weight.clone()
    operands = _operands(weight, operation)
    _apply(expected, operands, operation, alpha=1, beta=1)
    op = torch.ops.aten.add_.Tensor if operation == "add" else torch.ops.aten.addmm_.default
    assert op(weight, *operands) is weight
    assert torch.equal(weight.qdata, expected.qdata)
    with pytest.raises(ValueError, match="logical dtype"):
        _apply(weight, tuple(value.float() for value in operands), operation)
    with pytest.raises((TypeError, ValueError), match="unsigned 64-bit"):
        _apply(weight, operands, operation, seed=-1)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        _apply(weight, tuple(value.requires_grad_() for value in operands), operation)
