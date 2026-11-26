"""Tests for the semantic ConvRot NVFP4 tensor."""

from __future__ import annotations

import pytest
import torch
from _accelerators import exact_sm120_available
from _compile_capture import TargetCapturePass
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F  # noqa: N812

from piper_kernels import compile_options
from piper_kernels.linear.convrot.nvfp4 import convrot_nvfp4_linear
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import reference as nvfp4_reference
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import NVFP4ActQuant, PiperNVFP4Tensor
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout


def _act_quant(dynamic: bool) -> NVFP4ActQuant:
    return "dynamic" if dynamic else "static"


def _meta_weight(*, group_size: int = 16) -> ConvRotNVFP4Tensor:
    output_features, input_features = 128, 256
    return ConvRotNVFP4Tensor.from_quantized(
        torch.empty(
            output_features,
            input_features // 2,
            dtype=torch.uint8,
            device="meta",
        ),
        torch.empty(
            nvfp4_layout.scale_shape(output_features, input_features),
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
        group_size=group_size,
        logical_dtype=torch.bfloat16,
        per_tensor_scale=torch.empty((), device="meta"),
        is_swizzled_scales=True,
        act_quant=_act_quant(True),
    )


def test_from_quantized_reuses_storage_and_validates_group_size() -> None:
    source = _meta_weight()
    rebuilt = ConvRotNVFP4Tensor.from_quantized(
        source.qdata,
        source.scale,
        group_size=16,
        logical_dtype=source.dtype,
        per_tensor_scale=source.per_tensor_scale,
        is_swizzled_scales=True,
        act_quant=source.act_quant,
    )

    assert type(rebuilt) is ConvRotNVFP4Tensor
    assert rebuilt.qdata is source.qdata
    assert rebuilt.scale is source.scale
    assert rebuilt.per_tensor_scale is source.per_tensor_scale
    assert rebuilt.group_size == 16
    with pytest.raises(ValueError, match="group size"):
        ConvRotNVFP4Tensor.from_quantized(source.qdata, source.scale, group_size=12)
    with pytest.raises(TypeError, match="group_size"):
        ConvRotNVFP4Tensor.from_quantized(source.qdata, source.scale)  # type: ignore[call-arg]


def test_high_first_preserves_storage_and_rotation_metadata() -> None:
    canonical = torch.arange(32, dtype=torch.uint8).reshape(1, 32)
    high_first = (((canonical & 0x0F) << 4) | (canonical >> 4)).to(torch.uint8)

    weight = ConvRotNVFP4Tensor(
        high_first,
        torch.ones(4, dtype=torch.float8_e4m3fn),
        16,
        torch.bfloat16,
        64,
        high_first=True,
    )

    assert type(weight) is ConvRotNVFP4Tensor
    assert weight.group_size == 64
    assert weight.qdata is high_first
    assert weight.high_first is True


@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_from_hp_matches_explicit_rotation_and_quantization(group_size: int) -> None:
    torch.manual_seed(610 + group_size)
    logical_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    rotated_weight = rotate_groups(logical_weight.float(), group_size)
    per_tensor_scale = nvfp4_layout.per_tensor_amax_to_scale(rotated_weight.abs().amax())
    activation_scale = torch.tensor(0.5)

    weight = ConvRotNVFP4Tensor.from_hp(
        logical_weight.requires_grad_(),
        group_size=group_size,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant=_act_quant(False),
    )
    expected = PiperNVFP4Tensor.from_hp(
        rotated_weight,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant=_act_quant(False),
    )

    assert type(weight) is ConvRotNVFP4Tensor
    assert weight.group_size == group_size
    assert not weight.requires_grad
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))
    assert weight.per_tensor_scale is per_tensor_scale
    assert weight.act_per_tensor_scale is activation_scale
    assert weight.act_quant == expected.act_quant


def test_from_hp_computes_global_scale_in_the_rotated_basis() -> None:
    torch.manual_seed(874)
    logical_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    rotated_weight = rotate_groups(logical_weight.float(), 64)
    expected_scale = nvfp4_layout.per_tensor_amax_to_scale(rotated_weight.float().abs().amax())

    weight = ConvRotNVFP4Tensor.from_hp(
        logical_weight,
        group_size=64,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_quant=_act_quant(True),
    )

    assert weight.per_tensor_scale is not None
    assert torch.equal(weight.per_tensor_scale, expected_scale)


def test_from_hp_computed_global_scale_handles_an_all_zero_weight() -> None:
    weight = ConvRotNVFP4Tensor.from_hp(
        torch.zeros(128, 256, dtype=torch.bfloat16),
        group_size=64,
        compute_per_tensor_scale=True,
    )

    assert weight.per_tensor_scale is not None
    assert torch.isfinite(weight.per_tensor_scale)
    assert weight.per_tensor_scale > 0
    assert not torch.count_nonzero(weight.qdata)


def test_from_hp_rejects_ambiguous_per_tensor_scale_configuration() -> None:
    with pytest.raises(ValueError, match="both compute and receive"):
        ConvRotNVFP4Tensor.from_hp(
            torch.zeros(128, 256, dtype=torch.bfloat16),
            group_size=64,
            per_tensor_scale=torch.ones(()),
            compute_per_tensor_scale=True,
        )


def test_from_hp_handles_meta_fake_and_post_rotation_overflow() -> None:
    meta_weight = ConvRotNVFP4Tensor.from_hp(
        torch.empty(64, 64, dtype=torch.bfloat16, device="meta"),
        group_size=64,
    )
    assert meta_weight.shape == (64, 64)
    assert meta_weight.device.type == "meta"

    with FakeTensorMode():
        fake_weight = ConvRotNVFP4Tensor.from_hp(
            torch.empty(64, 64, dtype=torch.bfloat16, device="cpu"),
            group_size=64,
        )
    assert fake_weight.shape == (64, 64)

    sign16 = torch.tensor(
        [1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1], dtype=torch.float32
    )
    overflow = sign16.repeat(16, 1) * 1e38
    for compute_scale in (False, True):
        with pytest.raises(ValueError, match="non-finite"):
            ConvRotNVFP4Tensor.from_hp(
                overflow,
                group_size=16,
                compute_per_tensor_scale=compute_scale,
            )


def test_device_and_dtype_copies_preserve_wrapper_and_group_size() -> None:
    source = ConvRotNVFP4Tensor(
        torch.empty(128, 128, dtype=torch.uint8),
        torch.empty(
            nvfp4_layout.scale_shape(128, 256),
            dtype=torch.float8_e4m3fn,
        ),
        16,
        torch.bfloat16,
        64,
        torch.empty(()),
        None,
        True,
        _act_quant(True),
    )

    moved = source.to(device="meta", dtype=torch.float16)

    assert type(moved) is ConvRotNVFP4Tensor
    assert moved.device.type == "meta"
    assert moved.dtype is torch.float16
    assert moved.group_size == 64


def test_dtype_copy_reuses_packed_storage() -> None:
    source = _meta_weight(group_size=64)

    moved = source.to(dtype=torch.float16)

    assert type(moved) is ConvRotNVFP4Tensor
    assert moved.dtype is torch.float16
    assert moved.group_size == source.group_size
    assert moved.qdata is source.qdata
    assert moved.scale is source.scale
    assert moved.per_tensor_scale is source.per_tensor_scale
    assert moved.act_per_tensor_scale is source.act_per_tensor_scale


def test_explicit_dtype_copy_duplicates_packed_storage() -> None:
    source = _meta_weight(group_size=64)

    moved = source.to(dtype=torch.float16, copy=True)

    assert type(moved) is ConvRotNVFP4Tensor
    assert moved.dtype is torch.float16
    assert moved.group_size == source.group_size
    assert moved.qdata is not source.qdata
    assert moved.scale is not source.scale
    assert moved.per_tensor_scale is not source.per_tensor_scale


def test_tensor_flatten_round_trip_preserves_storage_and_metadata() -> None:
    source = _meta_weight(group_size=64)
    source.high_first = True
    names, metadata = source.__tensor_flatten__()
    tensors = {name: getattr(source, name) for name in names}

    rebuilt = ConvRotNVFP4Tensor.__tensor_unflatten__(
        tensors,
        metadata,
        source.shape,
        source.stride(),
    )

    assert type(rebuilt) is ConvRotNVFP4Tensor
    assert rebuilt.qdata is source.qdata
    assert rebuilt.scale is source.scale
    assert rebuilt.per_tensor_scale is source.per_tensor_scale
    assert rebuilt.act_per_tensor_scale is source.act_per_tensor_scale
    assert rebuilt.block_size == source.block_size
    assert rebuilt.dtype is source.dtype
    assert rebuilt.group_size == source.group_size
    assert rebuilt.is_swizzled_scales is source.is_swizzled_scales
    assert rebuilt.act_quant == source.act_quant
    assert rebuilt.high_first is True


def test_stable_hash_distinguishes_rotation_groups() -> None:
    source = _meta_weight(group_size=16)
    other = ConvRotNVFP4Tensor.from_quantized(
        source.qdata,
        source.scale,
        group_size=64,
        logical_dtype=source.dtype,
        block_size=source.block_size,
        per_tensor_scale=source.per_tensor_scale,
        act_per_tensor_scale=source.act_per_tensor_scale,
        is_swizzled_scales=source.is_swizzled_scales,
        act_quant=source.act_quant,
    )

    assert source._stable_hash_for_caching() != other._stable_hash_for_caching()


@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_dequantize_returns_the_unrotated_logical_weight(output_dtype: torch.dtype | None) -> None:
    torch.manual_seed(611)
    logical_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    rotated_weight = rotate_groups(logical_weight, 16)
    source = PiperNVFP4Tensor.from_hp(rotated_weight)
    weight = ConvRotNVFP4Tensor.from_quantized(
        source.qdata,
        source.scale,
        group_size=16,
        logical_dtype=source.dtype,
        block_size=source.block_size,
        per_tensor_scale=source.per_tensor_scale,
        act_per_tensor_scale=source.act_per_tensor_scale,
        is_swizzled_scales=source.is_swizzled_scales,
        act_quant=source.act_quant,
        high_first=source.high_first,
    )

    actual = weight.dequantize() if output_dtype is None else weight.dequantize(output_dtype)
    expected = rotate_groups(source.dequantize(output_dtype), 16)

    assert actual.dtype is (output_dtype or logical_weight.dtype)
    assert torch.equal(actual, expected)


def _cpu_weight_case(
    *,
    group_size: int = 16,
    seed: int = 620,
    two_level_scaling: bool = True,
) -> ConvRotNVFP4Tensor:
    generator = torch.Generator().manual_seed(seed)
    logical_weight = torch.randn(32, 256, dtype=torch.bfloat16, generator=generator)
    activation_scale = torch.tensor(0.5)
    return ConvRotNVFP4Tensor.from_hp(
        logical_weight,
        group_size=group_size,
        compute_per_tensor_scale=two_level_scaling,
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant=_act_quant(False),
    )


def _cpu_addmm_case(
    *,
    group_size: int = 16,
    seed: int = 620,
    two_level_scaling: bool = True,
) -> tuple[ConvRotNVFP4Tensor, torch.Tensor, torch.Tensor]:
    weight = _cpu_weight_case(
        group_size=group_size,
        seed=seed,
        two_level_scaling=two_level_scaling,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    mat1 = torch.randn(weight.shape[0], 4, dtype=weight.dtype, generator=generator)
    mat2 = torch.randn(4, weight.shape[1], dtype=weight.dtype, generator=generator)
    return weight, mat1, mat2


def _cpu_add_case(
    *,
    group_size: int = 16,
    seed: int = 620,
) -> tuple[ConvRotNVFP4Tensor, torch.Tensor]:
    weight = _cpu_weight_case(group_size=group_size, seed=seed)
    generator = torch.Generator().manual_seed(seed + 2)
    update = torch.randn(weight.shape, dtype=weight.dtype, generator=generator)
    return weight, update


@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize(("beta", "alpha"), [(1, 1), (0.25, 1.75), (0, -0.5)])
def test_addmm_updates_rotated_storage_in_place(
    group_size: int,
    beta: float,
    alpha: float,
) -> None:
    weight, mat1, mat2 = _cpu_addmm_case(group_size=group_size)
    rotated_before = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    expected_dense = torch.addmm(
        rotated_before,
        mat1.float(),
        rotate_groups(mat2.float(), group_size),
        beta=beta,
        alpha=alpha,
    )
    expected = PiperNVFP4Tensor.from_hp(
        expected_dense,
        block_size=weight.block_size,
        per_tensor_scale=nvfp4_layout.per_tensor_amax_to_scale(
            expected_dense.abs().amax()
        ).clamp_min(torch.finfo(torch.float32).tiny),
        act_per_tensor_scale=weight.act_per_tensor_scale,
        is_swizzled_scales=weight.is_swizzled_scales,
        act_quant=weight.act_quant,
    )
    qdata = weight.qdata
    scale = weight.scale
    per_tensor_scale = weight.per_tensor_scale
    act_per_tensor_scale = weight.act_per_tensor_scale

    result = weight.addmm_(mat1, mat2, beta=beta, alpha=alpha)

    assert result is weight
    assert weight.qdata is qdata
    assert weight.scale is scale
    assert weight.per_tensor_scale is per_tensor_scale
    assert weight.act_per_tensor_scale is act_per_tensor_scale
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))
    assert weight.per_tensor_scale is not None
    assert expected.per_tensor_scale is not None
    assert torch.equal(weight.per_tensor_scale, expected.per_tensor_scale)


@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize(("beta", "alpha"), [(1, 1), (0.25, 0.25), (-1, -0.5), (0, 0.5)])
def test_add_updates_rotated_storage_in_place(group_size: int, beta: float, alpha: float) -> None:
    weight, update = _cpu_add_case(group_size=group_size, seed=623)
    rotated_before = PiperNVFP4Tensor.dequantize(weight, torch.float32)
    expected_dense = torch.add(
        rotated_before * beta,
        rotate_groups(update.float(), group_size),
        alpha=alpha,
    )
    expected = PiperNVFP4Tensor.from_hp(
        expected_dense,
        block_size=weight.block_size,
        per_tensor_scale=nvfp4_layout.per_tensor_amax_to_scale(
            expected_dense.abs().amax()
        ).clamp_min(torch.finfo(torch.float32).tiny),
        act_per_tensor_scale=weight.act_per_tensor_scale,
        is_swizzled_scales=weight.is_swizzled_scales,
        act_quant=weight.act_quant,
    )
    qdata = weight.qdata
    scale = weight.scale
    per_tensor_scale = weight.per_tensor_scale
    act_per_tensor_scale = weight.act_per_tensor_scale

    result = weight.add_(update, beta=beta, alpha=alpha)

    assert result is weight
    assert weight.qdata is qdata
    assert weight.scale is scale
    assert weight.per_tensor_scale is per_tensor_scale
    assert weight.act_per_tensor_scale is act_per_tensor_scale
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))
    assert weight.per_tensor_scale is not None
    assert expected.per_tensor_scale is not None
    assert torch.equal(weight.per_tensor_scale, expected.per_tensor_scale)


def test_aten_add_tensor_dispatches_to_convrot_nvfp4_update() -> None:
    weight, update = _cpu_add_case()
    expected = weight.clone()

    expected.add_(update, alpha=0.25)
    result = torch.ops.aten.add_.Tensor(weight, update, alpha=0.25)

    assert result is weight
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))


def test_addmm_preserves_one_level_weight_scaling() -> None:
    weight, mat1, mat2 = _cpu_addmm_case(two_level_scaling=False)

    weight.addmm_(mat1, mat2)

    assert weight.per_tensor_scale is None


@pytest.mark.parametrize("operation", ["add_", "addmm_"])
def test_no_op_does_not_requantize_storage(operation: str) -> None:
    make_case = _cpu_add_case if operation == "add_" else _cpu_addmm_case
    weight, *operands = make_case()
    storage = (weight.qdata, weight.scale, weight.per_tensor_scale)
    before = [(tensor.clone(), tensor._version) for tensor in storage]

    assert getattr(weight, operation)(*operands, alpha=0) is weight

    assert weight.qdata is storage[0]
    assert weight.scale is storage[1]
    assert weight.per_tensor_scale is storage[2]
    for tensor, (original, version) in zip(storage, before, strict=True):
        assert torch.equal(tensor, original)
        assert tensor._version == version


def test_addmm_encodes_an_all_zero_result_without_invalid_scales() -> None:
    weight, mat1, mat2 = _cpu_addmm_case()

    weight.addmm_(mat1, mat2, beta=0, alpha=0)

    assert not bool(weight.dequantize().any())
    assert weight.per_tensor_scale is not None
    assert bool(torch.isfinite(weight.per_tensor_scale))
    assert bool((weight.per_tensor_scale > 0).all())


@pytest.mark.parametrize("operation", ["add_", "addmm_"])
def test_stochastic_rounding_replays_without_consuming_global_rng(operation: str) -> None:
    seed = (1 << 64) - 1
    first, *operands = _cpu_add_case(seed=624) if operation == "add_" else _cpu_addmm_case(seed=621)
    replay = first.clone()
    other = first.clone()
    deterministic = first.clone()
    torch.manual_seed(1701)
    rng_before = torch.random.get_rng_state()

    getattr(first, operation)(*operands, rounding_seed=seed)
    getattr(replay, operation)(*operands, rounding_seed=seed)
    getattr(other, operation)(*operands, rounding_seed=seed - 1)
    getattr(deterministic, operation)(*operands)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(first.qdata, replay.qdata)
    assert torch.equal(first.scale.view(torch.uint8), replay.scale.view(torch.uint8))
    assert not torch.equal(first.qdata, other.qdata)
    assert torch.equal(first.scale.view(torch.uint8), other.scale.view(torch.uint8))
    assert not torch.equal(first.qdata, deterministic.qdata)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_addmm_stochastic_rounding_replays() -> None:
    weight, mat1, mat2 = _cpu_addmm_case(seed=622)
    first = weight.cuda()
    replay = weight.cuda()
    mat1 = mat1.cuda()
    mat2 = mat2.cuda()

    first.addmm_(mat1, mat2, rounding_seed=12345)
    replay.addmm_(mat1, mat2, rounding_seed=12345)

    assert torch.equal(first.qdata, replay.qdata)
    assert torch.equal(first.scale.view(torch.uint8), replay.scale.view(torch.uint8))


@pytest.mark.parametrize(
    ("rounding_seed", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        (-1, ValueError),
        (1 << 64, ValueError),
    ],
)
def test_addmm_rejects_invalid_stochastic_rounding_seed(
    rounding_seed: object,
    error: type[Exception],
) -> None:
    weight, mat1, mat2 = _cpu_addmm_case()
    qdata_before = weight.qdata.clone()

    with pytest.raises(error, match="unsigned 64-bit integer"):
        weight.addmm_(mat1, mat2, rounding_seed=rounding_seed)  # type: ignore[arg-type]

    assert torch.equal(weight.qdata, qdata_before)


@pytest.mark.parametrize(
    ("mat1", "mat2", "message"),
    [
        (
            torch.empty(32, 4, 1, dtype=torch.bfloat16),
            torch.empty(4, 256, dtype=torch.bfloat16),
            "matrices must be 2-D",
        ),
        (
            torch.empty(31, 4, dtype=torch.bfloat16),
            torch.empty(4, 256, dtype=torch.bfloat16),
            "shape mismatch",
        ),
        (
            torch.empty(32, 4, dtype=torch.bfloat16),
            torch.empty(5, 256, dtype=torch.bfloat16),
            "shape mismatch",
        ),
        (
            torch.empty(32, 4, dtype=torch.float16),
            torch.empty(4, 256, dtype=torch.bfloat16),
            "logical dtype",
        ),
    ],
)
def test_addmm_rejects_invalid_matrices(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    message: str,
) -> None:
    weight = _cpu_weight_case()

    with pytest.raises(ValueError, match=message):
        weight.addmm_(mat1, mat2)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (torch.empty(31, 256, dtype=torch.bfloat16), "shape mismatch"),
        (torch.empty(32, 256, dtype=torch.float16), "logical dtype"),
        (torch.ones(32, 256, dtype=torch.bfloat16).to_sparse(), "strided layout"),
    ],
)
def test_add_rejects_invalid_update(update: torch.Tensor, message: str) -> None:
    weight = _cpu_weight_case()

    with pytest.raises(ValueError, match=message):
        weight.add_(update)


@pytest.mark.parametrize("operation", ["add_", "addmm_"])
def test_updates_reject_autograd_inputs(operation: str) -> None:
    make_case = _cpu_add_case if operation == "add_" else _cpu_addmm_case
    weight, *operands = make_case()
    operands[0].requires_grad_(True)
    update = getattr(weight, operation)

    with pytest.raises(RuntimeError, match="does not support autograd"):
        update(*operands)

    with torch.no_grad():
        assert update(*operands) is weight


def test_meta_linear_supports_functional_and_keyword_forms() -> None:
    input = torch.empty(3, 5, 256, dtype=torch.bfloat16, device="meta")  # noqa: A001
    weight = _meta_weight()
    bias = torch.empty(128, dtype=torch.bfloat16, device="meta")

    functional = F.linear(input=input, weight=weight, bias=bias)
    explicit = convrot_nvfp4_linear(input, weight, bias)

    assert functional.shape == (3, 5, 128)
    assert functional.dtype is torch.bfloat16
    assert explicit.shape == functional.shape


@pytest.mark.parametrize("group_size", [15, 32, 128])
def test_constructor_rejects_unsupported_group_size(group_size: int) -> None:
    with pytest.raises(ValueError, match="group size must be one of"):
        _meta_weight(group_size=group_size)


def test_constructor_rejects_non_matrix_weight() -> None:
    with pytest.raises(ValueError, match="weight must be two-dimensional"):
        ConvRotNVFP4Tensor(
            torch.empty(2, 128, 128, dtype=torch.uint8, device="meta"),
            torch.empty(2, 32, 64, dtype=torch.float8_e4m3fn, device="meta"),
            16,
            torch.bfloat16,
            16,
        )


def _cuda_case(
    dynamic: bool,
    group_size: int,
) -> tuple[torch.Tensor, ConvRotNVFP4Tensor, torch.Tensor]:
    input = torch.randn(257, 256, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    logical_weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    rotated_input = rotate_groups(input, group_size)
    activation_scale = (
        None if dynamic else nvfp4_layout.per_tensor_amax_to_scale(rotated_input.abs().amax())
    )
    weight = ConvRotNVFP4Tensor.from_hp(
        logical_weight,
        group_size=group_size,
        compute_per_tensor_scale=True,
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant=_act_quant(dynamic),
    )
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    return input, weight, bias


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_linear_matches_materialized_rotation(
    dynamic: bool, group_size: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(612 + group_size + dynamic)
    activation, weight, bias = _cuda_case(dynamic, group_size)
    activation = activation.to(dtype)
    weight = weight.to(dtype)
    rotated_input = rotate_groups(activation.float(), group_size)
    prepared_input = nvfp4_ops._prepare_compiled(
        rotated_input,
        weight.act_per_tensor_scale,
        dynamic,
    )
    expected = nvfp4_ops._execute_prepared(
        *prepared_input,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        bias,
        activation.dtype,
    )
    portable = nvfp4_reference.linear_prepared(
        *prepared_input,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        bias,
        activation.dtype,
    )

    actual = F.linear(activation, weight, bias)

    assert torch.equal(actual, expected)
    relative_l2 = (actual.float() - portable.float()).norm() / portable.float().norm()
    assert relative_l2 < 0.002


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_linear_supports_mixed_precision_bias(
    bias_dtype: torch.dtype,
    compiled: bool,
) -> None:
    torch.manual_seed(616)
    activation, weight, original_bias = _cuda_case(False, 64)
    bias = original_bias.to(bias_dtype)

    def projection(value: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return F.linear(value, weight, offset)

    expected = convrot_nvfp4_linear(activation, weight, bias)
    call = (
        torch.compile(projection, fullgraph=True, options=compile_options())
        if compiled
        else projection
    )
    actual = call(activation, bias)

    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_high_first_linear_matches_low_first() -> None:
    torch.manual_seed(615)
    activation, low, bias = _cuda_case(False, 64)
    high = ConvRotNVFP4Tensor(
        ((low.qdata & 0x0F) << 4) | (low.qdata >> 4),
        low.scale,
        low.block_size,
        low.dtype,
        low.group_size,
        low.per_tensor_scale,
        low.act_per_tensor_scale,
        low.is_swizzled_scales,
        low.act_quant,
        high_first=True,
    )

    actual = F.linear(activation, high, bias)
    expected = F.linear(activation, low, bias)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.002


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("dynamic", [False, True])
def test_cuda_compile_preserves_semantic_linear(dynamic: bool) -> None:
    torch.manual_seed(613 + dynamic)
    activation, weight, bias = _cuda_case(dynamic, 16)

    def projection(value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, weight, bias)

    expected = projection(activation)
    capture = TargetCapturePass()
    actual = torch.compile(
        projection,
        fullgraph=True,
        options={"post_grad_custom_pre_pass": capture},
    )(activation)

    assert torch.equal(actual, expected)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_nvfp4_linear.default) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("grad_mode", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_autocast_normalizes_semantic_linear(
    compiled: bool,
    grad_mode: type[torch.no_grad | torch.inference_mode],
    autocast_dtype: torch.dtype,
) -> None:
    torch.manual_seed(614)
    input = torch.randn(17, 256, device="cuda", dtype=torch.float32) * 0.01  # noqa: A001
    source = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.01
    weight = ConvRotNVFP4Tensor.from_hp(
        source,
        group_size=16,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_quant=_act_quant(True),
    )
    bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.01

    def projection(value: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return F.linear(value, weight, offset)

    call = (
        torch.compile(projection, fullgraph=True, options=compile_options())
        if compiled
        else projection
    )
    with grad_mode(), torch.autocast("cuda", dtype=autocast_dtype):
        actual = call(input, bias)
    with torch.no_grad():
        expected = F.linear(
            input.to(autocast_dtype), weight.to(dtype=autocast_dtype), bias.to(autocast_dtype)
        )

    assert actual.dtype is autocast_dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)
