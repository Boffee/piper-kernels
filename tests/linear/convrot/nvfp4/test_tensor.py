"""Tests for the semantic ConvRot NVFP4 tensor."""

from __future__ import annotations

import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from piper_kernels.linear.convrot._rotation import rotate_groups
from piper_kernels.linear.convrot.nvfp4 import (
    ConvRotNVFP4Tensor,
    convrot_nvfp4_compile_options,
    convrot_nvfp4_linear,
)
from piper_kernels.linear.nvfp4 import _layout as nvfp4_layout
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _quantization(dynamic: bool) -> QuantizeTensorToNVFP4Kwargs:
    return QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=dynamic,
    )


def _meta_weight(*, group_size: int = 16) -> ConvRotNVFP4Tensor:
    output_features, input_features = 128, 256
    source = TorchAONVFP4Tensor(
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
        16,
        torch.bfloat16,
        torch.empty((), device="meta"),
        None,
        True,
        False,
        _quantization(True),
    )
    return ConvRotNVFP4Tensor.from_torchao(source, group_size=group_size)


def test_from_torchao_reuses_storage_and_attaches_rotation_metadata() -> None:
    source = _meta_weight()

    wrapped = ConvRotNVFP4Tensor.from_torchao(source, group_size=16)

    assert wrapped is source
    assert wrapped.group_size == 16
    with pytest.raises(ValueError, match="uses group size 16, not 64"):
        ConvRotNVFP4Tensor.from_torchao(source, group_size=64)
    with pytest.raises(TypeError, match="requires a group size"):
        ConvRotNVFP4Tensor.from_torchao(source)


@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_from_hp_matches_explicit_rotation_and_torchao_quantization(group_size: int) -> None:
    torch.manual_seed(610 + group_size)
    logical_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    rotated_weight = rotate_groups(logical_weight, group_size)
    per_tensor_scale = per_tensor_amax_to_scale(rotated_weight.abs().amax())
    activation_scale = torch.tensor(0.5)

    weight = ConvRotNVFP4Tensor.from_hp(
        logical_weight.requires_grad_(),
        group_size=group_size,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(False),
    )
    expected = TorchAONVFP4Tensor.to_nvfp4(
        rotated_weight,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(False),
    )

    assert type(weight) is ConvRotNVFP4Tensor
    assert weight.group_size == group_size
    assert not weight.requires_grad
    assert torch.equal(weight.qdata, expected.qdata)
    assert torch.equal(weight.scale.view(torch.uint8), expected.scale.view(torch.uint8))
    assert weight.per_tensor_scale is per_tensor_scale
    assert weight.act_per_tensor_scale is activation_scale
    assert weight.act_quant_kwargs == expected.act_quant_kwargs


def test_from_hp_computes_global_scale_in_the_rotated_basis() -> None:
    torch.manual_seed(874)
    logical_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    rotated_weight = rotate_groups(logical_weight, 64)
    expected_scale = per_tensor_amax_to_scale(rotated_weight.float().abs().amax())

    weight = ConvRotNVFP4Tensor.from_hp(
        logical_weight,
        group_size=64,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(True),
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
        False,
        _quantization(True),
    )

    moved = source.to(device="meta", dtype=torch.float16)

    assert type(moved) is ConvRotNVFP4Tensor
    assert moved.device.type == "meta"
    assert moved.orig_dtype is torch.float16
    assert moved.group_size == 64


def test_tensor_flatten_round_trip_preserves_storage_and_metadata() -> None:
    source = _meta_weight(group_size=64)
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
    assert rebuilt.orig_dtype is source.orig_dtype
    assert rebuilt.group_size == source.group_size
    assert rebuilt.is_swizzled_scales is source.is_swizzled_scales
    assert rebuilt.use_triton_kernel is source.use_triton_kernel
    assert rebuilt.act_quant_kwargs == source.act_quant_kwargs


def test_stable_hash_distinguishes_rotation_groups() -> None:
    source = _meta_weight(group_size=16)
    other = ConvRotNVFP4Tensor.from_torchao(
        TorchAONVFP4Tensor(
            source.qdata,
            source.scale,
            source.block_size,
            source.orig_dtype,
            source.per_tensor_scale,
            source.act_per_tensor_scale,
            source.is_swizzled_scales,
            source.use_triton_kernel,
            source.act_quant_kwargs,
        ),
        group_size=64,
    )

    assert source._stable_hash_for_caching() != other._stable_hash_for_caching()


def test_dequantize_returns_the_unrotated_logical_weight() -> None:
    torch.manual_seed(611)
    logical_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    rotated_weight = rotate_groups(logical_weight, 16)
    source = TorchAONVFP4Tensor.to_nvfp4(rotated_weight)
    weight = ConvRotNVFP4Tensor.from_torchao(source, group_size=16)

    expected = rotate_groups(source.dequantize(), 16)

    assert torch.equal(weight.dequantize(), expected)


def test_dequantize_accepts_an_output_dtype() -> None:
    torch.manual_seed(612)
    logical_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    rotated_weight = rotate_groups(logical_weight, 16)
    source = TorchAONVFP4Tensor.to_nvfp4(rotated_weight)
    weight = ConvRotNVFP4Tensor.from_torchao(source, group_size=16)

    actual = weight.dequantize(torch.float32)
    expected = rotate_groups(source.dequantize(torch.float32), 16)

    assert actual.dtype is torch.float32
    assert torch.equal(actual, expected)


def _cpu_weight_case(
    *,
    group_size: int = 16,
    seed: int = 620,
    two_level_scaling: bool = True,
) -> ConvRotNVFP4Tensor:
    generator = torch.Generator().manual_seed(seed)
    logical_weight = torch.randn(32, 256, dtype=torch.bfloat16, generator=generator)
    rotated_weight = rotate_groups(logical_weight, group_size)
    activation_scale = torch.tensor(0.5)
    source = TorchAONVFP4Tensor.to_nvfp4(
        rotated_weight,
        per_tensor_scale=(
            per_tensor_amax_to_scale(rotated_weight.abs().amax()) if two_level_scaling else None
        ),
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        act_quant_kwargs=_quantization(False),
    )
    return ConvRotNVFP4Tensor.from_torchao(source, group_size=group_size)


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
    mat1 = torch.randn(weight.shape[0], 4, dtype=weight.orig_dtype, generator=generator)
    mat2 = torch.randn(4, weight.shape[1], dtype=weight.orig_dtype, generator=generator)
    return weight, mat1, mat2


def _cpu_add_case(
    *,
    group_size: int = 16,
    seed: int = 620,
) -> tuple[ConvRotNVFP4Tensor, torch.Tensor]:
    weight = _cpu_weight_case(group_size=group_size, seed=seed)
    generator = torch.Generator().manual_seed(seed + 2)
    update = torch.randn(weight.shape, dtype=weight.orig_dtype, generator=generator)
    return weight, update


@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize(("beta", "alpha"), [(1, 1), (0.25, 1.75), (0, -0.5)])
def test_addmm_updates_rotated_storage_in_place(
    group_size: int,
    beta: float,
    alpha: float,
) -> None:
    weight, mat1, mat2 = _cpu_addmm_case(group_size=group_size)
    rotated_before = TorchAONVFP4Tensor.dequantize(weight, weight.orig_dtype)
    expected_dense = torch.addmm(
        rotated_before,
        mat1,
        rotate_groups(mat2, group_size),
        beta=beta,
        alpha=alpha,
    )
    expected = TorchAONVFP4Tensor.to_nvfp4(
        expected_dense,
        block_size=weight.block_size,
        per_tensor_scale=per_tensor_amax_to_scale(expected_dense.abs().amax()).clamp_min(
            torch.finfo(torch.float32).tiny
        ),
        act_per_tensor_scale=weight.act_per_tensor_scale,
        is_swizzled_scales=weight.is_swizzled_scales,
        use_triton_kernel=False,
        act_quant_kwargs=weight.act_quant_kwargs,
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
@pytest.mark.parametrize("alpha", [1, 0.25, -0.5])
def test_add_updates_rotated_storage_in_place(group_size: int, alpha: float) -> None:
    weight, update = _cpu_add_case(group_size=group_size, seed=623)
    rotated_before = TorchAONVFP4Tensor.dequantize(weight, weight.orig_dtype)
    expected_dense = torch.add(
        rotated_before,
        rotate_groups(update, group_size),
        alpha=alpha,
    )
    expected = TorchAONVFP4Tensor.to_nvfp4(
        expected_dense,
        block_size=weight.block_size,
        per_tensor_scale=per_tensor_amax_to_scale(expected_dense.abs().amax()).clamp_min(
            torch.finfo(torch.float32).tiny
        ),
        act_per_tensor_scale=weight.act_per_tensor_scale,
        is_swizzled_scales=weight.is_swizzled_scales,
        use_triton_kernel=False,
        act_quant_kwargs=weight.act_quant_kwargs,
    )
    qdata = weight.qdata
    scale = weight.scale
    per_tensor_scale = weight.per_tensor_scale
    act_per_tensor_scale = weight.act_per_tensor_scale

    result = weight.add_(update, alpha=alpha)

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


def test_addmm_no_op_does_not_requantize_storage() -> None:
    weight, mat1, mat2 = _cpu_addmm_case()
    qdata_before = weight.qdata.clone()
    scale_before = weight.scale.clone()
    per_tensor_scale_before = weight.per_tensor_scale.clone()
    versions = (
        weight.qdata._version,
        weight.scale._version,
        weight.per_tensor_scale._version,
    )

    weight.addmm_(mat1, mat2, alpha=0)

    assert torch.equal(weight.qdata, qdata_before)
    assert torch.equal(weight.scale, scale_before)
    assert torch.equal(weight.per_tensor_scale, per_tensor_scale_before)
    assert versions == (
        weight.qdata._version,
        weight.scale._version,
        weight.per_tensor_scale._version,
    )


def test_add_no_op_does_not_requantize_storage() -> None:
    weight, update = _cpu_add_case()
    qdata_before = weight.qdata.clone()
    scale_before = weight.scale.clone()
    per_tensor_scale_before = weight.per_tensor_scale.clone()
    versions = (
        weight.qdata._version,
        weight.scale._version,
        weight.per_tensor_scale._version,
    )

    weight.add_(update, alpha=0)

    assert torch.equal(weight.qdata, qdata_before)
    assert torch.equal(weight.scale, scale_before)
    assert torch.equal(weight.per_tensor_scale, per_tensor_scale_before)
    assert versions == (
        weight.qdata._version,
        weight.scale._version,
        weight.per_tensor_scale._version,
    )


def test_addmm_encodes_an_all_zero_result_without_invalid_scales() -> None:
    weight, mat1, mat2 = _cpu_addmm_case()

    weight.addmm_(mat1, mat2, beta=0, alpha=0)

    assert not bool(weight.dequantize().any())
    assert weight.per_tensor_scale is not None
    assert bool(torch.isfinite(weight.per_tensor_scale))
    assert bool((weight.per_tensor_scale > 0).all())


def test_addmm_stochastic_rounding_replays_without_consuming_global_rng() -> None:
    seed = (1 << 64) - 1
    first, mat1, mat2 = _cpu_addmm_case(seed=621)
    replay = first.clone()
    other = first.clone()
    deterministic = first.clone()
    torch.manual_seed(1701)
    rng_before = torch.random.get_rng_state()

    first.addmm_(mat1, mat2, rounding_seed=seed)
    replay.addmm_(mat1, mat2, rounding_seed=seed)
    other.addmm_(mat1, mat2, rounding_seed=seed - 1)
    deterministic.addmm_(mat1, mat2)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(first.qdata, replay.qdata)
    assert torch.equal(first.scale.view(torch.uint8), replay.scale.view(torch.uint8))
    assert not torch.equal(first.qdata, other.qdata)
    assert torch.equal(first.scale.view(torch.uint8), other.scale.view(torch.uint8))
    assert not torch.equal(first.qdata, deterministic.qdata)


def test_add_stochastic_rounding_replays_without_consuming_global_rng() -> None:
    seed = (1 << 64) - 1
    first, update = _cpu_add_case(seed=624)
    replay = first.clone()
    other = first.clone()
    deterministic = first.clone()
    torch.manual_seed(1703)
    rng_before = torch.random.get_rng_state()

    first.add_(update, rounding_seed=seed)
    replay.add_(update, rounding_seed=seed)
    other.add_(update, rounding_seed=seed - 1)
    deterministic.add_(update)

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


def test_addmm_rejects_autograd_inputs() -> None:
    weight, mat1, mat2 = _cpu_addmm_case()
    mat1.requires_grad_(True)

    with pytest.raises(RuntimeError, match="does not support autograd"):
        weight.addmm_(mat1, mat2)

    with torch.no_grad():
        assert weight.addmm_(mat1, mat2) is weight


def test_add_rejects_autograd_inputs() -> None:
    weight = _cpu_weight_case()
    update = torch.randn(weight.shape, dtype=weight.orig_dtype, requires_grad=True)

    with pytest.raises(RuntimeError, match="does not support autograd"):
        weight.add_(update)

    with torch.no_grad():
        assert weight.add_(update) is weight


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


class _TargetCapturePass(CustomInferenceAwareGraphPass):
    def __init__(self) -> None:
        self.targets: list[object] = []
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid


def _cuda_case(
    dynamic: bool,
    group_size: int,
) -> tuple[torch.Tensor, TorchAONVFP4Tensor, ConvRotNVFP4Tensor, torch.Tensor]:
    input = torch.randn(257, 256, device="cuda", dtype=torch.bfloat16)  # noqa: A001
    logical_weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    rotated_input = rotate_groups(input, group_size)
    rotated_weight = rotate_groups(logical_weight, group_size)
    activation_scale = None if dynamic else per_tensor_amax_to_scale(rotated_input.abs().amax())
    torchao_weight = TorchAONVFP4Tensor.to_nvfp4(
        rotated_weight,
        per_tensor_scale=per_tensor_amax_to_scale(rotated_weight.abs().amax()),
        act_per_tensor_scale=activation_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(dynamic),
    )
    weight = ConvRotNVFP4Tensor.from_torchao(torchao_weight, group_size=group_size)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    return input, torchao_weight, weight, bias


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_cuda_linear_matches_materialized_rotation(dynamic: bool, group_size: int) -> None:
    torch.manual_seed(612 + group_size + dynamic)
    activation, torchao_weight, weight, bias = _cuda_case(dynamic, group_size)
    rotated_input = rotate_groups(activation, group_size)
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
    torchao_reference = F.linear(rotated_input, torchao_weight, bias)

    actual = F.linear(activation, weight, bias)

    assert torch.equal(actual, expected)
    relative_l2 = (
        actual.float() - torchao_reference.float()
    ).norm() / torchao_reference.float().norm()
    assert relative_l2 < 0.02


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("dynamic", [False, True])
def test_cuda_compile_preserves_semantic_linear(dynamic: bool) -> None:
    torch.manual_seed(613 + dynamic)
    activation, _torchao_weight, weight, bias = _cuda_case(dynamic, 16)

    def projection(value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, weight, bias)

    expected = projection(activation)
    capture = _TargetCapturePass()
    actual = torch.compile(
        projection,
        fullgraph=True,
        options={"post_grad_custom_pre_pass": capture},
    )(activation)

    assert torch.equal(actual, expected)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_nvfp4_linear.default) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_fp16_autocast_normalizes_semantic_linear(compiled: bool) -> None:
    torch.manual_seed(614)
    input = torch.randn(17, 256, device="cuda", dtype=torch.float32) * 0.01  # noqa: A001
    source = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.01
    weight = ConvRotNVFP4Tensor.from_hp(
        source,
        group_size=16,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_quant_kwargs=_quantization(True),
    )
    bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.01

    def projection(value: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return F.linear(value, weight, offset)

    call = (
        torch.compile(projection, fullgraph=True, options=convrot_nvfp4_compile_options())
        if compiled
        else projection
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        actual = call(input, bias)
    with torch.no_grad():
        expected = F.linear(input.half(), weight.to(dtype=torch.float16), bias.half())

    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)
