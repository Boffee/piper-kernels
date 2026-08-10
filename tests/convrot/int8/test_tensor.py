"""Tests for the ConvRot tensor representation."""

import pytest
import torch

from piper_kernels.convrot import ConvRotInt8Tensor
from piper_kernels.convrot._rotation import rotate_groups


def test_dequantize_unrotates_the_stored_weight() -> None:
    qdata = torch.arange(-128, 128, dtype=torch.int8).reshape(16, 16)
    scale = torch.linspace(0.001, 0.016, 16).reshape(16, 1)
    wrapped = ConvRotInt8Tensor.from_packed(
        qdata,
        scale,
        group_size=16,
        dtype=torch.float32,
    )
    expected = rotate_groups(qdata.float() * scale, 16)
    assert torch.equal(wrapped.dequantize(), expected)


def test_meta_tensor_preserves_storage_and_rotation_metadata() -> None:
    wrapped = ConvRotInt8Tensor.from_packed(
        torch.empty(8, 64, dtype=torch.int8, device="meta"),
        torch.empty(8, 1, dtype=torch.float32, device="meta"),
        group_size=64,
    )

    assert wrapped.device.type == "meta"
    assert wrapped.dtype is torch.bfloat16
    assert wrapped.group_size == 64
    assert wrapped.qdata.shape == (8, 64)
    assert wrapped.scale.shape == (8, 1)


def test_from_packed_normalizes_flat_scale_to_column() -> None:
    scale = torch.arange(1, 9, dtype=torch.float32)
    wrapped = ConvRotInt8Tensor.from_packed(
        torch.empty(8, 64, dtype=torch.int8),
        scale,
        group_size=64,
    )

    assert wrapped.scale.shape == (8, 1)
    assert torch.equal(wrapped.scale[:, 0], scale)


def test_from_quantized_canonicalizes_storage_and_names_logical_dtype() -> None:
    qdata = torch.randint(-128, 128, (8, 128), dtype=torch.int8)[:, ::2]
    scale = torch.arange(1, 17, dtype=torch.float32).reshape(16, 1)[::2]
    assert not qdata.is_contiguous()
    assert not scale.is_contiguous()

    wrapped = ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=64,
        logical_dtype=torch.float32,
    )

    assert wrapped.dtype is torch.float32
    assert wrapped.qdata.is_contiguous()
    assert wrapped.scale.is_contiguous()
    assert torch.equal(wrapped.qdata, qdata)
    assert torch.equal(wrapped.scale, scale)


def test_from_packed_remains_compatible_with_from_quantized() -> None:
    qdata = torch.randint(-128, 128, (8, 64), dtype=torch.int8)
    scale = torch.rand(8, dtype=torch.float32)

    packed = ConvRotInt8Tensor.from_packed(
        qdata,
        scale,
        group_size=64,
        dtype=torch.float16,
    )
    quantized = ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=64,
        logical_dtype=torch.float16,
    )

    assert packed.dtype is quantized.dtype is torch.float16
    assert torch.equal(packed.qdata, quantized.qdata)
    assert torch.equal(packed.scale, quantized.scale)


def test_from_packed_preserves_legacy_scale_reshape() -> None:
    scale = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)

    wrapped = ConvRotInt8Tensor.from_packed(
        torch.empty(8, 64, dtype=torch.int8),
        scale,
        group_size=64,
    )

    assert wrapped.scale.shape == (8, 1)
    assert torch.equal(wrapped.scale[:, 0], scale.flatten())


def test_from_packed_canonicalizes_noncontiguous_storage() -> None:
    qdata = torch.empty(8, 128, dtype=torch.int8)[:, ::2]
    scale = torch.empty(16, 1, dtype=torch.float32)[::2]

    wrapped = ConvRotInt8Tensor.from_packed(qdata, scale, group_size=64)

    assert wrapped.qdata.is_contiguous()
    assert wrapped.scale.is_contiguous()


@pytest.mark.parametrize("scale_shape", [(1, 8), (2, 4), (8, 1, 1)])
def test_from_quantized_rejects_ambiguous_scale_shapes(
    scale_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="from_quantized scale must have shape"):
        ConvRotInt8Tensor.from_quantized(
            torch.empty(8, 64, dtype=torch.int8),
            torch.empty(scale_shape, dtype=torch.float32),
            group_size=64,
        )


def test_constructor_rejects_noncanonical_storage_layouts() -> None:
    qdata = torch.empty(8, 128, dtype=torch.int8)[:, ::2]
    scale = torch.empty(16, 1, dtype=torch.float32)[::2]

    with pytest.raises(ValueError, match="qdata and scale must be contiguous"):
        ConvRotInt8Tensor(qdata, torch.empty(8, 1), 64)
    with pytest.raises(ValueError, match="qdata and scale must be contiguous"):
        ConvRotInt8Tensor(torch.empty(8, 64, dtype=torch.int8), scale, 64)


@pytest.mark.parametrize("storage_name", ["qdata", "scale"])
def test_dequantize_revalidates_canonical_storage_layout(storage_name: str) -> None:
    wrapped = ConvRotInt8Tensor.from_hp(torch.randn(8, 64), group_size=64)
    if storage_name == "qdata":
        wrapped.qdata = torch.empty(8, 128, dtype=torch.int8)[:, ::2]
    else:
        wrapped.scale = torch.empty(8, 2, dtype=torch.float32)[:, ::2]
    assert not getattr(wrapped, storage_name).is_contiguous()

    with pytest.raises(ValueError, match="qdata and scale must be contiguous"):
        wrapped.dequantize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_from_hp_rotates_and_quantizes_each_weight_row(dtype: torch.dtype) -> None:
    torch.manual_seed(12)
    weight = torch.randn(7, 32, dtype=dtype)
    rotated = rotate_groups(weight, 16)
    expected_scale = (rotated.float().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    expected_qdata = (rotated / expected_scale.to(dtype)).round().clamp(-128, 127).to(torch.int8)

    wrapped = ConvRotInt8Tensor.from_hp(weight, group_size=16)

    assert wrapped.dtype is dtype
    assert wrapped.group_size == 16
    assert wrapped.qdata.dtype is torch.int8
    assert wrapped.scale.dtype is torch.float32
    assert wrapped.scale.shape == (7, 1)
    assert torch.equal(wrapped.qdata, expected_qdata)
    assert torch.equal(wrapped.scale, expected_scale)


def test_from_hp_detaches_quantized_storage_from_autograd() -> None:
    weight = torch.randn(3, 16, requires_grad=True)

    wrapped = ConvRotInt8Tensor.from_hp(weight, group_size=16)

    assert not wrapped.qdata.requires_grad
    assert not wrapped.scale.requires_grad


@pytest.mark.parametrize(("beta", "alpha"), [(1, 1), (0.25, 1.75), (0, -0.5)])
def test_addmm_updates_logical_weight_and_requantizes_in_place(
    beta: float,
    alpha: float,
) -> None:
    torch.manual_seed(21)
    weight = torch.randn(7, 32)
    mat1 = torch.randn(7, 5)
    mat2 = torch.randn(5, 32)
    wrapped = ConvRotInt8Tensor.from_hp(weight, group_size=16)
    qdata = wrapped.qdata
    scale = wrapped.scale
    logical_before = wrapped.dequantize()
    expected = ConvRotInt8Tensor.from_hp(
        torch.addmm(logical_before, mat1, mat2, beta=beta, alpha=alpha),
        group_size=16,
    )

    result = wrapped.addmm_(mat1, mat2, beta=beta, alpha=alpha)

    assert result is wrapped
    assert wrapped.qdata is qdata
    assert wrapped.scale is scale
    assert torch.equal(wrapped.qdata, expected.qdata)
    assert torch.allclose(wrapped.scale, expected.scale, rtol=1e-6, atol=1e-7)


def test_addmm_no_op_does_not_requantize_storage() -> None:
    wrapped = ConvRotInt8Tensor.from_hp(torch.randn(7, 32), group_size=16)
    mat1 = torch.randn(7, 5)
    mat2 = torch.randn(5, 32)
    qdata_before = wrapped.qdata.clone()
    scale_before = wrapped.scale.clone()
    qdata_version = wrapped.qdata._version
    scale_version = wrapped.scale._version

    wrapped.addmm_(mat1, mat2, alpha=0)

    assert torch.equal(wrapped.qdata, qdata_before)
    assert torch.equal(wrapped.scale, scale_before)
    assert wrapped.qdata._version == qdata_version
    assert wrapped.scale._version == scale_version


def _stochastic_addmm_fixture(
    *,
    rows: int = 128,
    cols: int = 128,
) -> tuple[ConvRotInt8Tensor, torch.Tensor, torch.Tensor]:
    rotated_update = torch.ones(rows, cols, dtype=torch.bfloat16)
    rotated_update[:, -1] = 2.0
    mat1 = torch.eye(rows, dtype=torch.bfloat16)
    mat2 = rotate_groups(rotated_update, 16)
    weight = ConvRotInt8Tensor.from_packed(
        torch.zeros(rows, cols, dtype=torch.int8),
        torch.ones(rows, 1, dtype=torch.float32),
        group_size=16,
        dtype=torch.bfloat16,
    )
    return weight, mat1, mat2


def test_addmm_stochastic_rounding_replays_without_consuming_global_rng() -> None:
    seed = (1 << 64) - 1
    first, mat1, mat2 = _stochastic_addmm_fixture()
    replay = first.clone()
    other = first.clone()
    deterministic = first.clone()
    torch.manual_seed(1701)
    rng_before = torch.random.get_rng_state()

    first.addmm_(mat1, mat2, beta=0, rounding_seed=seed)
    replay.addmm_(mat1, mat2, beta=0, rounding_seed=seed)
    other.addmm_(mat1, mat2, beta=0, rounding_seed=seed - 1)
    deterministic.addmm_(mat1, mat2, beta=0)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(first.qdata, replay.qdata)
    assert torch.equal(first.scale, replay.scale)
    assert not torch.equal(first.qdata, other.qdata)
    assert torch.equal(first.scale, other.scale)
    assert torch.equal(first.scale, deterministic.scale)


def test_addmm_stochastic_rounding_uses_unbiased_fp32_scaled_probability() -> None:
    weight, mat1, mat2 = _stochastic_addmm_fixture(rows=256, cols=256)

    weight.addmm_(mat1, mat2, beta=0, rounding_seed=12345)

    samples = weight.qdata[:, :-1]
    assert bool(((samples == 63) | (samples == 64)).all())
    assert samples.to(torch.float32).mean().item() == pytest.approx(63.5, abs=0.01)
    assert bool((weight.qdata[:, -1] == 127).all())
    torch.testing.assert_close(
        weight.scale,
        torch.full_like(weight.scale, 2.0 / 127.0),
        rtol=0,
        atol=0,
    )


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
    weight, mat1, mat2 = _stochastic_addmm_fixture(rows=4, cols=16)
    qdata_before = weight.qdata.clone()
    scale_before = weight.scale.clone()

    with pytest.raises(error, match="unsigned 64-bit integer"):
        weight.addmm_(mat1, mat2, rounding_seed=rounding_seed)  # type: ignore[arg-type]

    assert torch.equal(weight.qdata, qdata_before)
    assert torch.equal(weight.scale, scale_before)


@pytest.mark.parametrize(
    ("mat1", "mat2", "message"),
    [
        (torch.empty(7, 5, 1), torch.empty(5, 32), "matrices must be 2-D"),
        (torch.empty(8, 5), torch.empty(5, 32), "shape mismatch"),
        (torch.empty(7, 5), torch.empty(6, 32), "shape mismatch"),
        (torch.empty(7, 5), torch.empty(5, 48), "shape mismatch"),
        (torch.empty(7, 5, dtype=torch.float16), torch.empty(5, 32), "logical dtype"),
    ],
)
def test_addmm_rejects_invalid_matrices(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    message: str,
) -> None:
    wrapped = ConvRotInt8Tensor.from_hp(torch.randn(7, 32), group_size=16)

    with pytest.raises(ValueError, match=message):
        wrapped.addmm_(mat1, mat2)


def test_addmm_rejects_autograd_inputs() -> None:
    wrapped = ConvRotInt8Tensor.from_hp(torch.randn(7, 32), group_size=16)
    mat1 = torch.randn(7, 5, requires_grad=True)
    mat2 = torch.randn(5, 32)

    with pytest.raises(RuntimeError, match="does not support autograd"):
        wrapped.addmm_(mat1, mat2)

    with torch.no_grad():
        assert wrapped.addmm_(mat1, mat2) is wrapped


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_from_hp_quantizes_cuda_weight() -> None:
    weight = torch.randn(9, 64, dtype=torch.bfloat16, device="cuda")

    wrapped = ConvRotInt8Tensor.from_hp(weight, group_size=64)

    assert wrapped.device.type == "cuda"
    assert wrapped.qdata.device.type == "cuda"
    assert wrapped.scale.device.type == "cuda"
    assert wrapped.qdata.shape == weight.shape
    assert wrapped.scale.shape == (weight.shape[0], 1)


@pytest.mark.parametrize(
    ("weight", "message"),
    [
        (torch.empty(2, 3, 16), "must be 2-D"),
        (torch.empty(2, 16, dtype=torch.int32), "must use float16, bfloat16, or float32"),
        (torch.empty(2, 16, device="meta"), "cannot quantize a meta tensor"),
        (torch.empty(2, 24), "is not divisible by group size"),
    ],
)
def test_from_hp_rejects_unsupported_dense_weight(
    weight: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ConvRotInt8Tensor.from_hp(weight, group_size=16)


@pytest.mark.parametrize("scale_shape", [(8,), (1, 8), (2, 4), (7, 1)])
def test_constructor_rejects_noncanonical_scale_shape(
    scale_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match=r"scale must be float32 with shape \(8, 1\)"):
        ConvRotInt8Tensor(
            torch.empty(8, 64, dtype=torch.int8),
            torch.empty(scale_shape, dtype=torch.float32),
            64,
        )


@pytest.mark.parametrize("group_size", [15, 32, 128])
def test_rejects_unsupported_group_size(group_size: int) -> None:
    with pytest.raises(ValueError, match="group size must be one of"):
        ConvRotInt8Tensor.from_packed(
            torch.empty(8, 256, dtype=torch.int8),
            torch.empty(8, 1, dtype=torch.float32),
            group_size=group_size,
        )


@pytest.mark.parametrize("out_features", [0, 3])
def test_zero_feature_weight_round_trips_through_quantization_and_dequantization(
    out_features: int,
) -> None:
    dense = torch.empty(out_features, 0, dtype=torch.bfloat16)

    weight = ConvRotInt8Tensor.from_hp(dense, group_size=16)
    result = weight.dequantize()

    assert weight.qdata.shape == dense.shape
    assert weight.scale.shape == (out_features, 1)
    assert torch.all(weight.scale == 1e-30)
    assert result.shape == dense.shape
    assert result.dtype is dense.dtype
    assert result.device == dense.device


@pytest.mark.parametrize("out_features", [0, 3])
def test_cpu_addmm_handles_zero_feature_weight(out_features: int) -> None:
    weight = ConvRotInt8Tensor.from_packed(
        torch.empty(out_features, 0, dtype=torch.int8),
        torch.ones(out_features, 1, dtype=torch.float32),
        group_size=16,
        dtype=torch.float32,
    )
    mat1 = torch.randn(out_features, 5)
    mat2 = torch.empty(5, 0)

    result = weight.addmm_(mat1, mat2, beta=0.5, alpha=1.25)

    assert result is weight
    assert weight.qdata.shape == (out_features, 0)
    assert weight.scale.shape == (out_features, 1)
    assert torch.all(weight.scale == 1e-30)
    assert weight.dequantize().shape == (out_features, 0)
