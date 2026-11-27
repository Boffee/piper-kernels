"""Tests for trellis encoding."""

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from piper_kernels.trellis import (
    TrellisHessian,
    TrellisLayout,
    TrellisStorage,
    resolve_block_size,
    resolve_layout,
)
from piper_kernels.trellis._reference import compute_states
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor
from trellis_format._encoding_vectors import ENCODER_VECTORS, EncoderVector, build_rows

_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="trellis encoding requires an NVIDIA GPU",
)


@pytest.mark.parametrize(
    ("in_features", "bits_per_weight", "layout"),
    [
        (5120, 4.5, TrellisLayout(256, 1128)),
        (2688, 4.5, TrellisLayout(224, 984)),
        (5120, 4.09375, TrellisLayout(256, 1024)),
        (5120, 4, TrellisLayout(256, 1000)),
        (5120, 3.5, TrellisLayout(256, 872)),
        (5120, 3.09375, TrellisLayout(256, 768)),
        (5120, 2.5, TrellisLayout(256, 616)),
        (5120, 2.09375, TrellisLayout(256, 512)),
        (5120, 6.1, TrellisLayout(256, 1536)),
        (336, 4.5, TrellisLayout(112, 480)),
        (272, 3.5, TrellisLayout(16, 32)),
        (272, 4.5, TrellisLayout(16, 48)),
    ],
)
def test_resolve_layout_takes_the_most_whole_bytes_within_the_rate(
    in_features: int, bits_per_weight: float, layout: TrellisLayout
) -> None:
    assert resolve_layout(in_features, bits_per_weight) == layout


@pytest.mark.parametrize(
    ("in_features", "bits_per_weight", "message"),
    [
        (5120, 2.05, "store 2.09375 to 6.09375 bits per weight"),
        (5120, 6.15, "store 2.09375 to 6.09375 bits per weight"),
        (100, 4.5, "multiple of 16 wide, got 100"),
    ],
)
def test_resolve_layout_rejects_unsupported_rates(
    in_features: int, bits_per_weight: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_layout(in_features, bits_per_weight)


@pytest.mark.parametrize(
    "layout",
    [
        TrellisLayout(256, 512),
        TrellisLayout(256, 1144),
        TrellisLayout(224, 1000),
        TrellisLayout(16, 32),
        TrellisLayout(16, 96),
    ],
    ids=str,
)
def test_pack_states_inverts_the_decoder_windows(layout: TrellisLayout) -> None:
    from piper_kernels.trellis._encoder import pack_states  # noqa: PLC0415 - needs Triton

    generator = torch.Generator().manual_seed(0)
    raw = torch.randint(0, 256, (5, layout.code_bytes), dtype=torch.uint8, generator=generator)
    bitstreams = torch.cat((raw, raw[:, :2]), dim=1)

    assert torch.equal(pack_states(compute_states(bitstreams, layout), layout), bitstreams)


@pytest.mark.gpu
@_nvidia
@pytest.mark.parametrize("vector", ENCODER_VECTORS, ids=lambda v: f"{v.cols}x{v.block_bits}")
def test_encoder_meets_sqnr_across_vectors(vector: EncoderVector) -> None:
    from piper_kernels.trellis._encoder import encode_matrix  # noqa: PLC0415 - needs Triton

    rows = build_rows(vector.rows, vector.cols, vector.seed).cuda()
    layout = TrellisLayout(resolve_block_size(vector.cols), vector.block_bits)
    bitstreams, block_scales, scale = encode_matrix(rows, layout)

    decoded = ConvRotInt8Tensor.from_trellis(
        bitstreams, block_scales, scale, block_size=layout.block_size, group_size=16
    )
    assert _sqnr_db(rows, decoded.qdata.float() * decoded.scale) > vector.min_sqnr_db


def _sqnr_db(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    noise = (reference.double() - approximation.double()).square().sum()
    return float(10 * torch.log10(reference.double().square().sum() / noise))


@pytest.mark.parametrize(
    ("bits_per_weight", "min_sqnr_db"),
    [(2.09375, 10.5), (2.5, 13.5), (3.5, 19.5), (4.5, 25.5), (5.5, 31.0)],
)
@pytest.mark.gpu
@_nvidia
def test_from_hp_round_trips_through_int8_decode(
    bits_per_weight: float, min_sqnr_db: float
) -> None:
    weight = (torch.randn(48, 1024, generator=torch.Generator().manual_seed(1)) * 0.02).cuda()
    weight = weight.to(torch.bfloat16)

    storage = TrellisStorage.from_hp(weight, group_size=256, bits_per_weight=bits_per_weight)
    decoded = ConvRotInt8Tensor.from_trellis(
        storage.bitstreams,
        storage.block_scales,
        storage.scale,
        block_size=storage.block_size,
        group_size=256,
    )
    direct = ConvRotTrellisTensor.from_trellis_storage(
        storage, group_size=256, logical_dtype=torch.bfloat16
    )

    layout = resolve_layout(1024, bits_per_weight)
    assert storage.block_size == layout.block_size == 256
    assert storage.bitstreams.shape == (48, 4, layout.bytes_per_block)
    assert storage.bitstreams.dtype == torch.uint8
    assert storage.block_scales.shape == (48, 4)
    assert storage.block_scales.dtype == torch.float8_e4m3fn
    assert storage.scale.shape == (48, 1)
    assert storage.scale.dtype == torch.float32
    assert decoded.dtype == torch.bfloat16
    assert _sqnr_db(weight, decoded.dequantize(torch.float32)) > min_sqnr_db
    assert _sqnr_db(weight, direct.dequantize(torch.float32)) > min_sqnr_db


@pytest.mark.gpu
@_nvidia
def test_from_hp_encodes_outlier_rows() -> None:
    generator = torch.Generator().manual_seed(2)
    heterogeneous = torch.randn(8, 512, generator=generator) * 0.02
    heterogeneous[:, 0:64] *= 4.0
    heterogeneous[:, 256:320] *= 4.0
    heterogeneous_cuda = heterogeneous.cuda()

    heterogeneous_storage = TrellisStorage.from_hp(
        heterogeneous_cuda, group_size=64, bits_per_weight=4.5
    )
    heterogeneous_decoded = ConvRotInt8Tensor.from_trellis(
        heterogeneous_storage.bitstreams,
        heterogeneous_storage.block_scales,
        heterogeneous_storage.scale,
        block_size=heterogeneous_storage.block_size,
        group_size=64,
    ).dequantize(torch.float32)
    assert _sqnr_db(heterogeneous_cuda, heterogeneous_decoded) > 23.0

    h4_row = torch.tensor((1.0, 1.0, 1.0, -1.0)) / 2.0
    h256_row = h4_row
    for _ in range(3):
        h256_row = torch.kron(h256_row, h4_row)
    spike_row = torch.randn(4, 512, generator=generator) * 0.02
    spike_row[:, :256] += 10.0 * 0.02 * h256_row
    spike_cuda = spike_row.cuda()

    spike_storage = TrellisStorage.from_hp(spike_cuda, group_size=256, bits_per_weight=4.5)
    spike_decoded = ConvRotInt8Tensor.from_trellis(
        spike_storage.bitstreams,
        spike_storage.block_scales,
        spike_storage.scale,
        block_size=spike_storage.block_size,
        group_size=256,
    ).dequantize(torch.float32)
    assert _sqnr_db(spike_cuda, spike_decoded) > 21.0


@pytest.mark.gpu
@_nvidia
def test_row_encodings_are_independent_of_neighboring_rows() -> None:
    # At group size 16 a lone impulse spreads over 16 of the block's 256 weights.
    # Its peak seed exceeds its RMS seed, while a constant row has just one seed.
    flat = torch.full((1, 256), 0.02, device="cuda")
    impulse = torch.zeros_like(flat)
    impulse[0, 0] = 1
    flat_storage, impulse_storage = (
        TrellisStorage.from_hp(row, group_size=16, bits_per_weight=4.5) for row in (flat, impulse)
    )
    separate = (flat_storage, impulse_storage, flat_storage)
    mixed = TrellisStorage.from_hp(
        torch.cat((flat, impulse, flat)), group_size=16, bits_per_weight=4.5
    )

    assert torch.equal(mixed.bitstreams, torch.cat([s.bitstreams for s in separate]))
    assert torch.equal(mixed.block_scales, torch.cat([s.block_scales for s in separate]))
    assert torch.equal(mixed.scale, torch.cat([s.scale for s in separate]))


@pytest.mark.gpu
@_nvidia
def test_from_hp_encodes_zero_rows_and_no_rows() -> None:
    weight = torch.zeros(3, 256, device="cuda")

    storage = TrellisStorage.from_hp(weight, group_size=16, bits_per_weight=4.5)
    decoded = ConvRotInt8Tensor.from_trellis(
        storage.bitstreams,
        storage.block_scales,
        storage.scale,
        block_size=storage.block_size,
        group_size=16,
    ).dequantize(torch.float32)
    empty = TrellisStorage.from_hp(weight[:0], group_size=16, bits_per_weight=4.5)

    assert bool(torch.isfinite(storage.scale).all())
    assert float(decoded.abs().max()) < 1e-5
    assert empty.bitstreams.shape == (0, 1, 143)
    assert empty.scale.shape == (0, 1)


@pytest.mark.gpu
@_nvidia
def test_from_hp_rejects_non_finite_weights() -> None:
    weight = torch.zeros(2, 256, device="cuda")
    weight[1, 7] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        TrellisStorage.from_hp(weight, group_size=256, bits_per_weight=4.5)


@pytest.mark.gpu
@_nvidia
def test_from_hp_encodes_large_finite_weights_without_overflow() -> None:
    weight = (torch.randn(8, 512, generator=torch.Generator().manual_seed(3)) * 1e20).cuda()

    storage = TrellisStorage.from_hp(weight, group_size=256, bits_per_weight=4.5)
    decoded = ConvRotInt8Tensor.from_trellis(
        storage.bitstreams,
        storage.block_scales,
        storage.scale,
        block_size=storage.block_size,
        group_size=256,
    ).dequantize(torch.float32)

    assert bool(torch.isfinite(storage.scale).all())
    assert bool(torch.isfinite(storage.block_scales.float()).all())
    assert bool(torch.isfinite(decoded).all())
    assert _sqnr_db(weight, decoded) > 25.0


@pytest.mark.parametrize(
    ("cols", "bytes_per_block", "bits_per_weight"),
    [
        (512, 66, 2.09375),
        (512, 143, 4.5),
        (448, 58, 2.11),
        (448, 72, 2.61),
        (448, 86, 3.11),
        (448, 125, 4.5),
        (448, 170, 6.11),
    ],
)
@pytest.mark.gpu
@_nvidia
def test_from_hp_bytes_per_block_matches_bits_per_weight(
    cols: int, bytes_per_block: int, bits_per_weight: float
) -> None:
    weight = (torch.randn(4, cols, generator=torch.Generator().manual_seed(cols)) * 0.02).cuda()
    weight = weight.to(torch.bfloat16)

    exact = TrellisStorage.from_hp(weight, group_size=64, bytes_per_block=bytes_per_block)
    by_rate = TrellisStorage.from_hp(weight, group_size=64, bits_per_weight=bits_per_weight)

    assert exact.block_size == by_rate.block_size
    assert torch.equal(exact.bitstreams, by_rate.bitstreams)
    assert torch.equal(exact.block_scales, by_rate.block_scales)
    assert torch.equal(exact.scale, by_rate.scale)


@pytest.mark.parametrize(
    ("weight", "group_size", "bits_per_weight", "message"),
    [
        (torch.zeros(2, 256), 256, 4.5, "NVIDIA CUDA device"),
        (torch.zeros(2, 256, 3), 256, 4.5, "2-D"),
        (torch.zeros(2, 256, dtype=torch.float64), 256, 4.5, "float64"),
        (torch.zeros(2, 320), 256, 4.5, "not divisible by group size 256"),
        (torch.zeros(2, 256), 32, 4.5, "group size must be one of"),
        (torch.zeros(2, 256), 256, 1.5, "bits per weight"),
    ],
)
def test_from_hp_validates_its_arguments(
    weight: torch.Tensor, group_size: int, bits_per_weight: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrellisStorage.from_hp(weight, group_size=group_size, bits_per_weight=bits_per_weight)


@pytest.mark.parametrize(
    ("kwargs", "exc", "message"),
    [
        ({"group_size": 256}, ValueError, "specify exactly one"),
        (
            {"group_size": 256, "bits_per_weight": 4.5, "bytes_per_block": 143},
            ValueError,
            "specify exactly one",
        ),
        (
            {"group_size": 256, "bytes_per_block": True},
            TypeError,
            "bytes_per_block must be an integer",
        ),
        ({"group_size": 256, "bytes_per_block": 65}, ValueError, "take 64 to 192 whole code bytes"),
        (
            {"group_size": 256, "bytes_per_block": 195},
            ValueError,
            "take 64 to 192 whole code bytes",
        ),
    ],
)
def test_from_hp_validates_bytes_per_block_arguments(
    kwargs: dict[str, object], exc: type[Exception], message: str
) -> None:
    weight = torch.zeros(2, 256)
    with pytest.raises(exc, match=message):
        TrellisStorage.from_hp(weight, **kwargs)  # type: ignore[arg-type]


def _regular_hadamard_block(size: int, dtype: torch.dtype) -> torch.Tensor:
    h4 = torch.tensor(
        ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1)),
        dtype=dtype,
    )
    mat = h4
    cur = 4
    while cur < size:
        mat = torch.kron(mat, h4)
        cur *= 4
    return mat / math.sqrt(size)


def _rotate_groups_ref(value: torch.Tensor, group_size: int) -> torch.Tensor:
    mat = _regular_hadamard_block(group_size, value.dtype).to(value.device)
    features = value.shape[-1]
    return torch.matmul(value.reshape(-1, features // group_size, group_size), mat).reshape(
        value.shape
    )


def _correlated_hessian(
    cols: int,
    *,
    num_samples: int = 2048,
    seed: int = 11,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    chunks = [torch.randn((num_samples, 256), generator=generator)]
    for _ in range(1, (cols + 255) // 256):
        chunks.append(
            0.85 * chunks[-1] + 0.52 * torch.randn((num_samples, 256), generator=generator)
        )
    activations = torch.cat(chunks, dim=1)[:, :cols].to(device=device)
    hessian = (activations.T @ activations) / num_samples
    return hessian, activations


@pytest.mark.parametrize(("group_size", "cols"), [(16, 512), (64, 448), (256, 512)])
def test_trellis_hessian_structure_and_factorization_identity(group_size: int, cols: int) -> None:
    generator = torch.Generator().manual_seed(group_size + cols)
    x = torch.randn(cols + 64, cols, generator=generator, dtype=torch.float64)
    hessian = (x.T @ x) / x.shape[0]
    factor = TrellisHessian.from_hessian(hessian.float(), group_size=group_size, damping=0.01)

    expected_block = resolve_block_size(cols)
    assert factor.group_size == group_size
    assert factor.block_size == expected_block
    assert factor.damping == pytest.approx(0.01)
    assert factor.feedback.dtype is torch.float32
    assert factor.feedback.is_contiguous()
    assert factor.feedback.shape == (cols, cols)
    assert factor.to("cpu") is factor

    eye_block = torch.eye(expected_block, dtype=torch.float32)
    blocks = cols // expected_block
    for b in range(blocks):
        s, e = b * expected_block, (b + 1) * expected_block
        torch.testing.assert_close(factor.feedback[s:e, s:e], eye_block, atol=1e-6, rtol=1e-6)
        if e < cols:
            assert float(factor.feedback[s:e, e:].abs().max()) == 0.0

    # Verify feedback[e:, s:e] @ L[s:e, s:e] == L[e:, s:e] and algebraic update cancellation.
    h_rot = _rotate_groups_ref(_rotate_groups_ref(hessian, group_size).t(), group_size)
    h_rot = 0.5 * (h_rot + h_rot.t())
    h_rot.diagonal().add_(factor.damping * float(h_rot.diagonal().mean().item()))
    l_chol = torch.linalg.cholesky(h_rot).float()
    for b in range(blocks - 1):
        s, e = b * expected_block, (b + 1) * expected_block
        torch.testing.assert_close(
            factor.feedback[e:, s:e] @ l_chol[s:e, s:e], l_chol[e:, s:e], atol=1e-4, rtol=1e-4
        )

    w_orig = torch.randn(12, cols, generator=generator, dtype=torch.float32)
    w_work = w_orig.clone()
    w_targets = torch.empty_like(w_orig)
    w_hat = torch.empty_like(w_orig)
    for b in range(blocks - 1, -1, -1):
        s, e = b * expected_block, (b + 1) * expected_block
        w_targets[:, s:e] = w_work[:, s:e]
        w_hat[:, s:e] = w_targets[:, s:e] if b < blocks - 1 else (w_targets[:, s:e] * 0.9)
        if b > 0:
            w_work[:, :s].addmm_(w_orig[:, s:e] - w_hat[:, s:e], factor.feedback[s:e, :s])
    torch.testing.assert_close(
        (w_orig - w_hat) @ factor.feedback, w_targets - w_hat, atol=1e-4, rtol=1e-4
    )


def test_trellis_hessian_scale_invariance_psd_and_zero_fallback() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(64, 512, generator=generator)
    psd = (x.T @ x) / x.shape[0]
    spd = psd + 0.1 * torch.eye(512)

    base = TrellisHessian.from_hessian(spd, group_size=64, damping=0.01)
    scaled = TrellisHessian.from_hessian(spd * 100.0, group_size=64, damping=0.01)
    torch.testing.assert_close(base.feedback, scaled.feedback, atol=1e-5, rtol=1e-5)

    # Undamped rank-deficient PSD matrix fails Cholesky; positive damping retries cleanly.
    with pytest.raises(ValueError, match="not positive definite"):
        TrellisHessian.from_hessian(psd, group_size=64, damping=0.0)
    slightly_indef = psd - 2e-5 * float(psd.diagonal().mean().item()) * torch.eye(512)
    escalated = TrellisHessian.from_hessian(slightly_indef, group_size=64, damping=1e-5)
    assert escalated.damping > 1e-5

    zeros = TrellisHessian.from_hessian(torch.zeros(256, 256), group_size=256, damping=0.01)
    assert zeros.damping == pytest.approx(0.01)
    torch.testing.assert_close(zeros.feedback, torch.eye(256))

    huge = torch.eye(256, dtype=torch.float64) * 1e308
    huge_ref = huge.clone()
    huge_factor = TrellisHessian.from_hessian(huge, group_size=256)
    torch.testing.assert_close(huge_factor.feedback, torch.eye(256))
    assert torch.equal(huge, huge_ref)

    tiny = torch.eye(256, dtype=torch.float64) * 1e-200
    tiny_factor = TrellisHessian.from_hessian(tiny, group_size=256)
    torch.testing.assert_close(tiny_factor.feedback, torch.eye(256))

    eye_256 = torch.eye(256, dtype=torch.float64)
    ill_spd = torch.zeros((512, 512), dtype=torch.float64)
    ill_spd[:256, :256] = 1e-80 * eye_256
    ill_spd[256:, :256] = 1e-40 * eye_256
    ill_spd[:256, 256:] = 1e-40 * eye_256
    ill_spd[256:, 256:] = 2.0 * eye_256
    with pytest.raises(ValueError, match="overflowed float32"):
        TrellisHessian.from_hessian(ill_spd, group_size=256, damping=0.0)


def test_trellis_hessian_validation() -> None:
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        TrellisHessian.from_hessian("invalid", group_size=64)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        TrellisHessian("invalid", group_size=64, block_size=256)  # type: ignore[arg-type]

    valid_h = torch.eye(256)
    meta_h = torch.empty((256, 256), device="meta")
    with pytest.raises(ValueError, match="meta"):
        TrellisHessian.from_hessian(meta_h, group_size=64)
    with pytest.raises(ValueError, match="meta"):
        TrellisHessian.from_hessian(valid_h, group_size=64, device="meta")
    with pytest.raises(ValueError, match="meta"):
        TrellisHessian(meta_h, group_size=64, block_size=256)
    with pytest.raises(ValueError, match="group size"):
        TrellisHessian.from_hessian(valid_h, group_size=32)
    with pytest.raises(ValueError, match="hold 16 to 256"):
        TrellisHessian.from_hessian(valid_h, group_size=64, block_size=512)
    with pytest.raises(ValueError, match="divisible by group size"):
        TrellisHessian.from_hessian(torch.eye(320), group_size=256)
    with pytest.raises(ValueError, match="divisible by block size"):
        TrellisHessian.from_hessian(valid_h, group_size=64, block_size=48)
    with pytest.raises(ValueError, match="square 2-D"):
        TrellisHessian.from_hessian(torch.ones(256, 128), group_size=64)
    with pytest.raises(ValueError, match="floating-point dtype"):
        TrellisHessian.from_hessian(torch.ones(256, 256, dtype=torch.int8), group_size=64)
    with pytest.raises(ValueError, match="non-negative"):
        TrellisHessian.from_hessian(valid_h, group_size=64, damping=-0.01)
    with pytest.raises(TypeError, match="real scalar"):
        TrellisHessian.from_hessian(valid_h, group_size=64, damping=True)
    with pytest.raises(ValueError, match="finite"):
        TrellisHessian.from_hessian(torch.full((256, 256), float("nan")), group_size=64)
    with pytest.raises(ValueError, match="not positive definite"):
        TrellisHessian.from_hessian(-torch.eye(256), group_size=64, damping=0.01)

    with pytest.raises(ValueError, match="contiguous float32"):
        TrellisHessian(torch.eye(256, dtype=torch.float64), group_size=64, block_size=256)
    with pytest.raises(ValueError, match="contiguous float32"):
        TrellisHessian(torch.eye(256).t(), group_size=64, block_size=256)


def test_from_hp_hessian_validation() -> None:
    valid_h = torch.eye(256)
    weight = torch.zeros(4, 256)
    factor = TrellisHessian.from_hessian(valid_h, group_size=256, block_size=256)
    with pytest.raises(ValueError, match="hessian_damping requires hessian"):
        TrellisStorage.from_hp(weight, group_size=256, bits_per_weight=4.5, hessian_damping=0.01)
    with pytest.raises(ValueError, match="cannot be passed with a pre-factored"):
        TrellisStorage.from_hp(
            weight, group_size=256, bits_per_weight=4.5, hessian=factor, hessian_damping=0.01
        )
    with pytest.raises(TypeError, match=r"torch\.Tensor, TrellisHessian, or None"):
        TrellisStorage.from_hp(
            weight,
            group_size=256,
            bits_per_weight=4.5,
            hessian="invalid",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"trellis Hessian shape .* does not match"):
        TrellisStorage.from_hp(weight, group_size=256, bits_per_weight=4.5, hessian=torch.eye(512))
    wrong_group = TrellisHessian.from_hessian(valid_h, group_size=64, block_size=256)
    with pytest.raises(ValueError, match="group_size 64 does not match 256"):
        TrellisStorage.from_hp(weight, group_size=256, bits_per_weight=4.5, hessian=wrong_group)
    wrong_block = TrellisHessian.from_hessian(valid_h, group_size=256, block_size=128)
    with pytest.raises(ValueError, match="block_size 128 does not match 256"):
        TrellisStorage.from_hp(weight, group_size=256, bits_per_weight=4.5, hessian=wrong_block)
    with pytest.raises(ValueError, match="meta"):
        TrellisStorage.from_hp(
            weight,
            group_size=256,
            bits_per_weight=4.5,
            hessian=torch.empty((256, 256), device="meta"),
        )


@pytest.mark.parametrize(
    ("bits_per_weight", "min_gain_db"),
    [(2.5, 1.5), (3.0, 2.0), (4.5, 2.0)],
)
@pytest.mark.gpu
@_nvidia
def test_from_hp_ldlq_improves_correlated_output_sqnr(
    bits_per_weight: float, min_gain_db: float
) -> None:
    cols, group_size = 1024, 256
    hessian_cpu, activations_cpu = _correlated_hessian(cols, seed=40 + int(bits_per_weight * 10))
    hessian = hessian_cpu.cuda()
    activations = activations_cpu.cuda()
    weight = (torch.randn(128, cols, generator=torch.Generator().manual_seed(5)) * 0.02).cuda()
    weight = weight.to(torch.bfloat16)

    datafree = TrellisStorage.from_hp(
        weight, group_size=group_size, bits_per_weight=bits_per_weight
    )
    pre_factored = TrellisHessian.from_hessian(hessian, group_size=group_size)
    ldlq_pre = TrellisStorage.from_hp(
        weight, group_size=group_size, bits_per_weight=bits_per_weight, hessian=pre_factored
    )
    ldlq_raw = TrellisStorage.from_hp(
        weight, group_size=group_size, bits_per_weight=bits_per_weight, hessian=hessian
    )
    ldlq_cpu_h = TrellisStorage.from_hp(
        weight, group_size=group_size, bits_per_weight=bits_per_weight, hessian=hessian_cpu
    )

    for other in (ldlq_raw, ldlq_cpu_h):
        assert torch.equal(ldlq_pre.bitstreams, other.bitstreams)
        assert torch.equal(ldlq_pre.block_scales, other.block_scales)
        assert torch.equal(ldlq_pre.scale, other.scale)

    ref_out = F.linear(activations, weight.float())
    df_w = ConvRotInt8Tensor.from_trellis(
        datafree.bitstreams,
        datafree.block_scales,
        datafree.scale,
        block_size=datafree.block_size,
        group_size=group_size,
    ).dequantize(torch.float32)
    ldlq_w = ConvRotInt8Tensor.from_trellis(
        ldlq_pre.bitstreams,
        ldlq_pre.block_scales,
        ldlq_pre.scale,
        block_size=ldlq_pre.block_size,
        group_size=group_size,
    ).dequantize(torch.float32)

    df_sqnr = _sqnr_db(ref_out, F.linear(activations, df_w))
    ldlq_sqnr = _sqnr_db(ref_out, F.linear(activations, ldlq_w))
    assert ldlq_sqnr - df_sqnr > min_gain_db


@pytest.mark.gpu
@_nvidia
def test_from_hp_ldlq_variable_rows_mixed_rates() -> None:
    cols, group_size = 1024, 256
    blocks = cols // group_size
    hessian_cpu, activations_cpu = _correlated_hessian(cols, seed=99)
    activations = activations_cpu.cuda()
    weight = (
        torch.randn(32, cols, generator=torch.Generator().manual_seed(101), dtype=torch.bfloat16)
        * 0.02
    ).cuda()
    factor = TrellisHessian.from_hessian(hessian_cpu, group_size=group_size, device="cuda")

    widths = [66 if i < 16 else 114 for i in range(32)]
    row_widths = torch.tensor(widths, dtype=torch.uint8, device="cuda")
    low_df = TrellisStorage.from_hp(weight[:16], group_size=group_size, bytes_per_block=66)
    hi_df = TrellisStorage.from_hp(weight[16:], group_size=group_size, bytes_per_block=114)
    low_h = TrellisStorage.from_hp(
        weight[:16], group_size=group_size, bytes_per_block=66, hessian=factor
    )
    hi_h = TrellisStorage.from_hp(
        weight[16:], group_size=group_size, bytes_per_block=114, hessian=factor
    )

    streams_df = torch.cat((low_df.bitstreams.reshape(-1), hi_df.bitstreams.reshape(-1)))
    bscales_df = torch.cat((low_df.block_scales, hi_df.block_scales), dim=0)
    scale_df = torch.cat((low_df.scale, hi_df.scale), dim=0)
    streams_h = torch.cat((low_h.bitstreams.reshape(-1), hi_h.bitstreams.reshape(-1)))
    bscales_h = torch.cat((low_h.block_scales, hi_h.block_scales), dim=0)
    scale_h = torch.cat((low_h.scale, hi_h.scale), dim=0)
    assert streams_h.numel() == blocks * sum(widths)

    df_int8 = ConvRotInt8Tensor.from_trellis(
        streams_df,
        bscales_df,
        scale_df,
        block_size=256,
        group_size=group_size,
        row_widths=row_widths,
    )
    h_int8 = ConvRotInt8Tensor.from_trellis(
        streams_h, bscales_h, scale_h, block_size=256, group_size=group_size, row_widths=row_widths
    )
    h_direct = ConvRotTrellisTensor.from_quantized(
        streams_h,
        bscales_h,
        scale_h,
        block_size=256,
        group_size=group_size,
        logical_dtype=torch.bfloat16,
        row_widths=row_widths,
    )
    torch.testing.assert_close(
        h_direct.dequantize(torch.float32).cpu(),
        h_direct.cpu().dequantize(torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    assert _sqnr_db(h_direct.dequantize(torch.float32), h_int8.dequantize(torch.float32)) > 25.0

    x_bf16 = activations[:16].to(torch.bfloat16)
    ref_out = F.linear(x_bf16, weight)
    df_out = F.linear(x_bf16, df_int8)
    h_out = F.linear(x_bf16, h_direct)
    assert _sqnr_db(ref_out, h_out) - _sqnr_db(ref_out, df_out) > 1.5
    ref_low_m = F.linear(x_bf16[:4], weight)
    df_low_m = F.linear(x_bf16[:4], df_int8)
    h_low_m = F.linear(x_bf16[:4], h_direct)
    assert _sqnr_db(ref_low_m, h_low_m) - _sqnr_db(ref_low_m, df_low_m) > 1.5


@pytest.mark.gpu
@_nvidia
def test_from_hp_ldlq_edge_cases() -> None:
    # Single block (K == 256) has no earlier blocks, so Block-LDLQ matches data-free bit-exactly.
    w_single = (
        torch.randn(8, 256, generator=torch.Generator().manual_seed(201), dtype=torch.bfloat16)
        * 0.02
    ).cuda()
    h_single, _ = _correlated_hessian(256, seed=202, device="cuda")
    s_df = TrellisStorage.from_hp(w_single, group_size=256, bits_per_weight=3.5)
    s_h = TrellisStorage.from_hp(w_single, group_size=256, bits_per_weight=3.5, hessian=h_single)
    assert torch.equal(s_df.bitstreams, s_h.bitstreams)
    assert torch.equal(s_df.block_scales, s_h.block_scales)
    assert torch.equal(s_df.scale, s_h.scale)

    # Empty, all-zero, large finite weights, row-independence, and non-pow2 width (448 -> 224).
    h512 = _correlated_hessian(512, seed=203)[0].cuda()
    factor_512 = TrellisHessian.from_hessian(h512, group_size=64)
    empty = TrellisStorage.from_hp(
        torch.empty((0, 512), device="cuda", dtype=torch.bfloat16),
        group_size=64,
        bits_per_weight=3.5,
        hessian=factor_512,
    )
    assert empty.bitstreams.shape == (0, 2, 111)

    zero_s = TrellisStorage.from_hp(
        torch.zeros((4, 512), device="cuda"),
        group_size=64,
        bits_per_weight=3.5,
        hessian=factor_512,
    )
    zero_dec = ConvRotInt8Tensor.from_trellis(
        zero_s.bitstreams, zero_s.block_scales, zero_s.scale, block_size=256, group_size=64
    ).dequantize(torch.float32)
    assert bool(torch.isfinite(zero_s.scale).all())
    assert float(zero_dec.abs().max()) < 1e-5

    large_w = (torch.randn(8, 512, generator=torch.Generator().manual_seed(204)) * 1e20).cuda()
    large_s = TrellisStorage.from_hp(
        large_w, group_size=64, bits_per_weight=4.5, hessian=factor_512
    )
    large_dec = ConvRotInt8Tensor.from_trellis(
        large_s.bitstreams, large_s.block_scales, large_s.scale, block_size=256, group_size=64
    ).dequantize(torch.float32)
    assert bool(torch.isfinite(large_dec).all())
    assert _sqnr_db(large_w, large_dec) > 22.0

    flat = torch.full((1, 512), 0.02, device="cuda")
    impulse = torch.zeros_like(flat)
    impulse[0, 0] = 1.0
    s_flat = TrellisStorage.from_hp(flat, group_size=64, bits_per_weight=3.5, hessian=factor_512)
    s_imp = TrellisStorage.from_hp(impulse, group_size=64, bits_per_weight=3.5, hessian=factor_512)
    s_cat = TrellisStorage.from_hp(
        torch.cat((flat, impulse, flat), dim=0),
        group_size=64,
        bits_per_weight=3.5,
        hessian=factor_512,
    )
    assert torch.equal(
        s_cat.bitstreams, torch.cat((s_flat.bitstreams, s_imp.bitstreams, s_flat.bitstreams))
    )
    assert torch.equal(
        s_cat.block_scales,
        torch.cat((s_flat.block_scales, s_imp.block_scales, s_flat.block_scales)),
    )
    torch.testing.assert_close(
        s_cat.scale, torch.cat((s_flat.scale, s_imp.scale, s_flat.scale)), rtol=1e-6, atol=1e-9
    )

    # Non-power-of-2 width (448 -> block_size 224) and CPU-to-GPU factor movement.
    h448_cpu, x448_cpu = _correlated_hessian(448, seed=205)
    f448_cpu = TrellisHessian.from_hessian(h448_cpu, group_size=64)
    w448 = (torch.randn(16, 448, generator=torch.Generator().manual_seed(206)) * 0.02).cuda()
    with pytest.raises(ValueError, match="does not match weight device"):
        TrellisStorage.from_hp(w448, group_size=64, bits_per_weight=3.5, hessian=f448_cpu)
    f448_gpu = f448_cpu.to("cuda")
    assert f448_gpu.to("cuda") is f448_gpu
    assert f448_gpu.to(f448_gpu.feedback.device) is f448_gpu
    s448 = TrellisStorage.from_hp(w448, group_size=64, bits_per_weight=3.5, hessian=f448_gpu)
    assert s448.block_size == 224
    w448_dec = ConvRotInt8Tensor.from_trellis(
        s448.bitstreams, s448.block_scales, s448.scale, block_size=224, group_size=64
    ).dequantize(torch.float32)
    x448 = x448_cpu.cuda()
    assert _sqnr_db(F.linear(x448, w448), F.linear(x448, w448_dec)) > 20.0


@pytest.mark.gpu
@_nvidia
def test_from_hp_ldlq_autograd_and_multichunk() -> None:
    # Autograd / no_grad safety when feedback or weights carry requires_grad=True.
    h768 = _correlated_hessian(768, seed=207)[0].cuda()
    f768 = TrellisHessian.from_hessian(h768, group_size=256)
    f768_grad = TrellisHessian(
        f768.feedback.clone().requires_grad_(True),
        group_size=256,
        block_size=256,
        damping=f768.damping,
    )
    w768_grad = (
        (torch.randn(8, 768, generator=torch.Generator().manual_seed(208)) * 0.02)
        .cuda()
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    s_clean = TrellisStorage.from_hp(w768_grad, group_size=256, bits_per_weight=3.5, hessian=f768)
    s_grad = TrellisStorage.from_hp(
        w768_grad, group_size=256, bits_per_weight=3.5, hessian=f768_grad
    )
    assert not s_grad.scale.requires_grad
    assert torch.equal(s_grad.bitstreams, s_clean.bitstreams)
    assert torch.equal(s_grad.block_scales, s_clean.block_scales)
    assert torch.equal(s_grad.scale, s_clean.scale)

    # Multi-chunk Block-LDLQ (> 4096 rows) matches encoding the tail chunk directly.
    h512_chunk, x512_chunk = _correlated_hessian(512, seed=209, device="cuda")
    factor_chunk = TrellisHessian.from_hessian(h512_chunk, group_size=256)
    w_big = (
        torch.randn(4100, 512, generator=torch.Generator().manual_seed(210), dtype=torch.bfloat16)
        * 0.02
    ).cuda()
    s_big = TrellisStorage.from_hp(w_big, group_size=256, bits_per_weight=3.5, hessian=factor_chunk)
    assert s_big.bitstreams.shape == (4100, 2, 111)
    s_tail_h = TrellisStorage.from_hp(
        w_big[4096:], group_size=256, bits_per_weight=3.5, hessian=factor_chunk
    )
    s_tail_df = TrellisStorage.from_hp(w_big[4096:], group_size=256, bits_per_weight=3.5)
    assert torch.equal(s_big.bitstreams[4096:], s_tail_h.bitstreams)
    assert torch.equal(s_big.block_scales[4096:], s_tail_h.block_scales)
    torch.testing.assert_close(s_big.scale[4096:], s_tail_h.scale, rtol=1e-6, atol=1e-9)

    tail_h_dec = ConvRotInt8Tensor.from_trellis(
        s_big.bitstreams[4096:],
        s_big.block_scales[4096:],
        s_big.scale[4096:],
        block_size=256,
        group_size=256,
    ).dequantize(torch.float32)
    tail_df_dec = ConvRotInt8Tensor.from_trellis(
        s_tail_df.bitstreams,
        s_tail_df.block_scales,
        s_tail_df.scale,
        block_size=256,
        group_size=256,
    ).dequantize(torch.float32)
    ref_tail_out = F.linear(x512_chunk, w_big[4096:].float())
    assert _sqnr_db(ref_tail_out, F.linear(x512_chunk, tail_h_dec)) > _sqnr_db(
        ref_tail_out, F.linear(x512_chunk, tail_df_dec)
    )
