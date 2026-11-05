"""Tests for trellis encoding."""

import pytest
import torch

from piper_kernels.trellis import TrellisStorage
from piper_kernels.trellis._layout import TrellisLayout, resolve_block_size, resolve_layout
from piper_kernels.trellis._reference import compute_states
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from trellis_format._encoding_vectors import ENCODER_VECTORS, EncoderVector, build_rows, digest

_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="trellis encoding requires an NVIDIA GPU",
)


@pytest.mark.parametrize(
    ("in_features", "bits_per_weight", "layout"),
    [
        (5120, 4.5, TrellisLayout(256, 1144)),
        (2688, 4.5, TrellisLayout(224, 1000)),
        (5120, 4.03125, TrellisLayout(256, 1024)),
        (5120, 4, TrellisLayout(256, 1016)),
        (5120, 3.5, TrellisLayout(256, 888)),
        (5120, 3.03125, TrellisLayout(256, 768)),
        (5120, 6.05, TrellisLayout(256, 1536)),
        (336, 4.5, TrellisLayout(112, 496)),
        (272, 4.5, TrellisLayout(16, 64)),
    ],
)
def test_resolve_layout_takes_the_most_whole_bytes_within_the_rate(
    in_features: int, bits_per_weight: float, layout: TrellisLayout
) -> None:
    assert resolve_layout(in_features, bits_per_weight) == layout


@pytest.mark.parametrize(
    ("in_features", "bits_per_weight", "message"),
    [
        (5120, 3, "store 3.03125 to 6.03125 bits per weight"),
        (5120, 6.1, "store 3.03125 to 6.03125 bits per weight"),
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
    [TrellisLayout(256, 1144), TrellisLayout(224, 1000), TrellisLayout(16, 96)],
    ids=str,
)
def test_pack_states_inverts_the_decoder_windows(layout: TrellisLayout) -> None:
    from piper_kernels.trellis._encoder import pack_states  # noqa: PLC0415 - needs Triton

    generator = torch.Generator().manual_seed(0)
    shape = (5, layout.bytes_per_block)
    bitstreams = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)

    assert torch.equal(pack_states(compute_states(bitstreams, layout), layout), bitstreams)


def _encode_vector(vector: EncoderVector) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from piper_kernels.trellis._encoder import encode_matrix  # noqa: PLC0415 - needs Triton

    layout = TrellisLayout(resolve_block_size(vector.cols), vector.block_bits)
    return encode_matrix(build_rows(vector.rows, vector.cols, vector.seed).cuda(), layout)


def _digests(storage: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[str, ...]:
    return tuple(digest(tensor) for tensor in storage)


def _expected_digests(vector: EncoderVector) -> tuple[str, ...]:
    return (vector.bitstreams_sha256, vector.block_scales_sha256, vector.scale_sha256)


@pytest.mark.gpu
@_nvidia
@pytest.mark.parametrize("vector", ENCODER_VECTORS, ids=lambda v: f"{v.cols}x{v.block_bits}")
@pytest.mark.parametrize("persistent", [True, False], ids=["persistent", "step_loop"])
def test_encoder_matches_research_encodings(
    vector: EncoderVector, persistent: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from piper_kernels.trellis import _gluon_forward, _viterbi  # noqa: PLC0415 - needs Triton

    if persistent and not _gluon_forward.supports_device(torch.device("cuda")):
        pytest.skip("this GPU or Triton release does not run the persistent forward")
    if not persistent:
        monkeypatch.setattr(_viterbi, "_select_persistent_forward", lambda *_: False)

    assert _digests(_encode_vector(vector)) == _expected_digests(vector)


def _sqnr_db(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    noise = (reference.double() - approximation.double()).square().sum()
    return float(10 * torch.log10(reference.double().square().sum() / noise))


@pytest.mark.parametrize(
    ("bits_per_weight", "min_sqnr_db"), [(3.5, 19.5), (4.5, 25.5), (5.5, 31.0)]
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


@pytest.mark.parametrize(
    ("weight", "group_size", "bits_per_weight", "message"),
    [
        (torch.zeros(2, 256), 256, 4.5, "NVIDIA CUDA device"),
        (torch.zeros(2, 256, 3), 256, 4.5, "2-D"),
        (torch.zeros(2, 256, dtype=torch.float64), 256, 4.5, "float64"),
        (torch.zeros(2, 320), 256, 4.5, "not divisible by group size 256"),
        (torch.zeros(2, 256), 32, 4.5, "group size must be one of"),
        (torch.zeros(2, 256), 256, 2.5, "bits per weight"),
    ],
)
def test_from_hp_validates_its_arguments(
    weight: torch.Tensor, group_size: int, bits_per_weight: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrellisStorage.from_hp(weight, group_size=group_size, bits_per_weight=bits_per_weight)
