"""Tests for trellis encoding."""

import pytest
import torch

from piper_kernels.trellis import TrellisStorage
from piper_kernels.trellis._layout import TrellisLayout, resolve_block_size, resolve_layout
from piper_kernels.trellis._reference import compute_states
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from trellis_format._encoding_vectors import ENCODER_VECTORS, EncoderVector, build_rows

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


@pytest.mark.gpu
@_nvidia
@pytest.mark.parametrize("vector", ENCODER_VECTORS, ids=lambda v: f"{v.cols}x{v.block_bits}")
def test_encoder_matches_across_backends_and_meets_sqnr(
    vector: EncoderVector, monkeypatch: pytest.MonkeyPatch
) -> None:
    from piper_kernels.trellis import _gluon_forward, _viterbi  # noqa: PLC0415 - needs Triton
    from piper_kernels.trellis._encoder import encode_matrix  # noqa: PLC0415 - needs Triton

    rows = build_rows(vector.rows, vector.cols, vector.seed).cuda()
    layout = TrellisLayout(resolve_block_size(vector.cols), vector.block_bits)
    persistent_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if _gluon_forward.supports_device(torch.device("cuda")):
        persistent_out = encode_matrix(rows, layout)

    monkeypatch.setattr(_viterbi, "_select_persistent_forward", lambda *_: False)
    bitstreams, block_scales, scale = encode_matrix(rows, layout)

    if persistent_out is not None:
        persistent_bitstreams, persistent_block_scales, persistent_scale = persistent_out
        assert torch.equal(persistent_bitstreams, bitstreams)
        assert torch.equal(persistent_block_scales, block_scales)
        assert torch.equal(persistent_scale, scale)

    decoded = ConvRotInt8Tensor.from_trellis(
        bitstreams, block_scales, scale, block_size=layout.block_size, group_size=16
    )
    assert _sqnr_db(rows, decoded.qdata.float() * decoded.scale) > vector.min_sqnr_db


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
