"""The persistent Gluon forward against the Triton step loop.

Final costs and every backpointer byte, written and unwritten, must be equal for the
encoder's state widths, step widths, decode modes and starts. Each batch mixes CTAs
whose rows share one scale with CTAs whose rows do not, so both decode paths run.
"""

import pytest
import torch

from piper_kernels.trellis._layout import TrellisLayout

_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="the persistent forward requires an NVIDIA GPU",
)

_VR45 = TrellisLayout(256, 1144).compute_step_widths()
_MIX45 = (4, 4, 4, 5, 5, 5, 4, 5)
"""A de Bruijn cycle: every step kind ``(k_t, k_t+1, k_t+2)`` once."""
_WIDTHS = {
    "k3": (3,) * 256,
    "k4": (4,) * 256,
    "k5": (5,) * 256,
    "k6": (6,) * 256,
    "vr34": TrellisLayout(256, 888).compute_step_widths(),
    "vr45": _VR45,
    "vr56": TrellisLayout(256, 1400).compute_step_widths(),
    "vr45_224": TrellisLayout(224, 1000).compute_step_widths(),
    "window45": _VR45[-64:] + _VR45[:64],
    "mix34": tuple(k - 1 for k in _MIX45),
    "mix45": _MIX45,
    "mix56": tuple(k + 1 for k in _MIX45),
}
_CASES = [
    (state_bits, name, int8)
    for state_bits, int8s in ((12, (False, True)), (16, (True,)))
    for name in _WIDTHS
    if state_bits == 16 or name != "window45"
    for int8 in int8s
]
"""What the encoder runs: float and INT8 searches over whole blocks with 12-bit states,
and INT8 searches over blocks and wrap windows with 16-bit states."""
_COUNT = 5  # candidates per sequence: row = candidate · batch + sequence
_BATCH = 41  # 205 rows: 35 CTAs of 6, the last one partly padding


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _inputs(
    kind: str, length: int, int8: bool, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(w [T, rows], scale [rows], p2 [rows])``, the scale before folding.

    The first half of the sequences take an FP8 window around ``1.1 · rms / s_c0``
    with ``p2 = s_c0``, so CTAs share their few values; large ones engage the INT8
    clamp. The rest take distinct jittered scales.
    """
    generator = _generator(seed)
    device = torch.device("cuda")
    if kind == "random":
        sequences = torch.randn(_BATCH, length, generator=generator, device=device)
        spikes = torch.rand(_BATCH, length, generator=generator, device=device) < 0.01
        sequences = torch.where(spikes, 6 * sequences, sequences) * 0.02
        second = torch.arange(_BATCH, device=device) >= 24
        top = sequences.abs().amax(1)
        s_c0 = torch.where(second, top[24:].max() / 8, top[:24].max()) / 127
    else:  # zero, constant and three-level sequences: many equal costs
        levels = torch.randint(-1, 2, (_BATCH, length), generator=generator, device=device)
        sequences = torch.where(
            (torch.arange(_BATCH, device=device) % 3 == 0)[:, None], 0.0, levels * 2.0**-7
        )
        sequences[1::3] = 2.0**-7
        s_c0 = torch.full((_BATCH,), 2.0**-12, device=device)
    half = _BATCH // 2
    rms = sequences.square().mean(1).sqrt()
    center = (1.1 * rms / s_c0).clamp(min=2.0**-9).to(torch.float8_e4m3fn).float()
    steps = torch.arange(-(_COUNT // 2), _COUNT // 2 + 1, device=device)
    window = center[:, None] * 2.0 ** (steps / 8)
    window = window.to(torch.float8_e4m3fn).float()
    jitter = 1 + torch.rand(_BATCH, _COUNT, generator=generator, device=device) * 0.5
    distinct = window * jitter if int8 else rms[:, None] / 0.909 * jitter
    scale = torch.cat([window[:half], distinct[half:]]).T.reshape(-1)
    p2 = s_c0 if int8 else torch.cat([s_c0[:half], torch.ones_like(s_c0[half:])])
    w = sequences.T.repeat(1, _COUNT).contiguous()
    return w, scale.contiguous(), p2.repeat(_COUNT).contiguous()


@pytest.mark.gpu
@_nvidia
@pytest.mark.parametrize("constrained", [False, True], ids=["free", "overlap"])
@pytest.mark.parametrize("kind", ["random", "ties"])
@pytest.mark.parametrize(("state_bits", "name", "int8"), _CASES, ids=str)
def test_forward_matches_the_step_loop(
    state_bits: int, name: str, kind: str, int8: bool, constrained: bool
) -> None:
    from piper_kernels.trellis import _gluon_forward, _viterbi  # noqa: PLC0415

    if not _gluon_forward.supports_device(torch.device("cuda")):
        pytest.skip("this GPU or Triton release does not run the persistent forward")
    widths = _WIDTHS[name]
    trellis = _viterbi.SearchTrellis(state_bits, widths)
    length = len(widths)
    seed = 10 * _CASES.index((state_bits, name, int8)) + int(constrained)
    w, scale, p2 = _inputs(kind, length, int8, seed)
    p1 = _viterbi.fold_block_scales(scale) if int8 else scale
    overlap = None
    if constrained:
        high = 1 << (state_bits - widths[0])
        overlap = torch.randint(0, high, (_BATCH,), generator=_generator(seed + 2), device="cuda")
    per_cta = _gluon_forward.compute_rows_per_cta(state_bits, widths)
    order = _gluon_forward._order(p1, per_cta).long()
    slots = p1.view(torch.int32)[order].view(-1, per_cta)
    assert {min(len(set(row)), 2) for row in slots.tolist()} == {1, 2}

    bp_expected = torch.full(
        (length, w.shape[1], 1 << (state_bits - min(widths))), 0xA5, dtype=torch.uint8,
        device="cuda",
    )  # fmt: skip
    expected = _viterbi._build_initial_prefix(trellis, _BATCH, overlap, w.device)
    table = (_viterbi.build_numerator_table if int8 else _viterbi.build_value_table)(
        state_bits, w.device
    )
    for t in range(length):
        fields = _viterbi._compute_step_fields(widths, t)
        expected = _viterbi._run_step(
            expected, table, w[t], p1, p2, bp_expected[t], int8, state_bits, fields
        )
    bp = torch.full_like(bp_expected, 0xA5)
    got = _gluon_forward.run_forward(
        state_bits, widths, int8=int8, w=w, p1=p1, p2=p2, overlap=overlap, bp=bp
    )

    assert torch.equal(got, expected)
    assert torch.equal(bp, bp_expected)


@pytest.mark.gpu
@_nvidia
@pytest.mark.parametrize("min_step_bits", [3, 4, 5, 6])
@pytest.mark.parametrize("state_bits", [16, 12])
def test_encoder_forward_kernels_do_not_spill(state_bits: int, min_step_bits: int) -> None:
    """A spill or a larger shared allocation would cost speed, or the GPUs the kernel
    runs on, without changing a result."""
    from piper_kernels.trellis import _gluon_forward  # noqa: PLC0415

    if not _gluon_forward.supports_device(torch.device("cuda")):
        pytest.skip("this GPU or Triton release does not run the persistent forward")
    k = min_step_bits
    widths = (k,) * 256 if k == 6 else TrellisLayout(256, 256 * k + 120).compute_step_widths()
    index = torch.cuda.current_device()
    for int8 in (False, True):
        specialization = _gluon_forward._specialization(state_bits, widths, int8)
        kernel = _gluon_forward._compiled(specialization, index)
        prefixes = 4 * _gluon_forward.compute_rows_per_cta(state_bits, widths) << (state_bits - k)
        assert kernel.n_spills == 0
        assert kernel.metadata.shared == prefixes <= _gluon_forward.SHARED_NEED


def test_persistent_forward_scope() -> None:
    from piper_kernels.trellis import _gluon_forward  # noqa: PLC0415

    assert all(_gluon_forward.supports_trellis(16, widths) for widths in _WIDTHS.values())
    assert all(_gluon_forward.supports_trellis(12, widths) for widths in _WIDTHS.values())
    assert not _gluon_forward.supports_trellis(14, _VR45)
    assert not _gluon_forward.supports_trellis(16, (2,) * 256)
    assert not _gluon_forward.supports_trellis(16, (4,) * 257)
    assert not _gluon_forward.supports_trellis(16, (4, 6) * 128)
    assert not _gluon_forward.supports_device(torch.device("cpu"))
    assert _gluon_forward._supports_properties((12, 0), _gluon_forward.SHARED_NEED)
    assert _gluon_forward._supports_properties((8, 0), 163 * 1024)
    assert not _gluon_forward._supports_properties((7, 5), 163 * 1024)
    assert not _gluon_forward._supports_properties((8, 6), _gluon_forward.SHARED_NEED - 1)
    assert [_gluon_forward.compute_rows_per_cta(state_bits, (3,)) for state_bits in (16, 12)] == [
        3,
        _gluon_forward.ROWS,
    ]
