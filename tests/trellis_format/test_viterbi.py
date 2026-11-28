"""Tests for the FP16x2 Gluon persistent Viterbi forward against a PyTorch oracle."""

from types import ModuleType

import pytest
import torch

from piper_kernels.trellis import TrellisLayout
from piper_kernels.trellis._layout import CODE_RECIPROCAL
from piper_kernels.trellis._reference import compute_code_numerators

_nvidia = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="the persistent forward requires an NVIDIA GPU",
)

_VR23 = TrellisLayout(256, 632).compute_step_widths()
_VR45 = TrellisLayout(256, 1144).compute_step_widths()
_MIX45 = (4, 4, 4, 5, 5, 5, 4, 5)
"""A de Bruijn cycle covering all 8 adjacent 2-width step pairs."""
_WIDTHS = {
    "k2": (2,) * 16,
    "k3": (3,) * 16,
    "k4": (4,) * 16,
    "k5": (5,) * 16,
    "k6": (6,) * 16,
    "vr23": _VR23,
    "b520": TrellisLayout(256, 520).compute_step_widths(),
    "b760": TrellisLayout(256, 760).compute_step_widths(),
    "vr34": TrellisLayout(256, 888).compute_step_widths(),
    "vr45": _VR45,
    "vr56": TrellisLayout(256, 1400).compute_step_widths(),
    "vr23_224": TrellisLayout(224, 552).compute_step_widths(),
    "vr45_224": TrellisLayout(224, 1000).compute_step_widths(),
    "window23": _VR23[-32:] + _VR23[:32],
    "window45": _VR45[-32:] + _VR45[:32],
    "mix23": tuple(k - 2 for k in _MIX45),
    "mix34": tuple(k - 1 for k in _MIX45),
    "mix45": _MIX45,
    "mix56": tuple(k + 1 for k in _MIX45),
}
_BATCH = 29  # odd batch size with mixed FP8 buckets and padding slots


def _gluon() -> ModuleType:
    from piper_kernels.trellis._nvidia import (  # noqa: PLC0415 - Triton is optional
        gluon_forward as _gluon_forward,
    )

    return _gluon_forward


def _pytorch_viterbi_fp16(
    state_bits: int,
    widths: tuple[int, ...],
    w_norm_h: torch.Tensor,
    alpha: torch.Tensor,
    overlap: torch.Tensor | None,
    *,
    cost_only: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch FP16 Viterbi oracle matching the Gluon kernel's arithmetic."""
    fp16_scale = _gluon().FP16_SCALE
    n, length = w_norm_h.shape
    device = w_norm_h.device
    num_table = compute_code_numerators(torch.arange(1 << state_bits, device=device)).float()
    q_table = (
        torch.clamp(torch.round(num_table[None, :] * alpha[:, None]), -127.0, 127.0) * fp16_scale
    ).to(torch.float16)
    k0 = widths[0]
    if overlap is None:
        costs = torch.zeros((n, 1 << (state_bits - k0)), dtype=torch.float16, device=device)
    else:
        costs = torch.full(
            (n, 1 << (state_bits - k0)), float("inf"), dtype=torch.float16, device=device
        )
        costs.scatter_(1, overlap[:, None], 0.0)
    backpointers: list[torch.Tensor] = []
    for t in range(length):
        ke, ka = widths[t], widths[(t + 1) % length]
        states = torch.arange(1 << state_bits, device=device).view(1 << ka, 1 << (state_bits - ka))
        diff = w_norm_h[:, t : t + 1, None] - q_table[:, states]
        costs, arg = torch.addcmul(costs[:, states >> ke], diff, diff).min(dim=1)
        backpointers.append(arg.to(torch.uint8))
    if cost_only:
        return costs.amin(dim=1)
    j = costs.argmin(dim=1) if overlap is None else overlap.clone()
    walk = torch.empty((n, length), dtype=torch.int64, device=device)
    for t in range(length - 1, -1, -1):
        ke, ka = widths[t], widths[(t + 1) % length]
        a = backpointers[t].gather(1, j[:, None]).squeeze(1).long()
        state = (a << (state_bits - ka)) | j
        walk[:, t] = state
        j = state >> ke
    return walk


def _make_inputs(
    state_bits: int, widths: tuple[int, ...], seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    fp16_scale = _gluon().FP16_SCALE
    gen = torch.Generator(device="cuda").manual_seed(seed)
    length = len(widths)
    w = torch.randn(_BATCH, length, generator=gen, device="cuda") * 2.5
    w_norm_h = (w.clamp(-256.0, 256.0) * fp16_scale).to(torch.float16)
    fp8_choices = torch.tensor([12.0, 24.0, 40.0, 64.0], device="cuda")
    alpha = fp8_choices[torch.arange(_BATCH, device="cuda") % 4] * CODE_RECIPROCAL
    return w_norm_h, alpha


@pytest.mark.gpu
@_nvidia
@pytest.mark.parametrize("name", list(_WIDTHS))
def test_viterbi_l12_cost_only_matches_oracle(name: str) -> None:
    gluon = _gluon()
    if not gluon.supports_device(torch.device("cuda")):
        pytest.skip("this GPU or Triton release does not run the persistent forward")
    widths = _WIDTHS[name]
    w_norm_h, alpha = _make_inputs(12, widths, seed=100 + list(_WIDTHS).index(name))
    got = gluon.run_viterbi(12, widths, w_norm_h, alpha, cost_only=True)
    expected = _pytorch_viterbi_fp16(12, widths, w_norm_h, alpha, None, cost_only=True)
    assert torch.equal(got, expected)


@pytest.mark.gpu
@_nvidia
@pytest.mark.parametrize("constrained", [False, True], ids=["free", "overlap"])
@pytest.mark.parametrize("name", list(_WIDTHS))
def test_viterbi_l16_walks_match_oracle(name: str, constrained: bool) -> None:
    gluon = _gluon()
    if not gluon.supports_device(torch.device("cuda")):
        pytest.skip("this GPU or Triton release does not run the persistent forward")
    widths = _WIDTHS[name]
    seed = 200 + 2 * list(_WIDTHS).index(name) + int(constrained)
    w_norm_h, alpha = _make_inputs(16, widths, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    overlap = (
        torch.randint(0, 1 << (16 - widths[0]), (_BATCH,), generator=gen, device="cuda")
        if constrained
        else None
    )

    got = gluon.run_viterbi(16, widths, w_norm_h, alpha, overlap, cost_only=False)
    expected = _pytorch_viterbi_fp16(16, widths, w_norm_h, alpha, overlap, cost_only=False)

    num_table = compute_code_numerators(torch.arange(1 << 16, device="cuda")).float()
    q_got = (
        torch.clamp(torch.round(num_table[got] * alpha[:, None]), -127.0, 127.0) * gluon.FP16_SCALE
    ).to(torch.float16)
    q_exp = (
        torch.clamp(torch.round(num_table[expected] * alpha[:, None]), -127.0, 127.0)
        * gluon.FP16_SCALE
    ).to(torch.float16)
    err_got = (w_norm_h.float() - q_got.float()).square().sum(-1)
    err_exp = (w_norm_h.float() - q_exp.float()).square().sum(-1)
    max_rel = ((err_got - err_exp).abs() / err_exp.clamp(min=1e-6)).max().item()
    assert max_rel < 5e-3
    for t in range(1, len(widths)):
        ke = widths[t]
        assert torch.equal(got[:, t] >> ke, got[:, t - 1] & ((1 << (16 - ke)) - 1))

    if constrained:
        assert overlap is not None
        assert torch.equal(got[:, 0] >> widths[0], overlap)
        assert torch.equal(got[:, -1] & ((1 << (16 - widths[0])) - 1), overlap)
    else:
        half = len(widths) // 2
        partial = gluon.run_viterbi(
            16, widths, w_norm_h, alpha, None, cost_only=False, write_bp_from=half
        )
        assert partial.shape == (_BATCH, len(widths) - half)
        assert torch.equal(partial, got[:, half:])


@pytest.mark.gpu
@_nvidia
def test_viterbi_handles_extreme_normalized_inputs_and_validates_overlap() -> None:
    gluon = _gluon()
    if not gluon.supports_device(torch.device("cuda")):
        pytest.skip("this GPU or Triton release does not run the persistent forward")
    widths = _VR45
    w_extreme = torch.full(
        (4, len(widths)), 256.0 * gluon.FP16_SCALE, dtype=torch.float16, device="cuda"
    )
    alpha = torch.full((4,), (2.0**-9) * CODE_RECIPROCAL, dtype=torch.float32, device="cuda")
    cost12 = gluon.run_viterbi(12, widths, w_extreme, alpha, cost_only=True)
    assert bool(torch.isfinite(cost12).all())

    noncontig_alpha = torch.stack([alpha, alpha], dim=1)[:, 0]
    assert not noncontig_alpha.is_contiguous()
    states16 = gluon.run_viterbi(16, widths, w_extreme, noncontig_alpha, cost_only=False)
    for t in range(1, len(widths)):
        ke = widths[t]
        assert torch.equal(states16[:, t] >> ke, states16[:, t - 1] & ((1 << (16 - ke)) - 1))

    overlap = torch.zeros(4, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="overlap is not supported with cost_only=True"):
        gluon.run_viterbi(12, widths, w_extreme, alpha, overlap, cost_only=True)


@pytest.mark.gpu
@_nvidia
@pytest.mark.parametrize("min_step_bits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("state_bits", [16, 12])
def test_encoder_forward_kernels_do_not_spill(state_bits: int, min_step_bits: int) -> None:
    """No register spills and shared memory fits opt-in limit across all schedules."""
    gluon = _gluon()
    if not gluon.supports_device(torch.device("cuda")):
        pytest.skip("this GPU or Triton release does not run the persistent forward")
    k = min_step_bits
    schedules = [(k,) * 256] if k in (2, 6) else []
    if k < 6:
        schedules.append(TrellisLayout(256, 256 * k + 120).compute_step_widths())
    index = torch.cuda.current_device()
    for widths in schedules:
        for cost_only in (False, True):
            spec = gluon._specialization(state_bits, widths, cost_only)
            kernel = gluon._compiled(spec, index)
            assert kernel.n_spills == 0
            assert kernel.metadata.shared <= gluon.SHARED_NEED


def test_persistent_forward_scope() -> None:
    gluon = _gluon()
    assert not gluon.supports_device(torch.device("cpu"))
    assert gluon._supports_properties((12, 0), gluon.SHARED_NEED)
    assert gluon._supports_properties((8, 0), 163 * 1024)
    assert not gluon._supports_properties((7, 5), 163 * 1024)
    assert not gluon._supports_properties((8, 6), gluon.SHARED_NEED - 1)
