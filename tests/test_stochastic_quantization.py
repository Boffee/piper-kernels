"""Tests for shared stochastic terminal-code selection."""

import pytest
import torch

from piper_kernels._stochastic_quantization import stochastic_codebook_indices


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available",
                ),
            ],
        ),
    ],
)
def test_codebook_selection_is_unbiased_reproducible_and_rng_isolated(
    device: str,
) -> None:
    values = torch.full((1 << 16,), 1.5, device=device)
    # Deliberately keep storage order different from numerical order.
    codebook = torch.tensor((0.0, 2.0, -1.0, 1.0), device=device)
    deterministic = torch.ones(values.shape, device=device, dtype=torch.int64)
    cpu_rng_before = torch.random.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state() if device == "cuda" else None

    selected = stochastic_codebook_indices(
        values,
        codebook,
        seed=(1 << 64) - 1,
        deterministic=deterministic,
    )
    replay = stochastic_codebook_indices(
        values,
        codebook,
        seed=(1 << 64) - 1,
        deterministic=deterministic,
    )
    other = stochastic_codebook_indices(
        values,
        codebook,
        seed=(1 << 64) - 2,
        deterministic=deterministic,
    )

    assert torch.equal(selected, replay)
    assert not torch.equal(selected, other)
    assert set(selected.cpu().unique().tolist()) == {1, 3}
    assert codebook[selected].mean().item() == pytest.approx(1.5, abs=0.01)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)


def test_codebook_selection_preserves_deterministic_terminal_codes() -> None:
    values = torch.tensor((-torch.inf, -1.0, 0.0, 1.0, torch.inf, torch.nan))
    codebook = torch.tensor((-1.0, 0.0, 1.0))
    deterministic = torch.tensor((2, 0, 1, 2, 0, 1))

    selected = stochastic_codebook_indices(
        values,
        codebook,
        seed=123,
        deterministic=deterministic,
    )

    assert torch.equal(selected, deterministic)
