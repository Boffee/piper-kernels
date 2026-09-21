"""Tests for shared stochastic terminal-code selection."""

import pytest
import torch

from piper_kernels.stochastic_quantization import (
    stochastic_codebook_indices,
    stochastic_round_to_int,
)


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
def test_integer_rounding_is_unbiased_reproducible_and_rng_isolated(device: str) -> None:
    values = torch.full((256, 512), 0.3, device=device)
    deterministic = torch.zeros_like(values, dtype=torch.int64)
    cpu_rng_before = torch.random.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state() if device == "cuda" else None

    rounded = stochastic_round_to_int(
        values,
        seed=7,
        quant_min=-2,
        quant_max=2,
        deterministic=deterministic,
    )
    replay = stochastic_round_to_int(
        values,
        seed=7,
        quant_min=-2,
        quant_max=2,
        deterministic=deterministic,
    )
    other = stochastic_round_to_int(
        values,
        seed=8,
        quant_min=-2,
        quant_max=2,
        deterministic=deterministic,
    )

    assert torch.equal(rounded, replay)
    assert not torch.equal(rounded, other)
    assert set(rounded.cpu().unique().tolist()) == {0, 1}
    assert rounded.to(torch.float32).mean().item() == pytest.approx(0.3, abs=0.005)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)


def test_integer_rounding_preserves_deterministic_terminal_codes() -> None:
    values = torch.tensor([[-torch.inf, -2.0, 0.0, 2.0, torch.inf, torch.nan]])
    deterministic = torch.tensor([[41, 42, 43, 44, 45, 46]])

    rounded = stochastic_round_to_int(
        values,
        seed=123,
        quant_min=-2,
        quant_max=2,
        deterministic=deterministic,
    )

    assert torch.equal(rounded, deterministic)


def test_codebook_selection_rejects_a_mismatched_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    values = torch.full((4,), 0.5, device="cuda")
    codebook = torch.tensor((0.0, 1.0))
    deterministic = torch.zeros((4,), device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match="not the values'"):
        stochastic_codebook_indices(
            values,
            codebook,
            seed=1,
            deterministic=deterministic,
        )
