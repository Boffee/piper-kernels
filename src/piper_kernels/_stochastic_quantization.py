"""Stochastic terminal-code selection for quantized updates."""

import torch

__all__: list[str] = []


def _uniform(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Draw reproducibly without consuming the process-global RNG."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(
        shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )


def _stochastic_round_to_int(
    values: torch.Tensor,
    *,
    seed: int,
    quant_min: int,
    quant_max: int,
    deterministic: torch.Tensor,
) -> torch.Tensor:
    """Round scaled values to adjacent integers with unbiased probability."""
    if deterministic.shape != values.shape:
        raise ValueError("Deterministic integer qdata does not match the values.")
    finite = torch.nan_to_num(
        values.to(torch.float32),
        nan=0.0,
        posinf=float(quant_max),
        neginf=float(quant_min),
    ).clamp_(quant_min, quant_max)
    lower = finite.floor()
    probability = finite - lower
    uniform = _uniform(
        tuple(values.shape),
        device=values.device,
        seed=seed,
    )
    rounded = lower.add_(uniform < probability).to(torch.int64)
    interior = (
        torch.isfinite(values) & (finite > quant_min) & (finite < quant_max) & (probability > 0)
    )
    return torch.where(
        interior,
        rounded,
        deterministic.to(device=values.device, dtype=torch.int64),
    )
