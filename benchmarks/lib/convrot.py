"""Model-neutral ConvRot workload and input helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from piper_kernels.convrot._rotation import SUPPORTED_GROUP_SIZES

type ConvRotInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
CONVROT_DTYPE_NAMES = tuple(_DTYPES)


@dataclass(slots=True, frozen=True)
class ConvRotShape:
    """One ConvRot linear shape, where ``in_features`` is linear K."""

    name: str
    rows: int
    out_features: int
    in_features: int
    input_activation: str | None = None
    has_bias: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ConvRot shape name cannot be empty")
        if any(value <= 0 for value in (self.rows, self.out_features, self.in_features)):
            raise ValueError("ConvRot rows, out_features, and in_features must be positive")
        if self.input_activation not in (None, "swiglu"):
            raise ValueError(
                f"ConvRot input activation must be None or 'swiglu', got {self.input_activation!r}"
            )

    def as_dict(self) -> dict[str, int | str]:
        """Return stable machine-readable case and dimension fields."""
        return {
            "case": self.name,
            "rows": self.rows,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "raw_input_features": raw_input_features(
                self.in_features,
                self.input_activation,
            ),
        }


@dataclass(slots=True, frozen=True)
class ConvRotConfig:
    """Common non-shape settings for a reproducible ConvRot workload."""

    dtype: torch.dtype
    group_size: int = 256
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES.values():
            raise ValueError(f"unsupported ConvRot workload dtype {self.dtype}")
        if self.group_size not in SUPPORTED_GROUP_SIZES:
            supported = ", ".join(map(str, SUPPORTED_GROUP_SIZES))
            raise ValueError(
                f"ConvRot workload group size must be one of {supported}, got {self.group_size}"
            )

    @property
    def dtype_name(self) -> str:
        """Return the stable CLI and record spelling for the logical dtype."""
        return str(self.dtype).removeprefix("torch.")

    def as_dict(self) -> dict[str, int | str]:
        """Return stable machine-readable workload settings."""
        return {
            "dtype": self.dtype_name,
            "group_size": self.group_size,
            "seed": self.seed,
        }


def convrot_dtype(name: str) -> torch.dtype:
    """Resolve a supported benchmark dtype name."""
    try:
        return _DTYPES[name]
    except KeyError as error:
        raise ValueError(f"unsupported ConvRot workload dtype {name!r}") from error


def parse_input_activation(value: str) -> str | None:
    """Parse the common ConvRot CLI input-activation spelling."""
    normalized = value.strip().lower()
    if normalized == "none":
        return None
    if normalized == "swiglu":
        return normalized
    raise argparse.ArgumentTypeError("input activation must be 'none' or 'swiglu'")


def raw_input_features(in_features: int, input_activation: str | None) -> int:
    """Return the source width before applying an optional input activation."""
    return in_features * (2 if input_activation == "swiglu" else 1)


def apply_input_activation(
    activation: torch.Tensor,
    input_activation: str | None,
) -> torch.Tensor:
    """Apply the public raw-input activation contract."""
    if input_activation is None:
        return activation
    if input_activation != "swiglu":
        raise ValueError(f"unsupported input activation {input_activation!r}")
    up, gate = activation.chunk(2, dim=-1)
    return up * torch.nn.functional.silu(gate)


def comfy_convrot_input(
    activation: torch.Tensor,
    input_activation: str | None,
) -> torch.Tensor:
    """Adapt Piper's ``[up | gate]`` input to Comfy Kitchen's ``[gate | up]``."""
    if input_activation is None:
        return activation
    if input_activation != "swiglu":
        raise ValueError(f"unsupported input activation {input_activation!r}")
    up, gate = activation.chunk(2, dim=-1)
    return torch.cat((gate, up), dim=-1)


def make_convrot_inputs(
    shape: ConvRotShape,
    config: ConvRotConfig,
    *,
    device: torch.device,
) -> ConvRotInputs:
    """Create one reproducible activation, quantized weight, scale, and bias set."""
    if shape.in_features % config.group_size:
        raise ValueError(
            f"ConvRot in_features {shape.in_features} is not divisible by "
            f"group size {config.group_size}"
        )
    generator = torch.Generator(device=device).manual_seed(config.seed)
    qdata = torch.randint(
        -127,
        128,
        (shape.out_features, shape.in_features),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    scale = (
        torch.rand(
            (shape.out_features, 1),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    )
    activation = torch.randn(
        (shape.rows, raw_input_features(shape.in_features, shape.input_activation)),
        device=device,
        dtype=config.dtype,
        generator=generator,
    )
    bias = (
        torch.randn(
            shape.out_features,
            device=device,
            dtype=config.dtype,
            generator=generator,
        )
        if shape.has_bias
        else None
    )
    return activation, qdata, scale, bias


def quality_row_indices(
    shape: ConvRotShape,
    *,
    maximum_rows: int = 256,
) -> tuple[int, ...]:
    """Stratify quality rows and retain every first signed-32-bit crossing."""
    if maximum_rows <= 0:
        raise ValueError("maximum quality rows must be positive")
    target = min(shape.rows, maximum_rows)
    if target == shape.rows:
        return tuple(range(shape.rows))
    if target == 1:
        return (shape.rows - 1,)

    sampled = {round(index * (shape.rows - 1) / (target - 1)) for index in range(target)}
    critical = {0, shape.rows - 1}
    for row_width in (
        raw_input_features(shape.in_features, shape.input_activation),
        shape.in_features,
        shape.out_features,
    ):
        boundary = ((1 << 31) + row_width - 1) // row_width
        critical.update(
            row for row in (boundary - 1, boundary, boundary + 1) if 0 <= row < shape.rows
        )

    for row in sorted(critical):
        if row in sampled:
            continue
        replaceable = sampled - critical
        if not replaceable:
            break
        victim = min(replaceable, key=lambda candidate: abs(candidate - row))
        sampled.remove(victim)
        sampled.add(row)
    return tuple(sorted(sampled))
