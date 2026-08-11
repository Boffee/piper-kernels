"""Model-neutral attention shapes and configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

type AttentionInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
ATTENTION_DTYPE_NAMES = tuple(_DTYPES)
ATTENTION_QKV_LAYOUT = "BHSD"


@dataclass(frozen=True, slots=True)
class AttentionShape:
    """The logical dimensions of an attention invocation."""

    batch_size: int
    num_query_heads: int
    query_length: int
    key_value_length: int
    head_dim: int
    num_key_value_heads: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.batch_size,
            self.num_query_heads,
            self.query_length,
            self.key_value_length,
            self.head_dim,
        )
        if any(value <= 0 for value in values):
            raise ValueError("attention dimensions must be positive")
        if self.num_key_value_heads is not None and self.num_key_value_heads <= 0:
            raise ValueError("key/value heads must be positive")
        if self.num_query_heads % self.effective_num_key_value_heads:
            raise ValueError("query heads must be divisible by key/value heads")

    @property
    def effective_num_key_value_heads(self) -> int:
        """Return the explicit or implicit number of key/value heads."""
        return self.num_key_value_heads or self.num_query_heads

    def as_dict(self) -> dict[str, int]:
        """Return stable machine-readable field names."""
        return {
            "batch_size": self.batch_size,
            "num_query_heads": self.num_query_heads,
            "num_key_value_heads": self.effective_num_key_value_heads,
            "query_length": self.query_length,
            "key_value_length": self.key_value_length,
            "head_dim": self.head_dim,
        }


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Common non-shape settings for attention providers."""

    dtype: torch.dtype
    is_causal: bool = False
    scale: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES.values():
            raise ValueError(f"unsupported attention workload dtype {self.dtype}")

    @property
    def dtype_name(self) -> str:
        """Return the stable CLI and record spelling for the logical dtype."""
        return str(self.dtype).removeprefix("torch.")

    def as_dict(self) -> dict[str, str | bool | float | int | None]:
        """Return stable machine-readable field names."""
        return {
            "dtype": self.dtype_name,
            "is_causal": self.is_causal,
            "scale": self.scale,
            "qkv_layout": ATTENTION_QKV_LAYOUT,
            "seed": self.seed,
        }


def attention_dtype(name: str) -> torch.dtype:
    """Resolve a supported benchmark dtype name."""
    try:
        return _DTYPES[name]
    except KeyError as error:
        raise ValueError(f"unsupported attention dtype {name!r}") from error


def make_attention_inputs(
    shape: AttentionShape,
    *,
    config: AttentionConfig,
    device: torch.device,
) -> AttentionInputs:
    """Create reproducible random Q/K/V tensors for an attention shape."""
    generator = torch.Generator(device=device).manual_seed(config.seed)
    query = torch.randn(
        (shape.batch_size, shape.num_query_heads, shape.query_length, shape.head_dim),
        device=device,
        dtype=config.dtype,
        generator=generator,
    )
    key_shape = (
        shape.batch_size,
        shape.effective_num_key_value_heads,
        shape.key_value_length,
        shape.head_dim,
    )
    key = torch.randn(key_shape, device=device, dtype=config.dtype, generator=generator)
    value = torch.randn(key_shape, device=device, dtype=config.dtype, generator=generator)
    return query, key, value


def run_sdpa(inputs: AttentionInputs, config: AttentionConfig) -> torch.Tensor:
    """Run the common full-precision quality reference."""
    query, key, value = inputs
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=config.scale,
        is_causal=config.is_causal,
    )
