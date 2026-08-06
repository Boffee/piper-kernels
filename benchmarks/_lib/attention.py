"""Model-neutral attention benchmark records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttentionShape:
    """The logical dimensions of an attention invocation."""

    batch: int
    query_heads: int
    query_sequence: int
    key_value_sequence: int
    head_dim: int
    key_value_heads: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.batch,
            self.query_heads,
            self.query_sequence,
            self.key_value_sequence,
            self.head_dim,
        )
        if any(value <= 0 for value in values):
            raise ValueError("attention dimensions must be positive")
        if self.key_value_heads is not None and self.key_value_heads <= 0:
            raise ValueError("key/value heads must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must be divisible by key/value heads")

    @property
    def kv_heads(self) -> int:
        """Return the explicit or implicit number of key/value heads."""
        return self.key_value_heads or self.query_heads

    def as_dict(self) -> dict[str, int]:
        """Return stable machine-readable field names."""
        return {
            "batch": self.batch,
            "query_heads": self.query_heads,
            "key_value_heads": self.kv_heads,
            "query_sequence": self.query_sequence,
            "key_value_sequence": self.key_value_sequence,
            "head_dim": self.head_dim,
        }


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Common non-shape settings for attention providers."""

    dtype: str
    is_causal: bool = False
    scale: float | None = None
    tensor_layout: str = "HND"

    def as_dict(self) -> dict[str, str | bool | float | None]:
        """Return stable machine-readable field names."""
        return {
            "dtype": self.dtype,
            "is_causal": self.is_causal,
            "scale": self.scale,
            "tensor_layout": self.tensor_layout,
        }
