"""Model-neutral attention shapes and configuration."""

from __future__ import annotations

from dataclasses import dataclass


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

    dtype: str
    is_causal: bool = False
    scale: float | None = None
    qkv_layout: str = "BHSD"

    def as_dict(self) -> dict[str, str | bool | float | None]:
        """Return stable machine-readable field names."""
        return {
            "dtype": self.dtype,
            "is_causal": self.is_causal,
            "scale": self.scale,
            "qkv_layout": self.qkv_layout,
        }
