"""Shared Triton-free contracts, layouts, schedules, and protocols for sparse Piper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch
from torch._custom_class_base import CustomClassBase
from torch._library.opaque_object import MemberType, register_custom_class

from piper_kernels.attention.kernels.sparse_piper.layout import TILE_ROWS, ValueFormat

if TYPE_CHECKING:
    from ._prepared import (
        _PreparedSparsePiperContext,
        _PreparedSparsePiperOperands,
        _PreparedSparsePiperQuery,
    )

# Activation and output dtypes supported by sparse Piper kernels.
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_output_dtype(dtype: torch.dtype) -> None:
    """Validate the floating output types supported by the kernels."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError("sparse Piper output dtype must be float16, bfloat16, or float32")


# Device-independent routing policy names and static operator modes.
_MINMAX_ROUTING = 0
_MEAN_ROUTING = 1
_ROUTING_MODE_BY_NAME = {
    "minmax": _MINMAX_ROUTING,
    "mean": _MEAN_ROUTING,
}
_ROUTING_NAME_BY_MODE = {mode: name for name, mode in _ROUTING_MODE_BY_NAME.items()}


def is_valid_routing_mode(routing_mode: object) -> bool:
    """Return whether a value names a supported static routing policy."""
    return (
        not isinstance(routing_mode, bool)
        and isinstance(routing_mode, int)
        and routing_mode in _ROUTING_NAME_BY_MODE
    )


def validate_routing_mode(routing_mode: int) -> None:
    """Reject routing modes outside the internal static operator contract."""
    if not is_valid_routing_mode(routing_mode):
        raise ValueError("sparse Piper routing mode must be minmax or mean")


def routing_mode_from_name(routing: str) -> int:
    """Resolve a public routing name to its static operator mode."""
    try:
        return _ROUTING_MODE_BY_NAME[routing]
    except (KeyError, TypeError) as exc:
        raise ValueError("sparse Piper routing must be 'mean' or 'minmax'") from exc


# K64 block-length layout and sparse-query scope helpers.
def validate_block_lengths(
    block_lengths: torch.Tensor,
    *,
    sequence_length: int,
    device: torch.device,
    context: str,
    require_contiguous: bool = True,
    check_values: bool = False,
) -> int:
    """Validate metadata (and optionally value range) for one length per K64 block."""
    if (
        block_lengths.ndim != 1
        or block_lengths.numel() < 1
        or block_lengths.dtype is not torch.int32
        or block_lengths.device != device
        or sequence_length != block_lengths.numel() * TILE_ROWS
        or (require_contiguous and not block_lengths.is_contiguous())
    ):
        contiguity = " contiguous" if require_contiguous else ""
        raise ValueError(
            f"{context} block lengths must be one{contiguity} device INT32 value per K64"
        )
    if check_values and not torch.compiler.is_compiling():
        torch._assert_async(
            torch.all((block_lengths >= 1) & (block_lengths <= TILE_ROWS)),
            f"{context} block lengths must lie in [1, {TILE_ROWS}]",
        )
    return block_lengths.numel()


def valid_block_rows(block_lengths: torch.Tensor) -> torch.Tensor:
    """Return the valid-prefix mask ``[blocks, 64]`` described by lengths."""
    return torch.arange(TILE_ROWS, device=block_lengths.device)[None, :] < block_lengths[:, None]


def validate_sparse_query_blocks(
    sparse_query_blocks: int | None,
    *,
    query_blocks: int,
    context: str,
) -> None:
    """Validate an optional leading sparse-query block count."""
    if sparse_query_blocks is None:
        return
    if isinstance(sparse_query_blocks, bool):
        raise TypeError(f"{context} sparse_query_blocks must be an integer or None")
    if torch.compiler.is_compiling():
        torch._check(
            sparse_query_blocks >= 0,
            lambda: f"{context} sparse_query_blocks must be nonnegative",
        )
        torch._check(
            sparse_query_blocks <= query_blocks,
            lambda: f"{context} sparse_query_blocks cannot exceed the query block count",
        )
        return
    if not isinstance(sparse_query_blocks, int):
        raise TypeError(f"{context} sparse_query_blocks must be an integer or None")
    if not 0 <= sparse_query_blocks <= query_blocks:
        raise ValueError(f"{context} sparse_query_blocks must fit the query block count")


# Fixed-budget caller-forced tiles (metadata validation only; no host readbacks).
def validate_force_tiles(
    ids: torch.Tensor | None,
    counts: torch.Tensor | None,
    *,
    query_blocks: int,
    device: torch.device,
) -> None:
    """Check only storage metadata for one padded ID list per query block."""
    if ids is None and counts is None:
        return
    if ids is None or counts is None:
        raise ValueError("force_tile_ids and force_tile_counts must be supplied together")
    if ids.ndim != 2 or ids.shape[0] != query_blocks or counts.shape != (query_blocks,):
        raise ValueError(
            "forced tiles require IDs [query_blocks,capacity] and counts [query_blocks]"
        )
    if any(t.dtype is not torch.int32 or t.device != device for t in (ids, counts)):
        raise ValueError("forced tile IDs and counts must be INT32 on the attention device")
    if any(t.layout is not torch.strided or not t.is_contiguous() for t in (ids, counts)):
        raise ValueError("forced tile IDs and counts must use contiguous strided storage")


def build_portable_force_tile_bitmask(
    ids: torch.Tensor, counts: torch.Tensor, sparse_key_blocks: int
) -> torch.Tensor:
    """Pack unique active IDs into 32-bit words without reading them on the host."""
    active = torch.arange(ids.shape[1], device=ids.device)[None, :] < counts[:, None]
    safe_ids = torch.where(active, ids, 0).long()
    bits = (1 << (safe_ids % 32)) * active
    words = torch.zeros(
        (ids.shape[0], (sparse_key_blocks + 31) // 32), dtype=torch.int64, device=ids.device
    )
    words.scatter_add_(1, safe_ids // 32, bits)
    return words.to(torch.int32)


# Runtime ratio configuration opaque to Dynamo except for head count.
# Consumers carry ``cudagraph_unsafe`` because automatic CUDA graph replay does
# not guard symbolic custom-class values; explicit CUDA graphs remain valid for
# a fixed profile.
@dataclass(frozen=True, slots=True)
class _SparsePiperRatioProfile(CustomClassBase):
    """Normalized fixed-point ratios; construct from validated Python configuration."""

    units: tuple[int, ...]

    @property
    def num_heads(self) -> int:
        """Number of query heads covered by the profile."""
        return len(self.units)


def _profile_guards(profile: _SparsePiperRatioProfile) -> list[int]:
    return [profile.num_heads]


register_custom_class(
    _SparsePiperRatioProfile,
    typ="symbolic",
    guard_fn=_profile_guards,
    members={"num_heads": MemberType.USE_REAL},
)


# Immutable launch schedules shared across target policies.
class AttentionSchedule(NamedTuple):
    """Query tile size and warp count selected for an attention launch."""

    block_m: int
    num_warps: int


class MinmaxScoreSchedule(NamedTuple):
    """Tile dimensions and fence policy selected for fused FP32 min/max summary scoring."""

    block_m: int
    block_n: int
    block_k: int
    lds_only_fences: bool = False


# Backend producer/consumer protocols.
class PrepareOperands(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        scale: float,
        *,
        sparse_key_blocks: int,
        combined_key: torch.Tensor,
        combined_value: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        sparse_query_blocks: int | None = None,
        emit_summaries: bool = False,
    ) -> _PreparedSparsePiperOperands: ...


class LaunchAttention(Protocol):
    """Run validated query windows against an immutable context."""

    @property
    def context(self) -> _PreparedSparsePiperContext: ...

    def __call__(
        self,
        query_state: _PreparedSparsePiperQuery,
        output: torch.Tensor,
        *,
        query_block_offset: int = 0,
        query_block_count: int | None = None,
        coarse_output: torch.Tensor | None = None,
        coarse_gate: torch.Tensor | None = None,
    ) -> None: ...


class BindAttention(Protocol):
    def __call__(self, context: _PreparedSparsePiperContext) -> LaunchAttention: ...


@dataclass(frozen=True, slots=True)
class AttentionBackend:
    """Producer storage contract and mandatory context binding operation."""

    value_format: ValueFormat
    prepare: PrepareOperands
    bind: BindAttention
    skip_dense_routing: bool = False

    def bind_context(self, context: _PreparedSparsePiperContext) -> LaunchAttention:
        """Validate the producer's format and own its immutable context."""
        if context.value_format is not self.value_format:
            raise ValueError(f"sparse Piper backend requires {self.value_format.name} V storage")
        return self.bind(context)


class SelectRoutes(Protocol):
    def __call__(
        self,
        scores: torch.Tensor,
        routes: torch.Tensor,
        head_keep_blocks: torch.Tensor,
        route_head_offsets: torch.Tensor,
        *,
        query_block_offset: int,
        force_tile_mask: torch.Tensor | None = None,
    ) -> None: ...


class MinmaxScores(Protocol):
    def __call__(
        self,
        query_summary: torch.Tensor,
        key_primary: torch.Tensor,
        key_aux: torch.Tensor,
        *,
        score_scale: float | None = None,
    ) -> torch.Tensor: ...


class SequenceSummaries(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        routing_mode: int,
        block_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
