"""Immutable execution choices for shared ConvRot INT8 projections."""

from dataclasses import dataclass
from typing import Literal

type ProjectionOperation = Literal["query", "key", "value"]


@dataclass(frozen=True, slots=True)
class ProjectionExecutionPlan:
    """Compute settings; each consumer owns its numerical groups and target tuning."""

    block_k: int
    heads_per_program: int
    num_warps: int
    num_stages: int
    block_m: int = 64
    group_m: int = 0
    round_rsqrt_to_nearest: bool = False
    value_mean_block_n: int | None = None
    # Fence only LDS at barriers; an RDNA workaround in _triton.amd_compiler_workarounds.
    lds_only_fences: bool = False

    def resolve_value_mean_block_n(self, head_dim: int) -> int:
        """Return explicit chunk-mean N tiling or fall back to one CTA per head."""
        if self.value_mean_block_n is not None:
            return self.value_mean_block_n
        return head_dim * self.heads_per_program

    @property
    def uses_gluon(self) -> bool:
        """Whether this plan dispatches to a Gluon kernel instead of Triton."""
        return False

    def iter_row_block_launches(
        self,
        batch: int,
        sequence_rows: int,
        input_features: int,
        heads: int,
        *,
        block_m: int | None = None,
        start_row_block: int = 0,
        num_row_blocks: int | None = None,
    ) -> tuple[tuple[tuple[int, int, int], int, bool, bool], ...]:
        """Return ``(grid, row_block_offset, aligned_rows, aligned_projection)`` launches."""
        resolved_block_m = self.block_m if block_m is None else block_m
        head_blocks = (heads + self.heads_per_program - 1) // self.heads_per_program
        aligned_shape = input_features % self.block_k == 0 and heads % self.heads_per_program == 0
        return tuple(
            (
                (row_blocks, head_blocks, batch),
                row_block_offset,
                aligned_rows,
                aligned_rows and aligned_shape,
            )
            for row_blocks, row_block_offset, aligned_rows in resolve_row_block_launches(
                sequence_rows,
                resolved_block_m,
                start_row_block=start_row_block,
                num_row_blocks=num_row_blocks,
            )
        )


def resolve_row_block_launches(
    sequence_rows: int,
    block_m: int,
    *,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> tuple[tuple[int, int, bool], ...]:
    """Return ``(row_blocks, row_block_offset, aligned_rows)`` launches for full or partial runs."""
    full_row_blocks = sequence_rows // block_m
    if start_row_block != 0 or num_row_blocks is not None:
        if start_row_block < 0:
            raise ValueError("start_row_block must be non-negative")
        if num_row_blocks is None:
            if sequence_rows % block_m != 0:
                raise ValueError("partial row-block launches require full block_m alignment")
            block_count = full_row_blocks - start_row_block
        else:
            block_count = num_row_blocks
        if block_count < 0 or start_row_block + block_count > full_row_blocks:
            raise ValueError(
                "partial row-block range must stay within full block_m blocks of sequence_length"
            )
        return ((block_count, start_row_block, True),) if block_count > 0 else ()

    launches: list[tuple[int, int, bool]] = []
    if full_row_blocks:
        launches.append((full_row_blocks, 0, True))
    if sequence_rows % block_m:
        launches.append((1, full_row_blocks, False))
    return tuple(launches)
