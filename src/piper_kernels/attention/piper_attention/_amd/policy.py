"""AMD RDNA4 Split-KV decode scheduling policy for Piper KV attention."""

from piper_kernels._triton.targets import GFX120X_ARCHITECTURES, AcceleratorTarget

from .._decode_schedule import (
    FAST_MAX_DECODE_SPLITS,
    DecodeSchedule,
    default_decode_schedule,
)

# Conservative RDNA4 baseline Split-KV schedule tuned on RX 9070 XT (gfx1201).
_RDNA4_BLOCK_N = 64
_RDNA4_NUM_WARPS = 4
_RDNA4_NUM_STAGES = 1
_RDNA4_MAX_TAIL_SPLITS = 4
_RDNA4_PREFILL_CTAS_PER_SM = 6
_RDNA4_MIN_WAVE_SPLIT_TILES = 128
_RDNA4_SHORT_CTX_CTAS_PER_SM = 4
_MAX_PREFILL_PARTIAL_OUT_BYTES = 64 * 1024 * 1024


def _wave_fill(base_blocks: int, wave_cap: int, splits: int) -> float:
    ctas = splits * base_blocks
    waves = (ctas + wave_cap - 1) // wave_cap
    return ctas / (waves * wave_cap)


def _best_wave_split(
    base_blocks: int,
    wave_cap: int,
    total_tiles: int,
    start_spl: int,
    end_spl: int,
    best_spl: int,
    best_fill: float,
    min_gain: float,
) -> int:
    for spl in range(start_spl, end_spl + 1):
        if total_tiles // spl >= 8:
            cand_fill = _wave_fill(base_blocks, wave_cap, spl)
            better = cand_fill > best_fill if min_gain == 0.0 else cand_fill >= best_fill + min_gain
            if better:
                best_spl = spl
                best_fill = cand_fill
    return best_spl


def select_frozen_prefill_splits(
    base_blocks: int,
    hint_chunks: int,
    sm_count: int,
    bytes_per_split: int,
    *,
    allow_subchunk_single: bool = False,
) -> int:
    """Select RDNA4 frozen chunk-prefill split count aligned to 6-CTA/WGP occupancy."""
    if (not allow_subchunk_single and hint_chunks < 2) or hint_chunks <= 0:
        return 1
    wave_cap = max(1, sm_count * _RDNA4_PREFILL_CTAS_PER_SM)
    total_tiles = hint_chunks * 16
    max_splits = min(
        32,
        total_tiles // 4,
        max(1, _MAX_PREFILL_PARTIAL_OUT_BYTES // max(1, bytes_per_split)),
    )
    if max_splits <= 1 or (
        total_tiles < _RDNA4_MIN_WAVE_SPLIT_TILES
        and base_blocks >= sm_count * _RDNA4_SHORT_CTX_CTAS_PER_SM
    ):
        return 1

    if base_blocks < wave_cap:
        if max_splits * base_blocks < wave_cap:
            return max_splits
        best_spl = min(max_splits, (wave_cap + base_blocks - 1) // base_blocks)
        best_fill = _wave_fill(base_blocks, wave_cap, best_spl)
        return _best_wave_split(
            base_blocks,
            wave_cap,
            total_tiles,
            best_spl + 1,
            min(max_splits, best_spl + 2),
            best_spl,
            best_fill,
            0.10,
        )

    if base_blocks < wave_cap * 2 and total_tiles >= _RDNA4_MIN_WAVE_SPLIT_TILES:
        base_fill = _wave_fill(base_blocks, wave_cap, 1)
        if base_fill < 0.80:
            chosen = _best_wave_split(
                base_blocks, wave_cap, total_tiles, 2, min(4, max_splits), 1, base_fill, 0.0
            )
            if _wave_fill(base_blocks, wave_cap, chosen) >= base_fill + 0.15:
                return chosen

    return 1


def select_decode_schedule(target: AcceleratorTarget, head_dim: int) -> DecodeSchedule:
    """Return the AMD Split-KV kernel and split schedule for ``target`` and ``head_dim``."""
    if not target.is_hip_architecture(*GFX120X_ARCHITECTURES):
        return default_decode_schedule(head_dim)
    from piper_kernels._triton import amd_compiler_workarounds  # noqa: PLC0415

    llvm_fn_attrs: tuple[tuple[str, str], ...] = (
        ("target-features", "-cumode"),
        *amd_compiler_workarounds.LDS_ONLY_FENCE_ATTRS,
    )
    # RDNA4 uses DecodeSchedule only for direct live-tail Split-KV decode; finalized pool
    # slots always execute the native INT8-Q/UINT8-P Gluon prefill kernel.
    return DecodeSchedule(
        block_n=_RDNA4_BLOCK_N,
        num_warps=_RDNA4_NUM_WARPS,
        num_stages=_RDNA4_NUM_STAGES,
        loop_num_stages=_RDNA4_NUM_STAGES,
        frozen_qk_mode="ieee",
        tail_qk_precision="native",
        pv_precision="bf16x2",
        max_splits=FAST_MAX_DECODE_SPLITS,
        target_ctas_per_sm=0,
        tiles_per_chunk_cap=1,
        max_tail_splits=_RDNA4_MAX_TAIL_SPLITS,
        compiler_options=(("llvm_fn_attrs", llvm_fn_attrs),),
    )


__all__ = ["select_decode_schedule", "select_frozen_prefill_splits"]
