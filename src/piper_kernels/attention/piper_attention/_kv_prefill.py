"""Native INT8-Q/UINT8-P chunked prefill and small-Q decode orchestration for Piper KV pools."""

# Host orchestration threads tensor operands across NVIDIA and AMD prefill backends.
# ruff: noqa: PLR0913, PLR0917

from __future__ import annotations

from typing import Literal

import torch
import triton

from piper_kernels._triton.runtime import device_context
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.linear._splitk import device_sm_count

from . import _backend, _quantization
from ._amd import policy as _amd_policy
from ._decode_schedule import (
    BLOCK_QH,
    CENTER_CHUNK,
    PREFILL_MIN_SEQ_Q,
    select_split_counts,
)
from ._lse_merge import merge_two_attention_waves
from ._nvidia import policy as _nvidia_policy
from ._validation import (
    PrefillKernelOperands,
    WindowMode,
    resolve_kv_window_modes,
)
from .triton import (
    launch_direct_tail_split_decode,
    launch_splitkv_reduce,
    pool_slot_strides,
    select_decode_schedule,
)


def _select_frozen_prefill_splits(
    device: torch.device,
    *,
    prefill_backend: Literal["amd", "nvidia"],
    batch: int,
    q_heads: int,
    seq_q: int,
    head_dim: int,
    max_chunks: int,
    active_chunks_hint: int,
    allow_subchunk_single: bool = False,
    logical_seq_q: int = 0,
    window_left: int = -1,
) -> int:
    sm_count = device_sm_count(device)
    base_blocks = max(1, batch * q_heads * triton.cdiv(seq_q, 64))
    hint_chunks = max_chunks if active_chunks_hint <= 0 else min(max_chunks, active_chunks_hint)
    eff_seq_q = logical_seq_q if logical_seq_q > 0 else seq_q
    if window_left >= 0:
        hint_chunks = min(hint_chunks, triton.cdiv(eff_seq_q + window_left, CENTER_CHUNK))
    policy = _amd_policy if prefill_backend == "amd" else _nvidia_policy
    return policy.select_frozen_prefill_splits(
        base_blocks,
        hint_chunks,
        sm_count,
        batch * q_heads * seq_q * head_dim * 4,
        allow_subchunk_single=allow_subchunk_single,
    )


def _launch_frozen_pool_prefill(
    query_int8: torch.Tensor,
    query_scale: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    seq_q: int,
    grouped_qk: bool,
    prefill_backend: Literal["amd", "nvidia"],
    active_chunks_hint: int,
    target: AcceleratorTarget | None,
    per_row_query_scale: bool = False,
    allow_subchunk_single: bool = False,
    frozen_splits: int | None = None,
    total_splits: int | None = None,
    partial_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    logical_seq_q: int = 0,
    tail_len: torch.Tensor | None = None,
    tail_cap: int = 0,
    window_mode: WindowMode = "none",
    window_left: int = -1,
    window_right: int = -1,
) -> None:
    """Assemble typed pool operands once and dispatch the backend frozen-pool kernel."""
    batch, q_heads, _, head_dim = output.shape
    num_slots, kv_heads, _, _ = key_int8.shape
    query_storage = query_int8.shape[2]
    pool_layout = pool_slot_strides(
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        chunk_to_slot,
        num_chunks,
        batch=batch,
    )
    gqa_group = q_heads // kv_heads

    resolved_splits = (
        _select_frozen_prefill_splits(
            query_int8.device,
            prefill_backend=prefill_backend,
            batch=batch,
            q_heads=q_heads,
            seq_q=seq_q,
            head_dim=head_dim,
            max_chunks=pool_layout.max_chunks,
            active_chunks_hint=active_chunks_hint,
            allow_subchunk_single=allow_subchunk_single,
            logical_seq_q=logical_seq_q,
            window_left=window_left,
        )
        if frozen_splits is None
        else frozen_splits
    )
    resolved_total = resolved_splits if total_splits is None else total_splits
    has_splits = partial_targets is not None or resolved_splits > 1

    out_target = output
    lse_target = lse
    partial_out: torch.Tensor | None = None
    partial_lse: torch.Tensor | None = None
    partial_bias: torch.Tensor | None = None
    partial_log_denom: torch.Tensor | None = None
    bias_target: torch.Tensor | None = None
    log_denom_target: torch.Tensor | None = None
    if partial_targets is not None:
        out_target, lse_target, bias_target, log_denom_target = partial_targets
    elif resolved_splits > 1:
        partial_out = torch.empty(
            (batch, resolved_splits, q_heads, seq_q, head_dim),
            device=output.device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            (batch, resolved_splits, q_heads, seq_q),
            device=output.device,
            dtype=torch.float32,
        )
        partial_bias = torch.empty_like(partial_lse)
        partial_log_denom = torch.empty_like(partial_lse)
        out_target = partial_out.view(batch * resolved_splits, q_heads, seq_q, head_dim)
        lse_target = partial_lse.view(batch * resolved_splits, q_heads, seq_q)
        bias_target = partial_bias.view(batch * resolved_splits, q_heads, seq_q)
        log_denom_target = partial_log_denom.view(batch * resolved_splits, q_heads, seq_q)

    operands = PrefillKernelOperands(
        query_int8=query_int8,
        query_scale=query_scale,
        key_int8=pool_layout.key_int8,
        key_scale=key_scale,
        key_center=key_center,
        value_int8=pool_layout.value_int8,
        value_scale=value_scale,
        value_mean=value_mean,
        chunk_to_slot=pool_layout.chunk_to_slot,
        num_chunks=num_chunks,
        output=out_target,
        lse=lse_target,
        seq_q=seq_q,
        query_storage=query_storage,
        q_heads=q_heads,
        num_slots=num_slots,
        max_chunks=pool_layout.max_chunks,
        stride_ki_s=pool_layout.stride_ki_s,
        stride_ks_s=pool_layout.stride_ks_s,
        stride_kc_s=pool_layout.stride_kc_s,
        stride_vi_s=pool_layout.stride_vi_s,
        stride_vs_s=pool_layout.stride_vs_s,
        stride_vm_s=pool_layout.stride_vm_s,
        stride_cs_b=pool_layout.stride_cs_b,
        stride_cs_c=pool_layout.stride_cs_c,
    )
    if prefill_backend == "amd":
        from ._amd import gluon as amd_gluon  # noqa: PLC0415

        amd_gluon.launch_frozen_pool_kernel(
            operands,
            batch=batch,
            gqa_group=gqa_group,
            head_dim=head_dim,
            has_chunk_map=pool_layout.has_chunk_map,
            split_chunks=resolved_splits,
            total_splits=resolved_total,
            has_chunk_splits=has_splits,
            per_row_query_scale=per_row_query_scale,
            partial_bias=bias_target,
            partial_log_denom=log_denom_target,
            tail_len=tail_len,
            tail_cap=tail_cap,
            window_left=window_left,
            window_right=window_right,
            logical_seq_q=logical_seq_q,
            window_mode=window_mode,
        )
    else:
        from ._nvidia import prefill as nvidia_prefill  # noqa: PLC0415

        nvidia_prefill.launch_prefill_kernel(
            operands,
            batch=batch,
            gqa_group=gqa_group,
            head_dim=head_dim,
            grouped_qk=grouped_qk,
            has_chunk_map=pool_layout.has_chunk_map,
            is_tail=False,
            is_causal=False,
            split_chunks=resolved_splits,
            total_splits=resolved_total,
            has_chunk_splits=has_splits,
            per_row_query_scale=per_row_query_scale,
            partial_bias=bias_target,
            partial_log_denom=log_denom_target,
            target=target,
            tail_len=tail_len,
            tail_cap=tail_cap,
            window_mode=window_mode,
            window_left=window_left,
            window_right=window_right,
            logical_seq_q=logical_seq_q,
        )
    if (
        partial_out is not None
        and partial_lse is not None
        and partial_bias is not None
        and partial_log_denom is not None
    ):
        po_view = partial_out.permute(0, 2, 3, 1, 4)
        pl_view, pb_view, pld_view = (
            t.permute(0, 2, 3, 1) for t in (partial_lse, partial_bias, partial_log_denom)
        )
        launch_splitkv_reduce(
            po_view,
            pl_view,
            pb_view,
            pld_view,
            output,
            lse,
            batch=batch,
            q_heads=q_heads,
            seq_q=seq_q,
            num_splits=resolved_splits,
            head_dim=head_dim,
            use_lds_only_fence=(prefill_backend == "amd"),
        )


def _quantize_query(
    query: torch.Tensor,
    scale: float,
    *,
    prefill_backend: Literal["amd", "nvidia"],
    grouped_qk: bool,
    per_row: bool,
    storage_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prefill_backend == "amd":
        from piper_kernels.attention.kernels.piper._amd import (  # noqa: PLC0415
            quantization as amd_quantization,
        )

        return amd_quantization.prepare_query(
            query,
            scale,
            per_row=per_row,
            storage_sequence_length=storage_length,
        )
    with device_context(query.device):
        return qk_quantization.prepare_query(
            query,
            scale,
            grouped=grouped_qk and not per_row,
            per_row=per_row,
            storage_query_length=storage_length,
        )


def _run_small_q_decode(
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float,
    is_causal: bool,
    grouped_qk: bool,
    active_chunks_hint: int,
    prefill_backend: Literal["amd", "nvidia"],
    target: AcceleratorTarget | None,
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute native GQA-packed INT8 pool decode plus optional direct BF16/FP16 tail."""
    batch, q_heads, seq_q, head_dim = query.shape
    num_slots, kv_heads, _, _ = key_int8.shape
    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    tail_cap = 0 if tail_key is None else tail_key.shape[2]
    has_chunks = max_chunks > 0 and num_slots > 0
    has_tail = tail_cap > 0
    gqa_group = q_heads // kv_heads
    packed_m = gqa_group * seq_q
    storage_m = triton.cdiv(packed_m, 64) * 64
    windows = resolve_kv_window_modes(
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
    )

    frozen_splits = (
        _select_frozen_prefill_splits(
            query.device,
            prefill_backend=prefill_backend,
            batch=batch,
            q_heads=kv_heads,
            seq_q=packed_m,
            head_dim=head_dim,
            max_chunks=max_chunks,
            active_chunks_hint=active_chunks_hint,
            allow_subchunk_single=True,
            logical_seq_q=seq_q,
            window_left=windows.frozen_left,
        )
        if has_chunks
        else 0
    )
    resolved_target = target
    schedule = None
    tail_splits = 0
    if has_tail:
        resolved_target = AcceleratorTarget.from_device(query.device) if target is None else target
        schedule = select_decode_schedule(resolved_target, head_dim)
        _, tail_splits = select_split_counts(
            batch=batch,
            kv_heads=kv_heads,
            gqa_group=gqa_group,
            seq_q=seq_q,
            max_chunks=0,
            num_slots=0,
            tail_cap=tail_cap,
            active_chunks_hint=0,
            sm_count=device_sm_count(query.device),
            schedule=schedule,
            window_left=windows.tail_left,
        )

    total_splits = frozen_splits + tail_splits
    needs_partials = has_tail or frozen_splits > 1
    partials: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    partial_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if needs_partials:
        partial_out = torch.empty(
            (batch, total_splits, q_heads, seq_q, head_dim),
            device=query.device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            (batch, total_splits, q_heads, seq_q),
            device=query.device,
            dtype=torch.float32,
        )
        partials = (
            partial_out,
            partial_lse,
            torch.empty_like(partial_lse),
            torch.empty_like(partial_lse),
        )
        partial_targets = (
            partials[0].view(batch * total_splits, kv_heads, packed_m, head_dim),
            partials[1].view(batch * total_splits, kv_heads, packed_m),
            partials[2].view(batch * total_splits, kv_heads, packed_m),
            partials[3].view(batch * total_splits, kv_heads, packed_m),
        )

    if has_chunks:
        query_int8, query_scale = _quantize_query(
            query.reshape(batch, kv_heads, packed_m, head_dim),
            scale,
            prefill_backend=prefill_backend,
            grouped_qk=grouped_qk,
            per_row=True,
            storage_length=storage_m,
        )
        _launch_frozen_pool_prefill(
            query_int8,
            query_scale,
            key_int8,
            key_scale,
            key_center,
            value_int8,
            value_scale,
            value_mean,
            chunk_to_slot,
            num_chunks,
            output.view(batch, kv_heads, packed_m, head_dim),
            lse.view(batch, kv_heads, packed_m),
            seq_q=packed_m,
            grouped_qk=grouped_qk,
            prefill_backend=prefill_backend,
            active_chunks_hint=active_chunks_hint,
            target=resolved_target,
            per_row_query_scale=True,
            allow_subchunk_single=True,
            frozen_splits=frozen_splits,
            total_splits=total_splits,
            partial_targets=partial_targets,
            logical_seq_q=seq_q,
            tail_len=tail_len,
            tail_cap=tail_cap,
            window_mode=windows.frozen_mode,
            window_left=windows.frozen_left,
            window_right=windows.frozen_right,
        )

    if partials is None:
        return output, lse

    po_view = partials[0].permute(0, 2, 3, 1, 4)
    pl_view, pb_view, pld_view = (t.permute(0, 2, 3, 1) for t in partials[1:])
    if has_tail:
        assert tail_key is not None
        assert tail_value is not None
        assert tail_len is not None
        assert schedule is not None
        launch_direct_tail_split_decode(
            query,
            tail_key,
            tail_value,
            tail_len,
            po_view,
            pl_view,
            pb_view,
            pld_view,
            scale=scale,
            kv_heads=kv_heads,
            gqa_group=gqa_group,
            q_tiles=triton.cdiv(packed_m, BLOCK_QH),
            seq_q=seq_q,
            tail_cap=tail_cap,
            int8_splits=frozen_splits,
            tail_splits=tail_splits,
            is_causal=is_causal,
            head_dim=head_dim,
            schedule=schedule,
            window_mode=windows.tail_mode,
            window_left=windows.tail_left,
            window_right=windows.tail_right,
        )
    launch_splitkv_reduce(
        po_view,
        pl_view,
        pb_view,
        pld_view,
        output,
        lse,
        batch=batch,
        q_heads=q_heads,
        seq_q=seq_q,
        num_splits=total_splits,
        head_dim=head_dim,
        use_lds_only_fence=(prefill_backend == "amd"),
    )
    return output, lse


def run_chunked_prefill(
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    *,
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float,
    is_causal: bool,
    grouped_qk: bool,
    pack_wmma: bool,
    active_chunks_hint: int,
    prefill_backend: Literal["amd", "nvidia"],
    target: AcceleratorTarget | None = None,
    window_left: int = -1,
    window_right: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run native INT8-Q/UINT8-P frozen pool execution plus direct or quantized live tail."""
    _backend.validate_prefill_backend(prefill_backend, grouped_qk=grouped_qk, pack_wmma=pack_wmma)
    batch, q_heads, seq_q, head_dim = query.shape
    num_slots = key_int8.shape[0]
    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    tail_cap = 0 if tail_key is None else tail_key.shape[2]
    has_chunks = max_chunks > 0 and num_slots > 0
    has_tail = tail_cap > 0

    windows = resolve_kv_window_modes(
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
    )

    output = torch.empty((batch, q_heads, seq_q, head_dim), device=query.device, dtype=query.dtype)
    lse = torch.empty((batch, q_heads, seq_q), device=query.device, dtype=torch.float32)
    if batch == 0:
        return output, lse
    if not has_chunks and not has_tail:
        output.zero_()
        lse.fill_(-float("inf"))
        return output, lse

    if seq_q < PREFILL_MIN_SEQ_Q:
        return _run_small_q_decode(
            query,
            key_int8,
            key_scale,
            key_center,
            value_int8,
            value_scale,
            value_mean,
            chunk_to_slot,
            num_chunks,
            output,
            lse,
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            scale=scale,
            is_causal=is_causal,
            grouped_qk=grouped_qk,
            active_chunks_hint=active_chunks_hint,
            prefill_backend=prefill_backend,
            target=target,
            window_left=window_left,
            window_right=window_right,
        )

    storage_q = triton.cdiv(seq_q, 64) * 64
    query_int8, query_scale = _quantize_query(
        query,
        scale,
        prefill_backend=prefill_backend,
        grouped_qk=grouped_qk,
        per_row=False,
        storage_length=storage_q,
    )

    if has_chunks:
        _launch_frozen_pool_prefill(
            query_int8,
            query_scale,
            key_int8,
            key_scale,
            key_center,
            value_int8,
            value_scale,
            value_mean,
            chunk_to_slot,
            num_chunks,
            output,
            lse,
            seq_q=seq_q,
            grouped_qk=grouped_qk,
            prefill_backend=prefill_backend,
            active_chunks_hint=active_chunks_hint,
            target=target,
            tail_len=tail_len,
            tail_cap=tail_cap,
            window_mode=windows.frozen_mode,
            window_left=windows.frozen_left,
            window_right=windows.frozen_right,
        )

    if has_tail:
        assert tail_key is not None
        assert tail_value is not None
        assert tail_len is not None
        tail_storage = triton.cdiv(tail_cap, 64) * 64
        tail_k_int8, tail_k_scale, tail_k_center = _quantization.prepare_chunk_centered_key(
            tail_key,
            grouped=grouped_qk,
            storage_key_length=tail_storage,
            scale_storage_length=tail_storage,
            key_lengths=tail_len,
        )
        tail_v_int8, tail_v_scale_or_mult, tail_v_log_scale, tail_v_mean = (
            _quantization.prepare_chunk_centered_value(
                tail_value,
                storage_key_length=tail_storage,
                store_log_scale=pack_wmma,
                scale_storage_length=tail_storage,
                pack_wmma=pack_wmma,
                log_scale_dtype=torch.float32,
                tile_layout=True,
                scale_by_p_range=pack_wmma,
                key_lengths=tail_len,
            )
        )
        out_tail = torch.empty_like(output) if has_chunks else output
        lse_tail = torch.empty_like(lse) if has_chunks else lse
        if prefill_backend == "amd":
            from ._amd import gluon as amd_gluon  # noqa: PLC0415

            amd_gluon.launch_tail_prefill(
                query_int8,
                query_scale,
                tail_k_int8,
                tail_k_scale,
                tail_k_center,
                tail_v_int8,
                tail_v_scale_or_mult,
                tail_v_log_scale,
                tail_v_mean,
                tail_len,
                out_tail,
                lse_tail,
                seq_q=seq_q,
                tail_cap=tail_cap,
                is_causal=is_causal,
                window_mode=windows.tail_mode,
                window_left=windows.tail_left,
                window_right=windows.tail_right,
            )
        else:
            from ._nvidia import prefill as nvidia_prefill  # noqa: PLC0415

            nvidia_prefill.launch_tail_prefill(
                query_int8,
                query_scale,
                tail_k_int8,
                tail_k_scale,
                tail_k_center,
                tail_v_int8,
                tail_v_scale_or_mult,
                tail_v_mean,
                tail_len,
                out_tail,
                lse_tail,
                seq_q=seq_q,
                tail_cap=tail_cap,
                is_causal=is_causal,
                grouped_qk=grouped_qk,
                target=target,
                window_mode=windows.tail_mode,
                window_left=windows.tail_left,
                window_right=windows.tail_right,
            )
        if has_chunks:
            merge_two_attention_waves(output, lse, out_tail, lse_tail)

    return output, lse
