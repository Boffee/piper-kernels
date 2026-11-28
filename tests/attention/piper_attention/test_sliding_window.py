"""Public-API tests for sliding-window piper_attention, cached decode, and merge_attention_lse."""

import math

import pytest
import torch
import triton

from piper_kernels import (
    PiperKVChunkPool,
    compile_options,
    freeze_kv_chunks_,
    merge_attention_lse,
    piper_attention,
    piper_attention_decode,
)

_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _explicit_dense_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    window_left: int | None = None,
    window_right: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, q_heads, seq_q, head_dim = query.shape
    _, kv_heads, seq_k, _ = key.shape
    group = q_heads // kv_heads
    q_fp32 = query.to(torch.float32)
    k_fp32 = key.repeat_interleave(group, dim=1).to(torch.float32)
    v_fp32 = value.repeat_interleave(group, dim=1).to(torch.float32)
    eff_scale = (head_dim**-0.5) if scale is None else scale
    scores = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) * eff_scale

    q_pos = torch.arange(seq_q, device=query.device).unsqueeze(1)
    k_pos = torch.arange(seq_k, device=query.device).unsqueeze(0)
    mask = torch.ones((seq_q, seq_k), device=query.device, dtype=torch.bool)
    if window_left is not None:
        mask = mask & (k_pos >= q_pos - window_left)
    eff_right = 0 if is_causal else window_right
    if eff_right is not None:
        mask = mask & (k_pos <= q_pos + eff_right)
    mask_4d = mask.view(1, 1, seq_q, seq_k).expand(batch, q_heads, seq_q, seq_k)

    scores = scores.masked_fill(~mask_4d, float("-inf"))
    has_any = mask_4d.any(dim=-1, keepdim=True)
    row_max = scores.amax(dim=-1, keepdim=True)
    safe_max = torch.where(has_any, row_max, torch.zeros_like(row_max))
    exp_scores = torch.where(mask_4d, torch.exp(scores - safe_max), torch.zeros_like(scores))
    denom = exp_scores.sum(dim=-1, keepdim=True)
    probs = torch.where(has_any, exp_scores / denom, torch.zeros_like(exp_scores))
    out = torch.matmul(probs, v_fp32).to(query.dtype)
    lse = torch.where(
        has_any.squeeze(-1),
        safe_max.squeeze(-1) + torch.log(denom.squeeze(-1)),
        torch.full(
            (batch, q_heads, seq_q), float("-inf"), device=query.device, dtype=torch.float32
        ),
    )
    return out, lse


def _explicit_cached_reference(
    query: torch.Tensor,
    full_keys: list[torch.Tensor],
    full_values: list[torch.Tensor],
    *,
    scale: float | None = None,
    is_causal: bool = True,
    window_left: int | None = None,
    window_right: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, q_heads, seq_q, head_dim = query.shape
    eff_scale = (head_dim**-0.5) if scale is None else scale
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    eff_right = 0 if is_causal else window_right
    for b in range(batch):
        k_b = full_keys[b]
        v_b = full_values[b]
        total_kv = k_b.shape[1]
        kv_heads = k_b.shape[0]
        group = q_heads // kv_heads
        q_fp32 = query[b].to(torch.float32)
        k_fp32 = k_b.repeat_interleave(group, dim=0).to(torch.float32)
        v_fp32 = v_b.repeat_interleave(group, dim=0).to(torch.float32)
        if total_kv == 0:
            outputs.append(torch.zeros_like(query[b]))
            lses.append(
                torch.full(
                    (q_heads, seq_q),
                    float("-inf"),
                    device=query.device,
                    dtype=torch.float32,
                )
            )
            continue
        scores = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) * eff_scale
        q_pos = (total_kv - seq_q) + torch.arange(seq_q, device=query.device).unsqueeze(1)
        k_pos = torch.arange(total_kv, device=query.device).unsqueeze(0)
        mask = torch.ones((seq_q, total_kv), device=query.device, dtype=torch.bool)
        if window_left is not None:
            mask = mask & (k_pos >= q_pos - window_left)
        if eff_right is not None:
            mask = mask & (k_pos <= q_pos + eff_right)
        mask_3d = mask.unsqueeze(0).expand(q_heads, seq_q, total_kv)
        scores = scores.masked_fill(~mask_3d, float("-inf"))
        has_any = mask_3d.any(dim=-1, keepdim=True)
        row_max = scores.amax(dim=-1, keepdim=True)
        safe_max = torch.where(has_any, row_max, torch.zeros_like(row_max))
        exp_scores = torch.where(mask_3d, torch.exp(scores - safe_max), torch.zeros_like(scores))
        denom = exp_scores.sum(dim=-1, keepdim=True)
        probs = torch.where(has_any, exp_scores / denom, torch.zeros_like(exp_scores))
        outputs.append(torch.matmul(probs, v_fp32).to(query.dtype))
        lses.append(
            torch.where(
                has_any.squeeze(-1),
                safe_max.squeeze(-1) + torch.log(denom.squeeze(-1)),
                torch.full(
                    (q_heads, seq_q),
                    float("-inf"),
                    device=query.device,
                    dtype=torch.float32,
                ),
            )
        )
    return torch.stack(outputs, dim=0), torch.stack(lses, dim=0)


# --- Validation tests (CPU + GPU) ---


@pytest.mark.parametrize(
    ("kwargs", "exc_type"),
    [
        ({"window_left": -1}, ValueError),
        ({"window_right": -2}, ValueError),
        ({"window_left": True}, TypeError),
        ({"window_right": False}, TypeError),
        ({"window_left": 1.5}, TypeError),
        ({"window_right": "10"}, TypeError),
    ],
)
def test_dense_piper_attention_rejects_invalid_window_args(
    kwargs: dict[str, object], exc_type: type[Exception]
) -> None:
    q = torch.randn(1, 2, 16, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 16, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 16, 64, dtype=torch.bfloat16)
    with pytest.raises(exc_type):
        piper_attention(q, k, v, **kwargs)  # type: ignore[arg-type]


def test_dense_piper_attention_causal_caps_positive_window_right() -> None:
    torch.manual_seed(77)
    q = torch.randn(1, 2, 16, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 16, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 16, 64, dtype=torch.bfloat16)
    out_capped, lse_capped = piper_attention(
        q, k, v, is_causal=True, window_left=4, window_right=8, return_lse=True
    )
    out_zero, lse_zero = piper_attention(
        q, k, v, is_causal=True, window_left=4, window_right=0, return_lse=True
    )
    torch.testing.assert_close(out_capped, out_zero, atol=0, rtol=0)
    torch.testing.assert_close(lse_capped, lse_zero, atol=0, rtol=0)


def test_dense_piper_attention_rejects_rectangular_causal_even_with_window_left() -> None:
    q = torch.randn(1, 2, 16, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="equal query and key lengths"):
        piper_attention(q, k, v, is_causal=True, window_left=8)


def test_cached_decode_rejects_invalid_window_args() -> None:
    pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu")
    chunk_to_slot = torch.tensor([[0, 1]], dtype=torch.int32)
    num_chunks = torch.tensor([1], dtype=torch.int32)
    q = torch.randn(1, 2, 1, 64, dtype=torch.bfloat16)
    for kwargs, exc_type in [
        ({"window_left": -1}, ValueError),
        ({"window_right": -1}, ValueError),
        ({"window_left": True}, TypeError),
        ({"window_right": 2.0}, TypeError),
    ]:
        with pytest.raises(exc_type):
            piper_attention_decode(q, pool, chunk_to_slot, num_chunks, **kwargs)  # type: ignore[arg-type]


# --- Dense piper_attention sliding window & return_lse ---


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=_REQUIRES_CUDA)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    ("seq_q", "seq_k", "is_causal", "window_left", "window_right"),
    [
        (1, 1, False, 0, 0),
        (17, 1, False, 0, 0),
        (129, 129, False, 63, 0),
        (129, 129, True, 63, None),
        (129, 129, False, None, 0),
        (129, 129, False, 32, 16),
        (193, 65, False, 16, 4),
        (65, 193, False, 10, 20),
    ],
)
def test_dense_sliding_window_and_lse_matches_reference(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    seq_q: int,
    seq_k: int,
    is_causal: bool,
    window_left: int | None,
    window_right: int | None,
) -> None:
    if device == "cpu" and seq_q > 65:
        pytest.skip("keep CPU reference test fast")
    torch.manual_seed(42 + seq_q + seq_k + head_dim)
    batch, q_heads, kv_heads = 2, 6, 2
    q = torch.randn(batch, q_heads, seq_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq_k, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_heads, seq_k, head_dim, device=device, dtype=dtype)

    out, lse = piper_attention(
        q,
        k,
        v,
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
        return_lse=True,
    )
    out_only = piper_attention(
        q,
        k,
        v,
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
    )
    ref_out, ref_lse = _explicit_dense_reference(
        q,
        k,
        v,
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
    )

    assert out.shape == q.shape
    assert lse.shape == (batch, q_heads, seq_q)
    assert lse.dtype is torch.float32
    torch.testing.assert_close(out, out_only, atol=3e-3, rtol=3e-3)

    empty_mask = torch.isneginf(ref_lse)
    if empty_mask.any():
        assert torch.all(out[empty_mask] == 0)
        assert torch.all(torch.isneginf(lse[empty_mask]))
    non_empty = ~empty_mask
    if non_empty.any():
        torch.testing.assert_close(out[non_empty], ref_out[non_empty], atol=0.05, rtol=0.05)
        torch.testing.assert_close(lse[non_empty], ref_lse[non_empty], atol=0.08, rtol=0.05)


@_REQUIRES_CUDA
@pytest.mark.parametrize("seq_len", [1023, 1024, 1025, 2049])
def test_dense_window_chunk_and_tile_boundaries_empty_first_tile_cuda(seq_len: int) -> None:
    torch.manual_seed(900 + seq_len)
    # Narrow backward window causes bottom queries (positions >= 1024) to skip early chunks/tiles,
    # and for causal window_left=63 skips the first tile on almost all rows.
    q = torch.randn(1, 4, seq_len, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, seq_len, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2, seq_len, 128, device="cuda", dtype=torch.bfloat16)

    out, lse = piper_attention(q, k, v, is_causal=True, window_left=127, return_lse=True)
    ref_out, ref_lse = _explicit_dense_reference(q, k, v, is_causal=True, window_left=127)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(out, ref_out, atol=0.05, rtol=0.05)
    torch.testing.assert_close(lse, ref_lse, atol=0.08, rtol=0.05)


@_REQUIRES_CUDA
def test_dense_bshd_strided_layout_with_window_and_lse() -> None:
    torch.manual_seed(177)
    q_bshd = torch.randn(2, 193, 6, 128, device="cuda", dtype=torch.bfloat16)
    k_bshd = torch.randn(2, 193, 2, 128, device="cuda", dtype=torch.bfloat16)
    v_bshd = torch.randn(2, 193, 2, 128, device="cuda", dtype=torch.bfloat16)
    q = q_bshd.transpose(1, 2)
    k = k_bshd.transpose(1, 2)
    v = v_bshd.transpose(1, 2)
    out, lse = piper_attention(q, k, v, window_left=64, window_right=32, return_lse=True)
    ref_out, ref_lse = _explicit_dense_reference(q, k, v, window_left=64, window_right=32)
    torch.testing.assert_close(out, ref_out, atol=0.05, rtol=0.05)
    torch.testing.assert_close(lse, ref_lse, atol=0.08, rtol=0.05)


@_REQUIRES_CUDA
def test_dense_left_none_right_zero_matches_is_causal() -> None:
    torch.manual_seed(314)
    q = torch.randn(2, 4, 129, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, 129, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 2, 129, 128, device="cuda", dtype=torch.bfloat16)
    out_w, lse_w = piper_attention(q, k, v, is_causal=False, window_right=0, return_lse=True)
    out_c, lse_c = piper_attention(q, k, v, is_causal=True, return_lse=True)
    torch.testing.assert_close(out_w, out_c, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_w, lse_c, atol=2e-2, rtol=2e-2)


@_REQUIRES_CUDA
def test_dense_torch_compile_window_and_lse_fullgraph_dynamic() -> None:
    torch.manual_seed(512)

    def _fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return piper_attention(q, k, v, window_left=95, window_right=16, return_lse=True)

    compiled = torch.compile(_fn, fullgraph=True, dynamic=True)
    for seq_k in (65, 193):
        q = torch.randn(2, 4, seq_k, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, 2, seq_k, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(2, 2, seq_k, 128, device="cuda", dtype=torch.bfloat16)
        out_c, lse_c = compiled(q, k, v)
        out_e, lse_e = _fn(q, k, v)
        torch.testing.assert_close(out_c, out_e, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(lse_c, lse_e, atol=1e-3, rtol=1e-3)


# --- Cached piper_attention_decode sliding window & Split-KV clipping ---


def _build_pool_and_inputs(
    *,
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    chunk_counts: list[int],
    tail_lens: list[int],
    seq_q: int,
    with_holes: bool = True,
) -> tuple[
    torch.Tensor,
    PiperKVChunkPool,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    batch = len(chunk_counts)
    kv_heads, q_heads = 2, 4
    max_chunks = max(1, *chunk_counts)
    extra_holes = 2 if with_holes else 0
    num_slots = sum(chunk_counts) + extra_holes
    pool = PiperKVChunkPool.empty(max(num_slots, 1), kv_heads, head_dim, device=device)
    chunk_to_slot = torch.full((batch, max_chunks), -1, device=device, dtype=torch.int32)
    full_keys: list[torch.Tensor] = []
    full_values: list[torch.Tensor] = []
    next_slot = extra_holes
    max_tail = max(tail_lens)
    tail_k = (
        torch.randn(batch, kv_heads, max_tail, head_dim, device=device, dtype=dtype)
        if max_tail > 0
        else None
    )
    tail_v = (
        torch.randn(batch, kv_heads, max_tail, head_dim, device=device, dtype=dtype)
        if max_tail > 0
        else None
    )
    for b in range(batch):
        nc = chunk_counts[b]
        keys_b: list[torch.Tensor] = []
        vals_b: list[torch.Tensor] = []
        for c in range(nc):
            slot = next_slot
            next_slot += 1
            chunk_to_slot[b, c] = slot
            kc = torch.randn(kv_heads, 1024, head_dim, device=device, dtype=dtype)
            vc = torch.randn(kv_heads, 1024, head_dim, device=device, dtype=dtype)
            freeze_kv_chunks_(
                kc.unsqueeze(0),
                vc.unsqueeze(0),
                pool,
                torch.tensor([slot], device=device, dtype=torch.int32),
            )
            keys_b.append(kc)
            vals_b.append(vc)
        tl = tail_lens[b]
        if tl > 0 and tail_k is not None and tail_v is not None:
            keys_b.append(tail_k[b, :, :tl, :])
            vals_b.append(tail_v[b, :, :tl, :])
        if keys_b:
            full_keys.append(torch.cat(keys_b, dim=1))
            full_values.append(torch.cat(vals_b, dim=1))
        else:
            full_keys.append(torch.empty((kv_heads, 0, head_dim), device=device, dtype=dtype))
            full_values.append(torch.empty((kv_heads, 0, head_dim), device=device, dtype=dtype))

    num_chunks_t = torch.tensor(chunk_counts, device=device, dtype=torch.int32)
    tail_len_t = torch.tensor(tail_lens, device=device, dtype=torch.int32) if max_tail > 0 else None
    q = torch.randn(batch, q_heads, seq_q, head_dim, device=device, dtype=dtype)
    return q, pool, chunk_to_slot, num_chunks_t, tail_k, tail_v, tail_len_t, full_keys, full_values


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=_REQUIRES_CUDA)])
@pytest.mark.parametrize("seq_q", [1, 5, 65, 129])
@pytest.mark.parametrize(
    ("window_left", "window_right", "is_causal"),
    [
        (255, None, True),
        (1023, None, True),
        (64, 32, False),
        (None, 0, False),
    ],
)
def test_cached_decode_sliding_window_matches_reference(
    device: str,
    seq_q: int,
    window_left: int | None,
    window_right: int | None,
    is_causal: bool,
) -> None:
    if device == "cpu" and seq_q > 5:
        pytest.skip("keep CPU cached reference test fast")
    torch.manual_seed(1200 + seq_q + (window_left or 0))
    chunk_counts = [2, 1]
    tail_lens = [37, 0] if seq_q == 1 else [max(seq_q, 65), max(seq_q, 32)]
    q, pool, c2s, nc, tk, tv, tl, full_k, full_v = _build_pool_and_inputs(
        device=device,
        dtype=torch.bfloat16,
        head_dim=128,
        chunk_counts=chunk_counts,
        tail_lens=tail_lens,
        seq_q=seq_q,
    )
    out, lse = piper_attention_decode(
        q,
        pool,
        c2s,
        nc,
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
        return_lse=True,
    )
    ref_out, ref_lse = _explicit_cached_reference(
        q,
        full_k,
        full_v,
        is_causal=is_causal,
        window_left=window_left,
        window_right=window_right,
    )
    torch.testing.assert_close(out, ref_out, atol=0.06, rtol=0.06)
    torch.testing.assert_close(lse, ref_lse, atol=0.1, rtol=0.08)


@_REQUIRES_CUDA
def test_cached_decode_empty_window_intersection_produces_zero_and_neginf_lse() -> None:
    torch.manual_seed(808)
    q, pool, c2s, nc, tk, tv, tl, _, _ = _build_pool_and_inputs(
        device="cuda",
        dtype=torch.bfloat16,
        head_dim=64,
        chunk_counts=[0, 1],
        tail_lens=[0, 16],
        seq_q=1,
        with_holes=False,
    )
    # Batch 0 has 0 KV tokens; window_left=3 for batch 0 is empty.
    out, lse = piper_attention_decode(
        q,
        pool,
        c2s,
        nc,
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        is_causal=True,
        window_left=3,
        return_lse=True,
    )
    assert torch.all(out[0] == 0)
    assert torch.all(torch.isneginf(lse[0]))
    assert torch.isfinite(out[1]).all()
    assert torch.isfinite(lse[1]).all()


@_REQUIRES_CUDA
def test_split_kv_window_clips_partial_splits_and_skips_out_of_window_chunks() -> None:
    torch.manual_seed(64000)
    num_chunks = 64
    pool = PiperKVChunkPool.empty(num_chunks, 2, 128, device="cuda")
    all_k = torch.randn(num_chunks, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    all_v = torch.randn(num_chunks, 2, 1024, 128, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(num_chunks, device="cuda", dtype=torch.int32)
    freeze_kv_chunks_(all_k, all_v, pool, slots)
    chunk_to_slot = slots.unsqueeze(0)
    num_chunks_t = torch.tensor([num_chunks], device="cuda", dtype=torch.int32)
    tail_k = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    tail_v = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    tail_len_t = torch.tensor([16], device="cuda", dtype=torch.int32)
    q = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16)

    for w in (None, 1023):
        piper_attention_decode(
            q,
            pool,
            chunk_to_slot,
            num_chunks_t,
            tail_key=tail_k,
            tail_value=tail_v,
            tail_len=tail_len_t,
            is_causal=True,
            window_left=w,
        )
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    base_unlim = torch.cuda.memory_allocated()
    piper_attention_decode(
        q,
        pool,
        chunk_to_slot,
        num_chunks_t,
        tail_key=tail_k,
        tail_value=tail_v,
        tail_len=tail_len_t,
        is_causal=True,
    )
    peak_unlimited = torch.cuda.max_memory_allocated() - base_unlim

    # Also verify out-of-window chunks [0..61] are skipped.
    pool.key_scale[:62].fill_(float("nan"))
    pool.value_mean[:62].fill_(float("nan"))
    torch.cuda.reset_peak_memory_stats()
    base_win = torch.cuda.memory_allocated()
    out_win, lse_win = piper_attention_decode(
        q,
        pool,
        chunk_to_slot,
        num_chunks_t,
        tail_key=tail_k,
        tail_value=tail_v,
        tail_len=tail_len_t,
        is_causal=True,
        window_left=1023,
        return_lse=True,
    )
    peak_windowed = torch.cuda.max_memory_allocated() - base_win

    assert peak_windowed < peak_unlimited
    assert torch.isfinite(out_win).all()
    assert torch.isfinite(lse_win).all()


# --- Fused merge_attention_lse ---


def _torch_merge_reference(
    outputs: list[torch.Tensor], lses: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    stacked_out = torch.stack([o.to(torch.float32) for o in outputs], dim=0)
    stacked_lse = torch.stack([s.to(torch.float32) for s in lses], dim=0).unsqueeze(-1)
    max_lse = stacked_lse.amax(dim=0, keepdim=True)
    safe_max = torch.where(torch.isneginf(max_lse), torch.zeros_like(max_lse), max_lse)
    weights = torch.exp(stacked_lse - safe_max)
    denom = weights.sum(dim=0)
    merged_out = torch.where(
        denom > 0,
        (weights * stacked_out).sum(dim=0) / denom,
        torch.zeros_like(stacked_out[0]),
    ).to(outputs[0].dtype)
    merged_lse = (safe_max.squeeze(0) + torch.log(denom)).squeeze(-1)
    return merged_out, merged_lse


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=_REQUIRES_CUDA)])
@pytest.mark.parametrize("num_waves", [1, 2, 3, 4, 8, 10])
@pytest.mark.parametrize("head_dim", [1, 3, 7, 64, 96, 128, 300])
def test_merge_attention_lse_waves_and_head_dims(
    device: str, num_waves: int, head_dim: int
) -> None:
    if device == "cpu" and num_waves not in (1, 2, 10):
        pytest.skip("redundant on CPU")
    torch.manual_seed(700 + num_waves + head_dim)
    shape = (2, 4, 19, head_dim)
    outputs = [torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(num_waves)]
    lses = [torch.randn(shape[:-1], device=device, dtype=torch.float32) for _ in range(num_waves)]
    actual_out, actual_lse = merge_attention_lse(outputs, lses)
    ref_out, ref_lse = _torch_merge_reference(outputs, lses)
    torch.testing.assert_close(actual_out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, ref_lse, atol=1e-4, rtol=1e-4)


@_REQUIRES_CUDA
@pytest.mark.parametrize("shape", [(17,), (3, 33), (2, 5, 19, 65), (2, 3, 4, 5, 17)])
def test_merge_attention_lse_ranks_and_non_contiguous_strided_layouts(
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(len(shape) * 101)
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for idx in range(3):
        # Alternate transposed and strided-last-dim views.
        base = torch.randn(*shape[:-1], shape[-1] * 2, device="cuda", dtype=torch.float16)
        out = (
            base[..., ::2]
            if idx % 2 == 1
            else torch.randn(shape, device="cuda", dtype=torch.float16)
        )
        lse_dtype = torch.float16 if idx == 1 else torch.float32
        lse_base = torch.randn(*shape[:-1], 2, device="cuda", dtype=lse_dtype)
        lse = lse_base[..., 0]
        outputs.append(out)
        lses.append(lse)
    actual_out, actual_lse = merge_attention_lse(outputs, lses)
    ref_out, ref_lse = _torch_merge_reference(outputs, lses)
    torch.testing.assert_close(actual_out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, ref_lse, atol=2e-3, rtol=2e-3)


@_REQUIRES_CUDA
def test_merge_attention_lse_special_float_edge_cases() -> None:
    # 1) -inf LSE wave containing NaN in output must NOT poison active wave.
    out0 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda", dtype=torch.float32)
    out1 = torch.full((1, 4), float("nan"), device="cuda", dtype=torch.float32)
    lse0 = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    lse1 = torch.tensor([float("-inf")], device="cuda", dtype=torch.float32)
    m_out, m_lse = merge_attention_lse([out0, out1], [lse0, lse1])
    torch.testing.assert_close(m_out, out0, atol=0, rtol=0)
    torch.testing.assert_close(m_lse, lse0, atol=0, rtol=0)

    # 2) All -inf LSE -> zero output and -inf LSE.
    z_out, z_lse = merge_attention_lse([out1, out1], [lse1, lse1])
    assert torch.all(z_out == 0)
    assert math.isinf(float(z_lse.item()))
    assert float(z_lse.item()) < 0

    # 3) +inf LSE -> NaN output and NaN LSE (matching PyTorch reference).
    p_inf = torch.tensor([float("inf")], device="cuda", dtype=torch.float32)
    pinf_out, pinf_lse = merge_attention_lse([out0, out0], [p_inf, p_inf])
    assert torch.isnan(pinf_out).all()
    assert torch.isnan(pinf_lse).all()

    # 4) Near-overflow finite LSE (2.0e38) stays finite.
    huge = torch.tensor([2.0e38], device="cuda", dtype=torch.float32)
    h_out, h_lse = merge_attention_lse([out0, out0 * 3.0], [huge, huge])
    torch.testing.assert_close(h_out, out0 * 2.0, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(h_lse).all()


def test_merge_attention_lse_fallbacks_float64_neg_and_autograd() -> None:
    o0 = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    o1 = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    l0 = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    l1 = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b, c, d: merge_attention_lse([a, b], [c, d])[0],
        (o0, o1, l0, l1),
        eps=1e-6,
        atol=1e-4,
    )
    if torch.cuda.is_available():
        base = torch.randn(2, 4, 16, device="cuda", dtype=torch.float32)
        neg_view = -base.detach()
        # Simulate negative stride / negated fallback vs ref.
        m_out, m_lse = merge_attention_lse([neg_view, base], [torch.zeros(2, 4, device="cuda")] * 2)
        ref_out, ref_lse = _torch_merge_reference(
            [neg_view, base], [torch.zeros(2, 4, device="cuda")] * 2
        )
        torch.testing.assert_close(m_out, ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(m_lse, ref_lse, atol=1e-5, rtol=1e-5)


@_REQUIRES_CUDA
def test_merge_attention_lse_torch_compile_single_triton_launch() -> None:
    launches: list[str] = []

    def _fn(
        o0: torch.Tensor,
        o1: torch.Tensor,
        o2: torch.Tensor,
        o3: torch.Tensor,
        l0: torch.Tensor,
        l1: torch.Tensor,
        l2: torch.Tensor,
        l3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return merge_attention_lse([o0, o1, o2, o3], [l0, l1, l2, l3])

    compiled = torch.compile(_fn, fullgraph=True, dynamic=True)
    inputs_warm = [
        *[torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16) for _ in range(4)],
        *[torch.randn(1, 2, 17, device="cuda", dtype=torch.float32) for _ in range(4)],
    ]
    _ = compiled(*inputs_warm)
    torch.cuda.synchronize()

    inputs_test = [
        *[torch.randn(2, 4, 33, 64, device="cuda", dtype=torch.bfloat16) for _ in range(4)],
        *[torch.randn(2, 4, 33, device="cuda", dtype=torch.float32) for _ in range(4)],
    ]

    def _hook(metadata: object) -> None:
        data = metadata.get() if hasattr(metadata, "get") else metadata  # type: ignore[union-attr]
        if isinstance(data, dict) and "name" in data:
            launches.append(str(data["name"]))

    triton.knobs.runtime.launch_enter_hook.add(_hook)
    try:
        out_c, lse_c = compiled(*inputs_test)
        torch.cuda.synchronize()
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(_hook)

    out_e, lse_e = _fn(*inputs_test)
    torch.testing.assert_close(out_c, out_e, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse_c, lse_e, atol=1e-4, rtol=1e-4)
    assert len(launches) == 1
    assert "_merge_two_attention_waves_kernel" in launches[0]


# --- ConvRot INT8 Piper FX guard safety ---


@_REQUIRES_CUDA
def test_convrot_int8_piper_compile_does_not_fuse_window_or_lse() -> None:
    torch.manual_seed(999)
    hidden = torch.randn(1, 65, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 65, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 2, 65, 64, device="cuda", dtype=torch.bfloat16)

    def _windowed_model(x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(x, weight).view(1, 65, 4, 64).transpose(1, 2)
        out, _lse = piper_attention(proj, k, v, is_causal=True, window_left=15, return_lse=True)
        return out

    compiled = torch.compile(
        _windowed_model, fullgraph=True, options=compile_options(attention=True)
    )
    with torch.no_grad():
        actual = compiled(hidden, key, value)
        expected = _windowed_model(hidden, key, value)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=_REQUIRES_CUDA)])
def test_large_int64_max_windows_do_not_overflow(device: str) -> None:
    int64_max = (1 << 63) - 1
    torch.manual_seed(1234)
    q = torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16)

    out_w1000, lse_w1000 = piper_attention(q, k, v, window_right=1000, return_lse=True)
    for kw in (
        {"window_right": int64_max},
        {"window_left": int64_max},
        {"window_left": int64_max, "window_right": int64_max},
        {"window_right": 1 << 40},
    ):
        out, lse = piper_attention(q, k, v, **kw, return_lse=True)
        assert torch.isfinite(out).all()
        assert torch.isfinite(lse).all()
        torch.testing.assert_close(out, out_w1000, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, lse_w1000, atol=2e-2, rtol=2e-2)

    pool = PiperKVChunkPool.empty(2, 2, 64, device=device)
    freeze_kv_chunks_(
        torch.randn(2, 2, 1024, 64, device=device, dtype=torch.bfloat16),
        torch.randn(2, 2, 1024, 64, device=device, dtype=torch.bfloat16),
        pool,
        torch.tensor([0, 1], device=device, dtype=torch.int32),
    )
    nc = torch.tensor([2], device=device, dtype=torch.int32)
    for seq_q in (1, 64):
        qd = torch.randn(1, 4, seq_q, 64, device=device, dtype=torch.bfloat16)
        ref_out, ref_lse = piper_attention_decode(
            qd, pool, None, nc, is_causal=False, window_right=4096, return_lse=True
        )
        big_out, big_lse = piper_attention_decode(
            qd, pool, None, nc, is_causal=False, window_right=int64_max, return_lse=True
        )
        assert torch.isfinite(big_out).all()
        assert torch.isfinite(big_lse).all()
        torch.testing.assert_close(big_out, ref_out, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(big_lse, ref_lse, atol=5e-2, rtol=5e-2)

    empty_pool = PiperKVChunkPool.empty(0, 2, 64, device=device)
    nc0 = torch.tensor([0], device=device, dtype=torch.int32)
    tk = torch.randn(1, 2, 16, 64, device=device, dtype=torch.bfloat16)
    tv = torch.randn(1, 2, 16, 64, device=device, dtype=torch.bfloat16)
    for seq_q, active_tail in ((4, 2), (64, 10)):
        qt = torch.randn(1, 4, seq_q, 64, device=device, dtype=torch.bfloat16)
        tl = torch.tensor([active_tail], device=device, dtype=torch.int32)
        ref_t_out, ref_t_lse = piper_attention_decode(
            qt,
            empty_pool,
            None,
            nc0,
            tail_key=tk,
            tail_value=tv,
            tail_len=tl,
            is_causal=False,
            return_lse=True,
        )
        wl_out, wl_lse = piper_attention_decode(
            qt,
            empty_pool,
            None,
            nc0,
            tail_key=tk,
            tail_value=tv,
            tail_len=tl,
            is_causal=False,
            window_left=int64_max,
            return_lse=True,
        )
        assert torch.isfinite(wl_out).all()
        assert torch.isfinite(wl_lse).all()
        torch.testing.assert_close(wl_out, ref_t_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(wl_lse, ref_t_lse, atol=2e-2, rtol=2e-2)
