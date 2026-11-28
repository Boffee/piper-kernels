"""Tests for GQA-grouped Split-KV decode (piper_attention_decode & merge_attention_lse)."""

from __future__ import annotations

import fractions
import math
import os
import shutil
import sys
from collections.abc import Callable
from typing import Any

import pytest
import torch
from _accelerators import (
    forbid_device_probes,
    forbid_triton_driver,
    rdna4_available,
    uint8_int8_mma_available,
)
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.autograd import forward_ad
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.nn import functional as fn
from torch.overrides import TorchFunctionMode

from piper_kernels import (
    PiperKVChunkPool,
    freeze_kv_chunks_,
    kv_chunk_storage_specs,
    merge_attention_lse,
    piper_attention_decode,
)
from piper_kernels.attention.piper_attention.kv_cache import KV_CHUNK_TOKENS
from piper_kernels.attention.piper_attention.reference import (
    freeze_kv_chunks_reference,
    piper_attention_decode_reference,
)

_SIGNED_HADAMARD_MASK = (0xF62B39F3, 0x802576E6, 0x910D4CFD, 0x4F877B5A)
_HAS_CPU_INDUCTOR = sys.platform != "win32" or shutil.which(os.environ.get("CXX", "cl")) is not None
_OPCHECK_UTILS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def _signed_hadamard_forward(values: torch.Tensor) -> torch.Tensor:
    head_dim = values.shape[-1]
    offsets = torch.arange(head_dim, device=values.device)
    word_index = offsets // 32
    selected = torch.full_like(word_index, _SIGNED_HADAMARD_MASK[3], dtype=torch.int64)
    for index in (2, 1, 0):
        selected = torch.where(word_index == index, _SIGNED_HADAMARD_MASK[index], selected)
    signs = 2 * ((selected >> (offsets % 32)) & 1) - 1
    out = values.float() * signs
    distance = 1
    while distance < head_dim:
        grouped = out.reshape(*out.shape[:-1], -1, 2, distance)
        low, high = grouped.unbind(dim=-2)
        out = torch.stack((low + high, low - high), dim=-2).flatten(-3)
        distance *= 2
    return out * (1.0 / math.sqrt(head_dim))


def _record_triton_launches[T](fn_call: Callable[[], T]) -> tuple[T, list[str]]:
    import triton  # noqa: PLC0415 - GPU-only launch hook keeps CPU collection Triton-free

    launched: list[str] = []

    def _record_launch(metadata: Any) -> None:
        data = metadata.get()
        if isinstance(data, dict) and "name" in data:
            launched.append(str(data["name"]))

    triton.knobs.runtime.launch_enter_hook.add(_record_launch)
    try:
        output = fn_call()
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(_record_launch)
    return output, launched


def _expected_frozen_split_kernel_name() -> str:
    if rdna4_available():
        return "_frozen_pool_piper_kernel"
    if uint8_int8_mma_available():
        return "_piper_chunked_prefill_kernel"
    return "_piper_decode_frozen_split_kernel"


def _assert_no_frozen_split_launch(launches: list[str]) -> None:
    assert "_piper_decode_frozen_split_kernel" not in launches


def _fill_pool_sentinels(pool: PiperKVChunkPool) -> None:
    max_val = 255 if pool.kv_dtype is torch.uint8 else 127
    pool.key_int8.fill_(max_val)
    pool.value_int8.fill_(max_val)
    pool.key_scale.fill_(float("nan"))
    pool.key_center.fill_(float("nan"))
    pool.value_scale.fill_(float("nan"))
    pool.value_mean.fill_(float("nan"))


def _make_strided_pool(
    num_slots: int,
    kv_heads: int,
    head_dim: int,
    *,
    device: torch.device | str,
    kv_dtype: torch.dtype,
    key_scales: int | None = None,
) -> PiperKVChunkPool:
    spec_target = ("sm80" if key_scales == 1024 else "sm120") if key_scales is not None else device
    specs = kv_chunk_storage_specs(
        num_slots,
        kv_heads,
        head_dim,
        device=spec_target,
        kv_dtype=kv_dtype,
    )
    resolved_scales = specs["key_scale"][0][2] if key_scales is None else key_scales
    interleaved: dict[str, torch.Tensor] = {}
    for name, (shape, dtype) in specs.items():
        actual_shape = (num_slots, kv_heads, resolved_scales) if name == "key_scale" else shape
        slab = torch.empty((num_slots, 2, *actual_shape[1:]), device=device, dtype=dtype)
        interleaved[name] = slab[:, 0]
    pool = PiperKVChunkPool.from_tensors(**interleaved)
    _fill_pool_sentinels(pool)
    return pool


def _populate_test_pool(
    pool: PiperKVChunkPool,
    slot_ids: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    kv_heads = int(pool.kv_heads)
    head_dim = int(pool.head_dim)
    num_new = len(slot_ids)
    keys = torch.randn((num_new, kv_heads, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype)
    vals = torch.randn((num_new, kv_heads, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype)
    slot_tensor = torch.tensor(slot_ids, device=device, dtype=torch.int32)
    freeze_kv_chunks_(keys, vals, pool, slot_tensor)
    return keys, vals


def _make_sentinel_tail(
    batch: int,
    kv_heads: int,
    tail_cap: int,
    head_dim: int,
    active_lengths: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tail_key = torch.randn((batch, kv_heads, tail_cap, head_dim), device=device, dtype=dtype)
    tail_value = torch.randn_like(tail_key)
    for b, length in enumerate(active_lengths):
        if length < tail_cap:
            tail_key[b, :, max(0, length) :].fill_(float("nan"))
            tail_value[b, :, max(0, length) :].fill_(float("inf"))
    tail_len = torch.tensor(active_lengths, device=device, dtype=torch.int32)
    return tail_key, tail_value, tail_len


def _exact_fp32_attention(
    query: torch.Tensor,
    slot_keys: dict[int, tuple[torch.Tensor, torch.Tensor]],
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    num_slots: int,
    *,
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, q_heads, seq_q, _head_dim = query.shape
    kv_heads = (
        next(iter(slot_keys.values()))[0].shape[0]
        if slot_keys
        else (0 if tail_key is None else tail_key.shape[1])
    )
    max_chunks = num_slots if chunk_to_slot is None else chunk_to_slot.shape[1]
    out = torch.zeros_like(query, dtype=torch.float32)
    lse_out = torch.full(
        (batch, q_heads, seq_q), -float("inf"), device=query.device, dtype=torch.float32
    )

    for b in range(batch):
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        active_c = min(max(0, int(num_chunks[b].item())), max_chunks)
        for c in range(active_c):
            slot = c if chunk_to_slot is None else int(chunk_to_slot[b, c].item())
            if 0 <= slot < num_slots and slot in slot_keys:
                k_chunk, v_chunk = slot_keys[slot]
                k_parts.append(k_chunk.float())
                v_parts.append(v_chunk.float())
        active_t = 0
        if tail_key is not None and tail_value is not None and tail_len is not None:
            active_t = min(max(0, int(tail_len[b].item())), tail_key.shape[2])
            if active_t > 0:
                k_parts.append(tail_key[b, :, :active_t].float())
                v_parts.append(tail_value[b, :, :active_t].float())
        if not k_parts:
            continue
        k_all = torch.cat(k_parts, dim=1)
        v_all = torch.cat(v_parts, dim=1)
        total_k = k_all.shape[1]
        repeat = q_heads // kv_heads
        k_rep = k_all.repeat_interleave(repeat, dim=0)
        v_rep = v_all.repeat_interleave(repeat, dim=0)

        attn_mask = None
        if is_causal and tail_key is not None:
            frozen_tokens = total_k - active_t
            q_idx = torch.arange(seq_q, device=query.device)[:, None]
            k_idx = torch.arange(total_k, device=query.device)[None, :]
            cutoff = frozen_tokens + active_t - seq_q + q_idx
            allowed = (k_idx < frozen_tokens) | (k_idx <= cutoff)
            if not bool(allowed.all()):
                attn_mask = allowed.unsqueeze(0)
        logits64 = torch.matmul(query[b].double(), k_rep.double().transpose(-1, -2)) * scale
        if attn_mask is None:
            out[b] = fn.scaled_dot_product_attention(
                query[b].unsqueeze(0).float(),
                k_rep.unsqueeze(0),
                v_rep.unsqueeze(0),
                scale=scale,
            ).squeeze(0)
            lse_out[b] = torch.logsumexp(logits64, dim=-1).float()
        else:
            row_has_any = attn_mask.any(dim=-1, keepdim=True)
            safe_mask = torch.where(row_has_any, attn_mask, True)
            sdpa = fn.scaled_dot_product_attention(
                query[b].unsqueeze(0).float(),
                k_rep.unsqueeze(0),
                v_rep.unsqueeze(0),
                attn_mask=safe_mask.unsqueeze(0),
                scale=scale,
            ).squeeze(0)
            out[b] = torch.where(row_has_any, sdpa, 0.0)
            masked_logits = logits64.masked_fill(~attn_mask, -float("inf"))
            lse_out[b] = torch.where(
                row_has_any.squeeze(-1),
                torch.logsumexp(masked_logits, dim=-1).float(),
                -float("inf"),
            )
    return out.to(query.dtype), lse_out


def test_validation_contracts() -> None:
    pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu")
    q = torch.randn((2, 4, 3, 64), dtype=torch.bfloat16)
    c2s = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)
    nc = torch.tensor([2, 1], dtype=torch.int32)

    class _NoReadbackTensor(torch.Tensor):
        def item(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("unexpected scalar host readback")

        def cpu(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("unexpected .cpu() transfer")

        def __repr__(self, *, tensor_contents: Any = None) -> str:
            raise AssertionError("unexpected tensor repr")

    with pytest.raises(TypeError, match="PiperKVChunkPool"):
        piper_attention_decode(q, "bad", c2s, nc)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="return_lse"):
        piper_attention_decode(q, pool, c2s, nc, return_lse=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="query"):
        piper_attention_decode("bad", pool, c2s, nc)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num_chunks"):
        piper_attention_decode(q, pool, c2s, [1, 1])  # type: ignore[arg-type]

    for bad_arg_name, kwargs in (
        ("chunk_to_slot", {"c2s": "bad"}),
        ("tail_key", {"tail_key": "bad"}),
        ("tail_value", {"tail_value": "bad"}),
        ("tail_len", {"tail_len": "bad"}),
    ):
        call_c2s = kwargs.get("c2s", c2s)
        tail_kw = {k: v for k, v in kwargs.items() if k != "c2s"}
        with pytest.raises(TypeError, match=bad_arg_name):
            piper_attention_decode(q, pool, call_c2s, nc, **tail_kw)  # type: ignore[arg-type]
        with FakeTensorMode(shape_env=ShapeEnv()):
            fake_pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu")
            fake_q = torch.empty((2, 4, 3, 64), dtype=torch.bfloat16)
            fake_c2s = (
                torch.empty((2, 2), dtype=torch.int32) if bad_arg_name != "chunk_to_slot" else "bad"
            )
            fake_nc = torch.empty((2,), dtype=torch.int32)
            with pytest.raises(TypeError, match=bad_arg_name):
                piper_attention_decode(fake_q, fake_pool, fake_c2s, fake_nc, **tail_kw)  # type: ignore[arg-type]

    spy_float = torch.tensor(0.5).as_subclass(_NoReadbackTensor)
    spy_bool = torch.tensor(True).as_subclass(_NoReadbackTensor)
    with pytest.raises(TypeError, match="scale"):
        piper_attention_decode(q, pool, c2s, nc, scale=spy_float)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="is_causal"):
        piper_attention_decode(q, pool, c2s, nc, is_causal=spy_bool)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="return_lse"):
        piper_attention_decode(q, pool, c2s, nc, return_lse=spy_bool)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="active_chunks_hint"):
        piper_attention_decode(q, pool, c2s, nc, active_chunks_hint=spy_bool)  # type: ignore[arg-type]
    for bad_text_scale in ("0.5", b"0.5", bytearray(b"0.5")):
        with pytest.raises(TypeError, match="scale"):
            piper_attention_decode(q, pool, c2s, nc, scale=bad_text_scale)  # type: ignore[arg-type]

    for bad_q in (
        q[:, :, 0],
        q.float(),
        q.transpose(-1, -2),
        torch.randn((2, 5, 3, 64), dtype=torch.bfloat16),
        torch.empty((2, 4, 0, 64), dtype=torch.bfloat16),
        torch.randn((2, 4, 3, 128), dtype=torch.bfloat16),
    ):
        with pytest.raises(ValueError, match=r"query"):
            piper_attention_decode(bad_q, pool, c2s, nc)

    with pytest.raises(ValueError, match="num_chunks"):
        piper_attention_decode(q, pool, c2s, nc.to(torch.int64))
    with pytest.raises(ValueError, match="num_chunks"):
        piper_attention_decode(q, pool, c2s, torch.zeros((2, 2), dtype=torch.int32)[::1, 0])
    with pytest.raises(ValueError, match="chunk_to_slot"):
        piper_attention_decode(q, pool, c2s.to(torch.int64), nc)
    with pytest.raises(ValueError, match="chunk_to_slot"):
        piper_attention_decode(q, pool, torch.zeros((3, 2), dtype=torch.int32), nc)


def test_real_scalar_validation_contracts() -> None:
    torch.manual_seed(330)
    pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu")
    q_full = torch.randn((2, 4, 3, 128), dtype=torch.bfloat16)
    c2s = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)

    for bad_scale in (memoryview(b"0.5"), 1 + 2j):
        with pytest.raises(TypeError, match="scale"):
            piper_attention_decode(
                q_full[..., :64],
                pool,
                c2s,
                torch.tensor([1, 1], dtype=torch.int32),
                scale=bad_scale,
            )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scale"):
        piper_attention_decode(
            q_full[..., :64], pool, c2s, torch.tensor([1, 1], dtype=torch.int32), scale=10**400
        )  # type: ignore[arg-type]
    torch._dynamo.reset()
    with pytest.raises(Exception, match="scale") as exc_info:
        torch.compile(
            lambda x: piper_attention_decode(x, pool, c2s, c2s.new_tensor([1, 1]), scale=10**400),
            backend="aot_eager",
            fullgraph=True,
        )(q_full[..., :64])
    assert "OverflowError" not in str(exc_info.value)
    torch._dynamo.reset()
    tk_f = torch.ones((2, 2, 2, 64), dtype=torch.bfloat16)
    tk_f[:, :, 1] = 2.0
    comp_frac_o, comp_frac_l = torch.compile(
        lambda x: piper_attention_decode(
            x,
            pool,
            c2s,
            c2s.new_tensor([0, 0]),
            tail_key=tk_f,
            tail_value=tk_f,
            tail_len=c2s.new_tensor([2, 2]),
            scale=fractions.Fraction(1, 2),  # type: ignore[arg-type]
            return_lse=True,
        ),
        backend="aot_eager",
        fullgraph=True,
    )(q_full[..., :64])
    ref_frac_o, ref_frac_l = piper_attention_decode(
        q_full[..., :64],
        pool,
        c2s,
        c2s.new_tensor([0, 0]),
        tail_key=tk_f,
        tail_value=tk_f,
        tail_len=c2s.new_tensor([2, 2]),
        scale=0.5,
        return_lse=True,
    )
    torch.testing.assert_close(comp_frac_o, ref_frac_o, atol=0.0, rtol=0.0)
    torch.testing.assert_close(comp_frac_l, ref_frac_l, atol=0.0, rtol=0.0)

    try:
        import numpy as np  # noqa: PLC0415 - skipped when running without NumPy
    except ImportError:
        np = None
    if np is not None:
        for np_bad in (np.complex64(1 + 2j), np.bool_(True), np.array(0.5)):
            with pytest.raises(TypeError, match="scale"):
                piper_attention_decode(
                    q_full[..., :64],
                    pool,
                    c2s,
                    torch.tensor([1, 1], dtype=torch.int32),
                    scale=np_bad,  # type: ignore[arg-type]
                )


def test_negative_view_no_clone_contracts() -> None:
    torch.manual_seed(331)
    pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu")
    q_full = torch.randn((2, 4, 3, 128), dtype=torch.bfloat16)
    q_stride2 = q_full[..., ::2]
    assert q_stride2.stride(-1) == 2
    q_neg_strided = torch._neg_view(q_stride2)
    c2s = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)
    nc_neg = torch._neg_view(torch.tensor([2, 9, 1, 9], dtype=torch.int32)[::2])

    from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC0415

    class _NoCloneMode(TorchDispatchMode):
        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            if func is torch.ops.aten.clone.default:
                raise AssertionError("unexpected clone on rejected view")
            return func(*args, **(kwargs or {}))

    nc = torch.tensor([2, 1], dtype=torch.int32)
    for ctx_factory in (torch.enable_grad, torch.no_grad, torch.inference_mode):
        with ctx_factory(), _NoCloneMode():
            with pytest.raises(ValueError, match="negated or conjugated"):
                piper_attention_decode(q_neg_strided, pool, c2s, nc)
            with pytest.raises(ValueError, match="negated or conjugated"):
                piper_attention_decode(q_full[..., :64], pool, c2s, nc_neg)

    with FakeTensorMode(shape_env=ShapeEnv()), _NoCloneMode():
        fpool = PiperKVChunkPool.empty(2, 2, 64, device="cpu")
        fq = torch._neg_view(torch.empty((2, 4, 3, 128), dtype=torch.bfloat16)[..., ::2])
        with pytest.raises(ValueError, match="negated or conjugated"):
            piper_attention_decode(fq, fpool, None, torch.empty((2,), dtype=torch.int32))

    mpool = PiperKVChunkPool.empty(2, 2, 64, device="meta")
    mq = torch._neg_view(torch.empty((2, 4, 3, 64), device="meta", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="negated or conjugated"):
        piper_attention_decode(mq, mpool, None, torch.empty((2,), device="meta", dtype=torch.int32))

    torch._dynamo.reset()
    with pytest.raises(Exception, match="negated or conjugated"):
        torch.compile(
            lambda x: piper_attention_decode(x, pool, c2s, nc),
            backend="aot_eager",
            fullgraph=True,
        )(torch._neg_view(q_full[..., :64]))

    # freeze_kv_chunks_ rejects negated views identically in compile & direct op calls.
    k_strided = torch._neg_view(
        torch.randn((1, 2, KV_CHUNK_TOKENS, 128), dtype=torch.bfloat16)[..., ::2]
    )
    v_ok = torch.randn((1, 2, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16)
    slots = torch.tensor([0], dtype=torch.int32)
    torch._dynamo.reset()
    with pytest.raises(Exception, match="negated or conjugated"):
        torch.compile(
            lambda k_in, v_in, s_in: freeze_kv_chunks_(k_in, v_in, pool, s_in),
            backend="aot_eager",
            fullgraph=True,
        )(k_strided, v_ok, slots)
    with torch.inference_mode(), pytest.raises(ValueError, match="negated or conjugated"):
        torch.ops.piper_kernels.piper_kv_freeze_.default(
            q_full[:1, :2, :1, :64].expand(1, 2, KV_CHUNK_TOKENS, 64),
            v_ok,
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
            torch._neg_view(slots),
        )


def test_tail_and_autograd_contracts() -> None:
    pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu")
    q = torch.randn((2, 4, 3, 64), dtype=torch.bfloat16)
    c2s = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)
    nc = torch.tensor([2, 1], dtype=torch.int32)
    tk = torch.randn((2, 2, 8, 64), dtype=torch.bfloat16)
    tv = torch.randn_like(tk)
    tl = torch.tensor([4, 8], dtype=torch.int32)

    for tail_kwargs in (
        {"tail_key": tk},
        {"tail_value": tv},
        {"tail_len": tl},
        {"tail_key": tk, "tail_value": tv},
    ):
        with pytest.raises(ValueError, match="tail_key, tail_value, and tail_len"):
            piper_attention_decode(q, pool, c2s, nc, **tail_kwargs)

    with pytest.raises(ValueError, match="tail_key"):
        piper_attention_decode(q, pool, c2s, nc, tail_key=tk.float(), tail_value=tv, tail_len=tl)
    with pytest.raises(ValueError, match="tail_value"):
        piper_attention_decode(q, pool, c2s, nc, tail_key=tk, tail_value=tv[:, :, :4], tail_len=tl)
    with pytest.raises(ValueError, match="tail_len"):
        piper_attention_decode(
            q, pool, c2s, nc, tail_key=tk, tail_value=tv, tail_len=tl.to(torch.int64)
        )
    with pytest.raises(ValueError, match="tail_key"):
        piper_attention_decode(
            q, pool, c2s, nc, tail_key=tk.transpose(-1, -2), tail_value=tv, tail_len=tl
        )

    for bad_scale in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scale"):
            piper_attention_decode(q, pool, c2s, nc, scale=bad_scale)
    with pytest.raises(TypeError, match="scale"):
        piper_attention_decode(q, pool, c2s, nc, scale=torch.tensor(0.125))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="is_causal"):
        piper_attention_decode(q, pool, c2s, nc, is_causal=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="active_chunks_hint"):
        piper_attention_decode(q, pool, c2s, nc, active_chunks_hint=True)
    with pytest.raises(ValueError, match="active_chunks_hint"):
        piper_attention_decode(q, pool, c2s, nc, active_chunks_hint=-1)
    with pytest.raises(ValueError, match="active_chunks_hint"):
        piper_attention_decode(q, pool, c2s, nc, active_chunks_hint=1 << 63)

    q_grad = q.clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="autograd"):
        piper_attention_decode(q_grad, pool, c2s, nc)
    # Autograd rejection fires before backend validation or kernel execution.
    with pytest.raises(RuntimeError, match="autograd"):
        piper_attention_decode(q_grad, pool, c2s, torch.zeros((99,), dtype=torch.int32))
    with pytest.raises(RuntimeError, match="autograd"):
        torch.ops.piper_kernels.piper_attention_decode.default(
            q_grad,
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
            c2s,
            nc,
            None,
            None,
            None,
            0.125,
            True,
            0,
        )
    tk_grad = tk.clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="autograd"):
        piper_attention_decode(q, pool, c2s, nc, tail_key=tk_grad, tail_value=tv, tail_len=tl)
    with forward_ad.dual_level():
        q_dual = forward_ad.make_dual(q, torch.ones_like(q))
        with pytest.raises(RuntimeError, match="autograd"):
            piper_attention_decode(q_dual, pool, c2s, nc)
        with pytest.raises(RuntimeError, match="autograd"):
            torch.ops.piper_kernels.piper_attention_decode.default(
                q_dual,
                pool.key_int8,
                pool.key_scale,
                pool.key_center,
                pool.value_int8,
                pool.value_scale,
                pool.value_mean,
                c2s,
                nc,
                None,
                None,
                None,
                0.125,
                True,
                0,
            )


def test_functorch_compiled_ad_and_torch_function_mode() -> None:
    pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu")
    q = torch.randn((2, 4, 3, 64), dtype=torch.bfloat16)
    c2s = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)
    nc = torch.tensor([2, 1], dtype=torch.int32)
    q_grad = q.clone().requires_grad_(True)

    with pytest.raises(RuntimeError, match="autograd"):
        torch.func.vjp(lambda q_in: piper_attention_decode(q_in, pool, c2s, nc), q)
    with pytest.raises(RuntimeError, match="autograd"):
        torch.func.jvp(
            lambda q_in: piper_attention_decode(q_in, pool, c2s, nc),
            (q,),
            (torch.ones_like(q),),
        )
    with torch.no_grad(), pytest.raises(RuntimeError, match="autograd"):
        torch.func.jvp(
            lambda q_in: piper_attention_decode(q_in, pool, c2s, nc),
            (q,),
            (torch.ones_like(q),),
        )
    torch._dynamo.reset()
    with pytest.raises(Exception, match="autograd"):
        torch.compile(
            lambda q_in, dq_in: torch.func.jvp(
                lambda inner_q: piper_attention_decode(inner_q, pool, c2s, nc),
                (q_in,),
                (dq_in,),
            ),
            backend="aot_eager",
            fullgraph=True,
        )(q, torch.ones_like(q))

    def _compiled_dual(q_in: torch.Tensor, dq_in: torch.Tensor) -> torch.Tensor:
        with forward_ad.dual_level():
            return piper_attention_decode(
                forward_ad.make_dual(q_in, dq_in),
                pool,
                c2s,
                nc,
            )

    torch._dynamo.reset()
    with pytest.raises(Exception, match="autograd"):
        torch.compile(_compiled_dual, backend="aot_eager", fullgraph=True)(q, torch.ones_like(q))

    class _RecordingMode(TorchFunctionMode):
        def __init__(self, swap_q: torch.Tensor | None = None) -> None:
            self.seen: list[object] = []
            self.swap_q = swap_q

        def __torch_function__(
            self,
            func: Any,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            self.seen.append(func)
            if (
                self.swap_q is not None
                and func is torch.ops.piper_kernels.piper_attention_decode.default
            ):
                args = (self.swap_q, *args[1:])
            return func(*args, **(kwargs or {}))

    with _RecordingMode() as mode:
        piper_attention_decode(q, pool, c2s, nc)
    assert torch.ops.piper_kernels.piper_attention_decode.default in mode.seen

    with _RecordingMode(swap_q=q.float()), pytest.raises(ValueError, match="query"):
        piper_attention_decode(q, pool, c2s, nc)
    with (
        _RecordingMode(swap_q=q.clone().requires_grad_(True)),
        pytest.raises(RuntimeError, match="autograd"),
    ):
        piper_attention_decode(q, pool, c2s, nc)

    # Swapping query + pool from head_dim=64 to 128 with scale=None resolves 128**-0.5 post-swap.
    pool_128 = PiperKVChunkPool.empty(2, 2, 128, device="cpu")
    _populate_test_pool(pool_128, [0, 1], device=torch.device("cpu"), dtype=torch.bfloat16, seed=83)
    q_128 = torch.randn((2, 4, 3, 128), dtype=torch.bfloat16)

    class _SwapHeadDimMode(TorchFunctionMode):
        def __torch_function__(
            self,
            func: Any,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            if func is torch.ops.piper_kernels.piper_attention_decode.default:
                args = (
                    q_128,
                    pool_128.key_int8,
                    pool_128.key_scale,
                    pool_128.key_center,
                    pool_128.value_int8,
                    pool_128.value_scale,
                    pool_128.value_mean,
                    *args[7:],
                )
            return func(*args, **(kwargs or {}))

    with _SwapHeadDimMode():
        swapped_o, swapped_l = piper_attention_decode(q, pool, c2s, nc, return_lse=True)
    ref_128_o, ref_128_l = piper_attention_decode(q_128, pool_128, c2s, nc, return_lse=True)
    stale_128_o, _ = piper_attention_decode(
        q_128, pool_128, c2s, nc, scale=64**-0.5, return_lse=True
    )
    torch.testing.assert_close(swapped_o, ref_128_o, atol=0.0, rtol=0.0)
    torch.testing.assert_close(swapped_l, ref_128_l, atol=0.0, rtol=0.0)
    assert not torch.allclose(swapped_o, stale_128_o)

    with torch.no_grad():
        res = piper_attention_decode(q_grad, pool, c2s, nc)
        assert not res.requires_grad


def test_merge_attention_lse_validation_and_edge_cases() -> None:
    torch.manual_seed(501)
    o0 = torch.randn((2, 4, 3, 64), dtype=torch.bfloat16)
    l0 = torch.randn((2, 4, 3), dtype=torch.float32)

    with pytest.raises(TypeError):
        merge_attention_lse("bad", [l0])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        merge_attention_lse(bytearray(b"ab"), [l0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        merge_attention_lse([], [])
    with pytest.raises(ValueError, match="equal lengths"):
        merge_attention_lse([o0, o0], [l0])
    with pytest.raises(ValueError, match="dtype"):
        merge_attention_lse([o0, o0.float()], [l0, l0])
    with pytest.raises(ValueError, match="shape"):
        merge_attention_lse([o0, o0], [l0, l0[:, :, :2]])
    with pytest.raises(ValueError, match="shape"):
        merge_attention_lse([o0, o0[:, :, :2]], [l0, l0])
    with pytest.raises(ValueError, match="floating"):
        merge_attention_lse([o0.to(torch.int32)], [l0])
    with pytest.raises(ValueError, match="floating"):
        merge_attention_lse([o0.to(torch.float8_e4m3fn)], [l0])
    with pytest.raises(ValueError, match="floating"):
        merge_attention_lse([o0], [l0.to(torch.float8_e4m3fn)])
    sparse_out = o0.to_sparse()
    with pytest.raises(ValueError, match="strided"):
        merge_attention_lse([sparse_out], [l0])
    with pytest.raises(ValueError, match="strided"):
        merge_attention_lse([o0], [l0.to_sparse()])

    # A wave whose lse is -inf with NaN output must not taint valid waves; all -inf -> (0, -inf).
    nan_wave = torch.full_like(o0, float("nan"))
    neg_inf = torch.full_like(l0, -float("inf"))
    l_partial = l0.clone()
    l_partial[1, 0, :] = -float("inf")
    merged_out, merged_lse = merge_attention_lse([o0, nan_wave], [l_partial, neg_inf])
    assert torch.isfinite(merged_out[0]).all()
    torch.testing.assert_close(merged_out[0], o0[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(merged_lse[0], l_partial[0], atol=1e-5, rtol=1e-5)
    assert (merged_out[1, 0] == 0.0).all()
    assert (merged_lse[1, 0] == -float("inf")).all()

    all_empty_o, all_empty_l = merge_attention_lse([nan_wave, nan_wave], [neg_inf, neg_inf])
    assert (all_empty_o == 0.0).all()
    assert (all_empty_l == -float("inf")).all()


def test_merge_attention_lse_nan_inf_fp64_and_negated_views() -> None:
    torch.manual_seed(601)
    o0 = torch.randn((2, 4, 3, 64), dtype=torch.bfloat16)
    o1 = torch.randn_like(o0)
    l0 = torch.randn((2, 4, 3), dtype=torch.float32)

    # NaN in LSE propagates cleanly and +inf in LSE must not collapse to (0.0, -inf).
    l_with_nan = l0.clone()
    l_with_nan[0, 0, 0] = float("nan")
    nan_out, nan_lse = merge_attention_lse([o0, o1], [l_with_nan, l0])
    assert torch.isnan(nan_out[0, 0, 0]).all()
    assert torch.isnan(nan_lse[0, 0, 0])
    assert torch.isfinite(nan_out[0, 0, 1]).all()

    pos_inf_o, pos_inf_l = merge_attention_lse(
        [torch.full((1, 2), 3.0), torch.full((1, 2), 5.0)],
        [torch.tensor([float("inf")]), torch.tensor([1.0])],
    )
    assert torch.isnan(pos_inf_o).all()
    assert torch.isnan(pos_inf_l).all()

    # Large finite wave outputs normalize weights before summing so they never overflow to inf.
    for big_dtype, big_val in (
        (torch.float64, 1.0e308),
        (torch.float32, 2.0e38),
        (torch.bfloat16, 2.0e38),
    ):
        big_wave = torch.full((2, 4), big_val, dtype=big_dtype)
        big_lse = torch.zeros(
            (2,), dtype=torch.float64 if big_dtype is torch.float64 else torch.float32
        )
        big_mo, big_ml = merge_attention_lse([big_wave, big_wave], [big_lse, big_lse])
        assert torch.isfinite(big_mo).all()
        torch.testing.assert_close(big_mo, big_wave, atol=0.0, rtol=2e-2)
        torch.testing.assert_close(
            big_ml,
            torch.full_like(big_ml, math.log(2.0)),
            atol=1e-6,
            rtol=1e-6,
        )

    # float64 inputs preserve float64 precision instead of rounding through float32.
    f64_o0 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    f64_o1 = torch.tensor([[1.0 + 1e-10, 2.0 - 1e-10]], dtype=torch.float64)
    f64_l = torch.zeros((1,), dtype=torch.float64)
    f64_merged_o, f64_merged_l = merge_attention_lse([f64_o0, f64_o1], [f64_l, f64_l])
    assert f64_merged_o.dtype is torch.float64
    assert f64_merged_l.dtype is torch.float64
    torch.testing.assert_close(
        f64_merged_o,
        torch.tensor([[1.0 + 0.5e-10, 2.0 - 0.5e-10]], dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )
    l1 = torch.randn_like(l0)
    mixed_o, mixed_l = merge_attention_lse([o0, o1], [l0, l1.half()])
    ref_mixed_o, ref_mixed_l = merge_attention_lse([o0, o1], [l0, l1])
    torch.testing.assert_close(mixed_o, ref_mixed_o, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(mixed_l, ref_mixed_l, atol=1e-2, rtol=1e-2)


def test_merge_attention_lse_negated_views() -> None:
    torch.manual_seed(602)
    o0 = torch.randn((2, 4, 3, 64), dtype=torch.float32)
    o1 = torch.randn_like(o0)
    l0 = torch.randn((2, 4, 3), dtype=torch.float32)
    l1 = torch.randn_like(l0)

    # Negated views preserve negation across eager, in-graph Inductor _neg_view, and aot_eager.
    o_neg = torch._neg_view(o0)
    l_neg = torch._neg_view(l0)
    assert o_neg.is_neg()
    assert l_neg.is_neg()
    ref_neg_o, ref_neg_l = merge_attention_lse([-o0, o1], [-l0, l1])
    eager_neg_o, eager_neg_l = merge_attention_lse([o_neg, o1], [l_neg, l1])
    torch.testing.assert_close(eager_neg_o, ref_neg_o, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(eager_neg_l, ref_neg_l, atol=1e-6, rtol=1e-6)

    if _HAS_CPU_INDUCTOR:
        torch._dynamo.reset()
        compiled_inner = torch.compile(
            lambda a0, a1, b0, b1: merge_attention_lse(
                [torch._neg_view(a0), a1], [torch._neg_view(b0), b1]
            ),
            fullgraph=True,
        )
        inner_o, inner_l = compiled_inner(o0, o1, l0, l1)
        torch.testing.assert_close(inner_o, ref_neg_o, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(inner_l, ref_neg_l, atol=1e-6, rtol=1e-6)

    torch._dynamo.reset()
    compiled_aot = torch.compile(
        lambda a0, a1, b0, b1: merge_attention_lse([a0, a1], [b0, b1]),
        backend="aot_eager",
        fullgraph=True,
    )
    aot_o, aot_l = compiled_aot(o_neg, o1, l_neg, l1)
    torch.testing.assert_close(aot_o, ref_neg_o, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(aot_l, ref_neg_l, atol=1e-6, rtol=1e-6)


def test_merge_attention_lse_autograd_and_edge_shapes() -> None:
    torch.manual_seed(603)
    o0 = torch.randn((2, 3, 4), dtype=torch.float64, requires_grad=True)
    o1 = torch.randn((2, 3, 4), dtype=torch.float64, requires_grad=True)
    l0 = torch.randn((2, 3), dtype=torch.float64, requires_grad=True)
    l1 = torch.randn((2, 3), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a0, a1, b0, b1: merge_attention_lse([a0, a1], [b0, b1]),
        (o0, o1, l0, l1),
    )
    assert torch.autograd.gradcheck(
        lambda a0, a1, b0: merge_attention_lse([a0, a1], [b0, b0]),
        (o0, o1, l0),
    )

    # Masked -inf and all-(-inf) waves yield finite zero gradients without NaN leakage.
    og0 = torch.randn((2, 4), dtype=torch.float32, requires_grad=True)
    og1 = torch.full((2, 4), float("nan"), dtype=torch.float32, requires_grad=True)
    lg0 = torch.tensor([1.0, -float("inf")], dtype=torch.float32, requires_grad=True)
    lg1 = torch.tensor([-float("inf"), -float("inf")], dtype=torch.float32, requires_grad=True)
    mo, ml = merge_attention_lse([og0, og1], [lg0, lg1])
    loss = mo[0].sum() + ml[0]
    loss.backward()
    assert og0.grad is not None
    assert torch.isfinite(og0.grad).all()
    assert lg0.grad is not None
    assert torch.isfinite(lg0.grad).all()
    assert (og0.grad[1] == 0.0).all()
    assert og1.grad is not None
    assert torch.isfinite(og1.grad).all()
    assert (og1.grad == 0.0).all()
    assert lg1.grad is not None
    assert torch.isfinite(lg1.grad).all()
    assert (lg1.grad == 0.0).all()

    og0_all = og0.detach().clone().requires_grad_(True)
    og1_all = og1.detach().clone().requires_grad_(True)
    lg0_all = lg0.detach().clone().requires_grad_(True)
    lg1_all = lg1.detach().clone().requires_grad_(True)
    mo_all, ml_all = merge_attention_lse([og0_all, og1_all], [lg0_all, lg1_all])
    (mo_all.sum() + ml_all.sum()).backward()
    for leaf in (og0_all, og1_all, lg0_all, lg1_all):
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()

    # 1-D outputs (0-D LSE), 0-size tensors, 3-wave float16 + float64 LSE promotion.
    s_o, s_l = merge_attention_lse([torch.randn(8)], [torch.tensor(0.5)])
    assert s_o.shape == (8,)
    assert s_l.shape == ()
    z_o, z_l = merge_attention_lse(
        [torch.empty((0, 4, 8)), torch.empty((0, 4, 8))],
        [torch.empty((0, 4)), torch.empty((0, 4))],
    )
    assert z_o.shape == (0, 4, 8)
    assert z_l.shape == (0, 4)

    f16_waves = [torch.randn((2, 4), dtype=torch.float16) for _ in range(3)]
    mixed_lses = [
        torch.randn((2,), dtype=torch.bfloat16),
        torch.randn((2,), dtype=torch.float64),
        torch.randn((2,), dtype=torch.float16),
    ]
    p_o, p_l = merge_attention_lse(f16_waves, mixed_lses)
    assert p_o.dtype is torch.float16
    assert p_l.dtype is torch.float64

    with pytest.raises(ValueError, match="at least 1 dim"):
        merge_attention_lse([torch.tensor(1.0)], [torch.tensor(1.0)])
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="device"):
            merge_attention_lse([torch.randn(2, 4)], [torch.randn(2, device="cuda")])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kv_dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("gqa_layout", [(2, 2), (2, 6), (1, 4)])
def test_cpu_frozen_tail_mtp_gqa_and_sentinels(
    dtype: torch.dtype,
    head_dim: int,
    kv_dtype: torch.dtype,
    gqa_layout: tuple[int, int],
) -> None:
    kv_heads, q_heads = gqa_layout
    batch = 3
    seq_q = 5  # pushes gqa_group * seq_q > 16 when q_heads // kv_heads >= 4
    num_slots = 5
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device="cpu", kv_dtype=kv_dtype)
    # Leave slot 0 untouched with NaN sentinels; populate slots [1, 3, 4].
    k_frozen, v_frozen = _populate_test_pool(
        pool, [1, 3, 4], device=torch.device("cpu"), dtype=dtype, seed=101
    )
    slot_keys = {
        1: (k_frozen[0], v_frozen[0]),
        3: (k_frozen[1], v_frozen[1]),
        4: (k_frozen[2], v_frozen[2]),
    }

    torch.manual_seed(202 + head_dim)
    query = torch.randn((batch, seq_q, q_heads, head_dim), dtype=dtype).transpose(1, 2)
    # Mix valid slots, duplicate slot 1, negative/out-of-range misses, and slice stride.
    c2s_wide = torch.full((batch, 8), -7, dtype=torch.int32)
    c2s_wide[:, ::2] = torch.tensor(
        [
            [1, 3, -1, 99],
            [4, 1, 1, 3],
            [-1, 5, 0, -3],
        ],
        dtype=torch.int32,
    )
    chunk_to_slot = c2s_wide[:, ::2]
    assert chunk_to_slot.stride(1) == 2
    num_chunks = torch.tensor([3, 99, 0], dtype=torch.int32)
    tail_key, tail_value, tail_len = _make_sentinel_tail(
        batch, kv_heads, 21, head_dim, [7, 21, 3], device=torch.device("cpu"), dtype=dtype
    )

    for is_causal in (True, False):
        out_default = piper_attention_decode(
            query,
            pool,
            chunk_to_slot,
            num_chunks,
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            is_causal=is_causal,
        )
        out, lse = piper_attention_decode(
            query,
            pool,
            chunk_to_slot,
            num_chunks,
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            is_causal=is_causal,
            return_lse=True,
        )
        torch.testing.assert_close(out, out_default, atol=0.0, rtol=0.0)
        assert out.shape == (batch, q_heads, seq_q, head_dim)
        assert out.dtype is dtype
        assert lse.shape == (batch, q_heads, seq_q)
        assert lse.dtype is torch.float32
        assert torch.isfinite(out).all()
        if is_causal:
            # Batch 2 has 0 frozen chunks and active_tail=3 < seq_q=5 -> q_pos 0, 1 see 0 tokens.
            assert (out[2, :, :2] == 0.0).all()
            assert (lse[2, :, :2] == -float("inf")).all()
            assert torch.isfinite(lse[2, :, 2:]).all()
        else:
            assert torch.isfinite(lse).all()

        # Verify against standalone wave merge (frozen chunks wave + tail wave).
        out_f, lse_f = piper_attention_decode(
            query, pool, chunk_to_slot, num_chunks, is_causal=is_causal, return_lse=True
        )
        out_t, lse_t = piper_attention_decode(
            query,
            pool,
            None,
            torch.zeros_like(num_chunks),
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            is_causal=is_causal,
            return_lse=True,
        )
        merged_out, merged_lse = merge_attention_lse([out_f, out_t], [lse_f, lse_t])
        tol = 2e-2 if dtype is torch.bfloat16 else 5e-3
        torch.testing.assert_close(merged_out, out, atol=tol, rtol=tol)
        torch.testing.assert_close(merged_lse, lse, atol=1e-4, rtol=1e-4)

    for causal_flag in (True, False):
        sdpa, sdpa_lse = _exact_fp32_attention(
            query[:2],
            slot_keys,
            chunk_to_slot[:2],
            num_chunks[:2],
            num_slots,
            tail_key=tail_key[:2],
            tail_value=tail_value[:2],
            tail_len=tail_len[:2],
            scale=head_dim**-0.5,
            is_causal=causal_flag,
        )
        out_cmp, lse_cmp = piper_attention_decode(
            query[:2],
            pool,
            chunk_to_slot[:2],
            num_chunks[:2],
            tail_key=tail_key[:2],
            tail_value=tail_value[:2],
            tail_len=tail_len[:2],
            is_causal=causal_flag,
            return_lse=True,
        )
        cos = fn.cosine_similarity(out_cmp.float().reshape(-1), sdpa.float().reshape(-1), dim=0)
        assert float(cos) > 0.99
        torch.testing.assert_close(lse_cmp, sdpa_lse, atol=5e-2, rtol=2e-2)

        sdpa_tail, sdpa_tail_lse = _exact_fp32_attention(
            query,
            {},
            None,
            torch.zeros_like(num_chunks),
            num_slots,
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            scale=head_dim**-0.5,
            is_causal=causal_flag,
        )
        out_tail_exact, lse_tail_exact = piper_attention_decode(
            query,
            pool,
            None,
            torch.zeros_like(num_chunks),
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            is_causal=causal_flag,
            return_lse=True,
        )
        torch.testing.assert_close(out_tail_exact, sdpa_tail, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse_tail_exact, sdpa_tail_lse, atol=5e-4, rtol=5e-4)


def test_cpu_empty_tail_only_chunk_map_none_and_wmma_reference() -> None:
    device = torch.device("cpu")
    pool = PiperKVChunkPool.empty(3, 2, 64, device=device)
    _populate_test_pool(pool, [0, 1, 2], device=device, dtype=torch.float16, seed=77)
    q = torch.randn((2, 6, 2, 64), dtype=torch.float16)
    nc = torch.tensor([2, 3], dtype=torch.int32)

    # Direct chunk_to_slot=None must match explicit [[0, 1], [0, 1, 2]].
    out_none, lse_none = piper_attention_decode(q, pool, None, nc, return_lse=True)
    explicit = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
    out_exp, lse_exp = piper_attention_decode(q, pool, explicit, nc, return_lse=True)
    torch.testing.assert_close(out_none, out_exp, atol=0.0, rtol=0.0)
    torch.testing.assert_close(lse_none, lse_exp, atol=0.0, rtol=0.0)

    # Active chunks hint must not alter outputs.
    out_hint = piper_attention_decode(q, pool, explicit, nc, active_chunks_hint=1)
    torch.testing.assert_close(out_hint, out_exp, atol=0.0, rtol=0.0)

    # tail_cap == 0 is a valid empty tail (with frozen chunks and with empty pool).
    empty_tk = torch.empty((2, 2, 0, 64), device=device, dtype=torch.float16)
    empty_tv = torch.empty_like(empty_tk)
    zero_tl = torch.zeros((2,), device=device, dtype=torch.int32)
    out_tc0, lse_tc0 = piper_attention_decode(
        q,
        pool,
        explicit,
        nc,
        tail_key=empty_tk,
        tail_value=empty_tv,
        tail_len=zero_tl,
        return_lse=True,
    )
    torch.testing.assert_close(out_tc0, out_exp, atol=0.0, rtol=0.0)
    torch.testing.assert_close(lse_tc0, lse_exp, atol=0.0, rtol=0.0)

    # Empty context & empty pool & tail_cap == 0 produce output=0, lse=-inf.
    empty_pool = PiperKVChunkPool.empty(0, 2, 64, device=device)
    zero_nc = torch.zeros((2,), dtype=torch.int32)
    out_empty, lse_empty = piper_attention_decode(
        q,
        empty_pool,
        None,
        zero_nc,
        tail_key=empty_tk,
        tail_value=empty_tv,
        tail_len=zero_tl,
        return_lse=True,
    )
    assert (out_empty == 0.0).all()
    assert (lse_empty == -float("inf")).all()

    # batch == 0 returns empty tensors with matching metadata.
    q0 = q[:0]
    out0, lse0 = piper_attention_decode(q0, pool, None, zero_nc[:0], return_lse=True)
    assert out0.shape == (0, 6, 2, 64)
    assert lse0.shape == (0, 6, 2)
    assert out0.dtype is torch.float16
    assert lse0.dtype is torch.float32

    # pack_wmma=True reference must decode a pack_wmma=True pool identically to pack_wmma=False.
    wmma_pool = PiperKVChunkPool.empty(1, 2, 64, device=device)
    k0 = torch.randn((1, 2, KV_CHUNK_TOKENS, 64), dtype=torch.float16)
    v0 = torch.randn_like(k0)
    s0 = torch.tensor([0], dtype=torch.int32)
    freeze_kv_chunks_reference(
        k0,
        v0,
        wmma_pool.key_int8,
        wmma_pool.key_scale,
        wmma_pool.key_center,
        wmma_pool.value_int8,
        wmma_pool.value_scale,
        wmma_pool.value_mean,
        s0,
        pack_wmma=True,
    )
    plain_pool = PiperKVChunkPool.empty(1, 2, 64, device=device)
    freeze_kv_chunks_(k0, v0, plain_pool, s0)
    nc1 = torch.ones((2,), dtype=torch.int32)
    out_p, lse_p = piper_attention_decode(q, plain_pool, None, nc1, return_lse=True)
    out_w, lse_w = piper_attention_decode_reference(
        q,
        wmma_pool.key_int8,
        wmma_pool.key_scale,
        wmma_pool.key_center,
        wmma_pool.value_int8,
        wmma_pool.value_scale,
        wmma_pool.value_mean,
        None,
        nc1,
        scale=64**-0.5,
        pack_wmma=True,
    )
    torch.testing.assert_close(out_w, out_p, atol=0.0, rtol=0.0)
    torch.testing.assert_close(lse_w, lse_p, atol=0.0, rtol=0.0)


def test_cpu_tail_only_sm8x_1024_and_custom_scale() -> None:
    torch.manual_seed(78)
    device = torch.device("cpu")
    q = torch.randn((2, 6, 2, 64), dtype=torch.float16)

    # Empty pool + active tail matches exact FP32 SDPA tightly.
    empty_pool = PiperKVChunkPool.empty(0, 2, 64, device=device)
    zero_nc = torch.zeros((2,), dtype=torch.int32)
    tk_only = torch.randn((2, 2, 11, 64), device=device, dtype=torch.float16)
    tv_only = torch.randn_like(tk_only)
    tl_only = torch.tensor([9, 5], device=device, dtype=torch.int32)
    out_tail_only, lse_tail_only = piper_attention_decode(
        q,
        empty_pool,
        None,
        zero_nc,
        tail_key=tk_only,
        tail_value=tv_only,
        tail_len=tl_only,
        return_lse=True,
    )
    sdpa_tail_only, sdpa_tail_lse = _exact_fp32_attention(
        q,
        {},
        None,
        zero_nc,
        0,
        tail_key=tk_only,
        tail_value=tv_only,
        tail_len=tl_only,
        scale=64**-0.5,
        is_causal=True,
    )
    torch.testing.assert_close(out_tail_only, sdpa_tail_only, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_tail_only, sdpa_tail_lse, atol=5e-4, rtol=5e-4)

    # Functional key_scales == 1024 (SM8x layout) CPU decode & custom scale parity.
    pool_1024 = _make_strided_pool(2, 2, 64, device=device, kv_dtype=torch.int8, key_scales=1024)
    k1024, v1024 = _populate_test_pool(
        pool_1024, [0, 1], device=device, dtype=torch.float16, seed=79
    )
    slot_1024 = {0: (k1024[0], v1024[0]), 1: (k1024[1], v1024[1])}
    nc_1024 = torch.tensor([2, 1], dtype=torch.int32)
    out_1024, lse_1024 = piper_attention_decode(
        q, pool_1024, None, nc_1024, scale=0.5, return_lse=True
    )
    sdpa_1024, sdpa_lse_1024 = _exact_fp32_attention(
        q,
        slot_1024,
        None,
        nc_1024,
        2,
        tail_key=None,
        tail_value=None,
        tail_len=None,
        scale=0.5,
        is_causal=True,
    )
    cos_1024 = fn.cosine_similarity(
        out_1024.float().reshape(-1), sdpa_1024.float().reshape(-1), dim=0
    )
    assert float(cos_1024) > 0.99
    torch.testing.assert_close(lse_1024, sdpa_lse_1024, atol=1e-1, rtol=5e-2)
    out_default_s = piper_attention_decode(q, pool_1024, None, nc_1024)
    assert not torch.allclose(out_1024, out_default_s)


def test_cpu_opcheck_meta_and_faketensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = PiperKVChunkPool.empty(5, 2, 64, device="cpu", kv_dtype=torch.uint8)
    _populate_test_pool(
        pool, [0, 1, 2, 3, 4], device=torch.device("cpu"), dtype=torch.bfloat16, seed=91
    )
    q = torch.randn((2, 6, 3, 64), dtype=torch.bfloat16)
    c2s = torch.tensor([[1, 3], [3, -1]], dtype=torch.int32)
    nc = torch.tensor([2, 1], dtype=torch.int32)
    tk = torch.randn((2, 2, 9, 64), dtype=torch.bfloat16)
    tv = torch.randn_like(tk)
    tl = torch.tensor([4, 9], dtype=torch.int32)

    res_full = torch.library.opcheck(
        torch.ops.piper_kernels.piper_attention_decode.default,
        (
            q,
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
            c2s,
            nc,
            tk,
            tv,
            tl,
            0.125,
            True,
            0,
        ),
        test_utils=_OPCHECK_UTILS,
    )
    assert set(res_full.values()) == {"SUCCESS"}

    res_min = torch.library.opcheck(
        torch.ops.piper_kernels.piper_attention_decode.default,
        (
            q,
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
            None,
            nc,
            None,
            None,
            None,
            None,
            False,
            2,
        ),
        test_utils=_OPCHECK_UTILS,
    )
    assert set(res_min.values()) == {"SUCCESS"}

    # Meta device & FakeTensor execution must perform zero device/driver probes.
    probe = forbid_device_probes(monkeypatch)
    with forbid_triton_driver():
        meta_pool = PiperKVChunkPool.empty(2, 2, 64, device="meta", kv_dtype=torch.uint8)
        meta_out, meta_lse = piper_attention_decode(
            torch.empty((2, 4, 3, 64), device="meta", dtype=torch.bfloat16),
            meta_pool,
            None,
            torch.empty((2,), device="meta", dtype=torch.int32),
            tail_key=torch.empty((2, 2, 4, 64), device="meta", dtype=torch.bfloat16),
            tail_value=torch.empty((2, 2, 4, 64), device="meta", dtype=torch.bfloat16),
            tail_len=torch.empty((2,), device="meta", dtype=torch.int32),
            return_lse=True,
        )
        assert meta_out.is_meta
        assert meta_out.shape == (2, 4, 3, 64)
        assert meta_lse.is_meta
        assert meta_lse.shape == (2, 4, 3)

        with FakeTensorMode(shape_env=ShapeEnv()):
            for key_scales in (16, 1024):
                fake_pool = _make_strided_pool(
                    4, 2, 64, device="cuda", kv_dtype=torch.uint8, key_scales=key_scales
                )
                fake_out, fake_lse = piper_attention_decode(
                    torch.empty((2, 4, 3, 64), device="cuda", dtype=torch.float16),
                    fake_pool,
                    torch.empty((2, 3), device="cuda", dtype=torch.int32),
                    torch.empty((2,), device="cuda", dtype=torch.int32),
                    return_lse=True,
                )
                assert fake_out.shape == (2, 4, 3, 64)
                assert fake_out.dtype is torch.float16
                assert fake_lse.shape == (2, 4, 3)
                assert fake_lse.dtype is torch.float32

            arena = torch.empty((4, 2, 2, KV_CHUNK_TOKENS, 64), device="cuda", dtype=torch.uint8)
            shared_pool = PiperKVChunkPool.from_tensors(
                key_int8=arena[:, 0],
                key_scale=torch.empty((4, 2, 16), device="cuda", dtype=torch.float32),
                key_center=torch.empty((4, 2, 64), device="cuda", dtype=torch.float32),
                value_int8=arena[:, 1].view(4, 2, 16, 64, 64),
                value_scale=torch.empty(
                    (4, 2, KV_CHUNK_TOKENS), device="cuda", dtype=torch.float32
                ),
                value_mean=torch.empty((4, 2, 64), device="cuda", dtype=torch.float32),
            )
            shared_out, shared_lse = piper_attention_decode(
                torch.empty((2, 4, 3, 64), device="cuda", dtype=torch.bfloat16),
                shared_pool,
                None,
                torch.empty((2,), device="cuda", dtype=torch.int32),
                return_lse=True,
            )
            assert shared_out.shape == (2, 4, 3, 64)
            assert shared_lse.shape == (2, 4, 3)
    probe.assert_not_called()


def test_faketensor_overlap_and_unbacked_symint() -> None:
    pool = PiperKVChunkPool.empty(5, 2, 64, device="cpu", kv_dtype=torch.uint8)
    _populate_test_pool(
        pool, [0, 1, 2, 3, 4], device=torch.device("cpu"), dtype=torch.bfloat16, seed=91
    )
    q = torch.randn((2, 6, 3, 64), dtype=torch.bfloat16)
    c2s = torch.tensor([[1, 3], [3, -1]], dtype=torch.int32)
    nc = torch.tensor([2, 1], dtype=torch.int32)

    with pytest.MonkeyPatch.context() as mp:
        probe = forbid_device_probes(mp)
        with forbid_triton_driver():
            fake_env = ShapeEnv()
            with FakeTensorMode(shape_env=fake_env):
                arena = torch.empty(
                    (4, 2, 2, KV_CHUNK_TOKENS, 64), device="cuda", dtype=torch.uint8
                )
                with pytest.raises(ValueError, match="overlap"):
                    PiperKVChunkPool.from_tensors(
                        key_int8=arena[:, 0],
                        key_scale=torch.empty((4, 2, 16), device="cuda", dtype=torch.float32),
                        key_center=torch.empty((4, 2, 64), device="cuda", dtype=torch.float32),
                        value_int8=arena[:, 0].view(4, 2, 16, 64, 64),
                        value_scale=torch.empty(
                            (4, 2, KV_CHUNK_TOKENS), device="cuda", dtype=torch.float32
                        ),
                        value_mean=torch.empty((4, 2, 64), device="cuda", dtype=torch.float32),
                    )

                u_hint = fake_env.create_unbacked_symint()
                u_pool = _make_strided_pool(
                    2, 2, 64, device="cuda", kv_dtype=torch.uint8, key_scales=16
                )
                u_out, u_lse = piper_attention_decode(
                    torch.empty((2, 4, 3, 64), device="cuda", dtype=torch.bfloat16),
                    u_pool,
                    None,
                    torch.empty((2,), device="cuda", dtype=torch.int32),
                    active_chunks_hint=u_hint,
                    return_lse=True,
                )
                assert u_out.shape == (2, 4, 3, 64)
                assert u_lse.shape == (2, 4, 3)
        probe.assert_not_called()

    def _compile_unbacked_hint(
        q_in: torch.Tensor,
        c2s_in: torch.Tensor,
        nc_in: torch.Tensor,
        hint_in: torch.Tensor,
        *pool_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dyn_pool = PiperKVChunkPool.from_tensors(*pool_tensors)
        return piper_attention_decode(
            q_in,
            dyn_pool,
            c2s_in,
            nc_in,
            active_chunks_hint=int(hint_in.item()),
            return_lse=True,
        )

    torch._dynamo.reset()
    compiled_unbacked = torch.compile(_compile_unbacked_hint, backend="aot_eager", fullgraph=True)
    with torch.no_grad():
        comp_u_out, comp_u_lse = compiled_unbacked(
            q,
            c2s,
            nc,
            torch.tensor(2, dtype=torch.int32),
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
        )
        ref_u_out, ref_u_lse = piper_attention_decode(
            q, pool, c2s, nc, active_chunks_hint=2, return_lse=True
        )
        torch.testing.assert_close(comp_u_out, ref_u_out, atol=0.0, rtol=0.0)
        torch.testing.assert_close(comp_u_lse, ref_u_lse, atol=0.0, rtol=0.0)


def test_cpu_dynamic_compile() -> None:
    num_slots = 7
    pool = PiperKVChunkPool.empty(num_slots, 2, 64, device="cpu", kv_dtype=torch.uint8)
    _populate_test_pool(
        pool, list(range(num_slots)), device=torch.device("cpu"), dtype=torch.bfloat16, seed=91
    )

    # Dynamic torch.compile over two calls with distinct dynamic dims >= 2.
    def _compiled_fn(
        q_in: torch.Tensor,
        c2s_in: torch.Tensor,
        nc_in: torch.Tensor,
        tk_in: torch.Tensor,
        tv_in: torch.Tensor,
        tl_in: torch.Tensor,
        *pool_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dyn_pool = PiperKVChunkPool.from_tensors(*pool_tensors)
        out_chunks, lse_chunks = piper_attention_decode(
            q_in, dyn_pool, c2s_in, nc_in, return_lse=True, active_chunks_hint=c2s_in.shape[1]
        )
        out_tail, lse_tail = piper_attention_decode(
            q_in,
            dyn_pool,
            None,
            torch.zeros_like(nc_in),
            tail_key=tk_in,
            tail_value=tv_in,
            tail_len=tl_in,
            return_lse=True,
        )
        return merge_attention_lse([out_chunks, out_tail], [lse_chunks, lse_tail])

    backends = ("aot_eager", "inductor") if _HAS_CPU_INDUCTOR else ("aot_eager",)
    for backend in backends:
        torch._dynamo.reset()
        counter = CompileCounterWithBackend(backend)
        compiled = torch.compile(_compiled_fn, backend=counter, fullgraph=True, dynamic=True)
        with torch.no_grad():
            # Avoid fixed dims (2 kv_heads, 6 q_heads, 7 num_slots, 64 head_dim) and collisions.
            for batch, seq_q, max_c, tail_cap in ((3, 4, 5, 11), (4, 5, 3, 17)):
                q_d = torch.randn((batch, 6, seq_q, 64), dtype=torch.bfloat16)
                c2s_d = torch.randint(-1, num_slots, (batch, max_c), dtype=torch.int32)
                nc_d = torch.full((batch,), max_c, dtype=torch.int32)
                tk_d = torch.randn((batch, 2, tail_cap, 64), dtype=torch.bfloat16)
                tv_d = torch.randn_like(tk_d)
                tl_d = torch.full((batch,), tail_cap - 1, dtype=torch.int32)

                actual_o, actual_l = compiled(
                    q_d,
                    c2s_d,
                    nc_d,
                    tk_d,
                    tv_d,
                    tl_d,
                    pool.key_int8,
                    pool.key_scale,
                    pool.key_center,
                    pool.value_int8,
                    pool.value_scale,
                    pool.value_mean,
                )
                ref_oc, ref_lc = piper_attention_decode(q_d, pool, c2s_d, nc_d, return_lse=True)
                ref_ot, ref_lt = piper_attention_decode(
                    q_d,
                    pool,
                    None,
                    torch.zeros_like(nc_d),
                    tail_key=tk_d,
                    tail_value=tv_d,
                    tail_len=tl_d,
                    return_lse=True,
                )
                exp_o, exp_l = merge_attention_lse([ref_oc, ref_ot], [ref_lc, ref_lt])
                torch.testing.assert_close(actual_o, exp_o, atol=1e-3, rtol=1e-3)
                torch.testing.assert_close(actual_l, exp_l, atol=1e-5, rtol=1e-5)
        assert counter.frame_count == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
def test_gpu_rejects_target_key_scales_mismatch() -> None:
    device = torch.device("cuda")
    gpu_scales = kv_chunk_storage_specs(2, 2, 64, device=device)["key_scale"][0][2]
    wrong_scales = 1024 if gpu_scales == 16 else 16
    pool = _make_strided_pool(
        2, 2, 64, device=device, kv_dtype=torch.uint8, key_scales=wrong_scales
    )
    q = torch.randn((1, 4, 2, 64), device=device, dtype=torch.bfloat16)
    nc = torch.tensor([1], device=device, dtype=torch.int32)
    with forbid_triton_driver(), pytest.raises(ValueError, match="key_scales"):
        piper_attention_decode(q, pool, None, nc)


def _assert_decode_parity_or_ref(
    query: torch.Tensor,
    pool: PiperKVChunkPool,
    chunk_to_slot: torch.Tensor,
    num_chunks: torch.Tensor,
    slot_keys: dict[int, tuple[torch.Tensor, torch.Tensor]],
    out: torch.Tensor,
    lse: torch.Tensor,
    tail: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    is_causal: bool,
) -> None:
    tail_key, tail_value, tail_len = tail
    head_dim = query.shape[-1]
    if _supports_native_prefill_gpu():
        sdpa, sdpa_lse = _exact_fp32_attention(
            query[:2],
            slot_keys,
            chunk_to_slot[:2],
            num_chunks[:2],
            pool.num_slots,
            tail_key=tail_key[:2],
            tail_value=tail_value[:2],
            tail_len=tail_len[:2],
            scale=head_dim**-0.5,
            is_causal=is_causal,
        )
        _assert_prefill_quality(out[:2], lse[:2], sdpa, sdpa_lse)
        if query.shape[0] > 2:
            sdpa_rest, sdpa_lse_rest = _exact_fp32_attention(
                query[2:],
                slot_keys,
                chunk_to_slot[2:],
                num_chunks[2:],
                pool.num_slots,
                tail_key=tail_key[2:],
                tail_value=tail_value[2:],
                tail_len=tail_len[2:],
                scale=head_dim**-0.5,
                is_causal=is_causal,
            )
            empty_rows = sdpa_lse_rest == -float("inf")
            assert (out[2:][empty_rows] == 0.0).all()
            assert (lse[2:][empty_rows] == -float("inf")).all()
            valid_rows = ~empty_rows
            if (num_chunks[2:] <= 0).all() and valid_rows.any():
                atol = 2e-2 if query.dtype is torch.bfloat16 else 1e-2
                torch.testing.assert_close(
                    out[2:][valid_rows], sdpa_rest[valid_rows], atol=atol, rtol=atol
                )
                torch.testing.assert_close(
                    lse[2:][valid_rows], sdpa_lse_rest[valid_rows], atol=5e-4, rtol=5e-4
                )
            elif valid_rows.any():
                _assert_prefill_quality(out[2:], lse[2:], sdpa_rest, sdpa_lse_rest)
        return
    ref_out, ref_lse = piper_attention_decode_reference(
        query,
        pool.key_int8,
        pool.key_scale,
        pool.key_center,
        pool.value_int8,
        pool.value_scale,
        pool.value_mean,
        chunk_to_slot,
        num_chunks,
        tail_key=tail_key,
        tail_value=tail_value,
        tail_len=tail_len,
        scale=head_dim**-0.5,
        is_causal=is_causal,
        pack_wmma=rdna4_available(),
    )
    atol = 2e-2 if query.dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=atol)
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_decode_parity_launches_and_modes(
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    device = torch.device("cuda")
    num_slots = 5
    kv_heads = 2
    q_heads = 8  # gqa_group = 4; seq_q = 5 -> 20 > 16 multi-tile query group
    batch = 3
    seq_q = 5

    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(pool, [1, 3, 4], device=device, dtype=dtype, seed=303)
    slot_keys = {
        1: (k_frozen[0], v_frozen[0]),
        3: (k_frozen[1], v_frozen[1]),
        4: (k_frozen[2], v_frozen[2]),
    }

    torch.manual_seed(404 + head_dim)
    query = torch.randn((batch, seq_q, q_heads, head_dim), device=device, dtype=dtype).transpose(
        1, 2
    )
    chunk_to_slot = torch.tensor(
        [
            [1, -1, 3, 99],
            [4, 1, 3, 1],
            [0, 5, -2, 0],
        ],
        device=device,
        dtype=torch.int32,
    )
    num_chunks = torch.tensor([3, 10, 0], device=device, dtype=torch.int32)
    tail_key, tail_value, tail_len = _make_sentinel_tail(
        batch, kv_heads, 25, head_dim, [9, 25, 3], device=device, dtype=dtype
    )

    for is_causal in (True, False):
        (out, lse), launches = _record_triton_launches(
            lambda is_causal=is_causal: piper_attention_decode(
                query,
                pool,
                chunk_to_slot,
                num_chunks,
                tail_key=tail_key,
                tail_value=tail_value,
                tail_len=tail_len,
                is_causal=is_causal,
                return_lse=True,
            )
        )
        assert _expected_frozen_split_kernel_name() in launches
        assert "_piper_decode_tail_split_kernel" in launches
        assert "_piper_decode_splitkv_reduce_kernel" in launches
        if _supports_native_prefill_gpu():
            _assert_no_frozen_split_launch(launches)
        _assert_decode_parity_or_ref(
            query,
            pool,
            chunk_to_slot,
            num_chunks,
            slot_keys,
            out,
            lse,
            (tail_key, tail_value, tail_len),
            is_causal=is_causal,
        )

    # Check frozen-only launch set (no tail kernel), real tail-only (no frozen kernel), and empty.
    (out_frozen, lse_frozen), f_launches = _record_triton_launches(
        lambda: piper_attention_decode(query, pool, chunk_to_slot, num_chunks, return_lse=True)
    )
    assert _expected_frozen_split_kernel_name() in f_launches
    assert "_piper_decode_tail_split_kernel" not in f_launches
    assert "_piper_decode_splitkv_reduce_kernel" in f_launches

    empty_c2s = torch.empty((batch, 0), device=device, dtype=torch.int32)
    (out_tail, lse_tail), t_launches = _record_triton_launches(
        lambda: piper_attention_decode(
            query,
            pool,
            empty_c2s,
            torch.zeros_like(num_chunks),
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            return_lse=True,
        )
    )
    _assert_no_frozen_split_launch(t_launches)
    assert _expected_frozen_split_kernel_name() not in t_launches
    assert "_piper_decode_tail_split_kernel" in t_launches
    assert "_piper_decode_splitkv_reduce_kernel" in t_launches

    sdpa_tail, sdpa_tail_lse = _exact_fp32_attention(
        query,
        {},
        empty_c2s,
        torch.zeros_like(num_chunks),
        num_slots,
        tail_key=tail_key,
        tail_value=tail_value,
        tail_len=tail_len,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    torch.testing.assert_close(
        out_tail, sdpa_tail, atol=2e-2 if dtype is torch.bfloat16 else 1e-2, rtol=2e-2
    )
    torch.testing.assert_close(lse_tail, sdpa_tail_lse, atol=5e-4, rtol=5e-4)

    merged_out, merged_lse = merge_attention_lse([out_frozen, out_tail], [lse_frozen, lse_tail])
    out_full, lse_full = piper_attention_decode(
        query,
        pool,
        chunk_to_slot,
        num_chunks,
        tail_key=tail_key,
        tail_value=tail_value,
        tail_len=tail_len,
        return_lse=True,
    )
    torch.testing.assert_close(merged_out, out_full, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(merged_lse, lse_full, atol=2e-4, rtol=2e-4)

    # Verify numerical quality against FP32 SDPA on active sequences.
    sdpa, sdpa_lse = _exact_fp32_attention(
        query[:2],
        slot_keys,
        chunk_to_slot[:2],
        num_chunks[:2],
        num_slots,
        tail_key=tail_key[:2],
        tail_value=tail_value[:2],
        tail_len=tail_len[:2],
        scale=head_dim**-0.5,
        is_causal=True,
    )
    cos = fn.cosine_similarity(out_full[:2].float().reshape(-1), sdpa.float().reshape(-1), dim=0)
    assert float(cos) > 0.99
    torch.testing.assert_close(lse_full[:2], sdpa_lse, atol=5e-2, rtol=2e-2)

    empty_pool = PiperKVChunkPool.empty(0, kv_heads, head_dim, device=device)
    (zero_pool_res, zero_batch_res), zero_launches = _record_triton_launches(
        lambda: (
            piper_attention_decode(query, empty_pool, None, num_chunks, return_lse=True),
            piper_attention_decode(query[:0], pool, None, num_chunks[:0], return_lse=True),
        )
    )
    assert zero_launches == []
    assert (zero_pool_res[0] == 0.0).all()
    assert (zero_pool_res[1] == -float("inf")).all()
    assert zero_batch_res[0].shape == (0, q_heads, seq_q, head_dim)
    assert zero_batch_res[1].shape == (0, q_heads, seq_q)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunk_map_none_strided_and_tail_splits(
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    device = torch.device("cuda")
    kv_heads = 2
    q_heads = 8
    batch = 3
    seq_q = 5
    num_slots = 4
    tol = 2e-2 if dtype is torch.bfloat16 else 1e-2

    torch.manual_seed(414 + head_dim)
    query = torch.randn((batch, q_heads, seq_q, head_dim), device=device, dtype=dtype)
    direct_nc = torch.tensor([2, 1, 0], device=device, dtype=torch.int32)
    direct_pool = _make_strided_pool(
        num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8
    )
    _populate_test_pool(direct_pool, [0, 1, 2], device=device, dtype=dtype, seed=313)

    # chunk_to_slot=None (has_chunk_map=False) must match explicit [[0, 1, ...]].
    (out_none, lse_none), none_launches = _record_triton_launches(
        lambda: piper_attention_decode(query, direct_pool, None, direct_nc, return_lse=True)
    )
    assert _expected_frozen_split_kernel_name() in none_launches
    explicit_c2s = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]], device=device, dtype=torch.int32)
    out_exp, lse_exp = piper_attention_decode(
        query, direct_pool, explicit_c2s, direct_nc, return_lse=True
    )
    torch.testing.assert_close(out_none, out_exp, atol=tol, rtol=tol)
    torch.testing.assert_close(lse_none, lse_exp, atol=2e-4, rtol=2e-4)

    # Non-contiguous chunk_to_slot (stride(1) == 2).
    c2s_wide = torch.full((batch, explicit_c2s.shape[1] * 2), -9, device=device, dtype=torch.int32)
    c2s_wide[:, ::2] = explicit_c2s
    c2s_strided = c2s_wide[:, ::2]
    assert c2s_strided.stride(1) == 2
    out_strided, lse_strided = piper_attention_decode(
        query,
        direct_pool,
        c2s_strided,
        direct_nc,
        return_lse=True,
    )
    torch.testing.assert_close(out_strided, out_exp, atol=tol, rtol=tol)
    torch.testing.assert_close(lse_strided, lse_exp, atol=2e-4, rtol=2e-4)

    # active_chunks_hint > 0 alters split count (base_blocks=2: hint=0 -> 32 vs hint=1 -> 8).
    q_hint = torch.randn((1, 4, 1, head_dim), device=device, dtype=dtype)
    c2s_hint = torch.tensor([[0, 1, 2, -1]], device=device, dtype=torch.int32)
    nc_hint = torch.tensor([3], device=device, dtype=torch.int32)
    out_h0, lse_h0 = piper_attention_decode(
        q_hint, direct_pool, c2s_hint, nc_hint, active_chunks_hint=0, return_lse=True
    )
    out_h1, lse_h1 = piper_attention_decode(
        q_hint, direct_pool, c2s_hint, nc_hint, active_chunks_hint=1, return_lse=True
    )
    torch.testing.assert_close(out_h1, out_h0, atol=tol, rtol=tol)
    torch.testing.assert_close(lse_h1, lse_h0, atol=2e-4, rtol=2e-4)

    # tail_splits == 2 path (tail_cap >= 128 and base_blocks <= 16) matches exact FP32 SDPA.
    q_t2 = torch.randn((1, 4, 2, head_dim), device=device, dtype=dtype)
    tk_t2, tv_t2, tl_t2 = _make_sentinel_tail(
        1, 1, 160, head_dim, [145], device=device, dtype=dtype
    )
    p_t2 = _make_strided_pool(1, 1, head_dim, device=device, kv_dtype=torch.uint8)
    out_t2, lse_t2 = piper_attention_decode(
        q_t2,
        p_t2,
        torch.empty((1, 0), device=device, dtype=torch.int32),
        torch.zeros((1,), device=device, dtype=torch.int32),
        tail_key=tk_t2,
        tail_value=tv_t2,
        tail_len=tl_t2,
        return_lse=True,
    )
    sdpa_t2, sdpa_t2_lse = _exact_fp32_attention(
        q_t2,
        {},
        torch.empty((1, 0), device=device, dtype=torch.int32),
        torch.zeros((1,), device=device, dtype=torch.int32),
        1,
        tail_key=tk_t2,
        tail_value=tv_t2,
        tail_len=tl_t2,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    torch.testing.assert_close(out_t2, sdpa_t2, atol=tol, rtol=tol)
    torch.testing.assert_close(lse_t2, sdpa_t2_lse, atol=5e-4, rtol=5e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("seq_q", [1, 9])
def test_gpu_splitkv_reduce_normalized_weights(seq_q: int) -> None:
    device = torch.device("cuda")
    pool = PiperKVChunkPool.empty(0, 1, 64, device=device)
    q = torch.zeros((1, 1, seq_q, 64), device=device, dtype=torch.bfloat16)
    q[..., 0] = 1.0
    tk = torch.zeros((1, 1, 128, 64), device=device, dtype=torch.bfloat16)
    tk[:, :, 0, 0] = 120.0
    tk[:, :, 64, 0] = 120.0
    tv = torch.zeros_like(tk)
    tv[:, :, 0] = 2.0e38
    tv[:, :, 64] = 2.0e38
    out, lse = piper_attention_decode(
        q,
        pool,
        None,
        torch.zeros((1,), device=device, dtype=torch.int32),
        tail_key=tk,
        tail_value=tv,
        tail_len=torch.tensor([128], device=device, dtype=torch.int32),
        scale=1.0,
        is_causal=False,
        return_lse=True,
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.full_like(out, 2.0e38), atol=0.0, rtol=2e-2)
    torch.testing.assert_close(
        lse, torch.full_like(lse, 120.0 + math.log(2.0)), atol=1e-4, rtol=1e-4
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_contracts_and_clamping(head_dim: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tol = 2e-2

    # Default kv_dtype=torch.int8 GPU freeze-to-decode + strided tail + clamping + custom scale.
    pool = PiperKVChunkPool.empty(2, 2, head_dim, device=device)  # default int8
    k_f, v_f = _populate_test_pool(pool, [0, 1], device=device, dtype=dtype, seed=420 + head_dim)
    slot_keys = {0: (k_f[0], v_f[0]), 1: (k_f[1], v_f[1])}
    torch.manual_seed(421 + head_dim)
    q_s = torch.randn((1, 3, 4, head_dim), device=device, dtype=dtype).transpose(1, 2)
    tk_s = torch.randn((1, 13, 2, head_dim), device=device, dtype=dtype).transpose(1, 2)
    tv_s = torch.randn((1, 13, 2, head_dim), device=device, dtype=dtype).transpose(1, 2)
    assert not q_s.is_contiguous()
    assert not tk_s.is_contiguous()
    assert not tv_s.is_contiguous()

    # batch == 1 with chunk_to_slot=None + tail_cap > 0 and tail_cap == 0.
    o_b1, l_b1 = piper_attention_decode(
        q_s,
        pool,
        None,
        torch.tensor([2], device=device, dtype=torch.int32),
        tail_key=tk_s,
        tail_value=tv_s,
        tail_len=torch.tensor([99], device=device, dtype=torch.int32),  # clamps to 13
        scale=0.5,
        is_causal=False,
        return_lse=True,
    )
    sdpa_b1, sdpa_l_b1 = _exact_fp32_attention(
        q_s,
        slot_keys,
        None,
        torch.tensor([2], device=device, dtype=torch.int32),
        2,
        tail_key=tk_s,
        tail_value=tv_s,
        tail_len=torch.tensor([13], device=device, dtype=torch.int32),
        scale=0.5,
        is_causal=False,
    )
    assert (
        float(fn.cosine_similarity(o_b1.float().reshape(-1), sdpa_b1.float().reshape(-1), dim=0))
        > 0.99
    )
    torch.testing.assert_close(l_b1, sdpa_l_b1, atol=5e-2, rtol=2e-2)

    # Non-causal strided tail-only with custom scale=0.5 matches exact FP32 SDPA tightly.
    o_tail_nc, l_tail_nc = piper_attention_decode(
        q_s,
        pool,
        torch.empty((1, 0), device=device, dtype=torch.int32),
        torch.tensor([-3], device=device, dtype=torch.int32),  # clamps negative to 0
        tail_key=tk_s,
        tail_value=tv_s,
        tail_len=torch.tensor([7], device=device, dtype=torch.int32),
        scale=0.5,
        is_causal=False,
        return_lse=True,
    )
    sdpa_tail_nc, sdpa_l_tail_nc = _exact_fp32_attention(
        q_s,
        {},
        torch.empty((1, 0), device=device, dtype=torch.int32),
        torch.tensor([0], device=device, dtype=torch.int32),
        2,
        tail_key=tk_s,
        tail_value=tv_s,
        tail_len=torch.tensor([7], device=device, dtype=torch.int32),
        scale=0.5,
        is_causal=False,
    )
    torch.testing.assert_close(o_tail_nc, sdpa_tail_nc, atol=tol, rtol=tol)
    torch.testing.assert_close(l_tail_nc, sdpa_l_tail_nc, atol=5e-4, rtol=5e-4)

    # Negative tail_len clamps to 0; tail_cap == 0 is valid on GPU.
    empty_tk = torch.empty((1, 2, 0, head_dim), device=device, dtype=dtype)
    o_tl_neg, l_tl_neg = piper_attention_decode(
        q_s,
        pool,
        torch.tensor([[0]], device=device, dtype=torch.int32),
        torch.tensor([1], device=device, dtype=torch.int32),
        tail_key=tk_s,
        tail_value=tv_s,
        tail_len=torch.tensor([-5], device=device, dtype=torch.int32),
        return_lse=True,
    )
    o_tc0, l_tc0 = piper_attention_decode(
        q_s,
        pool,
        torch.tensor([[0]], device=device, dtype=torch.int32),
        torch.tensor([1], device=device, dtype=torch.int32),
        tail_key=empty_tk,
        tail_value=empty_tk,
        tail_len=torch.tensor([0], device=device, dtype=torch.int32),
        return_lse=True,
    )
    torch.testing.assert_close(o_tl_neg, o_tc0, atol=tol, rtol=tol)
    torch.testing.assert_close(l_tl_neg, l_tc0, atol=2e-4, rtol=2e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_torch_compile_fullgraph_dynamic(head_dim: int) -> None:
    device = torch.device("cuda")
    is_rdna4 = rdna4_available()
    kv_heads = 2
    q_heads = 6
    num_slots = 7
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(
        pool, list(range(num_slots)), device=device, dtype=torch.bfloat16, seed=505
    )
    slot_keys = {i: (k_frozen[i], v_frozen[i]) for i in range(num_slots)}

    def _step(
        q_in: torch.Tensor,
        c2s_in: torch.Tensor,
        nc_in: torch.Tensor,
        tk_in: torch.Tensor,
        tv_in: torch.Tensor,
        tl_in: torch.Tensor,
        *pool_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dyn_pool = PiperKVChunkPool.from_tensors(*pool_tensors)
        return piper_attention_decode(
            q_in,
            dyn_pool,
            c2s_in,
            nc_in,
            tail_key=tk_in,
            tail_value=tv_in,
            tail_len=tl_in,
            return_lse=True,
            active_chunks_hint=c2s_in.shape[1],
        )

    torch._dynamo.reset()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(_step, backend=counter, fullgraph=True, dynamic=True)

    with torch.inference_mode():
        # Keep dynamic dims >= 2 and distinct from fixed 2 kv_heads, 6 q_heads, 7 slots, 64/128 dim.
        for batch, seq_q, max_c, tail_cap in ((3, 4, 5, 19), (5, 8, 4, 29)):
            q = torch.randn((batch, q_heads, seq_q, head_dim), device=device, dtype=torch.bfloat16)
            c2s = torch.randint(-1, num_slots, (batch, max_c), device=device, dtype=torch.int32)
            nc = torch.full((batch,), max_c, device=device, dtype=torch.int32)
            tk = torch.randn(
                (batch, kv_heads, tail_cap, head_dim), device=device, dtype=torch.bfloat16
            )
            tv = torch.randn_like(tk)
            tl = torch.full((batch,), tail_cap - 2, device=device, dtype=torch.int32)

            (out, lse), launches = _record_triton_launches(
                lambda q=q, c2s=c2s, nc=nc, tk=tk, tv=tv, tl=tl: compiled(
                    q,
                    c2s,
                    nc,
                    tk,
                    tv,
                    tl,
                    pool.key_int8,
                    pool.key_scale,
                    pool.key_center,
                    pool.value_int8,
                    pool.value_scale,
                    pool.value_mean,
                )
            )
            assert _expected_frozen_split_kernel_name() in launches
            assert "_piper_decode_tail_split_kernel" in launches
            assert "_piper_decode_splitkv_reduce_kernel" in launches

            if _supports_native_prefill_gpu():
                _assert_no_frozen_split_launch(launches)
                out_eager, lse_eager = piper_attention_decode(
                    q,
                    pool,
                    c2s,
                    nc,
                    tail_key=tk,
                    tail_value=tv,
                    tail_len=tl,
                    return_lse=True,
                    active_chunks_hint=c2s.shape[1],
                )
                torch.testing.assert_close(out, out_eager, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(lse, lse_eager, atol=2e-4, rtol=2e-4)
                sdpa, sdpa_lse = _exact_fp32_attention(
                    q,
                    slot_keys,
                    c2s,
                    nc,
                    num_slots,
                    tail_key=tk,
                    tail_value=tv,
                    tail_len=tl,
                    scale=head_dim**-0.5,
                    is_causal=True,
                )
                _assert_prefill_quality(out, lse, sdpa, sdpa_lse)
            else:
                ref_out, ref_lse = piper_attention_decode_reference(
                    q,
                    pool.key_int8,
                    pool.key_scale,
                    pool.key_center,
                    pool.value_int8,
                    pool.value_scale,
                    pool.value_mean,
                    c2s,
                    nc,
                    tail_key=tk,
                    tail_value=tv,
                    tail_len=tl,
                    scale=head_dim**-0.5,
                    is_causal=True,
                    pack_wmma=is_rdna4,
                )
                torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-4)
    assert counter.frame_count == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_cudagraph_replay(head_dim: int) -> None:
    device = torch.device("cuda")
    is_rdna4 = rdna4_available()
    kv_heads = 2
    q_heads = 6
    num_slots = 6
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(
        pool, [0, 1, 2, 4, 5], device=device, dtype=torch.bfloat16, seed=506
    )
    slot_keys = {slot: (k_frozen[idx], v_frozen[idx]) for idx, slot in enumerate([0, 1, 2, 4, 5])}

    q_buf = torch.randn((2, q_heads, 3, head_dim), device=device, dtype=torch.bfloat16)
    c2s_buf = torch.tensor([[1, 2, 4], [5, -1, 1]], device=device, dtype=torch.int32)
    nc_buf = torch.tensor([3, 2], device=device, dtype=torch.int32)
    tk_buf = torch.randn((2, kv_heads, 17, head_dim), device=device, dtype=torch.bfloat16)
    tv_buf = torch.randn_like(tk_buf)
    tl_buf = torch.tensor([10, 5], device=device, dtype=torch.int32)

    # Warmup prior to capture.
    for _ in range(2):
        piper_attention_decode(
            q_buf, pool, c2s_buf, nc_buf, tail_key=tk_buf, tail_value=tv_buf, tail_len=tl_buf
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_buf, lse_buf = piper_attention_decode(
            q_buf,
            pool,
            c2s_buf,
            nc_buf,
            tail_key=tk_buf,
            tail_value=tv_buf,
            tail_len=tl_buf,
            return_lse=True,
        )

    # Replay 1: new valid slots + tail lengths.
    q_buf.copy_(torch.randn_like(q_buf))
    c2s_buf.copy_(torch.tensor([[4, 5, -1], [0, 2, 1]], device=device, dtype=torch.int32))
    nc_buf.copy_(torch.tensor([2, 3], device=device, dtype=torch.int32))
    tk_buf.copy_(torch.randn_like(tk_buf))
    tv_buf.copy_(torch.randn_like(tv_buf))
    tl_buf.copy_(torch.tensor([17, 1], device=device, dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    if _supports_native_prefill_gpu():
        out_eager, lse_eager = piper_attention_decode(
            q_buf,
            pool,
            c2s_buf,
            nc_buf,
            tail_key=tk_buf,
            tail_value=tv_buf,
            tail_len=tl_buf,
            return_lse=True,
        )
        torch.testing.assert_close(out_buf, out_eager, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse_buf, lse_eager, atol=2e-4, rtol=2e-4)
        sdpa1, sdpa1_lse = _exact_fp32_attention(
            q_buf,
            slot_keys,
            c2s_buf,
            nc_buf,
            num_slots,
            tail_key=tk_buf,
            tail_value=tv_buf,
            tail_len=tl_buf,
            scale=head_dim**-0.5,
            is_causal=True,
        )
        _assert_prefill_quality(out_buf, lse_buf, sdpa1, sdpa1_lse)
    else:
        ref_o1, ref_l1 = piper_attention_decode_reference(
            q_buf,
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
            c2s_buf,
            nc_buf,
            tail_key=tk_buf,
            tail_value=tv_buf,
            tail_len=tl_buf,
            scale=head_dim**-0.5,
            is_causal=True,
            pack_wmma=is_rdna4,
        )
        torch.testing.assert_close(out_buf, ref_o1, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse_buf, ref_l1, atol=2e-4, rtol=2e-4)

    # Replay 2: batch row 0 zero active context -> output 0.0 and lse -inf.
    c2s_buf[0].fill_(-1)
    nc_buf[0] = 0
    tl_buf[0] = 0
    graph.replay()
    torch.cuda.synchronize()
    assert (out_buf[0] == 0.0).all()
    assert (lse_buf[0] == -float("inf")).all()


_DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU"),
        ],
    ),
]


@pytest.mark.parametrize("device_name", _DEVICE_PARAMS)
def test_decode_large_key_center_two_term_precision(device_name: str) -> None:
    device = torch.device(device_name)
    head_dim = 64
    scale = head_dim**-0.5
    pool = _make_strided_pool(1, 1, head_dim, device=device, kv_dtype=torch.int8)
    pool.key_int8.fill_(0)
    pool.key_int8[0, 0, :8, 0] = 20
    pool.key_int8[0, 0, 8:64, :] = -127
    pool.key_int8[0, 0, 64:72, 0] = -20
    pool.key_int8[0, 0, 72:, :] = -127
    pool.key_scale.fill_(0.5)
    pool.key_center.fill_(0.0)
    pool.key_center[0, 0, 0] = 1.0e9
    pool.value_int8.fill_(0)
    pool.value_int8[0, 0, 0, 0, :8] = 100
    pool.value_int8[0, 0, 1, 1, :8] = 100
    pool.value_scale.fill_(1.0)
    pool.value_mean.fill_(0.0)

    q = torch.zeros((1, 1, 1, head_dim), device=device, dtype=torch.float16)
    q[..., 0] = 1.0
    c2s = torch.tensor([[0]], device=device, dtype=torch.int32)
    nc = torch.tensor([1], device=device, dtype=torch.int32)

    out, lse = piper_attention_decode(q, pool, c2s, nc, scale=scale, return_lse=True)

    # Exact FP64 shifted softmax reference (tile=0/1 tok 0..7 are pack_wmma-invariant).
    q_rot64 = _signed_hadamard_forward(q.float()).reshape(1, head_dim).double()
    k_tok_scale = (
        pool.key_scale[0, 0].repeat_interleave(64)
        if pool.key_scale.shape[-1] == 16
        else pool.key_scale[0, 0]
    ).double()
    k_eff64 = pool.key_int8[0, 0].double() * k_tok_scale[:, None]
    centered_scores = (q_rot64 @ k_eff64.T).squeeze(0) * scale
    bias64 = (q_rot64 * pool.key_center[0, 0].double()).sum() * scale
    s_max = centered_scores.max()
    exp_s = torch.exp(centered_scores - s_max)
    v_eff64 = torch.zeros((KV_CHUNK_TOKENS, head_dim), device=device, dtype=torch.float64)
    v_eff64[:8, 0] = 100.0
    v_eff64[64:72, 1] = 100.0
    ref_out = ((exp_s / exp_s.sum()) @ v_eff64).to(q.dtype).reshape_as(out)
    ref_lse = ((s_max + bias64) + torch.log(exp_s.sum())).float().reshape_as(lse)

    torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device_name", _DEVICE_PARAMS)
def test_decode_nan_propagation(device_name: str) -> None:
    device = torch.device(device_name)
    head_dim = 64
    pool = _make_strided_pool(2, 2, head_dim, device=device, kv_dtype=torch.uint8)
    _populate_test_pool(pool, [0, 1], device=device, dtype=torch.bfloat16, seed=707)
    q = torch.randn((1, 2, 2, head_dim), device=device, dtype=torch.bfloat16)
    c2s = torch.tensor([[0]], device=device, dtype=torch.int32)
    nc = torch.tensor([1], device=device, dtype=torch.int32)

    # 1. NaN in query row 0 poisons row 0 without poisoning row 1.
    q_nan = q.clone()
    q_nan[0, 0, 0, 3] = float("nan")
    o_q, l_q = piper_attention_decode(q_nan, pool, c2s, nc, return_lse=True)
    assert torch.isnan(o_q[0, 0, 0]).all()
    assert torch.isnan(l_q[0, 0, 0])
    assert torch.isfinite(o_q[0, 0, 1]).all()
    assert torch.isfinite(l_q[0, 0, 1])

    # 2. NaN in key_scale / key_center poisons only the affected KV head.
    ks_backup = pool.key_scale[0, 0].clone()
    pool.key_scale[0, 0, 0] = float("nan")
    o_ks, l_ks = piper_attention_decode(q, pool, c2s, nc, return_lse=True)
    assert torch.isnan(o_ks[0, 0]).all()
    assert torch.isnan(l_ks[0, 0]).all()
    assert torch.isfinite(o_ks[0, 1]).all()
    assert torch.isfinite(l_ks[0, 1]).all()
    pool.key_scale[0, 0].copy_(ks_backup)

    kc_backup = pool.key_center[0, 0].clone()
    pool.key_center[0, 0, 0] = float("nan")
    o_kc, l_kc = piper_attention_decode(q, pool, c2s, nc, return_lse=True)
    assert torch.isnan(o_kc[0, 0]).all()
    assert torch.isnan(l_kc[0, 0]).all()
    assert torch.isfinite(o_kc[0, 1]).all()
    assert torch.isfinite(l_kc[0, 1]).all()
    pool.key_center[0, 0].copy_(kc_backup)

    # 3. Causal future tail NaN must not poison earlier query rows; attended NaN must propagate.
    q3 = torch.randn((1, 2, 3, head_dim), device=device, dtype=torch.bfloat16)
    tk = torch.randn((1, 2, 4, head_dim), device=device, dtype=torch.bfloat16)
    tv = torch.randn_like(tk)
    tl = torch.tensor([3], device=device, dtype=torch.int32)
    tk[:, 0, 2, 0] = float("nan")
    o_t, l_t = piper_attention_decode(
        q3,
        pool,
        torch.empty((1, 0), device=device, dtype=torch.int32),
        torch.zeros_like(nc),
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        is_causal=True,
        return_lse=True,
    )
    assert torch.isfinite(o_t[0, 0, :2]).all()
    assert torch.isfinite(l_t[0, 0, :2]).all()
    assert torch.isnan(o_t[0, 0, 2]).all()
    assert torch.isnan(l_t[0, 0, 2])
    assert torch.isfinite(o_t[0, 1]).all()
    assert torch.isfinite(l_t[0, 1]).all()


@pytest.mark.parametrize("device_name", _DEVICE_PARAMS)
def test_decode_split_invariant_and_indeterminate_nan(device_name: str) -> None:
    device = torch.device(device_name)
    head_dim = 64
    c2s = torch.tensor([[0]], device=device, dtype=torch.int32)
    nc = torch.tensor([1], device=device, dtype=torch.int32)

    # Split-invariant value_mean NaN propagation: both single-split (batch=64 -> int8_splits=1)
    # and multi-split (batch=1 -> int8_splits=16) must propagate NaN when a chunk has NaN v_mean.
    vm_pool = _make_strided_pool(2, 1, head_dim, device=device, kv_dtype=torch.int8)
    _populate_test_pool(vm_pool, [0, 1], device=device, dtype=torch.bfloat16, seed=711)
    q_dom = torch.ones((1, 1, 1, head_dim), device=device, dtype=torch.bfloat16)
    q_rot_dom = _signed_hadamard_forward(q_dom.float())[0, 0, 0]
    vm_pool.key_center[0, 0].copy_(q_rot_dom * 1.0e6)
    vm_pool.key_center[1, 0].zero_()
    vm_pool.value_mean[1, 0].fill_(float("nan"))
    c2s2 = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    nc2 = torch.tensor([2], device=device, dtype=torch.int32)
    for split_batch in (1, 64):
        o_vm, l_vm = piper_attention_decode(
            q_dom.repeat(split_batch, 1, 1, 1),
            vm_pool,
            c2s2.repeat(split_batch, 1),
            nc2.repeat(split_batch),
            return_lse=True,
        )
        assert torch.isnan(o_vm).all()
        assert torch.isfinite(l_vm).all()

    # Indeterminate (+inf) + (-inf) across scores and chunk_bias propagates NaN in out and lse.
    inf_pool = _make_strided_pool(1, 2, head_dim, device=device, kv_dtype=torch.int8)
    _populate_test_pool(inf_pool, [0], device=device, dtype=torch.bfloat16, seed=713)
    q_e0 = torch.zeros((1, 2, 1, head_dim), device=device, dtype=torch.bfloat16)
    q_e0[..., 0] = 1.0
    q_rot_e0 = _signed_hadamard_forward(q_e0.float())[0, 0, 0]
    sign_q = torch.where(q_rot_e0 > 0.0, 1, -1).to(torch.int8)
    pos_kc = torch.where(q_rot_e0 > 0.0, float("inf"), -float("inf"))
    neg_kc = -pos_kc
    inf_pool.key_scale[0].fill_(float("inf"))
    inf_pool.key_int8[0, 0, :, :] = sign_q  # head 0: scores=+inf, chunk_bias=-inf
    inf_pool.key_center[0, 0].copy_(neg_kc)
    inf_pool.key_int8[0, 1, :, :] = -sign_q  # head 1: scores=-inf, chunk_bias=+inf
    inf_pool.key_center[0, 1].copy_(pos_kc)
    o_indet, l_indet = piper_attention_decode(q_e0, inf_pool, c2s, nc, return_lse=True)
    assert torch.isnan(o_indet).all()
    assert torch.isnan(l_indet).all()


@pytest.mark.parametrize("device_name", _DEVICE_PARAMS)
def test_decode_inactive_block_nonfinite_values(device_name: str) -> None:
    device = torch.device(device_name)
    head_dim = 64
    pool = _make_strided_pool(2, 1, head_dim, device=device, kv_dtype=torch.int8)
    _populate_test_pool(pool, [0, 1], device=device, dtype=torch.bfloat16, seed=715)
    c2s1 = torch.tensor([[0]], device=device, dtype=torch.int32)
    nc1 = torch.tensor([1], device=device, dtype=torch.int32)

    # 1. Valid frozen chunk + live tail (seq_q > active_tail) with NaN in active tail_value:
    #    q_pos == 0 sees zero tail tokens and must match frozen-only output cleanly.
    q3 = torch.randn((1, 1, 3, head_dim), device=device, dtype=torch.bfloat16)
    tk = torch.randn((1, 1, 4, head_dim), device=device, dtype=torch.bfloat16)
    tv = torch.randn_like(tk)
    tv[0, 0, 1, 0] = float("nan")
    tl = torch.tensor([2], device=device, dtype=torch.int32)
    o_fonly, l_fonly = piper_attention_decode(q3, pool, c2s1, nc1, return_lse=True)
    o_mixed, l_mixed = piper_attention_decode(
        q3,
        pool,
        c2s1,
        nc1,
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        is_causal=True,
        return_lse=True,
    )
    torch.testing.assert_close(o_mixed[:, :, :1], o_fonly[:, :, :1], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(l_mixed[:, :, :1], l_fonly[:, :, :1], atol=2e-4, rtol=2e-4)

    # 2. Frozen -inf-bias chunk with value_scale=NaN alongside chunk 0 (1-split & multi-split).
    q1 = q3[:, :, :1]
    q_rot = _signed_hadamard_forward(q1.float())[0, 0, 0]
    pool.key_center[1, 0].copy_(
        torch.where(
            q_rot.abs() <= q_rot.abs().max() / 127.0,
            0.0,
            torch.where(q_rot > 0.0, -float("inf"), float("inf")),
        )
    )
    pool.value_scale[1, 0].fill_(float("nan"))
    c2s2 = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    nc2 = torch.tensor([2], device=device, dtype=torch.int32)
    o_valid, l_valid = piper_attention_decode(q1, pool, c2s1, nc1, return_lse=True)
    for split_batch in (1, 64):
        o_b, l_b = piper_attention_decode(
            q1.repeat(split_batch, 1, 1, 1),
            pool,
            c2s2.repeat(split_batch, 1),
            nc2.repeat(split_batch),
            return_lse=True,
        )
        torch.testing.assert_close(o_b, o_valid.expand_as(o_b), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(l_b, l_valid.expand_as(l_b), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("device_name", _DEVICE_PARAMS)
def test_decode_neg_inf_and_same_pos_inf_bias_contracts(device_name: str) -> None:
    device = torch.device(device_name)
    head_dim = 64
    pool = _make_strided_pool(1, 2, head_dim, device=device, kv_dtype=torch.uint8)
    _populate_test_pool(pool, [0], device=device, dtype=torch.bfloat16, seed=709)
    c2s = torch.tensor([[0]], device=device, dtype=torch.int32)
    nc = torch.tensor([1], device=device, dtype=torch.int32)

    q_one = torch.ones((1, 2, 1, head_dim), device=device, dtype=torch.bfloat16)
    q_rot = _signed_hadamard_forward(q_one.float())[0, 0, 0]
    neg_inf_kc = torch.where(
        q_rot.abs() <= q_rot.abs().max() / 127.0,
        0.0,
        torch.where(q_rot > 0.0, -float("inf"), float("inf")),
    )
    pool.key_center[0, 0].copy_(neg_inf_kc)
    o_ni, l_ni = piper_attention_decode(q_one, pool, c2s, nc, return_lse=True)
    assert (o_ni[0, 0] == 0.0).all()
    assert (l_ni[0, 0] == -float("inf")).all()
    assert torch.isfinite(o_ni[0, 1]).all()
    assert torch.isfinite(l_ni[0, 1]).all()

    tk_good = torch.randn((1, 2, 4, head_dim), device=device, dtype=torch.bfloat16)
    tv_good = torch.randn_like(tk_good)
    tl_good = torch.tensor([4], device=device, dtype=torch.int32)
    o_ni_t, l_ni_t = piper_attention_decode(
        q_one,
        pool,
        c2s,
        nc,
        tail_key=tk_good,
        tail_value=tv_good,
        tail_len=tl_good,
        return_lse=True,
    )
    o_tail_only, l_tail_only = piper_attention_decode(
        q_one,
        pool,
        torch.empty((1, 0), device=device, dtype=torch.int32),
        torch.zeros_like(nc),
        tail_key=tk_good,
        tail_value=tv_good,
        tail_len=tl_good,
        return_lse=True,
    )
    torch.testing.assert_close(o_ni_t[0, 0], o_tail_only[0, 0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(l_ni_t[0, 0], l_tail_only[0, 0], atol=1e-4, rtol=1e-4)

    # Single chunk with finite BF16 1e20 keys (chunk_bias == +inf) yields finite out, not NaN.
    pos_pool = _make_strided_pool(1, 1, head_dim, device=device, kv_dtype=torch.uint8)
    k_big = torch.full((1, 1, KV_CHUNK_TOKENS, head_dim), 1e20, device=device, dtype=torch.bfloat16)
    v_one = torch.ones_like(k_big)
    freeze_kv_chunks_(k_big, v_one, pos_pool, torch.tensor([0], device=device, dtype=torch.int32))
    q_big = torch.full((1, 1, 1, head_dim), 1e20, device=device, dtype=torch.bfloat16)
    o_pos, l_pos = piper_attention_decode(
        q_big,
        pos_pool,
        torch.tensor([[0]], device=device, dtype=torch.int32),
        torch.tensor([1], device=device, dtype=torch.int32),
        return_lse=True,
    )
    torch.testing.assert_close(o_pos, torch.ones_like(o_pos), atol=1e-2, rtol=1e-2)
    assert (l_pos == float("inf")).all()


@pytest.mark.parametrize("device_name", _DEVICE_PARAMS)
def test_decode_causal_seq_gt_tail_and_seq1(device_name: str) -> None:
    device = torch.device(device_name)
    is_rdna4 = device.type == "cuda" and rdna4_available()
    head_dim = 64
    pool = _make_strided_pool(1, 2, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(
        pool, [0], device=device, dtype=torch.bfloat16, seed=808
    )
    slot_keys = {0: (k_frozen[0], v_frozen[0])}
    c2s = torch.tensor([[0]], device=device, dtype=torch.int32)
    nc = torch.tensor([1], device=device, dtype=torch.int32)

    # seq_q > active_tail (seq_q=5, active_tail=2): rows 0..2 still attend all 1024 frozen tokens.
    tk, tv, tl = _make_sentinel_tail(1, 2, 8, head_dim, [2], device=device, dtype=torch.bfloat16)
    q5 = torch.randn((1, 4, 5, head_dim), device=device, dtype=torch.bfloat16)
    o5, l5 = piper_attention_decode(
        q5, pool, c2s, nc, tail_key=tk, tail_value=tv, tail_len=tl, is_causal=True, return_lse=True
    )
    assert torch.isfinite(o5).all()
    assert torch.isfinite(l5).all()
    if device.type == "cpu" or not _supports_native_prefill_gpu():
        ref_o5, ref_l5 = piper_attention_decode_reference(
            q5,
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
            c2s,
            nc,
            tail_key=tk,
            tail_value=tv,
            tail_len=tl,
            scale=head_dim**-0.5,
            is_causal=True,
            pack_wmma=is_rdna4,
        )
        torch.testing.assert_close(o5, ref_o5, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(l5, ref_l5, atol=2e-4, rtol=2e-4)
    sdpa5, sdpa5_lse = _exact_fp32_attention(
        q5,
        slot_keys,
        c2s,
        nc,
        1,
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    cos5 = fn.cosine_similarity(o5.float().reshape(-1), sdpa5.float().reshape(-1), dim=0)
    assert float(cos5) > 0.99
    torch.testing.assert_close(l5, sdpa5_lse, atol=5e-2, rtol=2e-2)

    # seq_q == 1 with is_causal=True across tail_len == 0 and tail_len > 0.
    q1 = torch.randn((1, 4, 1, head_dim), device=device, dtype=torch.bfloat16)
    for t_len in (0, 5):
        tk_t, tv_t, tl_cur = _make_sentinel_tail(
            1, 2, 8, head_dim, [t_len], device=device, dtype=torch.bfloat16
        )
        o1, l1 = piper_attention_decode(
            q1,
            pool,
            c2s,
            nc,
            tail_key=tk_t,
            tail_value=tv_t,
            tail_len=tl_cur,
            is_causal=True,
            return_lse=True,
        )
        assert torch.isfinite(l1).all()
        if device.type == "cpu" or not _supports_native_prefill_gpu():
            ref_o1, ref_l1 = piper_attention_decode_reference(
                q1,
                pool.key_int8,
                pool.key_scale,
                pool.key_center,
                pool.value_int8,
                pool.value_scale,
                pool.value_mean,
                c2s,
                nc,
                tail_key=tk_t,
                tail_value=tv_t,
                tail_len=tl_cur,
                scale=head_dim**-0.5,
                is_causal=True,
                pack_wmma=is_rdna4,
            )
            torch.testing.assert_close(o1, ref_o1, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(l1, ref_l1, atol=2e-4, rtol=2e-4)
        sdpa1, sdpa1_lse = _exact_fp32_attention(
            q1,
            slot_keys,
            c2s,
            nc,
            1,
            tail_key=tk_t,
            tail_value=tv_t,
            tail_len=tl_cur,
            scale=head_dim**-0.5,
            is_causal=True,
        )
        cos1 = fn.cosine_similarity(o1.float().reshape(-1), sdpa1.float().reshape(-1), dim=0)
        assert float(cos1) > 0.99
        torch.testing.assert_close(l1, sdpa1_lse, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("device_name", _DEVICE_PARAMS)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_decode_multi_tile_gqa(device_name: str, head_dim: int) -> None:
    device = torch.device(device_name)
    is_rdna4 = device.type == "cuda" and rdna4_available()
    kv_heads = 2
    q_heads = 16  # gqa_group=8, seq_q=4 -> gqa_group * seq_q = 32 (q_tiles=2)
    seq_q = 4
    num_slots = 3
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(
        pool, [0, 2], device=device, dtype=torch.bfloat16, seed=909
    )
    slot_keys = {0: (k_frozen[0], v_frozen[0]), 2: (k_frozen[1], v_frozen[1])}
    q = torch.randn((2, q_heads, seq_q, head_dim), device=device, dtype=torch.bfloat16)
    c2s = torch.tensor([[0, 2], [2, -1]], device=device, dtype=torch.int32)
    nc = torch.tensor([2, 1], device=device, dtype=torch.int32)
    tk, tv, tl = _make_sentinel_tail(
        2, kv_heads, 12, head_dim, [7, 3], device=device, dtype=torch.bfloat16
    )

    out, lse = piper_attention_decode(
        q, pool, c2s, nc, tail_key=tk, tail_value=tv, tail_len=tl, is_causal=True, return_lse=True
    )
    if device.type == "cpu" or not _supports_native_prefill_gpu():
        ref_out, ref_lse = piper_attention_decode_reference(
            q,
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
            c2s,
            nc,
            tail_key=tk,
            tail_value=tv,
            tail_len=tl,
            scale=head_dim**-0.5,
            is_causal=True,
            pack_wmma=is_rdna4,
        )
        torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-4)
    sdpa, sdpa_lse = _exact_fp32_attention(
        q,
        slot_keys,
        c2s,
        nc,
        num_slots,
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    cos = fn.cosine_similarity(out.float().reshape(-1), sdpa.float().reshape(-1), dim=0)
    assert float(cos) > 0.99
    torch.testing.assert_close(lse, sdpa_lse, atol=5e-2, rtol=2e-2)


def _supports_native_prefill_gpu() -> bool:
    return rdna4_available() or uint8_int8_mma_available()


def _assert_prefill_quality(
    out: torch.Tensor,
    lse: torch.Tensor,
    sdpa: torch.Tensor,
    sdpa_lse: torch.Tensor,
) -> None:
    for b in range(out.shape[0]):
        out_b = out[b].float().reshape(-1)
        sdpa_b = sdpa[b].float().reshape(-1)
        cos = fn.cosine_similarity(out_b, sdpa_b, dim=0)
        rel_err = float((out_b - sdpa_b).norm() / (sdpa_b.norm() + 1e-6))
        assert float(cos) > 0.99
        assert rel_err < 0.03
    torch.testing.assert_close(lse, sdpa_lse, atol=1e-1, rtol=3e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
def test_gpu_chunked_prefill_launch_routing() -> None:
    device = torch.device("cuda")
    is_rdna4 = rdna4_available()
    native_prefill = _supports_native_prefill_gpu()
    pool = _make_strided_pool(3, 2, 128, device=device, kv_dtype=torch.uint8)
    _populate_test_pool(pool, [0, 1, 2], device=device, dtype=torch.bfloat16, seed=1101)
    c2s = torch.tensor([[0, 2, 1]], device=device, dtype=torch.int32)
    nc = torch.tensor([3], device=device, dtype=torch.int32)
    tk, tv, tl = _make_sentinel_tail(1, 2, 96, 128, [80], device=device, dtype=torch.bfloat16)

    q63 = torch.randn((1, 6, 63, 128), device=device, dtype=torch.bfloat16)
    _, launches63 = _record_triton_launches(
        lambda: piper_attention_decode(
            q63, pool, c2s, nc, tail_key=tk, tail_value=tv, tail_len=tl, return_lse=True
        )
    )
    assert _expected_frozen_split_kernel_name() in launches63
    assert "_piper_decode_tail_split_kernel" in launches63
    assert "_piper_decode_splitkv_reduce_kernel" in launches63
    assert "_merge_two_attention_waves_kernel" not in launches63
    if native_prefill:
        _assert_no_frozen_split_launch(launches63)

    for seq_q in (64, 129):
        q = torch.randn((1, 6, seq_q, 128), device=device, dtype=torch.bfloat16)
        _, launches = _record_triton_launches(
            lambda q=q: piper_attention_decode(
                q, pool, c2s, nc, tail_key=tk, tail_value=tv, tail_len=tl, return_lse=True
            )
        )
        if not native_prefill:
            assert _expected_frozen_split_kernel_name() in launches
            continue
        _assert_no_frozen_split_launch(launches)
        assert "_piper_decode_tail_split_kernel" not in launches
        assert "_merge_two_attention_waves_kernel" in launches
        if is_rdna4:
            assert "_frozen_pool_piper_kernel" in launches
            assert "_dense_piper_kernel" in launches
        else:
            assert "_piper_chunked_prefill_kernel" in launches

    if native_prefill:
        q129 = torch.randn((1, 6, 129, 128), device=device, dtype=torch.bfloat16)
        _, f_only = _record_triton_launches(
            lambda: piper_attention_decode(q129, pool, c2s, nc, return_lse=True)
        )
        assert "_merge_two_attention_waves_kernel" not in f_only
        assert (
            "_frozen_pool_piper_kernel" if is_rdna4 else "_piper_chunked_prefill_kernel"
        ) in f_only
        if is_rdna4:
            assert "_dense_piper_kernel" not in f_only

        empty_c2s = torch.empty((1, 0), device=device, dtype=torch.int32)
        empty_nc = torch.zeros_like(nc)
        _, t_only = _record_triton_launches(
            lambda: piper_attention_decode(
                q129,
                pool,
                empty_c2s,
                empty_nc,
                tail_key=tk,
                tail_value=tv,
                tail_len=tl,
                return_lse=True,
            )
        )
        assert "_merge_two_attention_waves_kernel" not in t_only
        assert ("_dense_piper_kernel" if is_rdna4 else "_piper_chunked_prefill_kernel") in t_only
        if is_rdna4:
            assert "_frozen_pool_piper_kernel" not in t_only

        out_empty, lse_empty = piper_attention_decode(
            q129, pool, empty_c2s, empty_nc, return_lse=True
        )
        assert (out_empty == 0.0).all()
        assert (lse_empty == -float("inf")).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.int8, torch.uint8])
def test_gpu_chunked_prefill_non_aligned_causal_and_two_wave_parity(
    head_dim: int,
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> None:
    device = torch.device("cuda")
    num_slots = 6
    kv_heads = 2
    q_heads = 6
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=kv_dtype)
    k_frozen, v_frozen = _populate_test_pool(
        pool, [0, 1, 2, 3, 4, 5], device=device, dtype=dtype, seed=1201 + head_dim
    )
    slot_keys = {i: (k_frozen[i], v_frozen[i]) for i in range(num_slots)}

    # Scenario A: 5 frozen chunks + 380 prefix + seq_q=644 -> tail_len=1024 (holes, BSHD).
    torch.manual_seed(1202 + head_dim)
    seq_a = 644
    q_a = torch.randn((2, seq_a, q_heads, head_dim), device=device, dtype=dtype).transpose(1, 2)
    c2s_a = torch.tensor([[3, 0, -1, 5, 1, 99, 2], [4, 2, 1, 0, 5, 3, 0]], device=device).int()
    nc_a = torch.tensor([7, 3], device=device, dtype=torch.int32)
    tk_a_raw = torch.randn((2, 1040, kv_heads, head_dim), device=device, dtype=dtype)
    tv_a_raw = torch.randn_like(tk_a_raw)
    tk_a = tk_a_raw.transpose(1, 2)
    tv_a = tv_a_raw.transpose(1, 2)
    tk_a[0, :, 1024:].fill_(float("nan"))
    tv_a[0, :, 1024:].fill_(float("inf"))
    tk_a[1, :, 700:].fill_(float("nan"))
    tv_a[1, :, 700:].fill_(float("inf"))
    tl_a = torch.tensor([1024, 700], device=device, dtype=torch.int32)

    out_a, lse_a = piper_attention_decode(
        q_a,
        pool,
        c2s_a,
        nc_a,
        tail_key=tk_a,
        tail_value=tv_a,
        tail_len=tl_a,
        is_causal=True,
        return_lse=True,
    )
    sdpa_a, sdpa_lse_a = _exact_fp32_attention(
        q_a,
        slot_keys,
        c2s_a,
        nc_a,
        num_slots,
        tail_key=tk_a,
        tail_value=tv_a,
        tail_len=tl_a,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    _assert_prefill_quality(out_a, lse_a, sdpa_a, sdpa_lse_a)

    out_af, lse_af = piper_attention_decode(q_a, pool, c2s_a, nc_a, return_lse=True)
    out_at, lse_at = piper_attention_decode(
        q_a,
        pool,
        torch.empty((2, 0), device=device, dtype=torch.int32),
        torch.zeros_like(nc_a),
        tail_key=tk_a,
        tail_value=tv_a,
        tail_len=tl_a,
        is_causal=True,
        return_lse=True,
    )
    merged_o, merged_l = merge_attention_lse([out_af, out_at], [lse_af, lse_at])
    merge_atol = 2e-2 if dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(out_a, merged_o, atol=merge_atol, rtol=merge_atol)
    torch.testing.assert_close(lse_a, merged_l, atol=1e-4, rtol=1e-4)

    # Scenario B: chunk_to_slot=None + multi-chunk tail (tail_len=2380, seq_q=2000).
    seq_b = 2000
    q_b = torch.randn((1, q_heads, seq_b, head_dim), device=device, dtype=dtype)
    nc_b = torch.tensor([3], device=device, dtype=torch.int32)
    tk_b, tv_b, tl_b = _make_sentinel_tail(
        1, kv_heads, 2400, head_dim, [2380], device=device, dtype=dtype
    )
    for is_causal in (True, False):
        out_b, lse_b = piper_attention_decode(
            q_b,
            pool,
            None,
            nc_b,
            tail_key=tk_b,
            tail_value=tv_b,
            tail_len=tl_b,
            is_causal=is_causal,
            return_lse=True,
        )
        sdpa_b, sdpa_lse_b = _exact_fp32_attention(
            q_b,
            slot_keys,
            None,
            nc_b,
            num_slots,
            tail_key=tk_b,
            tail_value=tv_b,
            tail_len=tl_b,
            scale=head_dim**-0.5,
            is_causal=is_causal,
        )
        _assert_prefill_quality(out_b, lse_b, sdpa_b, sdpa_lse_b)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_boundary_offsets_and_seq_gt_tail(head_dim: int) -> None:
    device = torch.device("cuda")
    num_slots = 2
    kv_heads = 2
    q_heads = 4
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(
        pool, [0, 1], device=device, dtype=torch.bfloat16, seed=1301 + head_dim
    )
    slot_keys = {0: (k_frozen[0], v_frozen[0]), 1: (k_frozen[1], v_frozen[1])}
    c2s = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    nc = torch.tensor([2], device=device, dtype=torch.int32)

    torch.manual_seed(1302 + head_dim)
    for offset in (1, 62, 63, 65, 380, 1025):
        seq_q = 64
        tail_len_val = offset + seq_q
        q = torch.randn((1, q_heads, seq_q, head_dim), device=device, dtype=torch.bfloat16)
        tk, tv, tl = _make_sentinel_tail(
            1,
            kv_heads,
            tail_len_val + 17,
            head_dim,
            [tail_len_val],
            device=device,
            dtype=torch.bfloat16,
        )
        out, lse = piper_attention_decode(
            q,
            pool,
            c2s,
            nc,
            tail_key=tk,
            tail_value=tv,
            tail_len=tl,
            is_causal=True,
            return_lse=True,
        )
        sdpa, sdpa_lse = _exact_fp32_attention(
            q,
            slot_keys,
            c2s,
            nc,
            num_slots,
            tail_key=tk,
            tail_value=tv,
            tail_len=tl,
            scale=head_dim**-0.5,
            is_causal=True,
        )
        _assert_prefill_quality(out, lse, sdpa, sdpa_lse)

    # seq_q > active_tail (seq_q=128, active_tail=30): first 98 queries see zero tail tokens.
    seq_gt = 128
    q_gt = torch.randn((1, q_heads, seq_gt, head_dim), device=device, dtype=torch.bfloat16)
    tk_gt, tv_gt, tl_gt = _make_sentinel_tail(
        1, kv_heads, 96, head_dim, [30], device=device, dtype=torch.bfloat16
    )
    out_tail_only, lse_tail_only = piper_attention_decode(
        q_gt,
        pool,
        torch.empty((1, 0), device=device, dtype=torch.int32),
        torch.zeros_like(nc),
        tail_key=tk_gt,
        tail_value=tv_gt,
        tail_len=tl_gt,
        is_causal=True,
        return_lse=True,
    )
    assert (out_tail_only[:, :, :98] == 0.0).all()
    assert (lse_tail_only[:, :, :98] == -float("inf")).all()
    sdpa_tail_only, sdpa_lse_tail_only = _exact_fp32_attention(
        q_gt,
        {},
        None,
        torch.zeros_like(nc),
        0,
        tail_key=tk_gt,
        tail_value=tv_gt,
        tail_len=tl_gt,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    _assert_prefill_quality(
        out_tail_only[:, :, 98:],
        lse_tail_only[:, :, 98:],
        sdpa_tail_only[:, :, 98:],
        sdpa_lse_tail_only[:, :, 98:],
    )

    out_nc_gt, lse_nc_gt = piper_attention_decode(
        q_gt,
        pool,
        c2s,
        nc,
        tail_key=tk_gt,
        tail_value=tv_gt,
        tail_len=tl_gt,
        is_causal=False,
        return_lse=True,
    )
    sdpa_nc_gt, sdpa_lse_nc_gt = _exact_fp32_attention(
        q_gt,
        slot_keys,
        c2s,
        nc,
        num_slots,
        tail_key=tk_gt,
        tail_value=tv_gt,
        tail_len=tl_gt,
        scale=head_dim**-0.5,
        is_causal=False,
    )
    _assert_prefill_quality(out_nc_gt, lse_nc_gt, sdpa_nc_gt, sdpa_lse_nc_gt)

    out_with_pool, lse_with_pool = piper_attention_decode(
        q_gt,
        pool,
        c2s,
        nc,
        tail_key=tk_gt,
        tail_value=tv_gt,
        tail_len=tl_gt,
        is_causal=True,
        return_lse=True,
    )
    out_f_only, lse_f_only = piper_attention_decode(q_gt, pool, c2s, nc, return_lse=True)
    torch.testing.assert_close(out_with_pool[:, :, :98], out_f_only[:, :, :98], atol=0.0, rtol=0.0)
    torch.testing.assert_close(lse_with_pool[:, :, :98], lse_f_only[:, :, :98], atol=0.0, rtol=0.0)
    sdpa_gt, sdpa_lse_gt = _exact_fp32_attention(
        q_gt,
        slot_keys,
        c2s,
        nc,
        num_slots,
        tail_key=tk_gt,
        tail_value=tv_gt,
        tail_len=tl_gt,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    _assert_prefill_quality(out_with_pool, lse_with_pool, sdpa_gt, sdpa_lse_gt)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_ragged_batch(head_dim: int) -> None:
    device = torch.device("cuda")
    kv_heads = 2
    q_heads = 4
    num_slots = 4
    seq_q = 70
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(
        pool, [0, 1, 2, 3], device=device, dtype=torch.bfloat16, seed=1401 + head_dim
    )
    slot_keys = {i: (k_frozen[i], v_frozen[i]) for i in range(num_slots)}

    # Row 0: frozen+multi-chunk tail (clamped > tail_cap), Row 1: tail-only, Row 2: frozen-only,
    # Row 3: empty (negative num_chunks & tail_len clamped to 0).
    torch.manual_seed(1402 + head_dim)
    q = torch.randn((4, q_heads, seq_q, head_dim), device=device, dtype=torch.bfloat16)
    c2s = torch.tensor(
        [
            [2, -1, 0, 99],
            [0, 1, 2, 3],
            [3, 1, -5, 2],
            [0, 1, 2, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    nc = torch.tensor([9, 0, 2, -2], device=device, dtype=torch.int32)
    tk, tv, tl = _make_sentinel_tail(
        4, kv_heads, 1100, head_dim, [1500, 50, 0, -3], device=device, dtype=torch.bfloat16
    )
    out, lse = piper_attention_decode(
        q, pool, c2s, nc, tail_key=tk, tail_value=tv, tail_len=tl, is_causal=True, return_lse=True
    )
    assert (out[1, :, :20] == 0.0).all()
    assert (lse[1, :, :20] == -float("inf")).all()
    assert (out[3] == 0.0).all()
    assert (lse[3] == -float("inf")).all()
    sdpa, sdpa_lse = _exact_fp32_attention(
        q[:3],
        slot_keys,
        c2s[:3],
        nc[:3],
        num_slots,
        tail_key=tk[:3],
        tail_value=tv[:3],
        tail_len=tl[:3],
        scale=head_dim**-0.5,
        is_causal=True,
    )
    _assert_prefill_quality(out[:3], lse[:3], sdpa, sdpa_lse)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_ieee754_edges(head_dim: int) -> None:
    device = torch.device("cuda")
    q64 = torch.ones((1, 2, 64, head_dim), device=device, dtype=torch.bfloat16)
    edge_pool = _make_strided_pool(2, 2, head_dim, device=device, kv_dtype=torch.uint8)
    _populate_test_pool(
        edge_pool, [0, 1], device=device, dtype=torch.bfloat16, seed=1403 + head_dim
    )
    edge_pool.value_scale[0, 0].zero_()
    c2s_one = torch.tensor([[0]], device=device, dtype=torch.int32)
    nc_one = torch.tensor([1], device=device, dtype=torch.int32)
    o_zero_vs, l_zero_vs = piper_attention_decode(q64, edge_pool, c2s_one, nc_one, return_lse=True)
    assert torch.isfinite(o_zero_vs).all()
    assert torch.isfinite(l_zero_vs).all()
    torch.testing.assert_close(
        o_zero_vs[:, 0].float(),
        edge_pool.value_mean[0, 0].view(1, 1, head_dim).expand(1, 64, head_dim),
        atol=1e-2,
        rtol=1e-2,
    )

    # Multi-chunk pool: valid chunk 0 alongside -inf k_bias chunk 1 with poisoned NaN value meta.
    q_rot = _signed_hadamard_forward(q64.float())[0, 0, 0]
    neg_inf_kc = torch.where(
        q_rot == 0.0,
        0.0,
        torch.where(q_rot > 0.0, -float("inf"), float("inf")),
    )
    edge_pool.key_center[1, 0].copy_(neg_inf_kc)
    edge_pool.value_scale[1, 0].fill_(float("nan"))
    edge_pool.value_mean[1, 0].fill_(float("nan"))
    o_c0_only, l_c0_only = piper_attention_decode(
        q64,
        edge_pool,
        torch.tensor([[0]], device=device, dtype=torch.int32),
        nc_one,
        return_lse=True,
    )
    tk_g = torch.randn((1, 2, 70, head_dim), device=device, dtype=torch.bfloat16)
    tv_g = torch.randn_like(tk_g)
    tl_g = torch.tensor([70], device=device, dtype=torch.int32)
    o_neg_multi, l_neg_multi = piper_attention_decode(
        q64,
        edge_pool,
        torch.tensor([[0, 1]], device=device, dtype=torch.int32),
        torch.tensor([2], device=device, dtype=torch.int32),
        return_lse=True,
    )
    torch.testing.assert_close(o_neg_multi[:, 0], o_c0_only[:, 0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(l_neg_multi[:, 0], l_c0_only[:, 0], atol=1e-4, rtol=1e-4)

    # +inf k_bias from finite 1e20 keys/queries yields constant 1.0 out and +inf LSE, not NaN.
    pos_pool = _make_strided_pool(1, 1, head_dim, device=device, kv_dtype=torch.uint8)
    k_big = torch.full((1, 1, KV_CHUNK_TOKENS, head_dim), 1e20, device=device, dtype=torch.bfloat16)
    v_one = torch.ones_like(k_big)
    freeze_kv_chunks_(k_big, v_one, pos_pool, torch.tensor([0], device=device, dtype=torch.int32))
    q_big = torch.full((1, 1, 64, head_dim), 1e20, device=device, dtype=torch.bfloat16)
    o_pos, l_pos = piper_attention_decode(
        q_big,
        pos_pool,
        torch.tensor([[0]], device=device, dtype=torch.int32),
        torch.tensor([1], device=device, dtype=torch.int32),
        return_lse=True,
    )
    torch.testing.assert_close(o_pos, torch.ones_like(o_pos), atol=1e-2, rtol=1e-2)
    assert (l_pos == float("inf")).all()

    # Active NaN in pool key_center propagates cleanly through wave merge.
    pos_pool.key_center[0, 0, 0] = float("nan")
    o_nan, l_nan = piper_attention_decode(
        q_big,
        pos_pool,
        torch.tensor([[0]], device=device, dtype=torch.int32),
        torch.tensor([1], device=device, dtype=torch.int32),
        tail_key=tk_g[:, :1],
        tail_value=tv_g[:, :1],
        tail_len=tl_g,
        return_lse=True,
    )
    assert torch.isnan(o_nan).all()
    assert torch.isnan(l_nan).all()

    # Zero key_scale with huge softmax scale must associate (query_scale * key_scale) before
    # multiplying integer QK scores so +/-inf * 0.0 never produces NaN (e.g. SM8x grouped_qk=False).
    zero_k_pool = _make_strided_pool(1, 1, head_dim, device=device, kv_dtype=torch.uint8)
    v_two = torch.full((1, 1, KV_CHUNK_TOKENS, head_dim), 2.0, device=device, dtype=torch.bfloat16)
    freeze_kv_chunks_(
        torch.zeros_like(v_two),
        v_two,
        zero_k_pool,
        torch.tensor([0], device=device, dtype=torch.int32),
    )
    zero_k_pool.key_int8.view(torch.int8).fill_(127)
    zero_k_pool.key_scale.zero_()
    zero_k_pool.key_center.zero_()
    q_ones = torch.ones((1, 1, 64, head_dim), device=device, dtype=torch.bfloat16)
    expected_lse = math.log(KV_CHUNK_TOKENS)
    for q_slice in (q_ones, q_ones[:, :, :1]):
        o_zero_ks, l_zero_ks = piper_attention_decode(
            q_slice,
            zero_k_pool,
            torch.tensor([[0]], device=device, dtype=torch.int32),
            nc_one,
            scale=1e36,
            return_lse=True,
        )
        assert torch.isfinite(o_zero_ks).all()
        assert torch.isfinite(l_zero_ks).all()
        torch.testing.assert_close(o_zero_ks, torch.full_like(o_zero_ks, 2.0), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            l_zero_ks, torch.full_like(l_zero_ks, expected_lse), atol=1e-4, rtol=1e-4
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_masked_pos_inf_kbias_and_two_wave_pos_inf(head_dim: int) -> None:
    device = torch.device("cuda")
    pool = _make_strided_pool(2, 1, head_dim, device=device, kv_dtype=torch.uint8)
    empty_c2s = torch.empty((1, 0), device=device, dtype=torch.int32)
    zero_nc = torch.zeros((1,), device=device, dtype=torch.int32)

    # 1. Causal non-64-aligned tail with +inf k_bias: query rows masked out of trailing tile 1
    #    must remain finite instead of letting (-inf) + (+inf) poison valid outputs/LSE.
    q_big = torch.full((1, 1, 64, head_dim), 1e20, device=device, dtype=torch.bfloat16)
    tk_big = torch.full((1, 1, 100, head_dim), 1e20, device=device, dtype=torch.bfloat16)
    tv_one = torch.ones_like(tk_big)
    tl_100 = torch.tensor([100], device=device, dtype=torch.int32)
    o_causal_tail, l_causal_tail = piper_attention_decode(
        q_big,
        pool,
        empty_c2s,
        zero_nc,
        tail_key=tk_big,
        tail_value=tv_one,
        tail_len=tl_100,
        is_causal=True,
        return_lse=True,
    )
    assert torch.isfinite(o_causal_tail).all()
    torch.testing.assert_close(o_causal_tail, torch.ones_like(o_causal_tail), atol=1e-2, rtol=1e-2)
    assert (l_causal_tail == float("inf")).all()

    # 2. Causal seq_q > active_tail mixing a finite frozen chunk with a +inf k_bias live tail:
    #    early query rows (rows 0..31) see zero tail tokens and must match the frozen chunk only.
    _populate_test_pool(pool, [0], device=device, dtype=torch.bfloat16, seed=1499 + head_dim)
    c2s_0 = torch.tensor([[0]], device=device, dtype=torch.int32)
    nc_1 = torch.tensor([1], device=device, dtype=torch.int32)
    o_frozen_only, l_frozen_only = piper_attention_decode(q_big, pool, c2s_0, nc_1, return_lse=True)
    assert torch.isfinite(o_frozen_only).all()
    assert torch.isfinite(l_frozen_only).all()

    torch.manual_seed(1505 + head_dim)
    tl_64 = torch.tensor([64], device=device, dtype=torch.int32)
    o_tail_large, l_tail_large = piper_attention_decode(
        q_big,
        pool,
        empty_c2s,
        zero_nc,
        tail_key=torch.randn((1, 1, 64, head_dim), device=device, dtype=torch.bfloat16),
        tail_value=tv_one[:, :, :64],
        tail_len=tl_64,
        is_causal=False,
        return_lse=True,
    )
    assert torch.isfinite(o_tail_large).all()
    assert torch.isfinite(l_tail_large).all()
    torch.testing.assert_close(o_tail_large, torch.ones_like(o_tail_large), atol=1e-2, rtol=1e-2)

    o_mix_seq_gt, l_mix_seq_gt = piper_attention_decode(
        q_big,
        pool,
        c2s_0,
        nc_1,
        tail_key=tk_big[:, :, :32],
        tail_value=tv_one[:, :, :32],
        tail_len=torch.tensor([32], device=device, dtype=torch.int32),
        is_causal=True,
        return_lse=True,
    )
    torch.testing.assert_close(
        o_mix_seq_gt[:, :, :32], o_frozen_only[:, :, :32], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        l_mix_seq_gt[:, :, :32], l_frozen_only[:, :, :32], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        o_mix_seq_gt[:, :, 32:], torch.ones_like(o_mix_seq_gt[:, :, 32:]), atol=1e-2, rtol=1e-2
    )
    assert (l_mix_seq_gt[:, :, 32:] == float("inf")).all()

    # 3. Two-wave merge: +inf LSE in exactly one wave dominates a finite opposite wave, whereas
    #    +inf in both waves stays indeterminate NaN (matching Split-KV and reference semantics).
    k_pos = torch.full((1, 1, KV_CHUNK_TOKENS, head_dim), 1e34, device=device, dtype=torch.bfloat16)
    freeze_kv_chunks_(
        k_pos, torch.ones_like(k_pos), pool, torch.tensor([1], device=device, dtype=torch.int32)
    )
    c2s_1 = torch.tensor([[1]], device=device, dtype=torch.int32)
    q_med = torch.full((1, 1, 64, head_dim), 1e6, device=device, dtype=torch.bfloat16)
    torch.manual_seed(1511 + head_dim)
    tk_fin = torch.randn((1, 1, 64, head_dim), device=device, dtype=torch.bfloat16)
    tv_fin = torch.zeros_like(tk_fin)
    o_a_split, l_a_split = piper_attention_decode(
        q_med[:, :, :1],
        pool,
        c2s_1,
        nc_1,
        tail_key=tk_fin,
        tail_value=tv_fin,
        tail_len=tl_64,
        is_causal=True,
        return_lse=True,
    )
    o_a_prefill, l_a_prefill = piper_attention_decode(
        q_med,
        pool,
        c2s_1,
        nc_1,
        tail_key=tk_fin,
        tail_value=tv_fin,
        tail_len=tl_64,
        is_causal=True,
        return_lse=True,
    )
    torch.testing.assert_close(o_a_prefill, o_a_split.expand_as(o_a_prefill), atol=1e-2, rtol=1e-2)
    assert (l_a_split == float("inf")).all()
    assert (l_a_prefill == float("inf")).all()

    # Finite frozen wave (chunk 0) + +inf tail wave -> tail wave wins with 1.0 out and +inf LSE.
    o_b_prefill, l_b_prefill = piper_attention_decode(
        q_big,
        pool,
        c2s_0,
        nc_1,
        tail_key=tk_big[:, :, :64],
        tail_value=tv_one[:, :, :64],
        tail_len=tl_64,
        is_causal=False,
        return_lse=True,
    )
    torch.testing.assert_close(o_b_prefill, torch.ones_like(o_b_prefill), atol=1e-2, rtol=1e-2)
    assert (l_b_prefill == float("inf")).all()

    # Both waves +inf LSE -> indeterminate NaN.
    o_both_inf, l_both_inf = piper_attention_decode(
        q_big,
        pool,
        c2s_1,
        nc_1,
        tail_key=tk_big[:, :, :64],
        tail_value=tv_one[:, :, :64],
        tail_len=tl_64,
        is_causal=False,
        return_lse=True,
    )
    assert torch.isnan(o_both_inf).all()
    assert torch.isnan(l_both_inf).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_zero_key_scale_and_nan_kbias_tail(head_dim: int) -> None:
    device = torch.device("cuda")
    pool = _make_strided_pool(1, 1, head_dim, device=device, kv_dtype=torch.uint8)
    _populate_test_pool(pool, [0], device=device, dtype=torch.bfloat16, seed=1404 + head_dim)
    c2s = torch.tensor([[0]], device=device, dtype=torch.int32)
    nc = torch.tensor([1], device=device, dtype=torch.int32)
    q = torch.randn((1, 1, 64, head_dim), device=device, dtype=torch.bfloat16)

    # Underflowing softmax scale turns query_scale * key_scale to 0.0 -> uniform value_mean.
    o_tiny, l_tiny = piper_attention_decode(q, pool, c2s, nc, scale=1e-40, return_lse=True)
    torch.testing.assert_close(
        o_tiny.float(),
        pool.value_mean[0, 0].view(1, 1, 1, head_dim).expand_as(o_tiny),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(l_tiny, torch.full_like(l_tiny, math.log(KV_CHUNK_TOKENS)))

    # Explicit zero key_scale must not leave unscaled INT8 dot products in softmax.
    pool.key_scale.zero_()
    o_zs, l_zs = piper_attention_decode(q, pool, c2s, nc, return_lse=True)
    _, l_split = piper_attention_decode(q[:, :, :1], pool, c2s, nc, return_lse=True)
    torch.testing.assert_close(
        o_zs.float(),
        pool.value_mean[0, 0].view(1, 1, 1, head_dim).expand_as(o_zs),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(l_zs[:, :, :1], l_split, atol=1e-3, rtol=1e-3)

    # Finite 2**125 (+inf + -inf) k_bias on 1-tile tail (active_tail=32) must propagate NaN to out.
    if _supports_native_prefill_gpu():
        had = _signed_hadamard_forward(torch.eye(head_dim, device=device))
        qt = torch.zeros((head_dim,), device=device)
        qt[0], qt[1] = 1.0, 1.0
        kt = torch.zeros((head_dim,), device=device)
        kt[0], kt[1] = 2.0**125, -(2.0**125)
        q_kb = (
            (qt @ had.T)
            .to(torch.bfloat16)
            .view(1, 1, 1, head_dim)
            .expand(1, 1, 64, head_dim)
            .clone()
        )
        tk_kb = (
            (kt @ had.T)
            .to(torch.bfloat16)
            .view(1, 1, 1, head_dim)
            .expand(1, 1, 64, head_dim)
            .clone()
        )
        o_kb, l_kb = piper_attention_decode(
            q_kb,
            pool,
            torch.empty((1, 0), device=device, dtype=torch.int32),
            torch.zeros_like(nc),
            tail_key=tk_kb,
            tail_value=torch.ones_like(tk_kb),
            tail_len=torch.tensor([32], device=device, dtype=torch.int32),
            is_causal=False,
            return_lse=True,
        )
        assert torch.isnan(o_kb).all()
        assert torch.isnan(l_kb).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_small_denom_and_large_key_center(head_dim: int) -> None:
    device = torch.device("cuda")
    # Large-variance BF16 values (~2e35) yield legitimate denominator < 1e-30; must not clamp.
    sd_pool = _make_strided_pool(1, 1, head_dim, device=device, kv_dtype=torch.uint8)
    q_sd = torch.zeros((1, 1, 64, head_dim), device=device, dtype=torch.bfloat16)
    v_sd_tail = torch.zeros_like(q_sd)
    v_sd_tail[:, :, 32:, :] = 2e35
    empty_c2s = torch.empty((1, 0), device=device, dtype=torch.int32)
    zero_nc = torch.zeros((1,), device=device, dtype=torch.int32)
    o_sd_t, l_sd_t = piper_attention_decode(
        q_sd,
        sd_pool,
        empty_c2s,
        zero_nc,
        tail_key=q_sd,
        tail_value=v_sd_tail,
        tail_len=torch.tensor([64], device=device, dtype=torch.int32),
        is_causal=False,
        return_lse=True,
    )
    torch.testing.assert_close(
        o_sd_t.float(),
        torch.full_like(o_sd_t, 1e35, dtype=torch.float32),
        atol=0.0,
        rtol=5e-2,
    )
    torch.testing.assert_close(l_sd_t, torch.full_like(l_sd_t, math.log(64)), atol=5e-2, rtol=1e-2)

    k_sd_full = torch.zeros((1, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=torch.bfloat16)
    v_sd_full = torch.zeros_like(k_sd_full)
    v_sd_full[:, :, KV_CHUNK_TOKENS // 2 :, :] = 2e35
    freeze_kv_chunks_(k_sd_full, v_sd_full, sd_pool, torch.tensor([0], device=device).int())
    o_sd_f, l_sd_f = piper_attention_decode(
        q_sd,
        sd_pool,
        torch.tensor([[0]], device=device, dtype=torch.int32),
        torch.tensor([1], device=device, dtype=torch.int32),
        is_causal=False,
        return_lse=True,
    )
    torch.testing.assert_close(
        o_sd_f.float(),
        torch.full_like(o_sd_f, 1e35, dtype=torch.float32),
        atol=0.0,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        l_sd_f,
        torch.full_like(l_sd_f, math.log(KV_CHUNK_TOKENS)),
        atol=5e-2,
        rtol=1e-2,
    )

    # Large finite key_center (~1e36) with unit query must apply query_scale before center dot sum.
    q_big_c = torch.zeros((1, 1, 64, head_dim), device=device, dtype=torch.bfloat16)
    q_big_c[:, :, :, 0] = 1.0
    tk_big_c = torch.zeros((1, 1, 1, head_dim), device=device, dtype=torch.bfloat16)
    tk_big_c[:, :, 0, 0] = 1e36
    tv_big_c = torch.ones_like(tk_big_c)
    tl_big_c = torch.tensor([1], device=device, dtype=torch.int32)
    expected_lse_big = 1e36 / math.sqrt(head_dim)
    o_bc, l_bc = piper_attention_decode(
        q_big_c,
        sd_pool,
        empty_c2s,
        zero_nc,
        tail_key=tk_big_c,
        tail_value=tv_big_c,
        tail_len=tl_big_c,
        is_causal=False,
        return_lse=True,
    )
    assert torch.isfinite(o_bc).all()
    assert torch.isfinite(l_bc).all()
    torch.testing.assert_close(o_bc, torch.ones_like(o_bc), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        l_bc,
        torch.full_like(l_bc, expected_lse_big),
        atol=0.0,
        rtol=5e-2,
    )

    freeze_kv_chunks_(
        k_sd_full,
        torch.ones_like(k_sd_full),
        sd_pool,
        torch.tensor([0], device=device, dtype=torch.int32),
    )
    o_bc_mix, l_bc_mix = piper_attention_decode(
        q_big_c,
        sd_pool,
        torch.tensor([[0]], device=device, dtype=torch.int32),
        torch.tensor([1], device=device, dtype=torch.int32),
        tail_key=tk_big_c,
        tail_value=tv_big_c,
        tail_len=tl_big_c,
        is_causal=False,
        return_lse=True,
    )
    assert torch.isfinite(o_bc_mix).all()
    assert torch.isfinite(l_bc_mix).all()
    torch.testing.assert_close(o_bc_mix, torch.ones_like(o_bc_mix), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        l_bc_mix,
        torch.full_like(l_bc_mix, expected_lse_big),
        atol=0.0,
        rtol=5e-2,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
def test_gpu_chunked_prefill_layouts_scales_and_empty_edges() -> None:
    device = torch.device("cuda")
    # Non-contiguous chunk_to_slot, custom scale, and MQA (kv_heads=1, q_heads=4).
    pool_mqa = _make_strided_pool(3, 1, 128, device=device, kv_dtype=torch.uint8)
    k_mqa, v_mqa = _populate_test_pool(
        pool_mqa, [0, 1, 2], device=device, dtype=torch.bfloat16, seed=1410
    )
    slots_mqa = {i: (k_mqa[i], v_mqa[i]) for i in range(3)}
    q_mqa = torch.randn((2, 4, 68, 128), device=device, dtype=torch.bfloat16)
    c2s_mqa = torch.tensor([[2, 9, 0, 8], [1, 7, 2, 6]], device=device, dtype=torch.int32)[:, ::2]
    assert not c2s_mqa.is_contiguous()
    nc_mqa = torch.tensor([2, 2], device=device, dtype=torch.int32)
    tk_mqa, tv_mqa, tl_mqa = _make_sentinel_tail(
        2, 1, 100, 128, [80, 90], device=device, dtype=torch.bfloat16
    )
    out_mqa, lse_mqa = piper_attention_decode(
        q_mqa,
        pool_mqa,
        c2s_mqa,
        nc_mqa,
        tail_key=tk_mqa,
        tail_value=tv_mqa,
        tail_len=tl_mqa,
        scale=0.07,
        is_causal=True,
        return_lse=True,
        active_chunks_hint=1,
    )
    sdpa_mqa, sdpa_lse_mqa = _exact_fp32_attention(
        q_mqa,
        slots_mqa,
        c2s_mqa,
        nc_mqa,
        3,
        tail_key=tk_mqa,
        tail_value=tv_mqa,
        tail_len=tl_mqa,
        scale=0.07,
        is_causal=True,
    )
    _assert_prefill_quality(out_mqa, lse_mqa, sdpa_mqa, sdpa_lse_mqa)
    out_nolse = piper_attention_decode(
        q_mqa,
        pool_mqa,
        c2s_mqa,
        nc_mqa,
        tail_key=tk_mqa,
        tail_value=tv_mqa,
        tail_len=tl_mqa,
        scale=0.07,
        is_causal=True,
        return_lse=False,
    )
    assert isinstance(out_nolse, torch.Tensor)
    torch.testing.assert_close(out_nolse, out_mqa, atol=1e-2, rtol=1e-2)

    # MHA with odd head count (kv_heads=q_heads=3, head_dim=64 -> mask_heads=True on AMD).
    pool_mha = _make_strided_pool(2, 3, 64, device=device, kv_dtype=torch.uint8)
    k_mha, v_mha = _populate_test_pool(
        pool_mha, [0, 1], device=device, dtype=torch.bfloat16, seed=1411
    )
    q_mha = torch.randn((1, 3, 65, 64), device=device, dtype=torch.bfloat16)
    c2s_mha = torch.tensor([[1, 0]], device=device, dtype=torch.int32)
    nc_mha = torch.tensor([2], device=device, dtype=torch.int32)
    tk_mha, tv_mha, tl_mha = _make_sentinel_tail(
        1, 3, 96, 64, [80], device=device, dtype=torch.bfloat16
    )
    out_mha, lse_mha = piper_attention_decode(
        q_mha,
        pool_mha,
        c2s_mha,
        nc_mha,
        tail_key=tk_mha,
        tail_value=tv_mha,
        tail_len=tl_mha,
        return_lse=True,
    )
    sdpa_mha, sdpa_lse_mha = _exact_fp32_attention(
        q_mha,
        {0: (k_mha[0], v_mha[0]), 1: (k_mha[1], v_mha[1])},
        c2s_mha,
        nc_mha,
        2,
        tail_key=tk_mha,
        tail_value=tv_mha,
        tail_len=tl_mha,
        scale=64**-0.5,
        is_causal=True,
    )
    _assert_prefill_quality(out_mha, lse_mha, sdpa_mha, sdpa_lse_mha)

    # batch==0 and tail_cap==0 empty edges at seq_q=64.
    q_b0 = torch.empty((0, 3, 64, 64), device=device, dtype=torch.bfloat16)
    o_b0, l_b0 = piper_attention_decode(
        q_b0,
        pool_mha,
        torch.empty((0, 2), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.int32),
        return_lse=True,
    )
    assert o_b0.shape == q_b0.shape
    assert l_b0.shape == (0, 3, 64)
    o_t0, l_t0 = piper_attention_decode(
        q_mha[:, :, :64],
        pool_mha,
        torch.empty((1, 0), device=device, dtype=torch.int32),
        torch.zeros((1,), device=device, dtype=torch.int32),
        tail_key=torch.empty((1, 3, 0, 64), device=device, dtype=torch.bfloat16),
        tail_value=torch.empty((1, 3, 0, 64), device=device, dtype=torch.bfloat16),
        tail_len=torch.zeros((1,), device=device, dtype=torch.int32),
        return_lse=True,
    )
    assert (o_t0 == 0.0).all()
    assert (l_t0 == -float("inf")).all()

    # Fallback Split-KV GPU path (piper_attention_decode_triton) stays self-contained
    # without prefill forwarding and matches the portable reference across seq_q.
    from piper_kernels.attention.piper_attention.triton import (  # noqa: PLC0415 - lazy GPU import
        piper_attention_decode_triton,
    )

    grouped_qk_mha = pool_mha.key_scales == 16
    for q_fb in (q_mha[:, :, :1], q_mha):
        fb_out, fb_lse = piper_attention_decode_triton(
            q_fb,
            pool_mha.key_int8,
            pool_mha.key_scale,
            pool_mha.key_center,
            pool_mha.value_int8,
            pool_mha.value_scale,
            pool_mha.value_mean,
            c2s_mha,
            nc_mha,
            tail_key=tk_mha,
            tail_value=tv_mha,
            tail_len=tl_mha,
            scale=64**-0.5,
            is_causal=True,
            grouped_qk=grouped_qk_mha,
            pack_wmma=False,
            active_chunks_hint=0,
            window_left=1500,
            window_right=-1,
        )
        ref_out, ref_lse = piper_attention_decode_reference(
            q_fb,
            pool_mha.key_int8,
            pool_mha.key_scale,
            pool_mha.key_center,
            pool_mha.value_int8,
            pool_mha.value_scale,
            pool_mha.value_mean,
            c2s_mha,
            nc_mha,
            tail_key=tk_mha,
            tail_value=tv_mha,
            tail_len=tl_mha,
            scale=64**-0.5,
            is_causal=True,
            pack_wmma=False,
            window_left=1500,
            window_right=-1,
        )
        torch.testing.assert_close(fb_out, ref_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(fb_lse, ref_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_torch_compile_dynamic(head_dim: int) -> None:
    device = torch.device("cuda")
    kv_heads = 2
    q_heads = 6
    num_slots = 7
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(
        pool, list(range(num_slots)), device=device, dtype=torch.bfloat16, seed=1501 + head_dim
    )
    slot_keys = {i: (k_frozen[i], v_frozen[i]) for i in range(num_slots)}

    def _step(
        q_in: torch.Tensor,
        c2s_in: torch.Tensor,
        nc_in: torch.Tensor,
        tk_in: torch.Tensor,
        tv_in: torch.Tensor,
        tl_in: torch.Tensor,
        *pool_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dyn_pool = PiperKVChunkPool.from_tensors(*pool_tensors)
        return piper_attention_decode(
            q_in,
            dyn_pool,
            c2s_in,
            nc_in,
            tail_key=tk_in,
            tail_value=tv_in,
            tail_len=tl_in,
            return_lse=True,
            active_chunks_hint=c2s_in.shape[1],
        )

    torch._dynamo.reset()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(_step, backend=counter, fullgraph=True, dynamic=True)

    with torch.inference_mode():
        for seq_q, tail_cap, tl_val in ((5, 97, 81), (65, 113, 81), (130, 161, 145)):
            q = torch.randn((3, q_heads, seq_q, head_dim), device=device, dtype=torch.bfloat16)
            c2s = torch.tensor(
                [[1, 3, -1, 5], [4, 0, 2, 6], [6, -1, 2, 1]], device=device, dtype=torch.int32
            )
            nc = torch.tensor([4, 3, 2], device=device, dtype=torch.int32)
            tk, tv, tl = _make_sentinel_tail(
                3,
                kv_heads,
                tail_cap,
                head_dim,
                [tl_val, tl_val - 7, tl_val - 19],
                device=device,
                dtype=torch.bfloat16,
            )
            (out_c, lse_c), launches = _record_triton_launches(
                lambda q=q, c2s=c2s, nc=nc, tk=tk, tv=tv, tl=tl: compiled(
                    q,
                    c2s,
                    nc,
                    tk,
                    tv,
                    tl,
                    pool.key_int8,
                    pool.key_scale,
                    pool.key_center,
                    pool.value_int8,
                    pool.value_scale,
                    pool.value_mean,
                )
            )
            if seq_q < 64:
                assert _expected_frozen_split_kernel_name() in launches
                assert "_piper_decode_tail_split_kernel" in launches
                assert "_piper_decode_splitkv_reduce_kernel" in launches
                if _supports_native_prefill_gpu():
                    _assert_no_frozen_split_launch(launches)
            elif _supports_native_prefill_gpu():
                _assert_no_frozen_split_launch(launches)
                assert "_merge_two_attention_waves_kernel" in launches
            out_e, lse_e = piper_attention_decode(
                q, pool, c2s, nc, tail_key=tk, tail_value=tv, tail_len=tl, return_lse=True
            )
            torch.testing.assert_close(out_c, out_e, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(lse_c, lse_e, atol=1e-4, rtol=1e-4)
            sdpa, sdpa_lse = _exact_fp32_attention(
                q,
                slot_keys,
                c2s,
                nc,
                num_slots,
                tail_key=tk,
                tail_value=tv,
                tail_len=tl,
                scale=head_dim**-0.5,
                is_causal=True,
            )
            _assert_prefill_quality(out_c, lse_c, sdpa, sdpa_lse)
    assert counter.frame_count == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_cudagraph_replay(head_dim: int) -> None:
    device = torch.device("cuda")
    kv_heads = 2
    q_heads = 6
    num_slots = 5
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    _populate_test_pool(
        pool, [0, 1, 2, 3, 4], device=device, dtype=torch.bfloat16, seed=1502 + head_dim
    )
    q_buf = torch.randn((2, q_heads, 129, head_dim), device=device, dtype=torch.bfloat16)
    c2s_buf = torch.tensor([[0, 2, -1], [4, 1, 3]], device=device, dtype=torch.int32)
    nc_buf = torch.tensor([2, 3], device=device, dtype=torch.int32)
    tk_buf = torch.randn((2, kv_heads, 160, head_dim), device=device, dtype=torch.bfloat16)
    tv_buf = torch.randn_like(tk_buf)
    tl_buf = torch.tensor([140, 132], device=device, dtype=torch.int32)

    for _ in range(2):
        piper_attention_decode(
            q_buf, pool, c2s_buf, nc_buf, tail_key=tk_buf, tail_value=tv_buf, tail_len=tl_buf
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_buf, lse_buf = piper_attention_decode(
            q_buf,
            pool,
            c2s_buf,
            nc_buf,
            tail_key=tk_buf,
            tail_value=tv_buf,
            tail_len=tl_buf,
            return_lse=True,
        )

    q_buf.copy_(torch.randn_like(q_buf))
    c2s_buf.copy_(torch.tensor([[4, 3, -1], [1, 0, 2]], device=device, dtype=torch.int32))
    nc_buf.copy_(torch.tensor([2, 3], device=device, dtype=torch.int32))
    tk_buf.copy_(torch.randn_like(tk_buf))
    tv_buf.copy_(torch.randn_like(tv_buf))
    tl_buf.copy_(torch.tensor([155, 130], device=device, dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    eager_o, eager_l = piper_attention_decode(
        q_buf,
        pool,
        c2s_buf,
        nc_buf,
        tail_key=tk_buf,
        tail_value=tv_buf,
        tail_len=tl_buf,
        return_lse=True,
    )
    torch.testing.assert_close(out_buf, eager_o, atol=0.0, rtol=0.0)
    torch.testing.assert_close(lse_buf, eager_l, atol=0.0, rtol=0.0)

    c2s_buf[0].fill_(-1)
    nc_buf[0] = 0
    tl_buf[0] = 0
    graph.replay()
    torch.cuda.synchronize()
    assert (out_buf[0] == 0.0).all()
    assert (lse_buf[0] == -float("inf")).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_frozen_chunk_splits(head_dim: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    kv_heads = 4
    q_heads = 16
    chunks = 8
    pool = _make_strided_pool(chunks, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(
        pool, list(range(chunks)), device=device, dtype=dtype, seed=1601 + head_dim
    )
    slot_keys = {i: (k_frozen[i], v_frozen[i]) for i in range(chunks)}

    torch.manual_seed(1602 + head_dim)
    q = torch.randn((1, q_heads, 64, head_dim), device=device, dtype=dtype)
    c2s = torch.arange(chunks, device=device, dtype=torch.int32).view(1, chunks)
    nc = torch.tensor([chunks], device=device, dtype=torch.int32)
    tk = torch.randn((1, kv_heads, 128, head_dim), device=device, dtype=dtype)
    tv = torch.randn_like(tk)
    tl = torch.tensor([111], device=device, dtype=torch.int32)

    (out_split, lse_split), split_launches = _record_triton_launches(
        lambda: piper_attention_decode(
            q, pool, c2s, nc, tail_key=tk, tail_value=tv, tail_len=tl, return_lse=True
        )
    )
    (out_single, lse_single), single_launches = _record_triton_launches(
        lambda: piper_attention_decode(
            q,
            pool,
            c2s,
            nc,
            tail_key=tk,
            tail_value=tv,
            tail_len=tl,
            active_chunks_hint=1,
            return_lse=True,
        )
    )
    if (
        (torch.version.hip is None and uint8_int8_mma_available()) or rdna4_available()
    ) and torch.cuda.get_device_properties(device).multi_processor_count >= 32:
        assert "_piper_decode_splitkv_reduce_kernel" in split_launches
        assert "_piper_decode_splitkv_reduce_kernel" not in single_launches

    torch.testing.assert_close(out_split, out_single, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_split, lse_single, atol=1e-4, rtol=1e-4)
    sdpa, sdpa_lse = _exact_fp32_attention(
        q,
        slot_keys,
        c2s,
        nc,
        chunks,
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    _assert_prefill_quality(out_split, lse_split, sdpa, sdpa_lse)

    # Exercise partial empty splits (active_chunks=1 < frozen_splits) and all-empty splits.
    nc1 = torch.tensor([1], device=device, dtype=torch.int32)
    o_s1, l_s1 = piper_attention_decode(q, pool, c2s, nc1, return_lse=True)
    o_s1_ref, l_s1_ref = piper_attention_decode(
        q, pool, c2s, nc1, active_chunks_hint=1, return_lse=True
    )
    torch.testing.assert_close(o_s1, o_s1_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(l_s1, l_s1_ref, atol=1e-4, rtol=1e-4)
    sdpa1, sdpa1_lse = _exact_fp32_attention(
        q,
        slot_keys,
        c2s,
        nc1,
        chunks,
        tail_key=None,
        tail_value=None,
        tail_len=None,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    _assert_prefill_quality(o_s1, l_s1, sdpa1, sdpa1_lse)

    nc0 = torch.tensor([0], device=device, dtype=torch.int32)
    o_s0, l_s0 = piper_attention_decode(q, pool, c2s, nc0, return_lse=True)
    assert (o_s0 == 0.0).all()
    assert (l_s0 == -float("inf")).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_chunked_prefill_frozen_chunk_splits_two_term_lse(head_dim: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    nc4 = torch.tensor([4], device=device, dtype=torch.int32)

    # Shared large finite k_bias across splits must preserve small centered LSE differences.
    bias_pool = _make_strided_pool(2, 1, head_dim, device=device, kv_dtype=torch.uint8)
    freeze_kv_chunks_(
        torch.full((1, 1, KV_CHUNK_TOKENS, head_dim), 1e8, device=device, dtype=dtype),
        torch.zeros((1, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype),
        bias_pool,
        torch.tensor([0], device=device, dtype=torch.int32),
    )
    k_pm = torch.cat(
        [
            torch.full((1, 1, KV_CHUNK_TOKENS // 2, head_dim), 1.0, device=device, dtype=dtype),
            torch.full((1, 1, KV_CHUNK_TOKENS // 2, head_dim), -1.0, device=device, dtype=dtype),
        ],
        dim=2,
    )
    freeze_kv_chunks_(
        k_pm,
        torch.ones((1, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype),
        bias_pool,
        torch.tensor([1], device=device, dtype=torch.int32),
    )
    bias_pool.key_center[1].copy_(bias_pool.key_center[0])
    q_bias = torch.ones((1, 1, 64, head_dim), device=device, dtype=dtype)
    c2s_bias = torch.tensor([[0, 0, 1, 1]], device=device, dtype=torch.int32)

    o_bias_split, l_bias_split = piper_attention_decode(
        q_bias, bias_pool, c2s_bias, nc4, return_lse=True
    )
    o_bias_unsplit, l_bias_unsplit = piper_attention_decode(
        q_bias, bias_pool, c2s_bias, nc4, active_chunks_hint=1, return_lse=True
    )
    torch.testing.assert_close(o_bias_split, o_bias_unsplit, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(l_bias_split, l_bias_unsplit, atol=1e-4, rtol=1e-4)
    assert (o_bias_split > 0.99).all()

    # +inf k_bias across frozen chunk splits matches unsplit / Split-KV same-bias semantics.
    inf_pool = _make_strided_pool(2, 1, head_dim, device=device, kv_dtype=torch.uint8)
    freeze_kv_chunks_(
        torch.full((1, 1, KV_CHUNK_TOKENS, head_dim), 1e34, device=device, dtype=dtype),
        torch.ones((1, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype),
        inf_pool,
        torch.tensor([0], device=device, dtype=torch.int32),
    )
    _populate_test_pool(inf_pool, [1], device=device, dtype=dtype, seed=1603 + head_dim)
    q_med = torch.full((1, 1, 64, head_dim), 1e6, device=device, dtype=dtype)

    o_one_inf, l_one_inf = piper_attention_decode(
        q_med,
        inf_pool,
        torch.tensor([[0, 1, 1, 1]], device=device, dtype=torch.int32),
        nc4,
        return_lse=True,
    )
    torch.testing.assert_close(o_one_inf, torch.ones_like(o_one_inf), atol=1e-2, rtol=1e-2)
    assert (l_one_inf == float("inf")).all()

    c2s_two_inf = torch.tensor([[0, 1, 0, 1]], device=device, dtype=torch.int32)
    o_two_inf, l_two_inf = piper_attention_decode(
        q_med, inf_pool, c2s_two_inf, nc4, return_lse=True
    )
    o_two_inf_unsplit, l_two_inf_unsplit = piper_attention_decode(
        q_med, inf_pool, c2s_two_inf, nc4, active_chunks_hint=1, return_lse=True
    )
    torch.testing.assert_close(o_two_inf, o_two_inf_unsplit, atol=1e-2, rtol=1e-2)
    assert (l_two_inf == l_two_inf_unsplit).all()
    assert (l_two_inf == float("inf")).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_gpu_split_large_centered_max_preserves_denom(head_dim: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = _make_strided_pool(2, 1, head_dim, device=device, kv_dtype=torch.uint8)

    k_slot0 = torch.cat(
        [
            torch.full((1, 1, 512, head_dim), 1.0, device=device, dtype=dtype),
            torch.full((1, 1, 512, head_dim), -1.0, device=device, dtype=dtype),
        ],
        dim=2,
    )
    freeze_kv_chunks_(
        k_slot0,
        torch.zeros((1, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype),
        pool,
        torch.tensor([0], device=device, dtype=torch.int32),
    )
    k_slot1 = torch.cat(
        [
            torch.full((1, 1, 256, head_dim), 1.0, device=device, dtype=dtype),
            torch.full((1, 1, 256, head_dim), -1.0, device=device, dtype=dtype),
            torch.zeros((1, 1, 512, head_dim), device=device, dtype=dtype),
        ],
        dim=2,
    )
    freeze_kv_chunks_(
        k_slot1,
        torch.ones((1, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype),
        pool,
        torch.tensor([1], device=device, dtype=torch.int32),
    )

    c2s = torch.tensor([[0, 0, 1, 1]], device=device, dtype=torch.int32)
    nc = torch.tensor([4], device=device, dtype=torch.int32)
    q_prefill = torch.full((1, 1, 64, head_dim), 1e8, device=device, dtype=dtype)
    expected = torch.full((1, 1, 64, head_dim), 1.0 / 3.0, device=device, dtype=dtype)

    o_split, l_split = piper_attention_decode(q_prefill, pool, c2s, nc, return_lse=True)
    o_unsplit, l_unsplit = piper_attention_decode(
        q_prefill, pool, c2s, nc, active_chunks_hint=1, return_lse=True
    )
    torch.testing.assert_close(o_split, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(o_split, o_unsplit, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(l_split, l_unsplit, atol=1e-4, rtol=1e-4)

    q_decode = torch.full((1, 1, 1, head_dim), 1e8, device=device, dtype=dtype)
    o_decode, _ = piper_attention_decode(q_decode, pool, c2s, nc, return_lse=True)
    torch.testing.assert_close(o_decode, expected[:, :, :1], atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_q", [1, 8, 9, 32, 63, 64])
def test_gpu_splitkv_reduce_tiled_and_multichunk_splits(head_dim: int, seq_q: int) -> None:
    """Exercise multi-chunk split-vectorized (seq_q<=8) and row-tiled (seq_q>8) Split-KV reduce."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    kv_heads = 2
    q_heads = 8
    num_slots = 4
    active_chunks = 16
    pool = _make_strided_pool(num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    k_frozen, v_frozen = _populate_test_pool(
        pool, list(range(num_slots)), device=device, dtype=dtype, seed=1700 + head_dim + seq_q
    )
    slot_keys = {i: (k_frozen[i], v_frozen[i]) for i in range(num_slots)}

    torch.manual_seed(1800 + head_dim + seq_q)
    q_backing = torch.randn((1, q_heads, seq_q, 2 * head_dim), device=device, dtype=dtype)
    q = q_backing[..., :head_dim]  # non-contiguous outer stride
    c2s = (
        torch.arange(active_chunks, device=device, dtype=torch.int32)
        .remainder(num_slots)
        .view(1, -1)
    )
    nc = torch.tensor([active_chunks], device=device, dtype=torch.int32)
    tk, tv, tl = _make_sentinel_tail(1, kv_heads, 128, head_dim, [95], device=device, dtype=dtype)

    out, lse = piper_attention_decode(
        q,
        pool,
        c2s,
        nc,
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        return_lse=True,
    )
    out_unsplit, lse_unsplit = piper_attention_decode(
        q,
        pool,
        c2s,
        nc,
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        active_chunks_hint=1,
        return_lse=True,
    )
    torch.testing.assert_close(out, out_unsplit, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, lse_unsplit, atol=2e-4, rtol=2e-4)

    # Also test frozen-only path (seq_q <= 8 hits >16 splits via allow_subchunk_single).
    out_f, lse_f = piper_attention_decode(q, pool, c2s, nc, return_lse=True)
    out_f_ref, lse_f_ref = piper_attention_decode(
        q, pool, c2s, nc, active_chunks_hint=1, return_lse=True
    )
    torch.testing.assert_close(out_f, out_f_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_f, lse_f_ref, atol=2e-4, rtol=2e-4)

    sdpa, sdpa_lse = _exact_fp32_attention(
        q,
        slot_keys,
        c2s,
        nc,
        num_slots,
        tail_key=tk,
        tail_value=tv,
        tail_len=tl,
        scale=head_dim**-0.5,
        is_causal=True,
    )
    _assert_prefill_quality(out, lse, sdpa, sdpa_lse)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("seq_q", [1, 9])
def test_gpu_splitkv_reduce_edge_semantics(seq_q: int) -> None:
    """Verify -inf empty, 1e8 shared bias, +inf/+inf equivalence, and per-row NaN isolation."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 64
    active_chunks = 16
    pool = _make_strided_pool(2, 1, head_dim, device=device, kv_dtype=torch.uint8)
    freeze_kv_chunks_(
        torch.full((1, 1, KV_CHUNK_TOKENS, head_dim), 1e8, device=device, dtype=dtype),
        torch.zeros((1, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype),
        pool,
        torch.tensor([0], device=device, dtype=torch.int32),
    )
    k_pm = torch.cat(
        [
            torch.full((1, 1, KV_CHUNK_TOKENS // 2, head_dim), 1.0, device=device, dtype=dtype),
            torch.full((1, 1, KV_CHUNK_TOKENS // 2, head_dim), -1.0, device=device, dtype=dtype),
        ],
        dim=2,
    )
    freeze_kv_chunks_(
        k_pm,
        torch.ones((1, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype),
        pool,
        torch.tensor([1], device=device, dtype=torch.int32),
    )
    pool.key_center[1].copy_(pool.key_center[0])
    c2s = torch.tensor([[0] * 8 + [1] * 8], device=device, dtype=torch.int32)
    nc = torch.tensor([active_chunks], device=device, dtype=torch.int32)

    q_nan = torch.ones((1, 1, seq_q, head_dim), device=device, dtype=dtype)
    if seq_q > 1:
        q_nan[:, :, -1].fill_(float("nan"))
    out_b, lse_b = piper_attention_decode(q_nan, pool, c2s, nc, return_lse=True)
    out_u, lse_u = piper_attention_decode(
        q_nan, pool, c2s, nc, active_chunks_hint=1, return_lse=True
    )
    if seq_q > 1:
        torch.testing.assert_close(out_b[:, :, :-1], out_u[:, :, :-1], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse_b[:, :, :-1], lse_u[:, :, :-1], atol=2e-4, rtol=2e-4)
        assert (out_b[:, :, :-1] > 0.99).all()
        assert torch.isnan(out_b[:, :, -1]).all()
        assert torch.isnan(lse_b[:, :, -1]).all()
    else:
        torch.testing.assert_close(out_b, out_u, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse_b, lse_u, atol=2e-4, rtol=2e-4)
        assert (out_b > 0.99).all()
        q_only_nan = torch.full((1, 1, 1, head_dim), float("nan"), device=device, dtype=dtype)
        o_n, l_n = piper_attention_decode(q_only_nan, pool, c2s, nc, return_lse=True)
        assert torch.isnan(o_n).all()
        assert torch.isnan(l_n).all()

    nc0 = torch.tensor([0], device=device, dtype=torch.int32)
    o0, l0 = piper_attention_decode(
        torch.ones((1, 1, seq_q, head_dim), device=device, dtype=dtype),
        pool,
        c2s,
        nc0,
        return_lse=True,
    )
    assert (o0 == 0.0).all()
    assert (l0 == -float("inf")).all()

    # Duplicate +inf biases across multi-chunk splits preserve +inf LSE semantics.
    inf_pool = _make_strided_pool(2, 1, head_dim, device=device, kv_dtype=torch.uint8)
    freeze_kv_chunks_(
        torch.full((1, 1, KV_CHUNK_TOKENS, head_dim), 1e34, device=device, dtype=dtype),
        torch.ones((1, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype),
        inf_pool,
        torch.tensor([0], device=device, dtype=torch.int32),
    )
    _populate_test_pool(inf_pool, [1], device=device, dtype=dtype, seed=1901)
    q_med = torch.full((1, 1, seq_q, head_dim), 1e6, device=device, dtype=dtype)
    c2s_inf = torch.tensor([[0, 1] * 8], device=device, dtype=torch.int32)
    o_inf, l_inf = piper_attention_decode(q_med, inf_pool, c2s_inf, nc, return_lse=True)
    torch.testing.assert_close(o_inf, torch.ones_like(o_inf), atol=1e-2, rtol=1e-2)
    assert (l_inf == float("inf")).all()
