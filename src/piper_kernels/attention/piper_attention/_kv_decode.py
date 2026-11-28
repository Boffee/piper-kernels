"""Chunk-major quantized KV cache decode and LSE merge custom operators."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from typing import Literal, overload

import torch

from piper_kernels._triton import runtime
from piper_kernels._validation import validate_positive_finite_float

from ._kv_overlap import _SymInt
from ._kv_pool import (
    _INT64_MAX,
    _SUPPORTED_KEY_DTYPES,
    _SUPPORTED_LSE_DTYPES,
    PiperKVChunkPool,
    _is_fake_or_compiling,
    _KVChunkConfig,
    _reject_autograd,
    _reject_neg_or_conj_views,
    _require_bool,
    _require_int,
    _require_tensor,
    _require_tensors,
    _resolve_checked_config,
    _validate_int32_1d,
    _validate_pool_tensors,
    _validate_strided_tensor,
)

_DECODE_AUTOGRAD_CHECKED: ContextVar[tuple[int, ...] | None] = ContextVar(
    "piper_kv_decode_autograd_checked", default=None
)


def _validate_lse_waves(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
) -> tuple[tuple[int, ...], torch.dtype, torch.device]:
    if (
        not isinstance(outputs, Sequence)
        or isinstance(outputs, (str, bytes, bytearray))
        or not isinstance(lses, Sequence)
        or isinstance(lses, (str, bytes, bytearray))
    ):
        raise TypeError("outputs and lses must be sequences of torch.Tensor")
    if len(outputs) == 0:
        raise ValueError("outputs and lses must not be empty")
    if len(outputs) != len(lses):
        raise ValueError(
            f"outputs and lses must have equal lengths, got {len(outputs)} and {len(lses)}"
        )

    ref_out = _require_tensor(outputs[0], name="outputs[0]")
    if ref_out.ndim < 1:
        raise ValueError(f"outputs[0] must have at least 1 dim, got {tuple(ref_out.shape)}")
    ref_shape = tuple(ref_out.shape)
    ref_lse_shape = ref_shape[:-1]
    ref_dtype = ref_out.dtype
    ref_device = ref_out.device

    for idx, (out, lse) in enumerate(zip(outputs, lses, strict=True)):
        checked_out = _require_tensor(out, name=f"outputs[{idx}]")
        checked_lse = _require_tensor(lse, name=f"lses[{idx}]")
        expected_out_dtype: torch.dtype | tuple[torch.dtype, ...] = (
            _SUPPORTED_LSE_DTYPES if idx == 0 else ref_dtype
        )
        for label, tensor, expected_shape, expected_dtype in (
            (f"outputs[{idx}]", checked_out, ref_shape, expected_out_dtype),
            (f"lses[{idx}]", checked_lse, ref_lse_shape, _SUPPORTED_LSE_DTYPES),
        ):
            if tensor.device != ref_device:
                raise ValueError(
                    f"outputs[{idx}] and lses[{idx}] must share device {ref_device}, "
                    f"got {checked_out.device} and {checked_lse.device}"
                )
            if tensor.layout is not torch.strided:
                raise ValueError(f"{label} must use strided layout")
            if (
                tensor.dtype not in expected_dtype
                if isinstance(expected_dtype, tuple)
                else tensor.dtype is not expected_dtype
            ):
                if isinstance(expected_dtype, tuple):
                    raise ValueError(
                        f"{label} must use a supported floating dtype "
                        f"(float16, bfloat16, float32, or float64), got {tensor.dtype}"
                    )
                raise ValueError(
                    f"{label} dtype ({tensor.dtype}) must match outputs[0] ({ref_dtype})"
                )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{label} shape {tuple(tensor.shape)} must match {expected_shape}")
    return ref_shape, ref_dtype, ref_device


def merge_attention_lse(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge multi-wave attention outputs and natural-log LSE values.

    Args:
        outputs: Non-empty sequence of strided attention outputs sharing shape
            ``[..., head_dim]`` (at least 1-D), device, and dtype in
            ``{float16, bfloat16, float32, float64}``.
        lses: Matching-length sequence of strided natural-log LSE tensors sharing
            shape ``outputs[0].shape[:-1]`` and device, each independently using a dtype
            in ``{float16, bfloat16, float32, float64}``.

    Returns:
        ``(merged_out, merged_lse)`` with ``merged_out`` cast back to
        ``outputs[0].dtype`` and ``merged_lse`` in ``float64`` if any input uses
        ``float64``, otherwise ``float32``. Accumulates under disabled autocast with
        exact ``-inf`` wave masking and ``NaN`` propagation (``+inf`` LSE yields
        ``NaN``), supporting differentiable autograd, eager execution, and
        ``torch.compile(fullgraph=True)``.
    """
    _, ref_dtype, ref_device = _validate_lse_waves(outputs, lses)
    acc_dtype = (
        torch.float64
        if ref_dtype is torch.float64 or any(lse.dtype is torch.float64 for lse in lses)
        else torch.float32
    )
    autocast_ctx = (
        torch.amp.autocast(device_type=ref_device.type, enabled=False)
        if torch.amp.is_autocast_available(ref_device.type)
        else nullcontext()
    )
    with autocast_ctx:
        out_stack = torch.stack([out.to(acc_dtype) for out in outputs], dim=0)
        lse_stack = torch.stack([lse.to(acc_dtype) for lse in lses], dim=0)
        valid = lse_stack != -float("inf")
        has_nan = torch.isnan(lse_stack).any(dim=0)
        max_lse = torch.where(has_nan, float("nan"), lse_stack.amax(dim=0))
        has_any = max_lse != -float("inf")
        safe_max = torch.where(has_any, max_lse, 0.0)
        weights = torch.where(
            valid,
            torch.exp(torch.where(valid, lse_stack - safe_max.unsqueeze(0), 0.0)),
            0.0,
        )
        safe_out = torch.where(valid.unsqueeze(-1), out_stack, 0.0)
        denom = weights.sum(dim=0)
        safe_denom = torch.where(has_any, denom, 1.0)
        norm_weights = torch.where(
            has_any.unsqueeze(0),
            weights / safe_denom.unsqueeze(0),
            0.0,
        )
        merged_out = torch.where(
            has_any.unsqueeze(-1),
            (norm_weights.unsqueeze(-1) * safe_out).sum(dim=0),
            0.0,
        )
        merged_lse = torch.where(
            has_any,
            safe_max + torch.log(safe_denom),
            -float("inf"),
        )
        return merged_out.to(ref_dtype), merged_lse


def _validate_decode_query(
    query: torch.Tensor,
    *,
    device: torch.device,
    kv_heads: _SymInt,
    head_dim: _SymInt,
) -> _SymInt:
    _validate_strided_tensor(
        query,
        name="query",
        device=device,
        dtype=_SUPPORTED_KEY_DTYPES,
        dtype_label="float16 or bfloat16",
        ndim=4,
        ndim_desc="4 dims [batch, q_heads, seq_q, head_dim]",
        last_dim_contiguous=True,
    )

    batch, q_heads, seq_q, q_head_dim = query.shape
    if batch < 0:
        raise ValueError(f"query batch must be >= 0, got {batch}")
    if q_heads < 1 or q_heads % kv_heads != 0:
        raise ValueError(
            f"query heads ({q_heads}) must be positive and divisible by kv_heads ({kv_heads})"
        )
    if seq_q < 1:
        raise ValueError(f"query sequence length must be >= 1, got {seq_q}")
    if q_head_dim != head_dim:
        raise ValueError(f"query head_dim ({q_head_dim}) must match pool head_dim ({head_dim})")
    return batch


def _validate_chunk_to_slot(
    chunk_to_slot: torch.Tensor | None,
    *,
    device: torch.device,
    batch: _SymInt,
) -> None:
    if chunk_to_slot is None:
        return
    _validate_strided_tensor(
        chunk_to_slot,
        name="chunk_to_slot",
        device=device,
        dtype=torch.int32,
        dtype_label="int32",
    )
    if chunk_to_slot.ndim != 2 or chunk_to_slot.shape[0] != batch:
        raise ValueError(
            f"chunk_to_slot must have shape [{batch}, max_chunks], got {tuple(chunk_to_slot.shape)}"
        )


def _validate_decode_tail_kv(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    batch: _SymInt,
    kv_heads: _SymInt,
    head_dim: _SymInt,
) -> None:
    _validate_strided_tensor(
        tensor,
        name=name,
        device=device,
        dtype=dtype,
        dtype_label=f"query dtype ({dtype})",
        ndim=4,
        ndim_desc="4 dims [batch, kv_heads, tail_cap, head_dim]",
        last_dim_contiguous=True,
    )
    if tensor.shape[0] != batch or tensor.shape[1] != kv_heads or tensor.shape[3] != head_dim:
        raise ValueError(
            f"{name} must have batch={batch}, kv_heads={kv_heads}, head_dim={head_dim}, "
            f"got {tuple(tensor.shape)}"
        )


def _validate_decode_tail(
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch: _SymInt,
    kv_heads: _SymInt,
    head_dim: _SymInt,
) -> None:
    present = (tail_key is not None, tail_value is not None, tail_len is not None)
    if any(present) and not all(present):
        raise ValueError(
            "tail_key, tail_value, and tail_len must all be provided together or all omitted"
        )
    if not all(present):
        return
    assert tail_key is not None
    assert tail_value is not None
    assert tail_len is not None

    for name, tensor in (("tail_key", tail_key), ("tail_value", tail_value)):
        _validate_decode_tail_kv(
            tensor,
            name=name,
            device=device,
            dtype=dtype,
            batch=batch,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
    if tail_value.shape[2] != tail_key.shape[2]:
        raise ValueError(
            f"tail_value length ({tail_value.shape[2]}) must match tail_key ({tail_key.shape[2]})"
        )
    _require_tensor(tail_len, name="tail_len")
    _validate_int32_1d(tail_len, name="tail_len", device=device, length=batch)


def _validate_decode_inputs(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float | None,
    is_causal: bool,
    active_chunks_hint: _SymInt,
    *,
    check_target: bool,
) -> tuple[_KVChunkConfig | None, float]:
    _, kv_heads, head_dim, key_scales = _validate_pool_tensors(
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        check_disjoint=False,
    )
    device = key_int8.device
    batch = _validate_decode_query(query, device=device, kv_heads=kv_heads, head_dim=head_dim)

    _require_tensor(num_chunks, name="num_chunks")
    _validate_int32_1d(num_chunks, name="num_chunks", device=device, length=batch)
    _validate_chunk_to_slot(chunk_to_slot, device=device, batch=batch)
    _validate_decode_tail(
        tail_key,
        tail_value,
        tail_len,
        device=device,
        dtype=query.dtype,
        batch=batch,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )

    resolved_scale = (
        float(query.shape[-1]) ** -0.5
        if scale is None
        else validate_positive_finite_float(
            scale,
            "piper_attention_decode scale must be finite and positive",
        )
    )
    _require_bool(is_causal, name="is_causal")
    _require_int(
        active_chunks_hint,
        name="active_chunks_hint",
        minimum=0,
        maximum=_INT64_MAX,
        allow_symint=True,
    )

    config = _resolve_checked_config(device, key_scales) if check_target else None
    return config, resolved_scale


@torch.library.custom_op("piper_kernels::piper_attention_decode", mutates_args=())
def _piper_attention_decode_op(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float | None = None,
    is_causal: bool = True,
    active_chunks_hint: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_tensors = (
        query,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        chunk_to_slot,
        num_chunks,
        tail_key,
        tail_value,
        tail_len,
    )
    non_null_tensors = tuple(t for t in all_tensors if t is not None)
    checked_ids = _DECODE_AUTOGRAD_CHECKED.get()
    if checked_ids is not None:
        _DECODE_AUTOGRAD_CHECKED.set(None)
    if checked_ids != tuple(id(t) for t in non_null_tensors):
        _reject_autograd(*non_null_tensors, operator="piper_attention_decode")
    config, resolved_scale = _validate_decode_inputs(
        query,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        chunk_to_slot,
        num_chunks,
        tail_key,
        tail_value,
        tail_len,
        scale,
        is_causal,
        active_chunks_hint,
        check_target=not _is_fake_or_compiling(non_null_tensors),
    )
    if query.device.type == "meta" or query.shape[0] == 0:
        return (
            torch.empty_like(query, memory_format=torch.contiguous_format),
            torch.empty(query.shape[:3], device=query.device, dtype=torch.float32),
        )

    assert config is not None
    if runtime.supports_device(query.device):
        if config.prefill_backend is not None:
            from . import _kv_prefill  # noqa: PLC0415 - GPU-only Triton dependency

            return _kv_prefill.run_chunked_prefill(
                query,
                key_int8,
                key_scale,
                key_center,
                value_int8,
                value_scale,
                value_mean,
                chunk_to_slot,
                num_chunks,
                tail_key=tail_key,
                tail_value=tail_value,
                tail_len=tail_len,
                scale=resolved_scale,
                is_causal=is_causal,
                grouped_qk=config.grouped_qk,
                pack_wmma=config.pack_wmma,
                active_chunks_hint=active_chunks_hint,
                prefill_backend=config.prefill_backend,
                target=config.target,
            )
        from . import triton as _decode_triton  # noqa: PLC0415 - GPU-only Triton dependency

        return _decode_triton.piper_attention_decode_triton(
            query,
            key_int8,
            key_scale,
            key_center,
            value_int8,
            value_scale,
            value_mean,
            chunk_to_slot,
            num_chunks,
            tail_key=tail_key,
            tail_value=tail_value,
            tail_len=tail_len,
            scale=resolved_scale,
            is_causal=is_causal,
            grouped_qk=config.grouped_qk,
            pack_wmma=config.pack_wmma,
            active_chunks_hint=active_chunks_hint,
            target=config.target,
        )

    from .reference import piper_attention_decode_reference  # noqa: PLC0415 - lazy CPU fallback

    return piper_attention_decode_reference(
        query,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        chunk_to_slot,
        num_chunks,
        tail_key=tail_key,
        tail_value=tail_value,
        tail_len=tail_len,
        scale=resolved_scale,
        is_causal=is_causal,
        pack_wmma=config.pack_wmma,
    )


@_piper_attention_decode_op.register_fake
def _piper_attention_decode_fake(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float | None = None,
    is_causal: bool = True,
    active_chunks_hint: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_tensors = (
        query,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        chunk_to_slot,
        num_chunks,
        tail_key,
        tail_value,
        tail_len,
    )
    non_null_tensors = tuple(t for t in all_tensors if t is not None)
    _reject_autograd(*non_null_tensors, operator="piper_attention_decode")
    _validate_decode_inputs(
        query,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        chunk_to_slot,
        num_chunks,
        tail_key,
        tail_value,
        tail_len,
        scale,
        is_causal,
        active_chunks_hint,
        check_target=False,
    )
    return (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty(query.shape[:3], device=query.device, dtype=torch.float32),
    )


@torch.library.impl("piper_kernels::piper_attention_decode", "Autograd")
def _piper_attention_decode_autograd(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    tail_key: torch.Tensor | None,
    tail_value: torch.Tensor | None,
    tail_len: torch.Tensor | None,
    scale: float | None = None,
    is_causal: bool = True,
    active_chunks_hint: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_null_tensors = tuple(
        t
        for t in (
            query,
            key_int8,
            key_scale,
            key_center,
            value_int8,
            value_scale,
            value_mean,
            chunk_to_slot,
            num_chunks,
            tail_key,
            tail_value,
            tail_len,
        )
        if t is not None
    )
    _reject_autograd(*non_null_tensors, operator="piper_attention_decode")
    token = _DECODE_AUTOGRAD_CHECKED.set(tuple(id(t) for t in non_null_tensors))
    try:
        with torch._C._AutoDispatchBelowAutograd():
            return torch.ops.piper_kernels.piper_attention_decode.default(
                query,
                key_int8,
                key_scale,
                key_center,
                value_int8,
                value_scale,
                value_mean,
                chunk_to_slot,
                num_chunks,
                tail_key,
                tail_value,
                tail_len,
                scale,
                is_causal,
                active_chunks_hint,
            )
    finally:
        _DECODE_AUTOGRAD_CHECKED.reset(token)


for _view_key in ("Negative", "Conjugate"):
    torch.library.impl("piper_kernels::piper_attention_decode", _view_key)(
        _reject_neg_or_conj_views
    )


@overload
def piper_attention_decode(
    query: torch.Tensor,
    pool: PiperKVChunkPool,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    *,
    tail_key: torch.Tensor | None = None,
    tail_value: torch.Tensor | None = None,
    tail_len: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = True,
    return_lse: Literal[False] = False,
    active_chunks_hint: int = 0,
) -> torch.Tensor: ...


@overload
def piper_attention_decode(
    query: torch.Tensor,
    pool: PiperKVChunkPool,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    *,
    tail_key: torch.Tensor | None = None,
    tail_value: torch.Tensor | None = None,
    tail_len: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = True,
    return_lse: Literal[True],
    active_chunks_hint: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def piper_attention_decode(
    query: torch.Tensor,
    pool: PiperKVChunkPool,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    *,
    tail_key: torch.Tensor | None = None,
    tail_value: torch.Tensor | None = None,
    tail_len: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = True,
    return_lse: bool = False,
    active_chunks_hint: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def piper_attention_decode(  # noqa: PLR0913
    query: torch.Tensor,
    pool: PiperKVChunkPool,
    chunk_to_slot: torch.Tensor | None,
    num_chunks: torch.Tensor,
    *,
    tail_key: torch.Tensor | None = None,
    tail_value: torch.Tensor | None = None,
    tail_len: torch.Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = True,
    return_lse: bool = False,
    active_chunks_hint: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Attend grouped queries against frozen K1024/V1024 ``pool`` chunks and a live tail.

    Args:
        query: Query states of shape ``[batch, q_heads, seq_q, head_dim]`` in
            ``float16`` or ``bfloat16`` on ``pool.device`` (``stride(-1) == 1``,
            ``seq_q >= 1``, ``q_heads`` a positive multiple of ``pool.kv_heads``).
        pool: Persistent ``PiperKVChunkPool`` holding finalized 1024-token chunks.
        chunk_to_slot: Optional 2-D ``int32`` mapping ``[batch, max_chunks]``
            from logical chunk index to physical ``pool`` slot; entries ``< 0`` or
            ``>= pool.num_slots`` are skipped. When ``None``, logical chunks map
            directly to slots ``0 .. pool.num_slots - 1``.
        num_chunks: Contiguous 1-D ``int32`` active chunk counts ``[batch]``
            clamped to ``[0, max_chunks]``.
        tail_key: Optional live tail keys ``[batch, kv_heads, tail_cap, head_dim]``
            matching ``query.dtype`` and ``pool.device`` (must be supplied alongside
            ``tail_value`` and ``tail_len``).
        tail_value: Optional live tail values matching ``tail_key`` shape and dtype.
        tail_len: Optional contiguous 1-D ``int32`` active tail lengths ``[batch]``
            clamped to ``[0, tail_cap]``.
        scale: Positive finite softmax scale; defaults to ``head_dim**-0.5``.
        is_causal: Defaults to ``True``. For causal decode, speculative
            verification (``seq_q < 64``), and chunked prefill (``seq_q >= 64``),
            frozen chunks remain visible prior to the live tail while tail attention
            enforces ``token_idx <= active_tail - seq_q + q_pos`` (supporting
            ``seq_q == 1``, non-64-aligned offsets, and ``seq_q > active_tail``).
            When ``False``, all active tail tokens up to ``tail_len`` are visible
            to every query token; ignored when no tail is provided.
        return_lse: If ``True``, return ``(output, lse)`` with ``float32`` natural-log
            ``lse`` of shape ``[batch, q_heads, seq_q]`` (``-inf`` where a query sees
            zero valid KV tokens while ``output`` is ``0.0``).
        active_chunks_hint: Non-negative upper-bound hint on active chunks per
            sequence used only to choose Split-KV launch parallelism (``0`` uses
            ``max_chunks``).

    Backends & graphs:
        Shares one operator API, zero host readbacks,
        ``torch.compile(fullgraph=True, dynamic=True)``, and CUDA Graph capture
        across paths:
        - On NVIDIA SM8x/SM12x and AMD RDNA4 (``gfx1200``/``gfx1201``), finalized
          ``pool`` slots always execute native GQA-packed INT8-Q/UINT8-P
          tensor-core attention across all ``seq_q >= 1``:
          - For ``seq_q < 64`` (decode and MTP verification), queries are packed by
            KV head with exact per-row quantization over ``pool`` and combined in a
            single Split-KV reduction with the direct BF16/FP16 live tail.
          - For ``seq_q >= 64`` (chunked prefill), frozen pool slots pair with a
            K1024/V1024 chunk-centered quantized live tail and normalized FP32
            two-wave LSE merging.
        - On CPU or fallback GPUs without native quantized tensor-core Piper
          support (e.g. SM75/SM90/SM100, ``gfx942``, ``gfx1100``, ``gfx1151``),
          ``piper_attention_decode`` keeps queries and softmax weights in FP32
          while dequantizing pool slots, running native Triton Split-KV kernels on
          GPU and the portable FP32-accumulation reference elsewhere.
        Numerical contracts: on SM8x/SM12x and RDNA4, native INT8-Q/UINT8-P
        frozen pool execution approximates FP32 SDPA within INT8/UINT8 Piper
        quantization error across all ``seq_q >= 1`` rather than matching
        FP32-Q/P Split-KV bitwise; at ``seq_q >= 64``, K1024/V1024 tail centering
        also shares local chunk statistics across causal queries within each
        1024-token tail chunk. Callers should keep rotated dot products
        ``q · k_center``, ``query``, active ``tail_key``, and active value
        buffers finite.
    """
    if not isinstance(pool, PiperKVChunkPool):
        raise TypeError(f"pool must be a PiperKVChunkPool, got {type(pool).__name__}")
    _require_bool(return_lse, name="return_lse")
    _require_bool(is_causal, name="is_causal")
    _require_int(
        active_chunks_hint,
        name="active_chunks_hint",
        minimum=0,
        maximum=_INT64_MAX,
        allow_symint=True,
    )
    _require_tensors((("query", query), ("num_chunks", num_chunks)))
    _require_tensors(
        (
            ("chunk_to_slot", chunk_to_slot),
            ("tail_key", tail_key),
            ("tail_value", tail_value),
            ("tail_len", tail_len),
        ),
        allow_none=True,
    )

    resolved_scale: float | None = (
        None
        if scale is None
        else validate_positive_finite_float(
            scale,
            "piper_attention_decode scale must be finite and positive",
        )
    )

    output, lse = torch.ops.piper_kernels.piper_attention_decode.default(
        query,
        pool.key_int8,
        pool.key_scale,
        pool.key_center,
        pool.value_int8,
        pool.value_scale,
        pool.value_mean,
        chunk_to_slot,
        num_chunks,
        tail_key,
        tail_value,
        tail_len,
        resolved_scale,
        is_causal,
        active_chunks_hint,
    )
    if return_lse:
        return output, lse
    return output
