"""Global chunk-major quantized KV cache storage, freeze, decode, and LSE merge operators."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

import torch

from piper_kernels._triton import runtime

from ._kv_decode import merge_attention_lse, piper_attention_decode
from ._kv_overlap import _input_overlaps_pool_tensor, _SymInt
from ._kv_pool import (
    _POOL_FIELD_NAMES,
    _SUPPORTED_KEY_DTYPES,
    KV_CHUNK_TOKENS,
    PiperKVChunkPool,
    _is_fake_or_compiling,
    _KVChunkConfig,
    _reject_autograd,
    _reject_neg_or_conj_views,
    _require_tensors,
    _resolve_checked_config,
    _validate_int32_1d,
    _validate_pool_tensors,
    _validate_strided_tensor,
    kv_chunk_storage_specs,
)

_VALIDATED_FREEZE: ContextVar[tuple[tuple[int, ...], _KVChunkConfig] | None] = ContextVar(
    "piper_kv_freeze_validated_state", default=None
)


@contextlib.contextmanager
def _prevalidated_freeze(
    tensors: tuple[torch.Tensor, ...],
    config: _KVChunkConfig | None,
) -> Iterator[None]:
    if config is None:
        yield
        return
    token = _VALIDATED_FREEZE.set((tuple(id(tensor) for tensor in tensors), config))
    try:
        yield
    finally:
        _VALIDATED_FREEZE.reset(token)


def _validate_freeze_kv_input(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    kv_heads: _SymInt,
    head_dim: _SymInt,
    num_new: _SymInt,
) -> None:
    _validate_strided_tensor(
        tensor,
        name=name,
        device=device,
        dtype=_SUPPORTED_KEY_DTYPES,
        dtype_label="float16 or bfloat16",
        ndim=4,
        ndim_desc=f"shape [num_new, kv_heads, {KV_CHUNK_TOKENS}, head_dim]",
        last_dim_contiguous=True,
    )
    expected_shape = (num_new, kv_heads, KV_CHUNK_TOKENS, head_dim)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")


def _validate_freeze_before_mutating(
    key: torch.Tensor,
    value: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    check_target: bool,
) -> _KVChunkConfig | None:
    _, kv_heads, head_dim, key_scales = _validate_pool_tensors(
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
    )
    pool_tensors = (key_int8, key_scale, key_center, value_int8, value_scale, value_mean)
    if not torch.is_inference_mode_enabled():
        for pool_name, pool_tensor in zip(_POOL_FIELD_NAMES, pool_tensors, strict=True):
            if pool_tensor.is_inference():
                raise RuntimeError(
                    f"Inplace update to inference tensor {pool_name} outside InferenceMode "
                    "is not allowed."
                )

    _require_tensors((("key", key), ("value", value), ("slot_ids", slot_ids)))
    if key.ndim != 4:
        raise ValueError(
            f"key must have shape [num_new, kv_heads, {KV_CHUNK_TOKENS}, head_dim], "
            f"got {tuple(key.shape)}"
        )

    device = key_int8.device
    num_new = key.shape[0]
    _validate_freeze_kv_input(
        key, name="key", device=device, kv_heads=kv_heads, head_dim=head_dim, num_new=num_new
    )
    _validate_freeze_kv_input(
        value, name="value", device=device, kv_heads=kv_heads, head_dim=head_dim, num_new=num_new
    )
    if value.dtype is not key.dtype:
        raise ValueError(f"key ({key.dtype}) and value ({value.dtype}) must share a dtype")
    _validate_int32_1d(slot_ids, name="slot_ids", device=device, length=num_new)

    if not torch.compiler.is_compiling():
        for read_name, read_tensor in (("key", key), ("value", value), ("slot_ids", slot_ids)):
            for pool_name, pool_tensor in zip(_POOL_FIELD_NAMES, pool_tensors, strict=True):
                if _input_overlaps_pool_tensor(read_tensor, pool_tensor):
                    raise ValueError(f"{read_name} must not overlap {pool_name} in storage")

    return _resolve_checked_config(device, key_scales) if check_target else None


@torch.library.custom_op(
    "piper_kernels::piper_kv_freeze_",
    mutates_args=(
        "key_int8",
        "key_scale",
        "key_center",
        "value_int8",
        "value_scale",
        "value_mean",
    ),
)
def _piper_kv_freeze_op(
    key: torch.Tensor,
    value: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    slot_ids: torch.Tensor,
) -> None:
    tensors = (
        key,
        value,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        slot_ids,
    )
    cached = _VALIDATED_FREEZE.get()
    if cached is not None:
        _VALIDATED_FREEZE.set(None)
    if cached is not None and cached[0] == tuple(id(tensor) for tensor in tensors):
        config = cached[1]
    else:
        _reject_autograd(*tensors)
        config = _validate_freeze_before_mutating(
            *tensors,
            check_target=not _is_fake_or_compiling(tensors),
        )
    if key.device.type == "meta" or key.shape[0] == 0 or key_int8.shape[0] == 0:
        return

    assert config is not None
    if runtime.supports_device(key.device):
        from . import _quantization  # noqa: PLC0415 - GPU-only Triton dependency

        _quantization.freeze_kv_chunks_triton(
            key,
            value,
            slot_ids,
            key_int8.view(torch.int8),
            key_scale,
            key_center,
            value_int8.view(torch.int8),
            value_scale,
            value_mean,
            grouped_qk=config.grouped_qk,
            pack_wmma=config.pack_wmma,
        )
        return

    from .reference import freeze_kv_chunks_reference  # noqa: PLC0415 - lazy CPU fallback

    freeze_kv_chunks_reference(
        key,
        value,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        slot_ids,
        pack_wmma=config.pack_wmma,
    )


@_piper_kv_freeze_op.register_fake
def _piper_kv_freeze_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    slot_ids: torch.Tensor,
) -> None:
    tensors = (
        key,
        value,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        slot_ids,
    )
    _reject_autograd(*tensors)
    _validate_freeze_before_mutating(*tensors, check_target=False)


@torch.library.impl("piper_kernels::piper_kv_freeze_", "Autograd")
def _piper_kv_freeze_autograd(
    key: torch.Tensor,
    value: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_center: torch.Tensor,
    value_int8: torch.Tensor,
    value_scale: torch.Tensor,
    value_mean: torch.Tensor,
    slot_ids: torch.Tensor,
) -> None:
    tensors = (
        key,
        value,
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
        slot_ids,
    )
    _reject_autograd(*tensors)
    # Validate eagerly above ADInplaceOrView so rejected calls leave pool versions intact.
    config = (
        None
        if _is_fake_or_compiling(tensors)
        else _validate_freeze_before_mutating(*tensors, check_target=True)
    )
    with _prevalidated_freeze(tensors, config), torch._C._AutoDispatchBelowAutograd():
        torch.ops.piper_kernels.piper_kv_freeze_.default(*tensors)


def freeze_kv_chunks_(
    key: torch.Tensor,
    value: torch.Tensor,
    pool: PiperKVChunkPool,
    slot_ids: torch.Tensor,
) -> None:
    """Quantize and scatter finalized 1024-token KV chunks into persistent ``pool`` slots.

    Args:
        key: Finalized unrotated keys with shape ``[num_new, kv_heads, 1024, head_dim]``
            in ``float16`` or ``bfloat16`` on ``pool.device`` (``stride(-1) == 1``;
            outer dims may be strided; must not alias ``pool`` storage).
        value: Finalized values matching ``key`` in shape, dtype, device, and layout.
        pool: Destination ``PiperKVChunkPool``. Mutated in place only at valid ``slot_ids``;
            if any pool tensor was created in ``torch.inference_mode()``, callers must
            invoke inside ``torch.inference_mode()``.
        slot_ids: Contiguous 1-D ``int32`` destination indices of shape ``[num_new]``
            on ``pool.device``. Entries ``< 0`` or ``>= pool.num_slots`` are skipped
            with zero host readbacks; valid entries in a single call must be distinct.

    Backends & graphs:
        Executes native Triton K1024/V1024 quantization kernels on supported GPU targets
        and the portable PyTorch reference on CPU or fallback devices. Supports
        ``torch.compile(..., fullgraph=True, dynamic=True)`` and CUDA Graph capture;
        inference-only (rejects tensors requiring gradients when grad is enabled and
        forward-AD dual tensors).
    """
    if not isinstance(pool, PiperKVChunkPool):
        raise TypeError(f"pool must be a PiperKVChunkPool, got {type(pool).__name__}")
    _require_tensors((("key", key), ("value", value), ("slot_ids", slot_ids)))
    tensors = (
        key,
        value,
        pool.key_int8,
        pool.key_scale,
        pool.key_center,
        pool.value_int8,
        pool.value_scale,
        pool.value_mean,
        slot_ids,
    )
    # InferenceMode skips the Autograd key; validate concrete calls before ADInplaceOrView.
    if not _is_fake_or_compiling(tensors) and torch.is_inference_mode_enabled():
        config = _validate_freeze_before_mutating(*tensors, check_target=True)
        with _prevalidated_freeze(tensors, config):
            torch.ops.piper_kernels.piper_kv_freeze_.default(*tensors)
        return
    torch.ops.piper_kernels.piper_kv_freeze_.default(*tensors)


for _view_key in ("Negative", "Conjugate"):
    torch.library.impl("piper_kernels::piper_kv_freeze_", _view_key)(_reject_neg_or_conj_views)


__all__ = [
    "KV_CHUNK_TOKENS",
    "PiperKVChunkPool",
    "freeze_kv_chunks_",
    "kv_chunk_storage_specs",
    "merge_attention_lse",
    "piper_attention_decode",
]
