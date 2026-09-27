"""Dense Piper boundaries for quantized Q and optional quantized K/V producers."""

import torch

from piper_kernels._triton import mixed_int8, reductions, runtime, targets
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.piper._amd import _wmma, fragments
from piper_kernels.attention.kernels.qk_quantization.int8.sage import _rotation as qk_rotation
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_quantization

from . import _quantization, _validation
from ._amd import gluon as amd_backend
from ._amd import policy as amd_policy
from ._amd import triton as amd_preparation
from ._nvidia import policy as nvidia_policy
from ._nvidia import triton as nvidia_backend


def source_files() -> tuple[str, ...]:
    """Include preparation, dispatch, and launch semantics in compiler cache keys."""
    modules = (
        mixed_int8,
        reductions,
        runtime,
        targets,
        _wmma,
        fragments,
        qk_rotation,
        qk_quantization,
        _quantization,
        _validation,
        amd_backend,
        amd_policy,
        amd_preparation,
        nvidia_policy,
        nvidia_backend,
    )
    return (__file__, *(module.__file__ for module in modules if module.__file__ is not None))


def supports_quantized_query(target: AcceleratorTarget) -> bool:
    """Require native dense attention with Q32/K64 quantization groups."""
    return target.is_cuda_capability(12) or amd_policy.supports_target(target)


def _validate_query(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_length: int,
    is_causal: bool,
) -> tuple[int, int, int, int]:
    """Check the shared Q64 storage/Q32 scale contract using host metadata."""
    if isinstance(query_length, bool) or not isinstance(query_length, (int, torch.SymInt)):
        raise TypeError("quantized Piper query_length must be an integer")
    if query_length < 1:
        raise ValueError("quantized Piper query_length must be positive")
    if not isinstance(is_causal, bool):
        raise TypeError("quantized Piper is_causal must be a boolean")
    if query.ndim != 4 or query.dtype is not torch.int8:
        raise ValueError("quantized Piper query must be [batch,heads,storage,head_dim] INT8")
    if query.layout is not torch.strided or not query.is_contiguous():
        raise ValueError("quantized Piper query must use contiguous strided storage")
    batch, heads, storage_length, head_dim = query.shape
    if heads < 1 or head_dim not in (64, 128):
        raise ValueError("quantized Piper query requires positive heads and head_dim 64 or 128")
    if storage_length != (query_length + 63) // 64 * 64:
        raise ValueError("quantized Piper query storage must pad its logical length to Q64")
    if (
        query_scale.shape != (batch, heads, storage_length // 32)
        or query_scale.dtype is not torch.float32
        or query_scale.layout is not torch.strided
        or not query_scale.is_contiguous()
    ):
        raise ValueError("quantized Piper query_scale must be contiguous FP32 Q32 scales")
    return batch, heads, query_length, head_dim


def _validate_quantized_query(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_length: int,
    is_causal: bool,
) -> tuple[int, int, int, int]:
    """Validate storage metadata without inspecting codes or numerical scales."""
    batch, heads, _, head_dim = _validate_query(query, query_scale, query_length, is_causal)
    for name, tensor in (("key", key), ("value", value)):
        if tensor.ndim != 4 or tensor.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"quantized Piper {name} must be [batch,heads,sequence,D] FP16/BF16")
        if tensor.layout is not torch.strided or tensor.stride(-1) != 1:
            raise ValueError(f"quantized Piper {name} must have contiguous head features")
    if key.dtype is not value.dtype or key.shape != value.shape:
        raise ValueError("quantized Piper key/value must have matching shapes and dtypes")
    if (
        key.shape[0] != batch
        or key.shape[3] != head_dim
        or key.shape[1] < 1
        or heads % key.shape[1]
        or key.shape[2] < 1
    ):
        raise ValueError(
            "quantized Piper requires matching batches, head dimensions, and GQA heads"
        )
    if is_causal and key.shape[2] != query_length:
        raise ValueError("causal quantized Piper requires equal logical query and key lengths")
    if any(tensor.device != query.device for tensor in (query_scale, key, value)):
        raise ValueError("quantized Piper operands must share a device")
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query_scale, key, value)
    ):
        raise RuntimeError("quantized Piper Attention is an inference-only operator")
    return batch, heads, query_length, head_dim


@torch.library.custom_op("piper_kernels::piper_attention_from_quantized_query", mutates_args=())
def _piper_attention_from_quantized_query_op(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_length: int,
    is_causal: bool,
) -> torch.Tensor:
    """Consume Q32 INT8 queries while retaining dense K/V preparation.

    Q and its FP32 scale groups have Q64-padded contiguous storage. Scales
    already include the attention scale and log2(e); padded rows and scale
    groups must be initialized by the producer. Numerical codes/scales are
    caller preconditions. K is centered using its global post-transform mean;
    V retains per-token scaling and causal centering rules.
    """
    shape = _validate_quantized_query(query, query_scale, key, value, query_length, is_causal)
    if shape[0] == 0:
        return key.new_empty(shape)
    target = AcceleratorTarget.from_device(query.device)
    if not supports_quantized_query(target):
        raise RuntimeError(f"quantized-Q Piper Attention is unavailable on {query.device}")
    if target.is_nvidia_cuda:
        plan = nvidia_policy.select_execution_plan(
            target,
            head_dim=shape[3],
            is_causal=is_causal,
            query_length=query_length,
            key_length=key.shape[2],
        )
        context = nvidia_backend._prepare_piper_context(
            key, value, is_causal=is_causal, execution_plan=plan
        )
        descriptor = (
            nvidia_backend._make_query_descriptor(query, plan.block_m)
            if plan.use_tensor_descriptors and plan.block_m == 128
            else None
        )
        prepared_query = nvidia_backend._PreparedPiperQuery(
            data=query,
            scale=query_scale,
            descriptor=descriptor,
            shape=shape,
            dtype=key.dtype,
        )
        return nvidia_backend._launch_piper_attention_into(
            context, prepared_query, key.new_empty(shape)
        )
    amd_context = amd_backend.prepare_context(key, value, is_causal=is_causal)
    amd_query = amd_backend.PreparedQuery(
        data=query, scale=query_scale, shape=shape, dtype=key.dtype
    )
    return amd_backend.launch_attention_into(amd_context, amd_query, key.new_empty(shape))


@_piper_attention_from_quantized_query_op.register_fake
def _piper_attention_from_quantized_query_fake(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_length: int,
    is_causal: bool,
) -> torch.Tensor:
    shape = _validate_quantized_query(query, query_scale, key, value, query_length, is_causal)
    return key.new_empty(shape)


def _validate_quantized(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    multiplier: torch.Tensor,
    log_scale: torch.Tensor,
    value_mean: torch.Tensor,
    query_length: int,
    key_length: int,
    is_causal: bool,
    output_dtype: torch.dtype,
) -> tuple[int, int, int, int]:
    shape = _validate_query(query, query_scale, query_length, is_causal)
    if query_scale.device != query.device:
        raise ValueError("quantized Piper operands must share a device")
    if torch.is_grad_enabled() and query_scale.requires_grad:
        raise RuntimeError("quantized Piper Attention is an inference-only operator")
    validate_quantized_context(
        key,
        key_scale,
        value,
        multiplier,
        log_scale,
        value_mean,
        query_shape=shape,
        query_device=query.device,
        key_length=key_length,
        is_causal=is_causal,
        output_dtype=output_dtype,
    )
    return shape


def validate_quantized_context(  # noqa: PLR0913
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    multiplier: torch.Tensor,
    log_scale: torch.Tensor,
    value_mean: torch.Tensor,
    *,
    query_shape: tuple[int, int, int, int],
    query_device: torch.device,
    key_length: int,
    is_causal: bool,
    output_dtype: torch.dtype,
) -> None:
    """Check native K/V metadata without allocating a query tensor."""
    batch, heads, query_length, head_dim = query_shape
    if not isinstance(is_causal, bool):
        raise TypeError("quantized Piper is_causal must be a boolean")
    if isinstance(key_length, bool) or not isinstance(key_length, (int, torch.SymInt)):
        raise TypeError("quantized Piper key_length must be an integer")
    if key_length < 1 or (is_causal and key_length != query_length):
        raise ValueError("quantized Piper requires positive key length and square causal attention")
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("quantized Piper output must use FP16/BF16")
    storage = (key_length + 63) // 64 * 64
    if key.ndim != 4 or key.shape[1] < 1 or heads % key.shape[1]:
        raise ValueError("quantized Piper key must have compatible GQA heads")
    kv_heads = key.shape[1]
    expected = (
        (key, (batch, kv_heads, storage, head_dim), torch.int8),
        (key_scale, (batch, kv_heads, storage // 64), torch.float32),
        (value, (batch, kv_heads, head_dim, storage), torch.int8),
        (multiplier, (batch, kv_heads, storage), torch.float32),
        (log_scale, (batch, kv_heads, storage), torch.float32),
        (value_mean, (batch, kv_heads, head_dim), torch.float32),
    )
    for tensor, expected_shape, dtype in expected:
        if tensor.shape != expected_shape or tensor.dtype is not dtype:
            raise ValueError("quantized Piper K/V metadata must match the padded native contract")
    for tensor in (key, key_scale, value, multiplier, log_scale, value_mean):
        if tensor.device != query_device:
            raise ValueError("quantized Piper operands must share a device")
        if tensor.layout is not torch.strided or not tensor.is_contiguous():
            raise ValueError("quantized Piper operands must be contiguous strided tensors")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise RuntimeError("quantized Piper Attention is an inference-only operator")


@torch.library.custom_op("piper_kernels::piper_attention_from_quantized", mutates_args=())
def _piper_attention_from_quantized_op(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    multiplier: torch.Tensor,
    log_scale: torch.Tensor,
    value_mean: torch.Tensor,
    query_length: int,
    key_length: int,
    is_causal: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Consume native Q32/K64 and dense per-token V without repeating preparation.

    K/V use K64-padded storage, including V metadata. V codes follow the
    producer's target-native layout. Numerical codes, scales, and centering
    are caller preconditions; all checks here inspect host metadata only.
    """
    shape = _validate_quantized(
        query,
        query_scale,
        key,
        key_scale,
        value,
        multiplier,
        log_scale,
        value_mean,
        query_length,
        key_length,
        is_causal,
        output_dtype,
    )
    if shape[0] == 0:
        return query.new_empty(shape, dtype=output_dtype)
    context = prepare_quantized_context(
        key,
        key_scale,
        value,
        multiplier,
        log_scale,
        value_mean,
        query_length=query_length,
        key_length=key_length,
        is_causal=is_causal,
    )
    return launch_quantized_attention_into(
        context, query, query_scale, query.new_empty(shape, dtype=output_dtype)
    )


type QuantizedContext = nvidia_backend._PreparedPiperContext | amd_backend.PreparedContext


def prepare_quantized_context(
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    multiplier: torch.Tensor,
    log_scale: torch.Tensor,
    value_mean: torch.Tensor,
    *,
    query_length: int,
    key_length: int,
    is_causal: bool,
) -> QuantizedContext:
    """Wrap validated native K/V once for full or chunked query execution."""
    target = AcceleratorTarget.from_device(key.device)
    if not supports_quantized_query(target):
        raise RuntimeError(f"quantized Piper Attention is unavailable on {key.device}")
    if target.is_nvidia_cuda:
        plan = nvidia_policy.select_execution_plan(
            target,
            head_dim=key.shape[-1],
            is_causal=is_causal,
            query_length=query_length,
            key_length=key_length,
        )
        with runtime.device_context(key.device):
            mixed_int8.install_uint8_int8_dot_hook()
            key_argument, value_argument = (
                nvidia_backend._make_key_value_descriptors(
                    key, value, split_pv_head_dim=plan.split_pv_head_dim
                )
                if plan.use_tensor_descriptors
                else (key, value)
            )
        return nvidia_backend._PreparedPiperContext(
            key=key_argument,
            value=value_argument,
            key_scale=key_scale,
            value_scale_multiplier=multiplier,
            value_log_scale=log_scale,
            value_mean=value_mean,
            key_length=key_length,
            is_causal=is_causal,
            plan=plan,
            padded_kv=True,
        )
    return amd_backend.PreparedContext(
        key=key,
        value=value,
        key_scale=key_scale,
        multiplier=multiplier,
        log_scale=log_scale,
        value_mean=value_mean,
        key_length=key_length,
        is_causal=is_causal,
    )


def launch_quantized_attention_into(
    context: QuantizedContext,
    query: torch.Tensor,
    query_scale: torch.Tensor,
    output: torch.Tensor,
    *,
    global_row_offset: int = 0,
) -> torch.Tensor:
    """Consume local Q storage and fill a validated BHSD output window."""
    batch, heads, rows, head_dim = output.shape
    shape = (batch, heads, rows, head_dim)
    if isinstance(context, nvidia_backend._PreparedPiperContext):
        plan = context.plan
        descriptor = (
            nvidia_backend._make_query_descriptor(query, plan.block_m)
            if plan.use_tensor_descriptors and plan.block_m == 128
            else None
        )
        prepared = nvidia_backend._PreparedPiperQuery(
            data=query,
            scale=query_scale,
            descriptor=descriptor,
            shape=shape,
            dtype=output.dtype,
            global_row_offset=global_row_offset,
        )
        return nvidia_backend._launch_piper_attention_into(context, prepared, output)
    prepared = amd_backend.PreparedQuery(
        data=query,
        scale=query_scale,
        shape=shape,
        dtype=output.dtype,
        global_row_offset=global_row_offset,
    )
    return amd_backend.launch_attention_into(context, prepared, output)


@_piper_attention_from_quantized_op.register_fake
def _piper_attention_from_quantized_fake(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    value: torch.Tensor,
    multiplier: torch.Tensor,
    log_scale: torch.Tensor,
    value_mean: torch.Tensor,
    query_length: int,
    key_length: int,
    is_causal: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    shape = _validate_quantized(
        query,
        query_scale,
        key,
        key_scale,
        value,
        multiplier,
        log_scale,
        value_mean,
        query_length,
        key_length,
        is_causal,
        output_dtype,
    )
    return query.new_empty(shape, dtype=output_dtype)
