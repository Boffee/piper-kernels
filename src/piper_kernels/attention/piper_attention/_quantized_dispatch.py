"""Dense Piper boundary for fused Q producers and ordinary floating K/V."""

import torch

from piper_kernels._triton import mixed_int8, runtime, targets
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


def _validate_quantized_query(  # noqa: PLR0912
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_length: int,
    is_causal: bool,
) -> tuple[int, int, int, int]:
    """Validate storage metadata without inspecting codes or numerical scales."""
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
            target, head_dim=shape[3], is_causal=is_causal, query_length=query_length
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
