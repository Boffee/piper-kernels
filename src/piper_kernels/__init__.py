"""Reusable PyTorch inference operators and optimized kernels."""

from __future__ import annotations

from collections.abc import Mapping

from .attention.piper_attention import (
    PiperKVChunkPool,
    freeze_kv_chunks_,
    kv_chunk_storage_specs,
    piper_attention,
)
from .attention.sage_attention_2pp import sage_attention_2pp
from .attention.sparse_piper_attention import (
    SparsePiperAttention,
    apply_coarse_attention_residual,
    coarse_attention,
    coarse_attention_residual,
    mean_pool_block_values,
    sparse_piper_coarse_residual,
)


def compile_options(
    options: Mapping[str, object] | None = None,
    *,
    linear: bool = True,
    ffn: bool = False,
    attention: bool = False,
    attention_output: bool = False,
    residual_updates: bool = False,
    conv3d: bool = False,
) -> dict[str, object]:
    """Lazily return Inductor compile options for Piper Kernels' unified compiler pipeline.

    Pass the returned mapping to ``torch.compile(..., options=...)`` without
    also supplying its mutually exclusive ``mode`` argument. Repeated calls
    union enabled flags into one phase-ordered post-grad pass and one pre-grad
    pass without mutating ``options`` or duplicating passes.

    Args:
        options: Existing Inductor options mapping to copy and extend.
        linear: Share quantized linear input preparation, fold exclusive
            ``tanh``-GELU linear inputs, and hoist ConvRot INT8 preparation
            ahead of sequence all-gathers.
        ffn: Fuse ConvRot INT8, NVFP4, and ConvRot NVFP4 SwiGLU and
            ``tanh``-GELU FFNs (including indexed gated updates and copy-back
            in-place residual adds).
        attention: Fuse ConvRot INT8 dense Piper Q/K/V projections, ConvRot
            INT8 sparse Piper Q/K/V/output/distributed blocks, and NVFP4 /
            ConvRot NVFP4 sparse Piper Q/K/V/output blocks.
        attention_output: Opt-in bounded attention-to-output projection (plus
            in-place residual and sequence-parallel reduce-scatter folding) for
            dense ConvRot INT8 Piper attention. Implies ``attention=True``.
        residual_updates: Rewrite copy-back gated residual sums
            (``hidden.copy_(...)``) into functional in-place adds and set
            ``allow_buffer_reuse=False``.
        conv3d: Install pre-grad ConvRot INT8 Conv3D zero/spatial-reflection
            padding fusion.
    """
    try:
        from ._inductor_options import (  # noqa: PLC0415 - keep Inductor imports lazy
            compile_options as build_compile_options,
        )
    except ImportError as error:
        raise RuntimeError(
            "Piper Kernels compiler integration requires a compatible PyTorch "
            "Inductor custom graph-pass API"
        ) from error

    return build_compile_options(
        options,
        linear=linear,
        ffn=ffn,
        attention=attention,
        attention_output=attention_output,
        residual_updates=residual_updates,
        conv3d=conv3d,
    )


__all__ = [
    "PiperKVChunkPool",
    "SparsePiperAttention",
    "apply_coarse_attention_residual",
    "coarse_attention",
    "coarse_attention_residual",
    "compile_options",
    "freeze_kv_chunks_",
    "kv_chunk_storage_specs",
    "mean_pool_block_values",
    "piper_attention",
    "sage_attention_2pp",
    "sparse_piper_coarse_residual",
]
