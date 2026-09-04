"""Bounded sparse attention followed by a static ConvRot NVFP4 projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.fusions.nvfp4_sparse_piper import _output
from piper_kernels.fusions.sparse_piper import _output as output_common
from piper_kernels.linear.convrot._rotation import validate_group_size
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4_backend

_DEFAULT_QUERY_CHUNK_ROWS = _output.DEFAULT_QUERY_CHUNK_ROWS


@dataclass(frozen=True, slots=True)
class _ConvRotPreparation:
    """ConvRot preparation for the shared NVFP4 attention-output runner."""

    group_size: int

    def __post_init__(self) -> None:
        validate_group_size(self.group_size)

    def prepare_static_out(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return convrot_nvfp4_backend.prepare_static_out(
            input,
            per_tensor_scale,
            self.group_size,
            out,
        )


@torch.library.custom_op(
    "piper_kernels::convrot_nvfp4_sparse_piper_projected_query_attention_output",
    mutates_args=(),
)
def _projected_query_attention_output_op(  # noqa: PLR0913, PLR0917
    query_input_qdata: torch.Tensor,
    query_input_scale: torch.Tensor,
    query_input_per_tensor_scale: torch.Tensor,
    query_weight_qdata: torch.Tensor,
    query_weight_scale: torch.Tensor,
    query_weight_per_tensor_scale: torch.Tensor | None,
    query_bias: torch.Tensor | None,
    query_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_norm_epsilon: float,
    softmax_scale: float,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_input_qdata: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_input_per_tensor_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_weight_per_tensor_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    gate_projection = _output.prepare_optional_gate_projection(
        key,
        logical_sequence_length,
        block_lengths,
        gate_input_qdata,
        gate_input_scale,
        gate_input_per_tensor_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_bias,
    )
    return _output.run_projected_query_attention_output(
        query_input_qdata,
        query_input_scale,
        query_input_per_tensor_scale,
        query_weight_qdata,
        query_weight_scale,
        query_weight_per_tensor_scale,
        query_bias,
        query_norm_weight,
        cos,
        sin,
        query_norm_epsilon,
        softmax_scale,
        key,
        key_scale,
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
        routing_mode,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        query_chunk_rows,
        _ConvRotPreparation(group_size),
        block_lengths,
        block_mean,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
    )


@_projected_query_attention_output_op.register_fake
def _projected_query_attention_output_op_fake(
    _query_input_qdata: torch.Tensor,
    _query_input_scale: torch.Tensor,
    _query_input_per_tensor_scale: torch.Tensor,
    _query_weight_qdata: torch.Tensor,
    _query_weight_scale: torch.Tensor,
    _query_weight_per_tensor_scale: torch.Tensor | None,
    _query_bias: torch.Tensor | None,
    _query_norm_weight: torch.Tensor,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _query_norm_epsilon: float,
    _softmax_scale: float,
    key: torch.Tensor,
    _key_scale: torch.Tensor,
    _key_summary: torch.Tensor,
    _key_aux: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    _routing_mode: int,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _activation_per_tensor_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    group_size: int,
    _query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    _block_mean: torch.Tensor | None = None,
    _coarse_gate: torch.Tensor | None = None,
    _coarse_scale: float | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    _gate_input_qdata: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_input_per_tensor_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_weight_per_tensor_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_group_size(group_size)
    input_features = 2 * weight_qdata.shape[1]
    if isinstance(input_features, int) and input_features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 projection input features {input_features} must be divisible "
            f"by group size {group_size}"
        )
    return output_common.new_projected_output(
        key,
        logical_sequence_length,
        block_lengths,
        weight_qdata.shape[0],
    )


@torch.library.custom_op(
    "piper_kernels::convrot_nvfp4_sparse_piper_attention_output",
    mutates_args=(),
)
def _attention_output_op(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_summary: torch.Tensor,
    key_aux: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    routing_mode: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    block_mean: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    sparse_query_blocks: int | None = None,
    gate_input_qdata: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    gate_input_per_tensor_scale: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_weight_per_tensor_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    gate_projection = _output.prepare_optional_gate_projection(
        query,
        logical_sequence_length,
        block_lengths,
        gate_input_qdata,
        gate_input_scale,
        gate_input_per_tensor_scale,
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_bias,
    )
    return _output.run_attention_output(
        query,
        query_scale,
        query_summary,
        key,
        key_scale,
        key_summary,
        key_aux,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
        routing_mode,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        query_chunk_rows,
        _ConvRotPreparation(group_size),
        block_lengths,
        block_mean,
        coarse_gate,
        coarse_scale,
        coarse_key_blocks,
        sparse_query_blocks,
        gate_projection,
    )


@_attention_output_op.register_fake
def _attention_output_op_fake(
    query: torch.Tensor,
    _query_scale: torch.Tensor,
    _query_summary: torch.Tensor,
    _key: torch.Tensor,
    _key_scale: torch.Tensor,
    _key_summary: torch.Tensor,
    _key_aux: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    _routing_mode: int,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _activation_per_tensor_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    group_size: int,
    _query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
    block_lengths: torch.Tensor | None = None,
    _block_mean: torch.Tensor | None = None,
    _coarse_gate: torch.Tensor | None = None,
    _coarse_scale: float | None = None,
    _coarse_key_blocks: int | None = None,
    _sparse_query_blocks: int | None = None,
    _gate_input_qdata: torch.Tensor | None = None,
    _gate_input_scale: torch.Tensor | None = None,
    _gate_input_per_tensor_scale: torch.Tensor | None = None,
    _gate_weight_qdata: torch.Tensor | None = None,
    _gate_weight_scale: torch.Tensor | None = None,
    _gate_weight_per_tensor_scale: torch.Tensor | None = None,
    _gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_group_size(group_size)
    input_features = 2 * weight_qdata.shape[1]
    if isinstance(input_features, int) and input_features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 projection input features {input_features} must be divisible "
            f"by group size {group_size}"
        )
    return output_common.new_projected_output(
        query,
        logical_sequence_length,
        block_lengths,
        weight_qdata.shape[0],
    )


__all__: list[str] = []
