"""Bounded sparse attention followed by a static ConvRot NVFP4 projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.fusions.nvfp4_sparse_piper import _output
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
    "piper_kernels::convrot_nvfp4_sparse_piper_attention_output",
    mutates_args=(),
)
def _attention_output_op(  # noqa: PLR0913, PLR0917
    query: torch.Tensor,
    query_scale: torch.Tensor,
    query_summary: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    key_max: torch.Tensor,
    key_min: torch.Tensor,
    value: torch.Tensor,
    value_scale_multiplier: torch.Tensor,
    value_mean: torch.Tensor,
    head_keep_ratio_units: list[int],
    sparse_key_blocks: int,
    logical_sequence_length: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_per_tensor_scale: torch.Tensor | None,
    activation_per_tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
) -> torch.Tensor:
    return _output.run_attention_output(
        query,
        query_scale,
        query_summary,
        key,
        key_scale,
        key_max,
        key_min,
        value,
        value_scale_multiplier,
        value_mean,
        head_keep_ratio_units,
        sparse_key_blocks,
        logical_sequence_length,
        weight_qdata,
        weight_scale,
        weight_per_tensor_scale,
        activation_per_tensor_scale,
        bias,
        query_chunk_rows,
        _ConvRotPreparation(group_size),
    )


@_attention_output_op.register_fake
def _attention_output_op_fake(
    query: torch.Tensor,
    _query_scale: torch.Tensor,
    _query_summary: torch.Tensor,
    _key: torch.Tensor,
    _key_scale: torch.Tensor,
    _key_max: torch.Tensor,
    _key_min: torch.Tensor,
    _value: torch.Tensor,
    _value_scale_multiplier: torch.Tensor,
    _value_mean: torch.Tensor,
    _head_keep_ratio_units: list[int],
    _sparse_key_blocks: int,
    logical_sequence_length: int,
    weight_qdata: torch.Tensor,
    _weight_scale: torch.Tensor,
    _weight_per_tensor_scale: torch.Tensor | None,
    _activation_per_tensor_scale: torch.Tensor,
    _bias: torch.Tensor | None,
    group_size: int,
    _query_chunk_rows: int = _DEFAULT_QUERY_CHUNK_ROWS,
) -> torch.Tensor:
    validate_group_size(group_size)
    input_features = 2 * weight_qdata.shape[1]
    if isinstance(input_features, int) and input_features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 projection input features {input_features} must be divisible "
            f"by group size {group_size}"
        )
    return query.new_empty(
        (query.shape[0], logical_sequence_length, weight_qdata.shape[0]),
        dtype=torch.bfloat16,
    )


__all__: list[str] = []
