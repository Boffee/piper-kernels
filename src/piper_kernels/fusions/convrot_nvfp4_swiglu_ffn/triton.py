"""Bounded-workspace composition of standard/ConvRot NVFP4 SwiGLU FFNs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.fusions.nvfp4_swiglu_ffn import _core
from piper_kernels.fusions.nvfp4_swiglu_ffn._preparation import StandardPreparation
from piper_kernels.fusions.swiglu_ffn import triton as gated_updates_backend
from piper_kernels.linear.convrot._rotation import validate_group_size
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4_backend

_DEFAULT_CHUNK_ROWS = _core.DEFAULT_CHUNK_ROWS


@dataclass(frozen=True, slots=True)
class _Preparation:
    """Standard or ConvRot activation preparation for the shared NVFP4 runner."""

    source_group_size: int | None
    down_group_size: int | None
    standard: StandardPreparation

    def __post_init__(self) -> None:
        if self.source_group_size is not None:
            validate_group_size(self.source_group_size)
        if self.down_group_size is not None:
            validate_group_size(self.down_group_size)

    def dynamic_source_scale(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
    ) -> torch.Tensor:
        if self.source_group_size is None:
            return self.standard.dynamic_source_scale(input)
        return convrot_nvfp4_backend.dynamic_scale(input, self.source_group_size)

    def prepare_source(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.source_group_size is None:
            return self.standard.prepare_source(input, per_tensor_scale, out)
        return convrot_nvfp4_backend.prepare_static_out(
            input,
            per_tensor_scale,
            self.source_group_size,
            out,
            high_first=self.standard.source_high_first,
        )

    def prepare_down(
        self,
        projections: torch.Tensor,
        activation_per_tensor_scale: torch.Tensor | None,
        dynamic_activation_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.down_group_size is None:
            return self.standard.prepare_down(
                projections, activation_per_tensor_scale, dynamic_activation_scale
            )
        if dynamic_activation_scale:
            return convrot_nvfp4_backend.prepare_dynamic(
                projections,
                self.down_group_size,
                "swiglu",
                high_first=self.standard.down_high_first,
            )
        assert activation_per_tensor_scale is not None
        return convrot_nvfp4_backend.prepare_static(
            projections,
            activation_per_tensor_scale,
            self.down_group_size,
            "swiglu",
            high_first=self.standard.down_high_first,
        )


def _preparation(
    gate_group_size: int | None,
    value_group_size: int | None,
    down_group_size: int | None,
    gate_high_first: bool,
    value_high_first: bool,
    down_high_first: bool,
) -> _Preparation:
    if gate_group_size != value_group_size:
        raise ValueError(
            "NVFP4 gate and value projections must share a group size or both disable rotation"
        )
    if gate_high_first != value_high_first:
        raise ValueError("NVFP4 gate and value projections must share nibble ordering")
    return _Preparation(
        gate_group_size,
        down_group_size,
        StandardPreparation(gate_high_first, down_high_first),
    )


@torch.library.custom_op("piper_kernels::convrot_nvfp4_swiglu_ffn", mutates_args=())
def _chunked_swiglu_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    gate_weight_qdata: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_weight_per_tensor_scale: torch.Tensor | None,
    gate_activation_per_tensor_scale: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    gate_dynamic_activation_scale: bool,
    gate_group_size: int | None,
    gate_high_first: bool,
    value_weight_qdata: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_weight_per_tensor_scale: torch.Tensor | None,
    value_activation_per_tensor_scale: torch.Tensor | None,
    value_bias: torch.Tensor | None,
    value_dynamic_activation_scale: bool,
    value_group_size: int | None,
    value_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    chunk_rows: int,
) -> torch.Tensor:
    gate, value, down = _core.linear_operands(
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_high_first,
        value_weight_qdata,
        value_weight_scale,
        value_weight_per_tensor_scale,
        value_activation_per_tensor_scale,
        value_bias,
        value_dynamic_activation_scale,
        value_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_high_first,
    )
    return _core.run_chunked_swiglu_ffn(
        input,
        gate,
        value,
        down,
        chunk_rows,
        _preparation(
            gate_group_size,
            value_group_size,
            down_group_size,
            gate_high_first,
            value_high_first,
            down_high_first,
        ),
    )


@_chunked_swiglu_ffn_op.register_fake
def _chunked_swiglu_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002
    _gate_weight_qdata: torch.Tensor,
    _gate_weight_scale: torch.Tensor,
    _gate_weight_per_tensor_scale: torch.Tensor | None,
    _gate_activation_per_tensor_scale: torch.Tensor | None,
    _gate_bias: torch.Tensor | None,
    _gate_dynamic_activation_scale: bool,
    _gate_group_size: int | None,
    _gate_high_first: bool,
    _value_weight_qdata: torch.Tensor,
    _value_weight_scale: torch.Tensor,
    _value_weight_per_tensor_scale: torch.Tensor | None,
    _value_activation_per_tensor_scale: torch.Tensor | None,
    _value_bias: torch.Tensor | None,
    _value_dynamic_activation_scale: bool,
    _value_group_size: int | None,
    _value_high_first: bool,
    down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_weight_per_tensor_scale: torch.Tensor | None,
    _down_activation_per_tensor_scale: torch.Tensor | None,
    _down_bias: torch.Tensor | None,
    _down_dynamic_activation_scale: bool,
    _down_group_size: int | None,
    _down_high_first: bool,
    _chunk_rows: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], down_weight_qdata.shape[0]))


@torch.library.custom_op(
    "piper_kernels::convrot_nvfp4_swiglu_ffn_gated_updates_",
    mutates_args=("reusable_update",),
)
def _chunked_swiglu_ffn_gated_updates_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    gate_weight_qdata: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_weight_per_tensor_scale: torch.Tensor | None,
    gate_activation_per_tensor_scale: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    gate_dynamic_activation_scale: bool,
    gate_group_size: int | None,
    gate_high_first: bool,
    value_weight_qdata: torch.Tensor,
    value_weight_scale: torch.Tensor,
    value_weight_per_tensor_scale: torch.Tensor | None,
    value_activation_per_tensor_scale: torch.Tensor | None,
    value_bias: torch.Tensor | None,
    value_dynamic_activation_scale: bool,
    value_group_size: int | None,
    value_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
) -> None:
    gate, value, down = _core.linear_operands(
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_high_first,
        value_weight_qdata,
        value_weight_scale,
        value_weight_per_tensor_scale,
        value_activation_per_tensor_scale,
        value_bias,
        value_dynamic_activation_scale,
        value_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_high_first,
    )
    _core.run_chunked_swiglu_ffn(
        input,
        gate,
        value,
        down,
        chunk_rows,
        _preparation(
            gate_group_size,
            value_group_size,
            down_group_size,
            gate_high_first,
            value_high_first,
            down_high_first,
        ),
        gated_updates=gated_updates_backend.IndexedGatedUpdates(
            base=base,
            reusable_update=reusable_update,
            update_gate=update_gate,
            ffn_gate=ffn_gate,
            gate_indices=gate_indices,
            python_indexing=python_indexing,
        ),
    )


@_chunked_swiglu_ffn_gated_updates_op.register_fake
def _chunked_swiglu_ffn_gated_updates_op_fake(
    _input: torch.Tensor,
    _gate_weight_qdata: torch.Tensor,
    _gate_weight_scale: torch.Tensor,
    _gate_weight_per_tensor_scale: torch.Tensor | None,
    _gate_activation_per_tensor_scale: torch.Tensor | None,
    _gate_bias: torch.Tensor | None,
    _gate_dynamic_activation_scale: bool,
    _gate_group_size: int | None,
    _gate_high_first: bool,
    _value_weight_qdata: torch.Tensor,
    _value_weight_scale: torch.Tensor,
    _value_weight_per_tensor_scale: torch.Tensor | None,
    _value_activation_per_tensor_scale: torch.Tensor | None,
    _value_bias: torch.Tensor | None,
    _value_dynamic_activation_scale: bool,
    _value_group_size: int | None,
    _value_high_first: bool,
    _down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_weight_per_tensor_scale: torch.Tensor | None,
    _down_activation_per_tensor_scale: torch.Tensor | None,
    _down_bias: torch.Tensor | None,
    _down_dynamic_activation_scale: bool,
    _down_group_size: int | None,
    _down_high_first: bool,
    _base: torch.Tensor,
    _reusable_update: torch.Tensor,
    _update_gate: torch.Tensor,
    _ffn_gate: torch.Tensor,
    _gate_indices: torch.Tensor,
    _python_indexing: bool,
    _chunk_rows: int,
) -> None:
    return None


__all__ = [
    "_chunked_swiglu_ffn_gated_updates_op",
    "_chunked_swiglu_ffn_op",
]
