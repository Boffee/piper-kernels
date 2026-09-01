"""Bounded-workspace composition of a ConvRot NVFP4 SwiGLU FFN."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.fusions.nvfp4_swiglu_ffn import _core
from piper_kernels.fusions.swiglu_ffn import triton as gated_updates_backend
from piper_kernels.linear.convrot._rotation import validate_group_size
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4_backend

_DEFAULT_CHUNK_ROWS = _core.DEFAULT_CHUNK_ROWS


@dataclass(frozen=True, slots=True)
class _ConvRotPreparation:
    """ConvRot activation preparation for the shared NVFP4 FFN runner."""

    up_group_size: int
    down_group_size: int

    def __post_init__(self) -> None:
        validate_group_size(self.up_group_size)
        validate_group_size(self.down_group_size)

    def dynamic_up_scale(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
    ) -> torch.Tensor:
        return convrot_nvfp4_backend.dynamic_scale(input, self.up_group_size)

    def prepare_up(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return convrot_nvfp4_backend.prepare_static_out(
            input,
            per_tensor_scale,
            self.up_group_size,
            out,
        )

    def dynamic_down_scale(
        self,
        packed: torch.Tensor,
        up_global_scale: torch.Tensor,
        up_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return convrot_nvfp4_backend.projected_swiglu_dynamic_scale(
            packed,
            up_global_scale,
            up_bias,
            self.down_group_size,
        )

    def prepare_down(
        self,
        packed: torch.Tensor,
        down_per_tensor_scale: torch.Tensor,
        up_global_scale: torch.Tensor,
        up_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return convrot_nvfp4_backend.prepare_static_projected_swiglu(
            packed,
            down_per_tensor_scale,
            up_global_scale,
            up_bias,
            self.down_group_size,
        )


@torch.library.custom_op("piper_kernels::convrot_nvfp4_swiglu_ffn", mutates_args=())
def _chunked_swiglu_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int,
    chunk_rows: int,
) -> torch.Tensor:
    up = _core.linear_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
    )
    down = _core.linear_operands(
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
    )
    return _core.run_chunked_swiglu_ffn(
        input,
        up,
        down,
        chunk_rows,
        _ConvRotPreparation(up_group_size, down_group_size),
    )


@_chunked_swiglu_ffn_op.register_fake
def _chunked_swiglu_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002
    _up_weight_qdata: torch.Tensor,
    _up_weight_scale: torch.Tensor,
    _up_weight_per_tensor_scale: torch.Tensor | None,
    _up_activation_per_tensor_scale: torch.Tensor | None,
    _up_bias: torch.Tensor | None,
    _up_dynamic_activation_scale: bool,
    _up_group_size: int,
    down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_weight_per_tensor_scale: torch.Tensor | None,
    _down_activation_per_tensor_scale: torch.Tensor | None,
    _down_bias: torch.Tensor | None,
    _down_dynamic_activation_scale: bool,
    _down_group_size: int,
    _chunk_rows: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], down_weight_qdata.shape[0]))


@torch.library.custom_op(
    "piper_kernels::convrot_nvfp4_swiglu_ffn_gated_updates_",
    mutates_args=("reusable_update",),
)
def _chunked_swiglu_ffn_gated_updates_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int,
    base: torch.Tensor,
    reusable_update: torch.Tensor,
    update_gate: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
) -> None:
    up = _core.linear_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
    )
    down = _core.linear_operands(
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
    )
    _core.run_chunked_swiglu_ffn(
        input,
        up,
        down,
        chunk_rows,
        _ConvRotPreparation(up_group_size, down_group_size),
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
    _up_weight_qdata: torch.Tensor,
    _up_weight_scale: torch.Tensor,
    _up_weight_per_tensor_scale: torch.Tensor | None,
    _up_activation_per_tensor_scale: torch.Tensor | None,
    _up_bias: torch.Tensor | None,
    _up_dynamic_activation_scale: bool,
    _up_group_size: int,
    _down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_weight_per_tensor_scale: torch.Tensor | None,
    _down_activation_per_tensor_scale: torch.Tensor | None,
    _down_bias: torch.Tensor | None,
    _down_dynamic_activation_scale: bool,
    _down_group_size: int,
    _base: torch.Tensor,
    _reusable_update: torch.Tensor,
    _update_gate: torch.Tensor,
    _ffn_gate: torch.Tensor,
    _gate_indices: torch.Tensor,
    _python_indexing: bool,
    _chunk_rows: int,
) -> None:
    return None


__all__ = ["_chunked_swiglu_ffn_gated_updates_op", "_chunked_swiglu_ffn_op"]
