"""Bounded-workspace composition of an NVFP4 SwiGLU feed-forward network."""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.fusions.swiglu_ffn import triton as gated_updates_backend
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4 import triton as nvfp4_backend

from . import _core

_DEFAULT_CHUNK_ROWS = _core.DEFAULT_CHUNK_ROWS


def _dynamic_swiglu_scale(input: torch.Tensor) -> torch.Tensor:  # noqa: A002
    up, gate = input.chunk(2, dim=-1)
    activated = up * F.silu(gate)
    return per_tensor_amax_to_scale(activated.abs().amax())


def _projected_swiglu_scale(
    input: torch.Tensor,  # noqa: A002
    global_scale: torch.Tensor,
) -> torch.Tensor:
    projected = (input.float() * global_scale.float()).to(input.dtype)
    return _dynamic_swiglu_scale(projected)


def _projected_swiglu_scale_and_add_bias(
    input: torch.Tensor,  # noqa: A002
    global_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    projected = (input.float() * global_scale.float() + bias.float()).to(input.dtype)
    return _dynamic_swiglu_scale(projected)


_compiled_projected_swiglu_scale = torch.compile(_projected_swiglu_scale, fullgraph=True)
_compiled_projected_swiglu_scale_and_add_bias = torch.compile(
    _projected_swiglu_scale_and_add_bias,
    fullgraph=True,
)


def _projected_swiglu_dynamic_scale(
    input: torch.Tensor,  # noqa: A002
    global_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Calculate a dynamic scale without materializing the projected SwiGLU."""
    if bias is None:
        return _compiled_projected_swiglu_scale(input, global_scale)
    return _compiled_projected_swiglu_scale_and_add_bias(input, global_scale, bias)


class _StandardPreparation:
    """Ordinary NVFP4 preparation used by the shared chunked runner."""

    def prepare_up(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        up: _core.LinearOperands,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return nvfp4_ops._prepare_compiled(
            input,
            up.activation_per_tensor_scale,
            up.dynamic_activation_scale,
        )

    def dynamic_down_scale(
        self,
        packed: torch.Tensor,
        up_global_scale: torch.Tensor,
        up_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return _projected_swiglu_dynamic_scale(packed, up_global_scale, up_bias)

    def prepare_down(
        self,
        packed: torch.Tensor,
        down_per_tensor_scale: torch.Tensor,
        up_global_scale: torch.Tensor,
        up_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return nvfp4_backend.prepare_static_projected_swiglu(
            packed,
            down_per_tensor_scale,
            up_global_scale,
            up_bias,
        )


_STANDARD_PREPARATION = _StandardPreparation()


@torch.library.custom_op("piper_kernels::nvfp4_swiglu_ffn", mutates_args=())
def _chunked_swiglu_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
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
        _STANDARD_PREPARATION,
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
    down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_weight_per_tensor_scale: torch.Tensor | None,
    _down_activation_per_tensor_scale: torch.Tensor | None,
    _down_bias: torch.Tensor | None,
    _down_dynamic_activation_scale: bool,
    _chunk_rows: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], down_weight_qdata.shape[0]))


@torch.library.custom_op(
    "piper_kernels::nvfp4_swiglu_ffn_gated_updates_",
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
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
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
        _STANDARD_PREPARATION,
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
    _down_weight_qdata: torch.Tensor,
    _down_weight_scale: torch.Tensor,
    _down_weight_per_tensor_scale: torch.Tensor | None,
    _down_activation_per_tensor_scale: torch.Tensor | None,
    _down_bias: torch.Tensor | None,
    _down_dynamic_activation_scale: bool,
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
