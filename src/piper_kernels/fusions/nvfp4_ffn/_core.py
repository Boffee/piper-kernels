"""Topology-aware bounded execution for standard and ConvRot NVFP4 FFNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch

from piper_kernels.fusions.ffn import triton as indexed_updates
from piper_kernels.linear._storage import same_tensor_storage
from piper_kernels.linear.nvfp4 import _projection as nvfp4_projection
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation
from piper_kernels.linear.nvfp4._storage import prepare_activation_storage
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

DEFAULT_CHUNK_ROWS = 1_536


@dataclass(frozen=True, slots=True)
class LinearOperands:
    """Canonical storage and scaling operands for one NVFP4 projection."""

    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    weight_per_tensor_scale: torch.Tensor | None
    activation_per_tensor_scale: torch.Tensor | None
    bias: torch.Tensor | None
    dynamic_activation_scale: bool
    high_first: bool


class SourcePreparationBackend(Protocol):
    """Format-specific preparation shared by the topology's source projections."""

    def dynamic_scale(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
    ) -> torch.Tensor:
        """Calculate one global scale for the complete FFN input."""
        ...

    def prepare(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        per_tensor_scale: torch.Tensor,
        out: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare one FFN input chunk into reusable storage."""
        ...


class ActivationPreparationBackend(Protocol):
    """Topology-specific activation and down-input preparation."""

    source_projection_count: ClassVar[int]

    def prepare(
        self,
        projections: torch.Tensor,
        activation_per_tensor_scale: torch.Tensor | None,
        dynamic_activation_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Activate source projections and prepare one down-projection input chunk."""
        ...


def _validate_inputs(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    sources: tuple[LinearOperands, ...],
    down: LinearOperands,
    chunk_rows: int,
    activation_preparation: ActivationPreparationBackend,
) -> tuple[int, int, int]:
    if len(sources) != activation_preparation.source_projection_count:
        raise ValueError(
            "NVFP4 FFN source projection count does not match its activation preparation"
        )
    if len(sources) not in (1, 2):
        raise ValueError("NVFP4 FFN activations must consume one or two source projections")
    source_shapes = tuple(
        nvfp4_validation.validate_semantic_linear(
            input,
            source.weight_qdata,
            source.weight_scale,
            source.weight_per_tensor_scale,
            source.activation_per_tensor_scale,
            source.bias,
            source.dynamic_activation_scale,
            f"NVFP4 FFN source projection {index}",
        )
        for index, source in enumerate(sources)
    )
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("NVFP4 FFN requires FP16 or BF16 activations")
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < nvfp4_layout.SCALE_ROW_TILE
        or chunk_rows % nvfp4_layout.SCALE_ROW_TILE
    ):
        raise ValueError("NVFP4 FFN chunk_rows must be a positive multiple of 128")
    if any(source.high_first != sources[0].high_first for source in sources[1:]):
        raise ValueError("NVFP4 FFN source projections must share nibble ordering")
    shape = source_shapes[0]
    if (
        any(
            source_shape.rows != shape.rows
            or source_shape.input_features != shape.input_features
            or source_shape.output_features != shape.output_features
            for source_shape in source_shapes[1:]
        )
        or not isinstance(shape.rows, int)
        or not isinstance(shape.output_features, int)
    ):
        raise ValueError("NVFP4 FFN source projections must have matching concrete shapes")
    intermediate_features = shape.output_features
    nvfp4_validation.validate_activation_scale(
        down.activation_per_tensor_scale,
        down.dynamic_activation_scale,
        input.device,
        "NVFP4 FFN down projection",
    )
    output_features = nvfp4_validation.validate_weight(
        down.weight_qdata,
        down.weight_scale,
        down.weight_per_tensor_scale,
        down.bias,
        input_features=intermediate_features,
        device=input.device,
        name="NVFP4 FFN down projection",
    )
    if not isinstance(output_features, int):
        raise ValueError("NVFP4 FFN requires concrete projection dimensions")
    linears = (*sources, down)
    differentiable_tensors = (
        input,
        *(linear.weight_scale for linear in linears),
        *(
            tensor
            for linear in linears
            for tensor in (
                linear.weight_per_tensor_scale,
                linear.activation_per_tensor_scale,
                linear.bias,
            )
            if tensor is not None
        ),
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in differentiable_tensors):
        raise RuntimeError("NVFP4 FFN is inference-only and does not support autograd")
    return shape.rows, intermediate_features, output_features


def _project_affine_chunk(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    linear: LinearOperands,
    rows: int,
    output: torch.Tensor,
) -> None:
    nvfp4_projection.matmul_prepared_chunk_affine_out(
        input_qdata,
        input_scale,
        input_per_tensor_scale,
        linear.weight_qdata,
        linear.weight_scale,
        linear.weight_per_tensor_scale,
        linear.bias,
        0,
        rows,
        output,
    )


def _source_per_tensor_scales(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    sources: tuple[LinearOperands, ...],
    preparation: SourcePreparationBackend,
) -> tuple[torch.Tensor, ...]:
    dynamic_scale: torch.Tensor | None = None
    scales = []
    for source in sources:
        if source.dynamic_activation_scale:
            if dynamic_scale is None:
                dynamic_scale = preparation.dynamic_scale(input)
            scales.append(dynamic_scale)
        else:
            assert source.activation_per_tensor_scale is not None
            scales.append(source.activation_per_tensor_scale)
    return tuple(scales)


def run_chunked_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    sources: tuple[LinearOperands, ...],
    down: LinearOperands,
    chunk_rows: int,
    source_preparation: SourcePreparationBackend,
    activation_preparation: ActivationPreparationBackend,
    *,
    gated_updates: indexed_updates.IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    """Project, activate, and down-project row chunks with topology-sized workspaces."""
    rows, intermediate_features, output_features = _validate_inputs(
        input,
        sources,
        down,
        chunk_rows,
        activation_preparation,
    )
    leading_shape = input.shape[:-1]
    input_features = input.shape[-1]
    input_2d = input.reshape(rows, input_features)
    source_per_tensor_scales = _source_per_tensor_scales(input, sources, source_preparation)
    shared_input_preparation = all(source.dynamic_activation_scale for source in sources) or (
        all(not source.dynamic_activation_scale for source in sources)
        and all(
            same_tensor_storage(source_per_tensor_scales[0], per_tensor_scale)
            for per_tensor_scale in source_per_tensor_scales[1:]
        )
    )
    update_layout = (
        None
        if gated_updates is None
        else indexed_updates.validate_indexed_gated_updates(
            input,
            gated_updates,
            output_features,
        )
    )
    output = (
        torch.empty((*leading_shape, output_features), device=input.device, dtype=input.dtype)
        if gated_updates is None
        else gated_updates.reusable_update
    )
    output_2d = output.reshape(rows, output_features)
    base_2d = None if gated_updates is None else gated_updates.base.reshape(rows, output_features)
    workspace_rows = min(rows, chunk_rows)
    source_storage = prepare_activation_storage(
        input,
        workspace_rows,
        input_features,
    )
    projection_workspace = torch.empty(
        (workspace_rows, len(sources) * intermediate_features),
        device=input.device,
        dtype=input.dtype,
    )
    projected_workspace = (
        None
        if gated_updates is None
        else torch.empty(
            (workspace_rows, output_features),
            device=input.device,
            dtype=input.dtype,
        )
    )

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        chunk_row_count = stop - start
        projections = projection_workspace[:chunk_row_count]
        first_input_qdata, first_input_scale = source_preparation.prepare(
            input_2d[start:stop],
            source_per_tensor_scales[0],
            source_storage,
        )
        _project_affine_chunk(
            first_input_qdata,
            first_input_scale,
            source_per_tensor_scales[0],
            sources[0],
            chunk_row_count,
            projections[:, :intermediate_features],
        )
        if len(sources) == 2:
            second_input_qdata, second_input_scale = (
                (first_input_qdata, first_input_scale)
                if shared_input_preparation
                else source_preparation.prepare(
                    input_2d[start:stop],
                    source_per_tensor_scales[1],
                    source_storage,
                )
            )
            _project_affine_chunk(
                second_input_qdata,
                second_input_scale,
                source_per_tensor_scales[1],
                sources[1],
                chunk_row_count,
                projections[:, intermediate_features:],
            )
        down_qdata, down_scale, down_per_tensor_scale = activation_preparation.prepare(
            projections,
            down.activation_per_tensor_scale,
            down.dynamic_activation_scale,
        )
        if gated_updates is None:
            _project_affine_chunk(
                down_qdata,
                down_scale,
                down_per_tensor_scale,
                down,
                chunk_row_count,
                output_2d[start:stop],
            )
            continue
        assert projected_workspace is not None
        assert base_2d is not None
        assert update_layout is not None
        projected = projected_workspace[:chunk_row_count]
        _project_affine_chunk(
            down_qdata,
            down_scale,
            down_per_tensor_scale,
            down,
            chunk_row_count,
            projected,
        )
        indexed_updates.apply_indexed_gated_updates(
            projected,
            base_2d[start:stop],
            output_2d[start:stop],
            gated_updates,
            update_layout,
            start,
        )
    return output


__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "ActivationPreparationBackend",
    "LinearOperands",
    "SourcePreparationBackend",
    "run_chunked_ffn",
]
