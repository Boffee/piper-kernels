"""Bounded-workspace standard and ConvRot NVFP4 FFN runner and custom ops.

This module imports without Triton, TorchAO, or ConvRot. Standard NVFP4 calls
(``group_size=None``) never import ConvRot packages or register ConvRot ops;
ConvRot calls import rotation validation or kernels lazily.
"""

# Explicit custom-op schemas take flat tensor arguments required by torch.library.
# ruff: noqa: PLR0913, PLR0917

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels._input_activations import InputActivation, input_activation_width
from piper_kernels._residual import ResidualAdd, ReusableUpdate, validate_residual_add
from piper_kernels.linear._storage import same_tensor_storage
from piper_kernels.linear.nvfp4 import _validation as nvfp4_validation
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

DEFAULT_CHUNK_ROWS = 1_536
_GELU_WORKSPACE_LIMIT_BYTES = 512 * 1_024 * 1_024
_CHUNK_ROW_GRANULARITY = 128


@dataclass(frozen=True, slots=True)
class LinearOperands:
    """Canonical storage, scaling, and optional rotation operands for one NVFP4 projection."""

    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    weight_per_tensor_scale: torch.Tensor | None
    activation_per_tensor_scale: torch.Tensor | None
    bias: torch.Tensor | None
    dynamic_activation_scale: bool
    group_size: int | None
    high_first: bool


def _prepared_bytes_per_row(features: int) -> int:
    """Return exact packed-data and padded-scale storage for an aligned row tile."""
    tile_rows = nvfp4_layout.SCALE_ROW_TILE
    qdata_shape = cast(tuple[int, int], nvfp4_layout.qdata_shape(tile_rows, features))
    scale_shape = cast(tuple[int, int], nvfp4_layout.scale_shape(tile_rows, features))
    return (math.prod(qdata_shape) + math.prod(scale_shape)) // tile_rows


def default_chunk_rows(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    down_weight_qdata: torch.Tensor,
    activation_fn: InputActivation,
    *,
    adds_residual: bool,
) -> int:
    """Return bounded row chunk size for standard and ConvRot NVFP4 FFNs.

    SwiGLU uses a fixed 1536-row chunk; GELU bounds reusable scratch to 512 MiB
    while amortizing long-sequence launches and accounting for residual scratch
    when the output is wider than the intermediate projection.
    """
    if activation_fn == "swiglu":
        return DEFAULT_CHUNK_ROWS
    input_features = 2 * up_weight_qdata.shape[1]
    intermediate_features = up_weight_qdata.shape[0]
    output_features = down_weight_qdata.shape[0]
    if not all(
        isinstance(features, int)
        for features in (input_features, intermediate_features, output_features)
    ):
        return DEFAULT_CHUNK_ROWS
    source_preparation_bytes = _prepared_bytes_per_row(input_features)
    activation_preparation_bytes = _prepared_bytes_per_row(intermediate_features)
    projection_bytes = intermediate_features * input.element_size()
    projected_output_bytes = (
        output_features * input.element_size()
        if adds_residual and output_features > intermediate_features
        else 0
    )
    bytes_per_row = (
        source_preparation_bytes
        + projection_bytes
        + activation_preparation_bytes
        + projected_output_bytes
    )
    rows = _GELU_WORKSPACE_LIMIT_BYTES // bytes_per_row
    return max(_CHUNK_ROW_GRANULARITY, rows // _CHUNK_ROW_GRANULARITY * _CHUNK_ROW_GRANULARITY)


def _dimension_matches(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    return str(left) == str(right)


def _validate_rotation_group(
    features: int | torch.SymInt,
    group_size: int | None,
    name: str,
) -> None:
    if group_size is None:
        return
    from piper_kernels.weights.convrot._rotation import validate_group_size  # noqa: PLC0415

    validate_group_size(group_size)
    if isinstance(features, int) and features % group_size:
        raise ValueError(
            f"ConvRot NVFP4 FFN {name} features {features} must be divisible "
            f"by group size {group_size}"
        )


def _parse_activation(activation_fn: str) -> InputActivation:
    if activation_fn not in ("gelu_tanh", "swiglu"):
        raise ValueError(
            f"NVFP4 FFN activation_fn must be 'gelu_tanh' or 'swiglu', got {activation_fn!r}"
        )
    return cast(InputActivation, activation_fn)


def _build_operands(
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    activation_fn: str,
    gate_weight_qdata: torch.Tensor | None,
    gate_weight_scale: torch.Tensor | None,
    gate_weight_per_tensor_scale: torch.Tensor | None,
    gate_activation_per_tensor_scale: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    gate_dynamic_activation_scale: bool,
    gate_group_size: int | None,
    gate_high_first: bool,
) -> tuple[InputActivation, tuple[LinearOperands, ...], LinearOperands]:
    activation = _parse_activation(activation_fn)
    up = LinearOperands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_group_size,
        up_high_first,
    )
    down = LinearOperands(
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_group_size,
        down_high_first,
    )
    if activation == "gelu_tanh":
        if any(
            value is not None
            for value in (
                gate_weight_qdata,
                gate_weight_scale,
                gate_weight_per_tensor_scale,
                gate_activation_per_tensor_scale,
                gate_bias,
                gate_group_size,
            )
        ):
            raise ValueError("NVFP4 gelu_tanh FFN does not accept gate projection operands")
        return activation, (up,), down
    if gate_weight_qdata is None or gate_weight_scale is None:
        raise ValueError("NVFP4 swiglu FFN requires gate_weight_qdata and gate_weight_scale")
    gate = LinearOperands(
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_group_size,
        gate_high_first,
    )
    return activation, (up, gate), down


def _build_residual_add(
    output: torch.Tensor,
    base: torch.Tensor | None,
    update_gate: torch.Tensor | None,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
) -> ResidualAdd:
    if (base is None) != (update_gate is None):
        raise ValueError(
            "NVFP4 FFN update base and update_gate must both be provided or both omitted"
        )
    if base is None:
        return ResidualAdd(
            base=output,
            gate=ffn_gate,
            gate_indices=gate_indices,
            python_indexing=python_indexing,
        )
    assert update_gate is not None
    return ResidualAdd(
        base=base,
        gate=ffn_gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
        reusable=ReusableUpdate(output, update_gate),
    )


def validate_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    sources: tuple[LinearOperands, ...],
    down: LinearOperands,
    activation_fn: InputActivation,
    chunk_rows: int,
) -> tuple[int | torch.SymInt, int | torch.SymInt, int | torch.SymInt]:
    """Validate bounded NVFP4 FFN metadata and return its logical dimensions."""
    if input.ndim == 0 or input.layout is not torch.strided or not input.is_contiguous():
        raise ValueError("NVFP4 FFN input must be a non-scalar contiguous strided tensor")
    expected_sources = input_activation_width(activation_fn)
    if len(sources) != expected_sources:
        raise ValueError(
            f"NVFP4 {activation_fn} FFN requires {expected_sources} source projection(s)"
        )
    if any(source.group_size != sources[0].group_size for source in sources[1:]):
        raise ValueError(
            "NVFP4 source projections must share a group size or both disable rotation"
        )
    if any(source.high_first != sources[0].high_first for source in sources[1:]):
        raise ValueError("NVFP4 source projections must share nibble ordering")
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
        or not isinstance(chunk_rows, (int, torch.SymInt))
        or chunk_rows < nvfp4_layout.SCALE_ROW_TILE
        or chunk_rows % nvfp4_layout.SCALE_ROW_TILE
    ):
        raise ValueError("NVFP4 FFN chunk_rows must be a positive multiple of 128")
    shape = source_shapes[0]
    if any(
        not _dimension_matches(source_shape.rows, shape.rows)
        or not _dimension_matches(source_shape.input_features, shape.input_features)
        or not _dimension_matches(source_shape.output_features, shape.output_features)
        for source_shape in source_shapes[1:]
    ):
        raise ValueError("NVFP4 FFN source projections must have matching shapes")
    _validate_rotation_group(shape.input_features, sources[0].group_size, "source")
    intermediate_features = shape.output_features
    _validate_rotation_group(intermediate_features, down.group_size, "activated")
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


@functools.cache
def _compiled_standard_dynamic_scale(
    activation_fn: InputActivation,
    dtype: torch.dtype,
) -> Callable[[torch.Tensor], torch.Tensor]:
    from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale  # noqa: PLC0415

    # Separate functions per (activation_fn, dtype) keep distinct Dynamo code objects
    # so mixed FP16/BF16, inference_mode/no_grad, and single-row workloads stay well
    # under Dynamo's recompile limit.
    if activation_fn == "gelu_tanh":
        if dtype is torch.float16:

            def _gelu_fp16(projections: torch.Tensor) -> torch.Tensor:
                return per_tensor_amax_to_scale(
                    F.gelu(projections.float(), approximate="tanh").abs().amax()
                )

            return torch.compile(_gelu_fp16, fullgraph=True, dynamic=True)

        def _gelu_bf16(projections: torch.Tensor) -> torch.Tensor:
            return per_tensor_amax_to_scale(
                F.gelu(projections.float(), approximate="tanh").abs().amax()
            )

        return torch.compile(_gelu_bf16, fullgraph=True, dynamic=True)

    if dtype is torch.float16:

        def _swiglu_fp16(projections: torch.Tensor) -> torch.Tensor:
            value, gate = projections.chunk(2, dim=-1)
            return per_tensor_amax_to_scale((value.float() * F.silu(gate.float())).abs().amax())

        return torch.compile(_swiglu_fp16, fullgraph=True, dynamic=True)

    def _swiglu_bf16(projections: torch.Tensor) -> torch.Tensor:
        value, gate = projections.chunk(2, dim=-1)
        return per_tensor_amax_to_scale((value.float() * F.silu(gate.float())).abs().amax())

    return torch.compile(_swiglu_bf16, fullgraph=True, dynamic=True)


def _standard_dynamic_scale(
    projections: torch.Tensor,
    activation_fn: InputActivation,
) -> torch.Tensor:
    with torch.no_grad():
        return _compiled_standard_dynamic_scale(activation_fn, projections.dtype)(projections)


def _source_dynamic_scale(input: torch.Tensor, group_size: int | None) -> torch.Tensor:  # noqa: A002
    if group_size is None:
        from piper_kernels.weights.nvfp4 import triton as nvfp4_primitives  # noqa: PLC0415

        return nvfp4_primitives.dynamic_scale(input)
    from piper_kernels.linear.convrot.nvfp4 import triton as convrot_backend  # noqa: PLC0415

    return convrot_backend.dynamic_scale(input, group_size)


def _prepare_source_chunk(
    input_chunk: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    group_size: int | None,
    out: tuple[torch.Tensor, torch.Tensor],
    *,
    high_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_size is None:
        from piper_kernels.linear.nvfp4 import triton as nvfp4_backend  # noqa: PLC0415

        return nvfp4_backend.prepare_static_out(
            input_chunk,
            per_tensor_scale,
            out,
            high_first=high_first,
        )
    from piper_kernels.linear.convrot.nvfp4 import triton as convrot_backend  # noqa: PLC0415

    return convrot_backend.prepare_static_out(
        input_chunk,
        per_tensor_scale,
        group_size,
        out,
        high_first=high_first,
    )


def _prepare_activated_chunk(
    projections: torch.Tensor,
    activation_fn: InputActivation,
    activation_per_tensor_scale: torch.Tensor | None,
    dynamic_activation_scale: bool,
    group_size: int | None,
    *,
    high_first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_size is None:
        from piper_kernels.linear.nvfp4 import triton as nvfp4_backend  # noqa: PLC0415

        per_tensor_scale = (
            _standard_dynamic_scale(projections, activation_fn)
            if dynamic_activation_scale
            else activation_per_tensor_scale
        )
        if per_tensor_scale is None:
            return nvfp4_backend.prepare_static(
                projections,
                None,
                activation_fn=activation_fn,
                high_first=high_first,
            )
        qdata, scale = nvfp4_backend._prepare_static_storage(
            projections,
            per_tensor_scale,
            activation_fn=activation_fn,
            high_first=high_first,
        )
        return qdata, scale, per_tensor_scale

    if activation_fn == "swiglu":
        from . import triton as ffn_triton  # noqa: PLC0415

        return ffn_triton.prepare_convrot_nvfp4_swiglu(
            projections,
            activation_per_tensor_scale,
            dynamic_activation_scale,
            group_size,
            high_first,
        )

    from piper_kernels.linear.convrot.nvfp4 import triton as convrot_backend  # noqa: PLC0415

    validated_input = convrot_backend._validate_input(projections, group_size, "gelu_tanh")
    per_tensor_scale = (
        convrot_backend._prepare_dynamic_scale(
            validated_input,
            group_size,
            activation_fn="gelu_tanh",
        )
        if dynamic_activation_scale
        else activation_per_tensor_scale
    )
    if per_tensor_scale is None:
        per_tensor_scale = projections.new_ones((), dtype=torch.float32)
    qdata, scale = convrot_backend._prepare_static_storage(
        validated_input,
        per_tensor_scale,
        group_size,
        activation_fn="gelu_tanh",
        high_first=high_first,
    )
    return qdata, scale, per_tensor_scale


def _project_affine_chunk(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    input_per_tensor_scale: torch.Tensor,
    linear: LinearOperands,
    rows: int,
    output: torch.Tensor,
) -> None:
    from piper_kernels.linear.nvfp4 import _projection as nvfp4_projection  # noqa: PLC0415

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
) -> tuple[torch.Tensor, ...]:
    dynamic_scale: torch.Tensor | None = None
    unit_scale: torch.Tensor | None = None
    group_size = sources[0].group_size
    scales = []
    for source in sources:
        if source.dynamic_activation_scale:
            if dynamic_scale is None:
                dynamic_scale = _source_dynamic_scale(input, group_size)
            scales.append(dynamic_scale)
        elif source.activation_per_tensor_scale is not None:
            scales.append(source.activation_per_tensor_scale)
        else:
            if unit_scale is None:
                unit_scale = input.new_ones((), dtype=torch.float32)
            scales.append(unit_scale)
    return tuple(scales)


def run_chunked_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    sources: tuple[LinearOperands, ...],
    down: LinearOperands,
    activation_fn: InputActivation,
    chunk_rows: int,
    *,
    residual_add: ResidualAdd | None = None,
) -> torch.Tensor:
    """Project, activate, and down-project row chunks with topology-sized workspaces."""
    dimensions = validate_ffn(input, sources, down, activation_fn, chunk_rows)
    if any(not isinstance(dimension, int) for dimension in dimensions):
        raise ValueError("NVFP4 FFN execution requires concrete dimensions")
    rows, intermediate_features, output_features = cast(tuple[int, int, int], dimensions)
    if residual_add is not None:
        validate_residual_add(input, residual_add, output_features)

    from piper_kernels.linear.nvfp4._storage import prepare_activation_storage  # noqa: PLC0415

    leading_shape = input.shape[:-1]
    input_features = input.shape[-1]
    input_2d = input.view(rows, input_features)
    source_per_tensor_scales = _source_per_tensor_scales(input, sources)
    shared_input_preparation = all(source.dynamic_activation_scale for source in sources) or (
        all(not source.dynamic_activation_scale for source in sources)
        and all(
            same_tensor_storage(source_per_tensor_scales[0], per_tensor_scale)
            for per_tensor_scale in source_per_tensor_scales[1:]
        )
    )
    output = (
        torch.empty((*leading_shape, output_features), device=input.device, dtype=input.dtype)
        if residual_add is None
        else residual_add.output
    )
    output_2d = output.view(rows, output_features)
    workspace_rows = min(rows, chunk_rows)
    source_storage = prepare_activation_storage(
        input,
        workspace_rows,
        input_features,
        initialize_padding=False,
    )
    projection_workspace = torch.empty(
        (workspace_rows, len(sources) * intermediate_features),
        device=input.device,
        dtype=input.dtype,
    )
    projected_workspace = None
    apply_residual = None
    if residual_add is not None:
        from piper_kernels._residual.triton import apply_residual_add  # noqa: PLC0415

        apply_residual = apply_residual_add
        projection_columns = projection_workspace.shape[1]
        projected_workspace = (
            projection_workspace[:, :output_features]
            if output_features <= projection_columns
            else torch.empty(
                (workspace_rows, output_features),
                device=input.device,
                dtype=input.dtype,
            )
        )

    source_group_size = sources[0].group_size
    source_high_first = sources[0].high_first
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        chunk_row_count = stop - start
        projections = projection_workspace[:chunk_row_count]
        first_input_qdata, first_input_scale = _prepare_source_chunk(
            input_2d[start:stop],
            source_per_tensor_scales[0],
            source_group_size,
            source_storage,
            high_first=source_high_first,
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
                else _prepare_source_chunk(
                    input_2d[start:stop],
                    source_per_tensor_scales[1],
                    source_group_size,
                    source_storage,
                    high_first=source_high_first,
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
        down_qdata, down_scale, down_per_tensor_scale = _prepare_activated_chunk(
            projections,
            activation_fn,
            down.activation_per_tensor_scale,
            down.dynamic_activation_scale,
            down.group_size,
            high_first=down.high_first,
        )
        if residual_add is None:
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
        assert apply_residual is not None
        projected = projected_workspace[:chunk_row_count]
        _project_affine_chunk(
            down_qdata,
            down_scale,
            down_per_tensor_scale,
            down,
            chunk_row_count,
            projected,
        )
        apply_residual(projected, residual_add, start)
    return output


@torch.library.custom_op("piper_kernels::nvfp4_ffn", mutates_args=())
def nvfp4_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    activation_fn: str,
    chunk_rows: int,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_weight_per_tensor_scale: torch.Tensor | None = None,
    gate_activation_per_tensor_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    gate_dynamic_activation_scale: bool = False,
    gate_group_size: int | None = None,
    gate_high_first: bool = False,
) -> torch.Tensor:
    """Run a bounded-workspace standard or ConvRot NVFP4 GELU-tanh or SwiGLU FFN."""
    activation, sources, down = _build_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_group_size,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_group_size,
        down_high_first,
        activation_fn,
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_group_size,
        gate_high_first,
    )
    return run_chunked_ffn(input, sources, down, activation, chunk_rows)


@nvfp4_ffn_op.register_fake
def _nvfp4_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    activation_fn: str,
    chunk_rows: int,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_weight_per_tensor_scale: torch.Tensor | None = None,
    gate_activation_per_tensor_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    gate_dynamic_activation_scale: bool = False,
    gate_group_size: int | None = None,
    gate_high_first: bool = False,
) -> torch.Tensor:
    activation, sources, down = _build_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_group_size,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_group_size,
        down_high_first,
        activation_fn,
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_group_size,
        gate_high_first,
    )
    output_features = validate_ffn(input, sources, down, activation, chunk_rows)[2]
    return input.new_empty((*input.shape[:-1], output_features))


@torch.library.custom_op(
    "piper_kernels::nvfp4_ffn_update_",
    mutates_args=("output",),
)
def nvfp4_ffn_update_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    activation_fn: str,
    output: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
    base: torch.Tensor | None = None,
    update_gate: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_weight_per_tensor_scale: torch.Tensor | None = None,
    gate_activation_per_tensor_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    gate_dynamic_activation_scale: bool = False,
    gate_group_size: int | None = None,
    gate_high_first: bool = False,
) -> None:
    """Update ``output`` in place with a gated NVFP4/ConvRot-NVFP4 FFN residual sum.

    When ``base`` and ``update_gate`` are omitted, adds ``ffn_gate[i] * ffn(input)``
    into ``output`` (which serves as ``base`` without aliasing another operand).
    When provided, writes ``base + update_gate[i] * output + ffn_gate[i] * ffn(input)``
    over ``output``.
    """
    activation, sources, down = _build_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_group_size,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_group_size,
        down_high_first,
        activation_fn,
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_group_size,
        gate_high_first,
    )
    residual_add = _build_residual_add(
        output,
        base,
        update_gate,
        ffn_gate,
        gate_indices,
        python_indexing,
    )
    run_chunked_ffn(input, sources, down, activation, chunk_rows, residual_add=residual_add)


@nvfp4_ffn_update_op.register_fake
def _nvfp4_ffn_update_op_fake(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_weight_per_tensor_scale: torch.Tensor | None,
    up_activation_per_tensor_scale: torch.Tensor | None,
    up_bias: torch.Tensor | None,
    up_dynamic_activation_scale: bool,
    up_group_size: int | None,
    up_high_first: bool,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_weight_per_tensor_scale: torch.Tensor | None,
    down_activation_per_tensor_scale: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    down_dynamic_activation_scale: bool,
    down_group_size: int | None,
    down_high_first: bool,
    activation_fn: str,
    output: torch.Tensor,
    ffn_gate: torch.Tensor,
    gate_indices: torch.Tensor,
    python_indexing: bool,
    chunk_rows: int,
    base: torch.Tensor | None = None,
    update_gate: torch.Tensor | None = None,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_weight_per_tensor_scale: torch.Tensor | None = None,
    gate_activation_per_tensor_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    gate_dynamic_activation_scale: bool = False,
    gate_group_size: int | None = None,
    gate_high_first: bool = False,
) -> None:
    activation, sources, down = _build_operands(
        up_weight_qdata,
        up_weight_scale,
        up_weight_per_tensor_scale,
        up_activation_per_tensor_scale,
        up_bias,
        up_dynamic_activation_scale,
        up_group_size,
        up_high_first,
        down_weight_qdata,
        down_weight_scale,
        down_weight_per_tensor_scale,
        down_activation_per_tensor_scale,
        down_bias,
        down_dynamic_activation_scale,
        down_group_size,
        down_high_first,
        activation_fn,
        gate_weight_qdata,
        gate_weight_scale,
        gate_weight_per_tensor_scale,
        gate_activation_per_tensor_scale,
        gate_bias,
        gate_dynamic_activation_scale,
        gate_group_size,
        gate_high_first,
    )
    output_features = validate_ffn(input, sources, down, activation, chunk_rows)[2]
    residual_add = _build_residual_add(
        output,
        base,
        update_gate,
        ffn_gate,
        gate_indices,
        python_indexing,
    )
    validate_residual_add(input, residual_add, output_features)


__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "LinearOperands",
    "default_chunk_rows",
    "nvfp4_ffn_op",
    "nvfp4_ffn_update_op",
    "run_chunked_ffn",
    "validate_ffn",
]
