"""Bounded-workspace ConvRot INT8 feed-forward network runner and custom ops.

This module imports without Triton or NVFP4. Runtime dispatch selects the active
ConvRot INT8 linear backend once per call and reuses chunk workspaces across
both ``gelu_tanh`` and ``swiglu`` activations.
"""

# Explicit custom-op schemas take flat tensor arguments required by torch.library.
# ruff: noqa: PLR0913, PLR0917

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from piper_kernels._input_activations import (
    InputActivation,
    input_activation_width,
    parse_input_activation,
)
from piper_kernels._residual import ResidualAdd, validate_residual_add
from piper_kernels.linear import _bias
from piper_kernels.linear._storage import same_tensor_storage
from piper_kernels.linear.convrot.int8 import _backend
from piper_kernels.weights.convrot.int8._quantization import (
    validate_activation_scale,
    validate_storage,
)

DEFAULT_CHUNK_ROWS = 4_096
_GELU_WORKSPACE_LIMIT_BYTES = 1_024 * 1_024 * 1_024
_CHUNK_ROW_GRANULARITY = 128


@dataclass(frozen=True, slots=True)
class LinearOperands:
    """Storage, affine, and preparation operands for one ConvRot INT8 projection."""

    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None
    group_size: int
    input_scale: torch.Tensor | None


def default_chunk_rows(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    down_weight_qdata: torch.Tensor,
    activation_fn: InputActivation,
    *,
    adds_residual: bool,
) -> int:
    """Return bounded row chunk size for ConvRot INT8 FFNs.

    SwiGLU uses a fixed chunk size; GELU bounds reusable scratch to 1 GiB while
    amortizing per-chunk launches and accounting for standalone residual scratch
    when the output is wider than the intermediate projection.
    """
    if activation_fn == "swiglu":
        return DEFAULT_CHUNK_ROWS
    input_features = up_weight_qdata.shape[1]
    intermediate_features = up_weight_qdata.shape[0]
    output_features = down_weight_qdata.shape[0]
    if not all(
        isinstance(features, int)
        for features in (input_features, intermediate_features, output_features)
    ):
        return DEFAULT_CHUNK_ROWS
    projection_bytes = intermediate_features * input.element_size()
    preparation_bytes = max(input_features, intermediate_features) * torch.int8.itemsize
    scale_bytes = torch.float32.itemsize
    projected_output_bytes = (
        output_features * input.element_size()
        if adds_residual and output_features > intermediate_features
        else 0
    )
    bytes_per_row = projection_bytes + preparation_bytes + scale_bytes + projected_output_bytes
    rows = _GELU_WORKSPACE_LIMIT_BYTES // bytes_per_row
    return max(_CHUNK_ROW_GRANULARITY, rows // _CHUNK_ROW_GRANULARITY * _CHUNK_ROW_GRANULARITY)


def _validate_bias(
    bias: torch.Tensor | None,
    *,
    features: int,
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    name: str,
) -> None:
    if bias is None:
        return
    if (
        bias.shape != (features,)
        or bias.device != input.device
        or bias.layout is not torch.strided
        or not bias.is_contiguous()
    ):
        raise ValueError(
            f"chunked ConvRot INT8 {name} bias must be a contiguous strided tensor "
            f"with shape ({features},) on {input.device}"
        )
    _bias.validate_dtype(bias, f"chunked ConvRot INT8 {name}")


def _build_operands(
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    activation_fn: str,
    gate_weight_qdata: torch.Tensor | None,
    gate_weight_scale: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    gate_group_size: int | None,
    up_input_scale: torch.Tensor | None,
    gate_input_scale: torch.Tensor | None,
    down_input_scale: torch.Tensor | None,
) -> tuple[InputActivation, tuple[LinearOperands, ...], LinearOperands]:
    activation = parse_input_activation(activation_fn, name="ConvRot INT8 FFN")
    up = LinearOperands(
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        up_input_scale,
    )
    down = LinearOperands(
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        down_input_scale,
    )
    if activation == "gelu_tanh":
        if any(
            value is not None
            for value in (
                gate_weight_qdata,
                gate_weight_scale,
                gate_bias,
                gate_group_size,
                gate_input_scale,
            )
        ):
            raise ValueError("ConvRot INT8 gelu_tanh FFN does not accept gate projection operands")
        return activation, (up,), down
    if (
        gate_weight_qdata is None
        or gate_weight_scale is None
        or gate_group_size is None
        or isinstance(gate_group_size, bool)
        or not isinstance(gate_group_size, (int, torch.SymInt))
    ):
        raise ValueError(
            "ConvRot INT8 swiglu FFN requires gate_weight_qdata, gate_weight_scale, "
            "and integer gate_group_size"
        )
    gate = LinearOperands(
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        gate_group_size,
        gate_input_scale,
    )
    return activation, (up, gate), down


def validate_ffn(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    sources: tuple[LinearOperands, ...],
    down: LinearOperands,
    activation_fn: InputActivation,
    chunk_rows: int,
) -> tuple[int, int, int]:
    """Validate bounded ConvRot INT8 FFN metadata and return its logical dimensions."""
    if input.ndim == 0 or input.layout is not torch.strided or not input.is_contiguous():
        raise ValueError("ConvRot INT8 FFN input must be a non-scalar contiguous strided tensor")
    rows = math.prod(input.shape[:-1])
    if isinstance(rows, int) and rows < 1:
        raise ValueError("ConvRot INT8 FFN requires at least one input row")
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, (int, torch.SymInt))
        or chunk_rows < 1
    ):
        raise ValueError("ConvRot INT8 FFN chunk_rows must be a positive integer")
    expected_sources = input_activation_width(activation_fn)
    if len(sources) != expected_sources:
        raise ValueError(
            f"ConvRot INT8 {activation_fn} FFN requires {expected_sources} source projection(s)"
        )
    source_group_size = sources[0].group_size
    if any(source.group_size != source_group_size for source in sources[1:]):
        raise ValueError("ConvRot INT8 FFN source projections must share one group size")

    for linear in (*sources, down):
        validate_storage(
            linear.weight_qdata,
            linear.weight_scale,
            linear.group_size,
            input.dtype,
        )
        if linear.weight_qdata.device != input.device or linear.weight_scale.device != input.device:
            raise ValueError("ConvRot INT8 FFN operands must share a device")
        validate_activation_scale(linear.input_scale, input.device)

    input_features = sources[0].weight_qdata.shape[1]
    intermediate_features = sources[0].weight_qdata.shape[0]
    output_features = down.weight_qdata.shape[0]
    if input.shape[-1] != input_features:
        raise ValueError(
            f"ConvRot INT8 FFN input has {input.shape[-1]} features, expected {input_features}"
        )
    if any(
        source.weight_qdata.shape != (intermediate_features, input_features)
        for source in sources[1:]
    ):
        raise ValueError("ConvRot INT8 FFN source projections must have matching shapes")
    if down.weight_qdata.shape[1] != intermediate_features:
        raise ValueError("ConvRot INT8 FFN down projection must consume the intermediate width")
    for index, source in enumerate(sources):
        _validate_bias(
            source.bias,
            features=intermediate_features,
            input=input,
            name=f"source {index}",
        )
    _validate_bias(down.bias, features=output_features, input=input, name="down")
    differentiable = (
        input,
        *(linear.weight_scale for linear in (*sources, down)),
        *(linear.bias for linear in (*sources, down) if linear.bias is not None),
    )
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in differentiable):
        raise RuntimeError("ConvRot INT8 FFN is inference-only and does not support autograd")
    return input_features, intermediate_features, output_features


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
    input_features, intermediate_features, output_features = validate_ffn(
        input,
        sources,
        down,
        activation_fn,
        chunk_rows,
    )
    if residual_add is not None:
        validate_residual_add(input, residual_add, output_features)
    backend = _backend.require_linear_backend(input)
    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)
    capacity = min(rows, chunk_rows)
    input_2d = input.view(rows, input_features)
    output = (
        torch.empty((*leading_shape, output_features), device=input.device, dtype=input.dtype)
        if residual_add is None
        else residual_add.output
    )
    output_2d = output.view(rows, output_features)
    projection_features = len(sources) * intermediate_features
    projection_workspace = torch.empty(
        (capacity, projection_features),
        device=input.device,
        dtype=input.dtype,
    )
    # Reuse dead up-projection storage if the backend needs a materialized down
    # projection. A fused residual epilogue ignores this view and allocates none.
    projection_scratch = (
        projection_workspace.view(-1)[: capacity * output_features].view(capacity, output_features)
        if residual_add is not None and output_features <= projection_features
        else None
    )
    prepared_storage = torch.empty(
        capacity * max(input_features, intermediate_features),
        device=input.device,
        dtype=torch.int8,
    )
    scale_storage = torch.empty(capacity, device=input.device, dtype=torch.float32)
    shared_source = all(
        same_tensor_storage(sources[0].input_scale, source.input_scale) for source in sources[1:]
    )
    second_projection = (
        (sources[1].weight_qdata, sources[1].weight_scale, sources[1].bias)
        if shared_source and len(sources) == 2
        else None
    )
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        chunk_row_count = stop - start
        prepared_input = prepared_storage[: chunk_row_count * input_features].view(
            chunk_row_count,
            input_features,
        )
        prepared_scale = scale_storage[:chunk_row_count]
        projections = projection_workspace[:chunk_row_count]
        if shared_source:
            source = sources[0]
            backend.prepare_input(
                input_2d[start:stop],
                source.group_size,
                input_scale=source.input_scale,
                out=(prepared_input, prepared_scale),
            )
            backend.linear_prepared(
                prepared_input,
                prepared_scale,
                source.weight_qdata,
                source.weight_scale,
                source.bias,
                input.dtype,
                out=projections,
                second_projection=second_projection,
            )
        else:
            for index, source in enumerate(sources):
                backend.prepare_input(
                    input_2d[start:stop],
                    source.group_size,
                    input_scale=source.input_scale,
                    out=(prepared_input, prepared_scale),
                )
                offset = index * intermediate_features
                backend.linear_prepared(
                    prepared_input,
                    prepared_scale,
                    source.weight_qdata,
                    source.weight_scale,
                    source.bias,
                    input.dtype,
                    out=projections[:, offset : offset + intermediate_features],
                )

        prepared_activation = prepared_storage[: chunk_row_count * intermediate_features].view(
            chunk_row_count,
            intermediate_features,
        )
        backend.prepare_input(
            projections,
            down.group_size,
            activation_fn=activation_fn,
            input_scale=down.input_scale,
            out=(prepared_activation, prepared_scale),
        )
        if residual_add is None:
            backend.linear_prepared(
                prepared_activation,
                prepared_scale,
                down.weight_qdata,
                down.weight_scale,
                down.bias,
                input.dtype,
                out=output_2d[start:stop],
            )
        else:
            backend.linear_prepared_residual(
                prepared_activation,
                prepared_scale,
                down.weight_qdata,
                down.weight_scale,
                down.bias,
                residual_add,
                row_offset=start,
                projection_scratch=None
                if projection_scratch is None
                else projection_scratch[:chunk_row_count],
            )
    return output


@torch.library.custom_op("piper_kernels::convrot_int8_ffn", mutates_args=())
def convrot_int8_ffn_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    activation_fn: str,
    chunk_rows: int,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    gate_group_size: int | None = None,
    up_input_scale: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a bounded-workspace ConvRot INT8 GELU-tanh or SwiGLU FFN."""
    activation, sources, down = _build_operands(
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        activation_fn,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        gate_group_size,
        up_input_scale,
        gate_input_scale,
        down_input_scale,
    )
    return run_chunked_ffn(input, sources, down, activation, chunk_rows)


@convrot_int8_ffn_op.register_fake
def _convrot_int8_ffn_op_fake(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
    activation_fn: str,
    chunk_rows: int,
    gate_weight_qdata: torch.Tensor | None = None,
    gate_weight_scale: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    gate_group_size: int | None = None,
    up_input_scale: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    activation, sources, down = _build_operands(
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        activation_fn,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        gate_group_size,
        up_input_scale,
        gate_input_scale,
        down_input_scale,
    )
    output_features = validate_ffn(input, sources, down, activation, chunk_rows)[2]
    return input.new_empty((*input.shape[:-1], output_features))


@torch.library.custom_op(
    "piper_kernels::convrot_int8_ffn_update_",
    mutates_args=("output",),
)
def convrot_int8_ffn_update_op(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
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
    gate_bias: torch.Tensor | None = None,
    gate_group_size: int | None = None,
    up_input_scale: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> None:
    """Update ``output`` in place with a gated ConvRot INT8 FFN residual sum.

    When ``base`` and ``update_gate`` are omitted, adds ``ffn_gate[i] * ffn(input)``
    into ``output`` (which serves as ``base`` without aliasing another operand).
    When provided, writes ``base + update_gate[i] * output + ffn_gate[i] * ffn(input)``
    over ``output``.
    """
    activation, sources, down = _build_operands(
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        activation_fn,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        gate_group_size,
        up_input_scale,
        gate_input_scale,
        down_input_scale,
    )
    residual_add = ResidualAdd.from_update(
        output,
        base,
        update_gate,
        ffn_gate,
        gate_indices,
        python_indexing,
        name="ConvRot INT8 FFN",
    )
    run_chunked_ffn(input, sources, down, activation, chunk_rows, residual_add=residual_add)


@convrot_int8_ffn_update_op.register_fake
def _convrot_int8_ffn_update_op_fake(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    up_weight_qdata: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_bias: torch.Tensor | None,
    up_group_size: int,
    down_weight_qdata: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_bias: torch.Tensor | None,
    down_group_size: int,
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
    gate_bias: torch.Tensor | None = None,
    gate_group_size: int | None = None,
    up_input_scale: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> None:
    activation, sources, down = _build_operands(
        up_weight_qdata,
        up_weight_scale,
        up_bias,
        up_group_size,
        down_weight_qdata,
        down_weight_scale,
        down_bias,
        down_group_size,
        activation_fn,
        gate_weight_qdata,
        gate_weight_scale,
        gate_bias,
        gate_group_size,
        up_input_scale,
        gate_input_scale,
        down_input_scale,
    )
    output_features = validate_ffn(input, sources, down, activation, chunk_rows)[2]
    residual_add = ResidualAdd.from_update(
        output,
        base,
        update_gate,
        ffn_gate,
        gate_indices,
        python_indexing,
        name="ConvRot INT8 FFN",
    )
    validate_residual_add(input, residual_add, output_features)


__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "LinearOperands",
    "convrot_int8_ffn_op",
    "convrot_int8_ffn_update_op",
    "default_chunk_rows",
    "run_chunked_ffn",
    "validate_ffn",
]
