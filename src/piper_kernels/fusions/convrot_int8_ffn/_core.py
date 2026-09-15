"""Topology-aware bounded execution for ConvRot INT8 feed-forward networks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from piper_kernels._input_activations import InputActivation, input_activation_width
from piper_kernels.fusions.ffn import triton as indexed_updates
from piper_kernels.linear import _bias
from piper_kernels.linear._storage import same_tensor_storage
from piper_kernels.linear.convrot.int8 import _backend
from piper_kernels.weights.convrot.int8._quantization import (
    validate_activation_scale,
    validate_storage,
)

DEFAULT_CHUNK_ROWS = 4_096


@dataclass(frozen=True, slots=True)
class LinearOperands:
    """Storage, affine, and preparation operands for one ConvRot INT8 projection."""

    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None
    group_size: int
    input_scale: torch.Tensor | None


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
    if math.prod(input.shape[:-1]) < 1:
        raise ValueError("ConvRot INT8 FFN requires at least one input row")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
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
    gated_updates: indexed_updates.IndexedGatedUpdates | None = None,
) -> torch.Tensor:
    """Project, activate, and down-project row chunks with topology-sized workspaces."""
    input_features, intermediate_features, output_features = validate_ffn(
        input,
        sources,
        down,
        activation_fn,
        chunk_rows,
    )
    backend = _backend.require_linear_backend(input)
    leading_shape = input.shape[:-1]
    rows = math.prod(leading_shape)
    capacity = min(rows, chunk_rows)
    input_2d = input.view(rows, input_features)
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
    output_2d = output.view(rows, output_features)
    base_2d = None if gated_updates is None else gated_updates.base.view(rows, output_features)
    projection_features = len(sources) * intermediate_features
    projection_workspace = torch.empty(
        (capacity, projection_features),
        device=input.device,
        dtype=input.dtype,
    )
    projected = None
    if gated_updates is not None:
        projected = (
            projection_workspace.reshape(-1)[: capacity * output_features].view(
                capacity,
                output_features,
            )
            if output_features <= projection_features
            else torch.empty(
                (capacity, output_features),
                device=input.device,
                dtype=input.dtype,
            )
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
            first, second = sources
            backend.prepare_input(
                input_2d[start:stop],
                first.group_size,
                input_scale=first.input_scale,
                out=(prepared_input, prepared_scale),
            )
            backend.linear_prepared(
                prepared_input,
                prepared_scale,
                first.weight_qdata,
                first.weight_scale,
                first.bias,
                input.dtype,
                out=projections[:, :intermediate_features],
            )
            # Reuse the bounded preparation buffers for the second source.
            backend.prepare_input(
                input_2d[start:stop],
                second.group_size,
                input_scale=second.input_scale,
                out=(prepared_input, prepared_scale),
            )
            backend.linear_prepared(
                prepared_input,
                prepared_scale,
                second.weight_qdata,
                second.weight_scale,
                second.bias,
                input.dtype,
                out=projections[:, intermediate_features:],
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
        output_chunk = output_2d[start:stop] if projected is None else projected[:chunk_row_count]
        backend.linear_prepared(
            prepared_activation,
            prepared_scale,
            down.weight_qdata,
            down.weight_scale,
            down.bias,
            input.dtype,
            out=output_chunk,
        )
        if gated_updates is not None:
            assert base_2d is not None
            assert update_layout is not None
            indexed_updates.apply_indexed_gated_updates(
                output_chunk,
                base_2d[start:stop],
                output_2d[start:stop],
                gated_updates,
                update_layout,
                start,
            )
    return output


__all__ = ["DEFAULT_CHUNK_ROWS", "LinearOperands", "run_chunked_ffn", "validate_ffn"]
