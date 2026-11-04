"""ConvRot INT8 output projection shared by dense and sparse attention pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from piper_kernels.fusions.attention._output import ChunkProjector
from piper_kernels.linear import _bias
from piper_kernels.linear.convrot.int8._interfaces import LinearBackend
from piper_kernels.weights.convrot.int8._quantization import validate_storage

if TYPE_CHECKING:
    from piper_kernels.fusions.residual_updates.triton import (
        IndexedGatedUpdateLayout,
        IndexedGatedUpdates,
    )


def validate_output_projection(
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    *,
    input_features: int,
    device: torch.device,
    output_dtype: torch.dtype,
    name: str,
) -> int:
    """Check shared projection metadata and return its output width without tensor work."""
    validate_storage(weight_qdata, weight_scale, group_size, output_dtype)
    output_features = weight_qdata.shape[0]
    if weight_qdata.shape[1] != input_features or output_features < 1:
        raise ValueError(f"{name} projection weight must consume all attention heads")
    if weight_qdata.device != device:
        raise ValueError(f"{name} projection operands must share a device")
    if bias is not None:
        _bias.validate_dtype(bias, name)
        if (
            bias.shape != (output_features,)
            or bias.device != device
            or bias.layout is not torch.strided
            or not bias.is_contiguous()
        ):
            raise ValueError(f"{name} bias must be contiguous with one value per output feature")
    if torch.is_grad_enabled() and (
        weight_scale.requires_grad or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError(f"{name} projection is inference-only and does not support autograd")
    return output_features


def _project_batch_rows(
    attention_rows: torch.Tensor,
    output_rows: torch.Tensor,
    prepared_input: torch.Tensor,
    prepared_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    backend: LinearBackend,
    output_input_scale: torch.Tensor | None,
) -> None:
    """Prepare one batch's attention rows, then project them into ``output_rows``.

    All attention reads finish in prepare_input before linear_prepared writes, so
    the two views may share storage; the INT8 preparation buffers are separate.
    """
    backend.prepare_input(
        attention_rows,
        group_size,
        activation_fn=None,
        input_scale=output_input_scale,
        out=(prepared_input, prepared_scale),
    )
    backend.linear_prepared(
        prepared_input,
        prepared_scale,
        weight_qdata,
        weight_scale,
        bias,
        output_rows.dtype,
        out=output_rows,
    )


def _project_attention_chunk(  # noqa: PLR0913, PLR0917
    attention_chunk: torch.Tensor,
    output: torch.Tensor,
    start: int,
    rows: int,
    prepared_input: torch.Tensor,
    prepared_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    backend: LinearBackend,
    output_input_scale: torch.Tensor | None = None,
) -> None:
    """Project each batch into its final output rows, which attention may occupy."""
    input_features = weight_qdata.shape[1]
    for batch_index in range(attention_chunk.shape[0]):
        _project_batch_rows(
            attention_chunk[batch_index, :rows].reshape(rows, input_features),
            output[batch_index, start : start + rows],
            prepared_input[:rows],
            prepared_scale[:rows],
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            backend,
            output_input_scale,
        )


def prepare_chunk_projector(
    sequence_length: int,
    query_chunk_rows: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    *,
    backend: LinearBackend,
    output_input_scale: torch.Tensor | None = None,
) -> tuple[ChunkProjector, tuple[torch.Tensor, ...]]:
    """Allocate bounded scratch for validated projection inputs on a consumer stream."""
    input_features = weight_qdata.shape[1]
    capacity = min(sequence_length, query_chunk_rows)
    prepared_input = torch.empty(
        (capacity, input_features),
        device=weight_qdata.device,
        dtype=torch.int8,
    )
    prepared_scale = torch.empty(
        capacity,
        device=weight_qdata.device,
        dtype=torch.float32,
    )

    def project_chunk(
        attention_chunk: torch.Tensor,
        output: torch.Tensor,
        start: int,
        rows: int,
    ) -> None:
        _project_attention_chunk(
            attention_chunk,
            output,
            start,
            rows,
            prepared_input,
            prepared_scale,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            backend,
            output_input_scale,
        )

    retained = tuple(
        tensor
        for tensor in (
            prepared_input,
            prepared_scale,
            weight_qdata,
            weight_scale,
            bias,
            output_input_scale,
        )
        if tensor is not None
    )
    return project_chunk, retained


def prepare_residual_add_projector(
    sequence_length: int,
    query_chunk_rows: int,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    updates: IndexedGatedUpdates,
    layout: IndexedGatedUpdateLayout,
    *,
    backend: LinearBackend,
    output_input_scale: torch.Tensor | None = None,
) -> tuple[ChunkProjector, tuple[torch.Tensor, ...]]:
    """Project each chunk into bounded scratch, then add its gated rows into the output.

    ``updates`` has no first update: the pipeline output is ``updates.base``, and each
    row becomes ``base + updates.gate[index] * projection`` with one rounding.
    Output rows hold live data, so attention must not share their storage.
    """
    # Triton loads only when the gated update runs.
    from piper_kernels.fusions.residual_updates import triton as indexed_updates  # noqa: PLC0415

    input_features = weight_qdata.shape[1]
    output_features = weight_qdata.shape[0]
    capacity = min(sequence_length, query_chunk_rows)
    device = weight_qdata.device
    prepared_input = torch.empty((capacity, input_features), device=device, dtype=torch.int8)
    prepared_scale = torch.empty(capacity, device=device, dtype=torch.float32)
    projected = torch.empty((capacity, output_features), device=device, dtype=updates.base.dtype)

    def project_chunk(
        attention_chunk: torch.Tensor,
        output: torch.Tensor,
        start: int,
        rows: int,
    ) -> None:
        chunk_projected = projected[:rows]
        for batch_index in range(attention_chunk.shape[0]):
            _project_batch_rows(
                attention_chunk[batch_index, :rows].reshape(rows, input_features),
                chunk_projected,
                prepared_input[:rows],
                prepared_scale[:rows],
                weight_qdata,
                weight_scale,
                bias,
                group_size,
                backend,
                output_input_scale,
            )
            output_rows = output[batch_index, start : start + rows]
            indexed_updates.apply_indexed_gated_updates(
                chunk_projected,
                output_rows,
                output_rows,
                updates,
                layout,
                batch_index * sequence_length + start,
            )

    retained = tuple(
        tensor
        for tensor in (
            prepared_input,
            prepared_scale,
            projected,
            weight_qdata,
            weight_scale,
            bias,
            output_input_scale,
            updates.gate,
            updates.gate_indices,
        )
        if tensor is not None
    )
    return project_chunk, retained
