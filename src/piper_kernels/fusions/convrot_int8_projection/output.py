"""ConvRot INT8 output projection shared by dense and sparse attention pipelines."""

from __future__ import annotations

import torch

from piper_kernels._residual import ResidualAdd
from piper_kernels.fusions.attention._output import ChunkProjector
from piper_kernels.linear import _bias
from piper_kernels.linear.convrot.int8._interfaces import LinearBackend
from piper_kernels.weights.convrot.int8._quantization import validate_storage


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
    residual_add: ResidualAdd | None = None,
) -> tuple[ChunkProjector, tuple[torch.Tensor, ...]]:
    """Allocate bounded preparation scratch and project each batch of a chunk.

    Without ``residual_add``, each batch projects into its final output rows, which
    attention may occupy: preparation reads the whole batch into separate INT8
    storage before the projection writes. A validated ``residual_add`` has no
    reusable update, and the pipeline output is its base: the linear backend adds
    ``residual_add.gate[index] * projection`` into each row with one rounding.
    Those rows hold live data, so attention must not share their storage.
    """
    input_features = weight_qdata.shape[1]
    capacity = min(sequence_length, query_chunk_rows)
    device = weight_qdata.device
    prepared_input = torch.empty((capacity, input_features), device=device, dtype=torch.int8)
    prepared_scale = torch.empty(capacity, device=device, dtype=torch.float32)

    def project_chunk(
        attention_chunk: torch.Tensor,
        output: torch.Tensor,
        start: int,
        rows: int,
    ) -> None:
        prepared = (prepared_input[:rows], prepared_scale[:rows])
        for batch_index in range(attention_chunk.shape[0]):
            backend.prepare_input(
                attention_chunk[batch_index, :rows].reshape(rows, input_features),
                group_size,
                input_scale=output_input_scale,
                out=prepared,
            )
            if residual_add is None:
                backend.linear_prepared(
                    *prepared,
                    weight_qdata,
                    weight_scale,
                    bias,
                    output.dtype,
                    out=output[batch_index, start : start + rows],
                )
            else:
                backend.linear_prepared_residual(
                    *prepared,
                    weight_qdata,
                    weight_scale,
                    bias,
                    residual_add,
                    row_offset=batch_index * sequence_length + start,
                )

    residual_tensors = (
        () if residual_add is None else (residual_add.gate, residual_add.gate_indices)
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
            *residual_tensors,
        )
        if tensor is not None
    )
    return project_chunk, retained
