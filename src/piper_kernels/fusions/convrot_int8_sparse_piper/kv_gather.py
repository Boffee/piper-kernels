"""Async INT8 input gather overlapped with own-rank K/V sparse projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels._distributed import _comm as distributed_comm
from piper_kernels.attention.kernels.sparse_piper.layout import ValueFormat, padded_sequence_length
from piper_kernels.attention.sparse_piper_attention._contracts import validate_routing_mode
from piper_kernels.fusions.projected_qk._validation import resolve_head_dim
from piper_kernels.linear.convrot.int8 import _backend as convrot_linear_backend
from piper_kernels.linear.convrot.int8._backend import LinearBackend

from . import _backend
from . import key as key_ops
from . import value as value_ops
from ._interfaces import KeyOutput, ValueOutput
from ._layout import validate_block_lengths


@dataclass(frozen=True, slots=True)
class _KeyProjectionSpec:
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    norm_weight: torch.Tensor | None
    cos: torch.Tensor
    sin: torch.Tensor
    norm_epsilon: float
    routing_mode: int
    block_lengths: torch.Tensor | None
    bias: torch.Tensor | None
    head_dim: int | None


@dataclass(frozen=True, slots=True)
class _ValueProjectionSpec:
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    block_lengths: torch.Tensor | None
    head_dim: int
    bias: torch.Tensor | None
    storage_format: ValueFormat


def _project_key_range(
    key_backend: _backend.ProjectionBackend,
    full_qdata: torch.Tensor,
    full_scale: torch.Tensor,
    key_spec: _KeyProjectionSpec,
    key_outputs: KeyOutput,
    *,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    key_backend.project_key(
        full_qdata,
        full_scale,
        key_spec.weight_qdata,
        key_spec.weight_scale,
        key_spec.norm_weight,
        key_spec.cos,
        key_spec.sin,
        key_spec.norm_epsilon,
        key_spec.routing_mode,
        key_spec.block_lengths,
        out=key_outputs,
        bias=key_spec.bias,
        start_row_block=start_row_block,
        num_row_blocks=num_row_blocks,
    )


def _project_value_range(
    value_backend: _backend.ProjectionBackend,
    full_qdata: torch.Tensor,
    full_scale: torch.Tensor,
    value_spec: _ValueProjectionSpec,
    value_outputs: ValueOutput,
    *,
    start_row_block: int = 0,
    num_row_blocks: int | None = None,
) -> None:
    value_backend.project_value(
        full_qdata,
        full_scale,
        value_spec.weight_qdata,
        value_spec.weight_scale,
        value_spec.block_lengths,
        value_format=value_spec.storage_format,
        out=value_outputs,
        bias=value_spec.bias,
        start_row_block=start_row_block,
        num_row_blocks=num_row_blocks,
    )


def _launch_peer_kv_ranges(
    key_backend: _backend.ProjectionBackend,
    value_backend: _backend.ProjectionBackend,
    full_qdata: torch.Tensor,
    full_scale: torch.Tensor,
    key_spec: _KeyProjectionSpec,
    value_spec: _ValueProjectionSpec,
    key_outputs: KeyOutput,
    value_outputs: ValueOutput,
    key_block_range: tuple[int, int, int],
    value_block_range: tuple[int, int, int],
) -> None:
    key_start, key_own_blocks, key_total_blocks = key_block_range
    value_start, value_own_blocks, value_total_blocks = value_block_range
    for start_block, count in (
        (0, key_start),
        (key_start + key_own_blocks, key_total_blocks - (key_start + key_own_blocks)),
    ):
        if count > 0:
            _project_key_range(
                key_backend,
                full_qdata,
                full_scale,
                key_spec,
                key_outputs,
                start_row_block=start_block,
                num_row_blocks=count,
            )
    for start_block, count in (
        (0, value_start),
        (value_start + value_own_blocks, value_total_blocks - (value_start + value_own_blocks)),
    ):
        if count > 0:
            _project_value_range(
                value_backend,
                full_qdata,
                full_scale,
                value_spec,
                value_outputs,
                start_row_block=start_block,
                num_row_blocks=count,
            )


def _gather_batched_inputs(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    process_group: torch.distributed.ProcessGroup,
    linear_backend: LinearBackend,
    prepare_group_size: int,
    input_scale: torch.Tensor | None,
    full_qdata: torch.Tensor,
    full_scale: torch.Tensor,
) -> None:
    world_size = process_group.size()
    rank = process_group.rank()
    batch, local_sequence_length, input_features = input.shape
    gathered_qdata = torch.empty(
        (world_size * batch, local_sequence_length, input_features),
        device=input.device,
        dtype=torch.int8,
    )
    gathered_scale = torch.empty(
        (world_size * batch, local_sequence_length),
        device=input.device,
        dtype=torch.float32,
    )
    own_qdata = gathered_qdata[rank * batch : (rank + 1) * batch]
    own_scale = gathered_scale[rank * batch : (rank + 1) * batch]
    linear_backend.prepare_input(
        input.view(batch * local_sequence_length, input_features),
        prepare_group_size,
        input_scale=input_scale,
        out=(
            own_qdata.view(batch * local_sequence_length, input_features),
            own_scale.view(batch * local_sequence_length),
        ),
    )
    works = distributed_comm.start_coalesced_all_gather(
        process_group,
        [gathered_qdata, gathered_scale],
        [own_qdata, own_scale],
    )
    distributed_comm.wait_all(works)
    for rank_index in range(world_size):
        row_start = rank_index * local_sequence_length
        row_end = row_start + local_sequence_length
        batch_start = rank_index * batch
        batch_end = batch_start + batch
        full_qdata[:, row_start:row_end].copy_(gathered_qdata[batch_start:batch_end])
        full_scale[:, row_start:row_end].copy_(gathered_scale[batch_start:batch_end])


def _validate_distributed_comm(comm_group_size: int, comm_group_name: str) -> None:
    if (
        isinstance(comm_group_size, bool)
        or not isinstance(comm_group_size, int)
        or comm_group_size < 2
    ):
        raise ValueError("distributed sparse Piper KV gather requires comm_group_size >= 2")
    if not isinstance(comm_group_name, str):
        raise TypeError("distributed sparse Piper KV gather requires string comm_group_name")


def _prepare_kv_outputs(
    full_qdata: torch.Tensor,
    full_scale: torch.Tensor,
    key_spec: _KeyProjectionSpec,
    value_spec: _ValueProjectionSpec,
    resolved_key_head_dim: int,
) -> tuple[KeyOutput, ValueOutput]:
    """Validate K/V projection metadata and allocate contiguous storage."""
    batch, full_sequence_length, _input_features = full_qdata.shape
    _, _, key_heads = key_ops.validate_key_inputs(
        full_qdata,
        full_scale,
        key_spec.weight_qdata,
        key_spec.weight_scale,
        key_spec.norm_weight,
        key_spec.cos,
        key_spec.sin,
        norm_epsilon=key_spec.norm_epsilon,
        head_dim=key_spec.head_dim,
        bias=key_spec.bias,
    )
    validate_block_lengths(key_spec.block_lengths, full_sequence_length, full_qdata.device)
    _, _, value_heads = value_ops.validate_value_inputs(
        full_qdata,
        full_scale,
        value_spec.weight_qdata,
        value_spec.weight_scale,
        value_spec.head_dim,
        bias=value_spec.bias,
    )
    validate_block_lengths(value_spec.block_lengths, full_sequence_length, full_qdata.device)
    storage_sequence_length = padded_sequence_length(full_sequence_length)
    key_outputs = key_ops.allocate_key_output(
        full_qdata,
        batch,
        key_heads,
        storage_sequence_length,
        resolved_key_head_dim,
        key_spec.routing_mode,
    )
    value_outputs = value_ops.allocate_value_output(
        full_qdata,
        batch,
        value_heads,
        storage_sequence_length,
        value_spec.head_dim,
        value_spec.storage_format,
    )
    return key_outputs, value_outputs


def _run_prepare_gather_project_kv(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    comm_group_size: int,
    comm_group_name: str,
    prepare_group_size: int,
    input_scale: torch.Tensor | None,
    key_spec: _KeyProjectionSpec,
    value_spec: _ValueProjectionSpec,
) -> tuple[torch.Tensor, ...]:
    _validate_distributed_comm(comm_group_size, comm_group_name)
    if input.ndim != 3 or input.layout is not torch.strided or not input.is_contiguous():
        raise ValueError("distributed sparse Piper KV gather requires contiguous 3D strided input")
    process_group = distributed_comm.resolve_process_group(comm_group_name)
    if process_group.size() != comm_group_size:
        raise ValueError(
            f"process group {comm_group_name!r} has size {process_group.size()}, "
            f"expected {comm_group_size}"
        )
    rank = process_group.rank()
    batch, local_sequence_length, input_features = input.shape
    full_sequence_length = local_sequence_length * comm_group_size
    resolved_key_head_dim = resolve_head_dim(key_spec.norm_weight, key_spec.head_dim)
    validate_routing_mode(key_spec.routing_mode)
    linear_backend = convrot_linear_backend.require_linear_backend(input)
    key_backend = _backend.require_projection_backend(input, head_dim=resolved_key_head_dim)
    value_backend = _backend.require_projection_backend(input, head_dim=value_spec.head_dim)

    full_qdata = torch.empty(
        (batch, full_sequence_length, input_features), device=input.device, dtype=torch.int8
    )
    full_scale = torch.empty(
        (batch, full_sequence_length), device=input.device, dtype=torch.float32
    )
    # Validate all K/V projection metadata before starting any distributed collective.
    key_outputs, value_outputs = _prepare_kv_outputs(
        full_qdata, full_scale, key_spec, value_spec, resolved_key_head_dim
    )

    if batch != 1:
        _gather_batched_inputs(
            input,
            process_group,
            linear_backend,
            prepare_group_size,
            input_scale,
            full_qdata,
            full_scale,
        )
        _project_key_range(key_backend, full_qdata, full_scale, key_spec, key_outputs)
        _project_value_range(value_backend, full_qdata, full_scale, value_spec, value_outputs)
        return (full_qdata, full_scale, *key_outputs, *value_outputs)

    gathered_qdata = full_qdata.view(comm_group_size, local_sequence_length, input_features)
    gathered_scale = full_scale.view(comm_group_size, local_sequence_length)
    own_qdata = gathered_qdata[rank]
    own_scale = gathered_scale[rank]
    linear_backend.prepare_input(
        input.view(local_sequence_length, input_features),
        prepare_group_size,
        input_scale=input_scale,
        out=(own_qdata, own_scale),
    )
    works = distributed_comm.start_coalesced_all_gather(
        process_group, [gathered_qdata, gathered_scale], [own_qdata, own_scale]
    )

    try:
        key_block_m = key_backend.partial_row_block_size(
            full_qdata,
            key_spec.weight_qdata,
            operation="key",
            head_dim=resolved_key_head_dim,
            rotary_dim=key_spec.cos.shape[1],
        )
        value_block_m = value_backend.partial_row_block_size(
            full_qdata,
            value_spec.weight_qdata,
            operation="value",
            head_dim=value_spec.head_dim,
            value_format=value_spec.storage_format,
        )
        can_overlap = (
            key_block_m is not None
            and value_block_m is not None
            and key_block_m > 0
            and value_block_m > 0
            and local_sequence_length % key_block_m == 0
            and local_sequence_length % value_block_m == 0
        )

        if not can_overlap:
            distributed_comm.wait_all(works)
            _project_key_range(key_backend, full_qdata, full_scale, key_spec, key_outputs)
            _project_value_range(value_backend, full_qdata, full_scale, value_spec, value_outputs)
            return (full_qdata, full_scale, *key_outputs, *value_outputs)

        assert key_block_m is not None
        assert value_block_m is not None
        own_row_start = rank * local_sequence_length
        key_own_start = own_row_start // key_block_m
        key_own_blocks = local_sequence_length // key_block_m
        value_own_start = own_row_start // value_block_m
        value_own_blocks = local_sequence_length // value_block_m

        # Project own-rank K and V row blocks while the async INT8 all-gather transfers peers.
        _project_key_range(
            key_backend,
            full_qdata,
            full_scale,
            key_spec,
            key_outputs,
            start_row_block=key_own_start,
            num_row_blocks=key_own_blocks,
        )
        _project_value_range(
            value_backend,
            full_qdata,
            full_scale,
            value_spec,
            value_outputs,
            start_row_block=value_own_start,
            num_row_blocks=value_own_blocks,
        )
    except BaseException:
        distributed_comm.wait_all(works)
        raise
    distributed_comm.wait_all(works)
    _launch_peer_kv_ranges(
        key_backend,
        value_backend,
        full_qdata,
        full_scale,
        key_spec,
        value_spec,
        key_outputs,
        value_outputs,
        (key_own_start, key_own_blocks, full_sequence_length // key_block_m),
        (value_own_start, value_own_blocks, full_sequence_length // value_block_m),
    )
    return (full_qdata, full_scale, *key_outputs, *value_outputs)


@torch.library.custom_op(
    "piper_kernels::convrot_int8_sparse_piper_prepare_gather_project_kv",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _prepare_gather_project_kv_op(  # noqa: PLR0913, PLR0917
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    comm_group_size: int,
    comm_group_name: str,
    prepare_group_size: int,
    input_scale: torch.Tensor | None,
    k_weight_qdata: torch.Tensor,
    k_weight_scale: torch.Tensor,
    k_norm_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_norm_epsilon: float,
    routing_mode: int,
    k_block_lengths: torch.Tensor | None,
    k_bias: torch.Tensor | None,
    k_head_dim: int | None,
    v_weight_qdata: torch.Tensor,
    v_weight_scale: torch.Tensor,
    v_block_lengths: torch.Tensor | None,
    v_head_dim: int,
    v_bias: torch.Tensor | None,
    v_value_format: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare INT8 inputs, overlap async all-gather with own K/V projection, and finish peers."""
    key_spec = _KeyProjectionSpec(
        weight_qdata=k_weight_qdata,
        weight_scale=k_weight_scale,
        norm_weight=k_norm_weight,
        cos=cos,
        sin=sin,
        norm_epsilon=k_norm_epsilon,
        routing_mode=routing_mode,
        block_lengths=k_block_lengths,
        bias=k_bias,
        head_dim=k_head_dim,
    )
    value_spec = _ValueProjectionSpec(
        weight_qdata=v_weight_qdata,
        weight_scale=v_weight_scale,
        block_lengths=v_block_lengths,
        head_dim=v_head_dim,
        bias=v_bias,
        storage_format=ValueFormat(v_value_format),
    )
    result = _run_prepare_gather_project_kv(
        input,
        comm_group_size,
        comm_group_name,
        prepare_group_size,
        input_scale,
        key_spec,
        value_spec,
    )
    assert len(result) == 11
    return result  # type: ignore[return-value]


@_prepare_gather_project_kv_op.register_fake
def _prepare_gather_project_kv_op_fake(
    input: torch.Tensor,  # noqa: A002 - match linear terminology
    comm_group_size: int,
    comm_group_name: str,
    _prepare_group_size: int,
    _input_scale: torch.Tensor | None,
    k_weight_qdata: torch.Tensor,
    _k_weight_scale: torch.Tensor,
    k_norm_weight: torch.Tensor | None,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
    _k_norm_epsilon: float,
    routing_mode: int,
    _k_block_lengths: torch.Tensor | None,
    _k_bias: torch.Tensor | None,
    k_head_dim: int | None,
    v_weight_qdata: torch.Tensor,
    _v_weight_scale: torch.Tensor,
    _v_block_lengths: torch.Tensor | None,
    v_head_dim: int,
    _v_bias: torch.Tensor | None,
    v_value_format: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _validate_distributed_comm(comm_group_size, comm_group_name)
    if input.ndim != 3:
        raise ValueError("distributed sparse Piper KV gather requires 3D input and group_size >= 2")
    validate_routing_mode(routing_mode)
    batch, local_sequence_length, input_features = input.shape
    full_sequence_length = local_sequence_length * comm_group_size
    full_qdata = input.new_empty((batch, full_sequence_length, input_features), dtype=torch.int8)
    full_scale = input.new_empty((batch, full_sequence_length), dtype=torch.float32)
    resolved_key_head_dim = resolve_head_dim(k_norm_weight, k_head_dim)
    storage_sequence_length = padded_sequence_length(full_sequence_length)
    key_heads = k_weight_qdata.shape[0] // resolved_key_head_dim
    value_heads = v_weight_qdata.shape[0] // v_head_dim
    key_outputs = key_ops.allocate_key_output(
        full_qdata, batch, key_heads, storage_sequence_length, resolved_key_head_dim, routing_mode
    )
    value_outputs = value_ops.allocate_value_output(
        full_qdata,
        batch,
        value_heads,
        storage_sequence_length,
        v_head_dim,
        ValueFormat(v_value_format),
    )
    return (full_qdata, full_scale, *key_outputs, *value_outputs)


__all__: list[str] = []
