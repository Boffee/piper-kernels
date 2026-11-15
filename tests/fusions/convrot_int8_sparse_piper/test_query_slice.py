"""Fused ConvRot INT8 sparse outputs for query-row slices match the whole-sequence rows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from _compile_capture import FunctionalTargetCapturePass, TargetCapturePass
from torch.nn import functional as F  # noqa: N812

from piper_kernels import SparsePiperAttention
from piper_kernels.attention.sparse_piper_attention._backend import require_attention_backend
from piper_kernels.attention.sparse_piper_attention._profile import _SparsePiperRatioProfile
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MINMAX_ROUTING
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    convrot_int8_sparse_piper_compile_options,
    key,
    query,
    value,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import output as output_fusion
from piper_kernels.fusions.residual_updates import residual_updates_compile_options
from piper_kernels.linear.convrot import convrot_int8_compile_options

from ._helpers import output_available, projection_available
from .test_attention import _HEAD_DIM, _HEADS, _INPUT_FEATURES, _Operands, _operands
from .test_compile import _make_output_projection, _SparseProjectionAttention

_ops = torch.ops.piper_kernels
_FEATURES = _HEADS * _HEAD_DIM
_OUTPUT_FEATURES = 320
_SPARSE_KEY_BLOCKS = 6
# Routed queries end inside the second slice, which starts at row 192.
_SPARSE_QUERY_BLOCKS = 4
_QUERY_CHUNK_ROWS = 128
_BOUNDARY = 192
_BLOCK_LENGTHS = (64, 17, 64, 40, 64, 64, 3)
# Head slices of one D128 head need an output group that divides their input columns.
_SLICE_GROUP_SIZE = 64

requires_output = pytest.mark.skipif(
    not output_available(), reason="requires fused sparse output support"
)

type _SliceCall = Callable[[int | None, slice], torch.Tensor]


@dataclass(frozen=True, slots=True)
class _Attention:
    """Whole-sequence projected Q/K/V, an output projection, and global routing inputs."""

    operands: _Operands
    query: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    key: tuple[torch.Tensor, ...]
    value: tuple[torch.Tensor, ...]
    block_lengths: torch.Tensor | None
    attention: SparsePiperAttention
    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return self.operands.input_qdata.shape[1]

    @property
    def query_blocks(self) -> int:
        return (self.sequence_length + 63) // 64

    @property
    def policy(self) -> tuple[object, ...]:
        return (
            self.attention._ratio_profile,
            _SPARSE_KEY_BLOCKS,
            self.sequence_length,
            self.attention._routing_mode,
        )

    def forced_tiles(self) -> dict[str, torch.Tensor]:
        blocks = torch.arange(self.query_blocks, device="cuda", dtype=torch.int32)
        return {
            "force_tile_ids": blocks.remainder(_SPARSE_KEY_BLOCKS)[:, None].contiguous(),
            "force_tile_counts": torch.ones(self.query_blocks, device="cuda", dtype=torch.int32),
        }

    def valid_rows(self) -> torch.Tensor:
        rows = torch.arange(self.sequence_length, device="cuda")
        if self.block_lengths is None:
            return torch.ones_like(rows, dtype=torch.bool)
        return rows % 64 < self.block_lengths.repeat_interleave(64)

    def output_rows(self, rows: slice) -> slice:
        return slice(rows.start, min(rows.stop, self.sequence_length))


def _prepare(*, padded: bool, output_features: int = _OUTPUT_FEATURES) -> _Attention:
    sequence_length = 64 * len(_BLOCK_LENGTHS) if padded else 449
    block_lengths = (
        torch.tensor(_BLOCK_LENGTHS, device="cuda", dtype=torch.int32) if padded else None
    )
    operands = _operands(batch=1, sequence_length=sequence_length)
    shared = (operands.cos, operands.sin, 1e-5)
    projected_query = query._project_query_op(
        operands.input_qdata,
        operands.input_scale,
        operands.query_weight,
        operands.query_weight_scale,
        operands.query_norm,
        *shared,
        _HEAD_DIM**-0.5,
        _MINMAX_ROUTING,
        block_lengths,
    )
    projected_key = key._project_key_op(
        operands.input_qdata,
        operands.input_scale,
        operands.key_weight,
        operands.key_weight_scale,
        operands.key_norm,
        *shared,
        _MINMAX_ROUTING,
        block_lengths,
    )
    projected_value = value._project_value_op(
        operands.input_qdata,
        operands.input_scale,
        operands.value_weight,
        operands.value_weight_scale,
        block_lengths,
        value_format=int(require_attention_backend(operands.input_qdata).value_format),
    )
    weight = torch.randint(-127, 128, (output_features, _FEATURES), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand((output_features, 1), device="cuda").mul_(0.01).add_(0.001)
    bias = torch.randn(output_features, device="cuda", dtype=torch.bfloat16)
    return _Attention(
        operands,
        projected_query,
        projected_key,
        projected_value,
        block_lengths,
        SparsePiperAttention((0.5, 1.0)),
        weight,
        weight_scale,
        bias,
    )


def _whole_and_sliced(state: _Attention, call: _SliceCall) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the whole-sequence call and the concatenation of two query-slice calls."""
    storage_rows = state.query_blocks * 64
    with torch.no_grad():
        whole = call(None, slice(0, storage_rows))
        sliced = torch.cat(
            (call(0, slice(0, _BOUNDARY)), call(_BOUNDARY, slice(_BOUNDARY, storage_rows))),
            dim=1,
        )
    return whole, sliced


def _assert_valid_rows_equal(state: _Attention, actual: torch.Tensor, expected: torch.Tensor):
    assert actual.shape == expected.shape
    valid_rows = state.valid_rows()
    assert torch.equal(actual[:, valid_rows], expected[:, valid_rows])


@pytest.mark.gpu
@requires_output
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("coarse", [False, True])
def test_attention_output_query_slices_reproduce_whole_sequence_rows(
    padded: bool, coarse: bool
) -> None:
    state = _prepare(padded=padded)
    torch.manual_seed(1601)
    coarse_gate = torch.randn(
        (1, state.sequence_length, _HEADS, _HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )

    def call(global_row_offset: int | None, rows: slice) -> torch.Tensor:
        query_data, query_scale, query_summary = state.query
        coarse_arguments = (
            (coarse_gate[:, state.output_rows(rows)], 0.125, _SPARSE_KEY_BLOCKS)
            if coarse
            else (None, None, None)
        )
        return output_fusion._attention_output_op(
            query_data[:, :, rows].contiguous(),
            query_scale[:, :, rows.start // 32 : rows.stop // 32].contiguous(),
            query_summary[:, :, rows.start // 64 : rows.stop // 64].contiguous(),
            *state.key,
            *state.value,
            *state.policy,
            state.weight,
            state.weight_scale,
            state.bias,
            _FEATURES,
            _QUERY_CHUNK_ROWS,
            state.block_lengths,
            *coarse_arguments,
            _SPARSE_QUERY_BLOCKS,
            **state.forced_tiles(),
            **({} if global_row_offset is None else {"global_row_offset": global_row_offset}),
        )

    whole, sliced = _whole_and_sliced(state, call)

    _assert_valid_rows_equal(state, sliced, whole)


@pytest.mark.gpu
@requires_output
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("variant", ["prepared", "chunk-prepared", "in-place"])
def test_projected_query_slices_reproduce_whole_sequence_rows(padded: bool, variant: str) -> None:
    state = _prepare(padded=padded, output_features=_INPUT_FEATURES)
    operands = state.operands
    torch.manual_seed(1602)
    hidden = torch.randn(
        (1, state.sequence_length, _INPUT_FEATURES), device="cuda", dtype=torch.bfloat16
    )

    def call(global_row_offset: int | None, rows: slice) -> torch.Tensor:
        rows = state.output_rows(rows)
        if variant == "prepared":
            query_input = operands.input_qdata[:, rows].contiguous()
            query_input_scale = operands.input_scale[:, rows].contiguous()
            group = {}
        else:
            query_input = hidden[:, rows].clone()
            query_input_scale = None
            group = {"input_group_size": _INPUT_FEATURES}
        arguments = (
            query_input,
            query_input_scale,
            operands.query_weight,
            operands.query_weight_scale,
            operands.query_norm,
            operands.cos[rows],
            operands.sin[rows],
            1e-5,
            _HEAD_DIM**-0.5,
            *state.key,
            *state.value,
            *state.policy,
            state.weight,
            state.weight_scale,
            state.bias,
            _FEATURES,
            _QUERY_CHUNK_ROWS,
            state.block_lengths,
            None,
            None,
            None,
            _SPARSE_QUERY_BLOCKS,
        )
        keywords = {
            **group,
            **state.forced_tiles(),
            **({} if global_row_offset is None else {"global_row_offset": global_row_offset}),
        }
        if variant == "in-place":
            output_fusion._projected_query_attention_output_inplace_op(*arguments, **keywords)
            return query_input
        return output_fusion._projected_query_attention_output_op(*arguments, **keywords)

    whole, sliced = _whole_and_sliced(state, call)

    _assert_valid_rows_equal(state, sliced, whole)


@pytest.mark.gpu
@requires_output
@pytest.mark.parametrize("num_head_slices", [1, 2])
def test_residual_add_query_slices_reproduce_whole_sequence_rows(num_head_slices: int) -> None:
    state = _prepare(padded=False)
    operands = state.operands
    torch.manual_seed(1603)
    base = torch.randn(
        (1, state.sequence_length, _OUTPUT_FEATURES), device="cuda", dtype=torch.bfloat16
    )
    gate = torch.randn((5, _OUTPUT_FEATURES), device="cuda", dtype=torch.bfloat16)
    gate_indices = torch.randint(-5, 5, (state.sequence_length,), device="cuda")
    value_format = int(require_attention_backend(operands.input_qdata).value_format)
    shared = (operands.cos, operands.sin, 1e-5)
    slice_features = _FEATURES // num_head_slices

    def call(global_row_offset: int | None, rows: slice) -> torch.Tensor:
        rows = state.output_rows(rows)
        result = base[:, rows].clone()
        for head_slice_index in range(num_head_slices):
            features = slice(
                head_slice_index * slice_features, (head_slice_index + 1) * slice_features
            )
            # Each slice projects its own heads' K/V from the whole sequence.
            context = (
                *key._project_key_op(
                    operands.input_qdata,
                    operands.input_scale,
                    operands.key_weight[features],
                    operands.key_weight_scale[features],
                    operands.key_norm,
                    *shared,
                    _MINMAX_ROUTING,
                ),
                *value._project_value_op(
                    operands.input_qdata,
                    operands.input_scale,
                    operands.value_weight[features],
                    operands.value_weight_scale[features],
                    value_format=value_format,
                ),
            )
            output_fusion._projected_query_attention_residual_add_op(
                result,
                gate,
                gate_indices[rows],
                True,
                operands.input_qdata[:, rows].contiguous(),
                operands.input_scale[:, rows].contiguous(),
                operands.query_weight[features],
                operands.query_weight_scale[features],
                operands.query_norm,
                operands.cos[rows],
                operands.sin[rows],
                1e-5,
                _HEAD_DIM**-0.5,
                *context,
                *state.policy,
                state.weight[:, features].contiguous(),
                state.weight_scale,
                state.bias if head_slice_index == 0 else None,
                _SLICE_GROUP_SIZE,
                _QUERY_CHUNK_ROWS,
                None,
                None,
                None,
                None,
                _SPARSE_QUERY_BLOCKS,
                **state.forced_tiles(),
                head_slice_index=head_slice_index,
                num_head_slices=num_head_slices,
                **({} if global_row_offset is None else {"global_row_offset": global_row_offset}),
            )
        return result

    whole, sliced = _whole_and_sliced(state, call)

    assert torch.equal(sliced, whole)


def _meta(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.empty(shape, device="meta", dtype=dtype)


def _fake_attention_tail() -> tuple[object, ...]:
    """K/V and policy for a compact 300-row, three-head sequence."""
    return (
        _meta(2, 3, 320, 128, dtype=torch.int8),
        _meta(2, 3, 5),
        _meta(2, 3, 5, 2, 128, dtype=torch.int8),
        _meta(2, 3, 5, 2),
        _meta(2, 3, 5, 128),
        _meta(2, 3, 5, 128),
        _meta(2, 3, 128, 320, dtype=torch.int8),
        _meta(2, 3, 5, 1),
        _meta(2, 3, 5, 128, dtype=torch.bfloat16),
        _SparsePiperRatioProfile((500_000, 1_000_000, 750_000)),
        2,
        300,
        0,
        _meta(_OUTPUT_FEATURES, 384, dtype=torch.int8),
        _meta(_OUTPUT_FEATURES, 1),
        None,
        128,
        64,
    )


def _fake_projected_arguments(query_rows: int) -> tuple[object, ...]:
    return (
        _meta(2, query_rows, 384, dtype=torch.int8),
        _meta(2, query_rows),
        _meta(384, 384, dtype=torch.int8),
        _meta(384, 1),
        None,
        _meta(query_rows, 96),
        _meta(query_rows, 96),
        1e-5,
        128**-0.5,
        *_fake_attention_tail(),
    )


def _fake_attention_output(storage_rows: int, global_row_offset: int) -> torch.Tensor:
    return output_fusion._attention_output_op_fake(
        _meta(2, 3, storage_rows, 128, dtype=torch.int8),
        _meta(2, 3, storage_rows // 32),
        _meta(2, 3, storage_rows // 64, 128),
        *_fake_attention_tail(),
        global_row_offset=global_row_offset,
    )


def test_fakes_size_outputs_by_the_query_slice() -> None:
    projected = output_fusion._projected_query_attention_output_op_fake(
        *_fake_projected_arguments(172), global_row_offset=128
    )
    base = _meta(2, 172, _OUTPUT_FEATURES, dtype=torch.bfloat16)
    gate = _meta(5, _OUTPUT_FEATURES, dtype=torch.bfloat16)
    residual_arguments = (base, gate, _meta(344, dtype=torch.int64), True)

    assert projected.shape == (2, 172, _OUTPUT_FEATURES)
    assert _fake_attention_output(192, 128).shape == (2, 172, _OUTPUT_FEATURES)
    assert _fake_attention_output(128, 64).shape == (2, 128, _OUTPUT_FEATURES)
    assert (
        output_fusion._projected_query_attention_residual_add_op_fake(
            *residual_arguments, *_fake_projected_arguments(172), global_row_offset=128
        )
        is None
    )
    with pytest.raises(ValueError, match="one row per output row"):
        output_fusion._projected_query_attention_residual_add_op_fake(
            *residual_arguments, *_fake_projected_arguments(172)
        )


@pytest.mark.parametrize(
    ("query_rows", "global_row_offset", "message"),
    [(64, 32, "multiple of 64"), (100, 0, "whole K64 blocks"), (128, 256, "must fit")],
    ids=["misaligned", "ragged-middle", "past-end"],
)
def test_projected_fakes_reject_misplaced_query_rows(
    query_rows: int, global_row_offset: int, message: str
) -> None:
    arguments = _fake_projected_arguments(query_rows)

    with pytest.raises(ValueError, match=message):
        output_fusion._projected_query_attention_output_op_fake(
            *arguments, global_row_offset=global_row_offset
        )
    with pytest.raises(ValueError, match=message):
        output_fusion._projected_query_attention_output_inplace_op_fake(
            *arguments, global_row_offset=global_row_offset
        )


@pytest.mark.parametrize(
    ("storage_rows", "global_row_offset", "message"),
    [(64, 32, "multiple of 64"), (256, 128, "extend past"), (64, 320, "must fit")],
    ids=["misaligned", "storage-past-end", "past-end"],
)
def test_attention_output_fake_rejects_misplaced_query_storage(
    storage_rows: int, global_row_offset: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _fake_attention_output(storage_rows, global_row_offset)


@pytest.mark.gpu
@requires_output
def test_query_slice_custom_ops_pass_opcheck() -> None:
    state = _prepare(padded=False)
    operands = state.operands
    rows = slice(_BOUNDARY, state.sequence_length)
    query_data, query_scale, query_summary = state.query
    attention_tail = (
        *state.key,
        *state.value,
        *state.policy,
        state.weight,
        state.weight_scale,
        state.bias,
        _FEATURES,
        _QUERY_CHUNK_ROWS,
    )
    keywords = {"global_row_offset": _BOUNDARY}

    with torch.no_grad():
        materialized = torch.library.opcheck(
            output_fusion._attention_output_op,
            (
                query_data[:, :, _BOUNDARY:].contiguous(),
                query_scale[:, :, _BOUNDARY // 32 :].contiguous(),
                query_summary[:, :, _BOUNDARY // 64 :].contiguous(),
                *attention_tail,
            ),
            keywords,
        )
        projected = torch.library.opcheck(
            output_fusion._projected_query_attention_output_op,
            (
                operands.input_qdata[:, rows].contiguous(),
                operands.input_scale[:, rows].contiguous(),
                operands.query_weight,
                operands.query_weight_scale,
                operands.query_norm,
                operands.cos[rows],
                operands.sin[rows],
                1e-5,
                _HEAD_DIM**-0.5,
                *attention_tail,
            ),
            keywords,
        )

    assert set(materialized.values()) == {"SUCCESS"}
    assert set(projected.values()) == {"SUCCESS"}


@pytest.mark.gpu
@requires_output
@pytest.mark.parametrize("project_query", [False, True])
def test_output_fold_keeps_the_global_row_offset(project_query: bool) -> None:
    state = _prepare(padded=False)
    operands = state.operands
    rows = slice(_BOUNDARY, state.sequence_length)
    num_rows = state.sequence_length - _BOUNDARY
    projection = _make_output_projection(_FEATURES, _OUTPUT_FEATURES, bias=True)
    query_input = (operands.input_qdata[:, rows].contiguous(), operands.input_scale[:, rows])
    query_rope = (operands.cos[rows].contiguous(), operands.sin[rows].contiguous())
    query_weights = (operands.query_weight, operands.query_weight_scale, operands.query_norm)
    query_data, query_scale, query_summary = state.query
    stored_query = (
        query_data[:, :, _BOUNDARY:].contiguous(),
        query_scale[:, :, _BOUNDARY // 32 :].contiguous(),
        query_summary[:, :, _BOUNDARY // 64 :].contiguous(),
    )

    def run(*local_query: torch.Tensor) -> torch.Tensor:
        if project_query:
            local_query = query._project_query_op(
                *local_query, *query_weights, *query_rope, 1e-5, _HEAD_DIM**-0.5, _MINMAX_ROUTING
            )
        attention = _sparse_piper_attention_from_quantized_op(
            *local_query,
            *state.key,
            *state.value,
            *state.policy,
            None,
            _SPARSE_QUERY_BLOCKS,
            global_row_offset=_BOUNDARY,
        )
        return projection(attention.reshape(1, num_rows, _FEATURES))

    capture = TargetCapturePass()
    weight = projection.weight
    output_operands = (weight.qdata, weight.scale, projection.bias, weight.group_size)
    torch._dynamo.reset()
    with torch.no_grad():
        if project_query:
            actual = torch.compile(
                run,
                fullgraph=True,
                options=capture.wrap_options(convrot_int8_sparse_piper_compile_options()),
            )(*query_input)
            expected = output_fusion._projected_query_attention_output_op(
                *query_input,
                *query_weights,
                *query_rope,
                1e-5,
                _HEAD_DIM**-0.5,
                *state.key,
                *state.value,
                *state.policy,
                *output_operands,
                sparse_query_blocks=_SPARSE_QUERY_BLOCKS,
                global_row_offset=_BOUNDARY,
            )
            fused = _ops.convrot_int8_sparse_piper_projected_query_attention_output.default
        else:
            actual = torch.compile(
                run,
                fullgraph=True,
                options=capture.wrap_options(convrot_int8_sparse_piper_compile_options()),
            )(*stored_query)
            expected = output_fusion._attention_output_op(
                *stored_query,
                *state.key,
                *state.value,
                *state.policy,
                *output_operands,
                sparse_query_blocks=_SPARSE_QUERY_BLOCKS,
                global_row_offset=_BOUNDARY,
            )
            fused = _ops.convrot_int8_sparse_piper_attention_output.default

    assert capture.targets.count(fused) == 1
    assert _ops.sparse_piper_attention_from_quantized.default not in capture.targets
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@requires_output
def test_residual_fold_keeps_the_global_row_offset_in_each_head_slice() -> None:
    state = _prepare(padded=False, output_features=_OUTPUT_FEATURES)
    operands = state.operands
    value_format = int(require_attention_backend(operands.input_qdata).value_format)
    torch.manual_seed(1604)
    modulation = torch.randn((3, _OUTPUT_FEATURES), device="cuda", dtype=torch.bfloat16)

    def block(hidden, gate_indices, query_input, query_input_scale, cos, sin, global_row_offset):
        context = (
            *key._project_key_op(
                operands.input_qdata,
                operands.input_scale,
                operands.key_weight,
                operands.key_weight_scale,
                operands.key_norm,
                operands.cos,
                operands.sin,
                1e-5,
                _MINMAX_ROUTING,
            ),
            *value._project_value_op(
                operands.input_qdata,
                operands.input_scale,
                operands.value_weight,
                operands.value_weight_scale,
                value_format=value_format,
            ),
        )
        attention = output_fusion._projected_query_attention_output_op(
            query_input,
            query_input_scale,
            operands.query_weight,
            operands.query_weight_scale,
            operands.query_norm,
            cos,
            sin,
            1e-5,
            _HEAD_DIM**-0.5,
            *context,
            *state.policy,
            state.weight,
            state.weight_scale,
            state.bias,
            _SLICE_GROUP_SIZE,
            _QUERY_CHUNK_ROWS,
            sparse_query_blocks=_SPARSE_QUERY_BLOCKS,
            **({} if global_row_offset is None else {"global_row_offset": global_row_offset}),
        )
        return hidden.copy_(hidden + modulation[gate_indices] * attention)

    torch.manual_seed(1605)
    hidden = torch.randn(
        (1, state.sequence_length, _OUTPUT_FEATURES), device="cuda", dtype=torch.bfloat16
    )
    gate_indices = torch.randint(0, 3, (state.sequence_length,), device="cuda")
    options = residual_updates_compile_options(convrot_int8_sparse_piper_compile_options())

    def compiled_call(global_row_offset: int | None, rows: slice) -> torch.Tensor:
        rows = state.output_rows(rows)
        capture = FunctionalTargetCapturePass()
        torch._dynamo.reset()
        # The add writes into its base only when that base has standard strides.
        result = hidden[:, rows].clone(memory_format=torch.contiguous_format)
        # Static shapes: the rewrite cannot prove that two inputs' dynamic rows match.
        compiled = torch.compile(
            block, fullgraph=True, dynamic=False, options=capture.wrap_options(options)
        )
        compiled(
            result,
            gate_indices[rows],
            operands.input_qdata[:, rows].contiguous(),
            operands.input_scale[:, rows].contiguous(),
            operands.cos[rows].contiguous(),
            operands.sin[rows].contiguous(),
            global_row_offset,
        )
        residual_add = _ops.convrot_int8_sparse_piper_projected_query_attention_residual_add_
        assert capture.targets.count(residual_add.default) == 2
        return result

    whole, sliced = _whole_and_sliced(state, compiled_call)

    assert torch.equal(sliced, whole)


class _QuerySliceAttention(_SparseProjectionAttention):
    """Project Q from one slice of the hidden rows and K/V from all of them."""

    def _rope_rows(self, projected: torch.Tensor, norm: torch.Tensor, rows: slice) -> torch.Tensor:
        normalized = F.rms_norm(
            projected.unflatten(-1, (self.heads, self.head_dim)), (self.head_dim,), norm, 1e-5
        )
        rotary = normalized[..., : self.rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        cos = self.cos[rows].to(projected.dtype)[None, :, None, :]
        sin = self.sin[rows].to(projected.dtype)[None, :, None, :]
        rotary = rotary * cos + torch.cat((-second, first), dim=-1) * sin
        return torch.cat((rotary, normalized[..., self.rotary_dim :]), dim=-1).contiguous()

    def forward(self, hidden_states: torch.Tensor, global_row_offset: int) -> torch.Tensor:  # type: ignore[override]
        rows = slice(global_row_offset, global_row_offset + 128)
        query = self._rope_rows(self.query(hidden_states[:, rows]), self.query_norm, rows)
        everything = slice(None)
        key = self._rope_rows(self.key(hidden_states), self.key_norm, everything)
        value = self.value(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        return self.sparse_attention(
            query,
            key,
            value,
            sparse_key_blocks=self.sparse_key_blocks,
            global_row_offset=global_row_offset,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not projection_available(), reason="requires fused sparse projection support")
@pytest.mark.parametrize("global_row_offset", [0, 64])
def test_projection_fold_leaves_query_slices_unfused(global_row_offset: int) -> None:
    torch.manual_seed(1606)
    model = _QuerySliceAttention().eval()
    hidden = torch.randn(
        (1, model.sequence_length, model.input_features), device="cuda", dtype=torch.bfloat16
    )
    capture = TargetCapturePass()
    options = capture.wrap_options(convrot_int8_sparse_piper_compile_options())

    with torch.no_grad():
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=options)(hidden, global_row_offset)
        torch._dynamo.reset()
        unfused = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())
        expected = unfused(hidden, global_row_offset)

    # The projection fusion needs Q and K/V projected from the same rows.
    assert capture.targets.count(_ops.sparse_piper_attention.default) == 1
    assert _ops.sparse_piper_attention_from_quantized.default not in capture.targets
    assert torch.equal(actual, expected)
