"""A fused sparse attention computed inside its residual add runs in two head slices."""

import re

import pytest
import torch
from _compile_capture import FunctionalTargetCapturePass
from torch._dynamo.utils import ReinplaceCounters
from torch._inductor.utils import run_and_get_code
from torch.nn import functional as F  # noqa: N812

from piper_kernels import SparsePiperAttention, SparsePiperKeyLayout
from piper_kernels.attention.sparse_piper_attention._backend import require_attention_backend
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    convrot_int8_sparse_piper_compile_options,
    key,
    value,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import output as output_fusion
from piper_kernels.fusions.residual_updates import residual_updates_compile_options
from piper_kernels.linear.convrot.int8 import _ops as int8_ops

from ._helpers import output_available
from .test_attention import _HEAD_DIM, _INPUT_FEATURES, _operands
from .test_compile import (
    _CoarseSparseProjectionAttentionOutput,
    _ProjectedGateCoarseSparseAttentionOutput,
    _SparseProjectionAttentionOutput,
)

_kernels = torch.ops.piper_kernels
_RESIDUAL_ADD = _kernels.convrot_int8_sparse_piper_projected_query_attention_residual_add_.default
_KEY = _kernels.convrot_int8_sparse_piper_project_key.default
_VALUE = _kernels.convrot_int8_sparse_piper_project_value.default
_VALUE_WITH_BLOCK_MEANS = _kernels.convrot_int8_sparse_piper_project_value_with_block_means.default

_HEADS = 4
_SEQUENCE = 512
_WIDTH = _INPUT_FEATURES
_KEY_BLOCKS = 8
# Rounding gives all four heads (3, 3, 2, 2) blocks, but the last two alone (3, 2), so
# each slice must take its heads' share of the whole budget.
_RATIOS = (0.3,) * _HEADS


class _Attention(_SparseProjectionAttentionOutput):
    """Four biased heads whose output projection splits into aligned halves."""

    heads = _HEADS
    sequence_length = _SEQUENCE
    num_sparse_blocks = _KEY_BLOCKS
    output_features = _WIDTH

    def __init__(self) -> None:
        super().__init__()
        self.set_projection_bias(torch.bfloat16)
        self.sparse_attention = SparsePiperAttention(_RATIOS)


class _CoarseAttention(_ProjectedGateCoarseSparseAttentionOutput):
    """Four heads with a coarse branch whose gate is projected from the same input."""

    heads = _HEADS
    sequence_length = _SEQUENCE
    num_sparse_blocks = _KEY_BLOCKS
    output_features = _WIDTH

    def __init__(self) -> None:
        super().__init__(routing="mean")
        self.sparse_attention = SparsePiperAttention(_RATIOS, routing="mean")


class _MaterializedGateAttention(_CoarseSparseProjectionAttentionOutput):
    """Four heads with a coarse branch whose gate the fusion receives whole."""

    heads = _HEADS
    sequence_length = _SEQUENCE
    num_sparse_blocks = _KEY_BLOCKS
    output_features = _WIDTH

    def __init__(self) -> None:
        super().__init__(routing="mean")
        self.sparse_attention = SparsePiperAttention(_RATIOS, routing="mean")
        self.register_buffer(
            "coarse_gate",
            torch.rand(1, _SEQUENCE, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden_states, self.coarse_gate)


class _ResidualBlock(torch.nn.Module):
    """A gated attention update of the hidden stream."""

    def __init__(self, attention: torch.nn.Module) -> None:
        super().__init__()
        torch.manual_seed(1401)
        self.attention = attention
        modulation = torch.randn(3, 3 * _WIDTH, device="cuda", dtype=torch.bfloat16)
        self.register_buffer("modulation", modulation.mul_(0.1))

    def forward(self, hidden: torch.Tensor, gate_indices: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation.split(_WIDTH, dim=-1)
        normalized = F.rms_norm(hidden, (_WIDTH,), eps=1e-6)
        normalized = normalized * (1 + scale[gate_indices]) + shift[gate_indices]
        return hidden + gate[gate_indices] * self.attention(normalized)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.randn(1, _SEQUENCE, _WIDTH, device="cuda", dtype=torch.bfloat16)
    return hidden, torch.randint(0, 3, (_SEQUENCE,), device="cuda")


def _compile_copy_back(model: torch.nn.Module, capture: FunctionalTargetCapturePass):
    def copy_back(hidden, gate_indices):
        return hidden.copy_(model(hidden, gate_indices))

    options = residual_updates_compile_options(convrot_int8_sparse_piper_compile_options())
    torch._dynamo.reset()
    return torch.compile(copy_back, fullgraph=True, options=capture.wrap_options(options))


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.gpu
@pytest.mark.skipif(not output_available(), reason="requires fused sparse output support")
@pytest.mark.parametrize(
    ("attention", "value_producer"),
    [(_Attention, _VALUE), (_CoarseAttention, _VALUE_WITH_BLOCK_MEANS)],
    ids=["fine", "coarse"],
)
def test_each_head_slice_projects_its_context_before_its_add(attention, value_producer):
    model = _ResidualBlock(attention())
    hidden, gate_indices = _inputs()
    torch._dynamo.reset()
    with torch.no_grad():
        # One add, without the residual rewrite.
        expected = torch.compile(
            model, fullgraph=True, options=convrot_int8_sparse_piper_compile_options()
        )(hidden.clone(), gate_indices)
    capture = FunctionalTargetCapturePass()
    compiled = _compile_copy_back(model, capture)
    ReinplaceCounters.clear()
    with torch.no_grad():
        result = compiled(hidden, gate_indices)

    assert result.data_ptr() == hidden.data_ptr()
    assert _relative_l2(hidden, expected) < 0.01
    assert ReinplaceCounters.get_total_missed() == 0
    assert capture.targets.count(_RESIDUAL_ADD) == 2
    assert capture.targets.count(_KEY) == 2
    assert capture.targets.count(value_producer) == 2


@pytest.mark.gpu
@pytest.mark.skipif(not output_available(), reason="requires fused sparse output support")
def test_attention_with_a_materialized_coarse_gate_stays_one_slice():
    model = _ResidualBlock(_MaterializedGateAttention())
    hidden, gate_indices = _inputs()
    capture = FunctionalTargetCapturePass()
    compiled = _compile_copy_back(model, capture)
    with torch.no_grad():
        result = compiled(hidden, gate_indices)

    # The gate holds every head along its third dimension, which slices don't cut.
    assert result.data_ptr() == hidden.data_ptr()
    assert capture.targets.count(_RESIDUAL_ADD) == 1
    assert capture.targets.count(_KEY) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not output_available(), reason="requires fused sparse output support")
def test_each_head_slice_projects_its_context_after_the_previous_slice_adds():
    model = _ResidualBlock(_Attention())
    compiled = _compile_copy_back(model, FunctionalTargetCapturePass())
    with torch.no_grad():
        _, (code, *_) = run_and_get_code(compiled, *_inputs())

    calls = re.findall(
        r"torch\.ops\.piper_kernels\.convrot_int8_sparse_piper_"
        r"(project_key|project_value|projected_query_attention_residual_add_)\.default\(",
        code,
    )
    # Inductor frees the first slice's K and V before the second slice allocates its own.
    first_add = calls.index("projected_query_attention_residual_add_")
    assert sorted(calls[:first_add]) == ["project_key", "project_value"]
    assert sorted(calls[first_add + 1 :]) == [
        "project_key",
        "project_value",
        "projected_query_attention_residual_add_",
    ]


@pytest.mark.gpu
@pytest.mark.skipif(not output_available(), reason="requires fused sparse output support")
def test_head_slices_add_what_one_add_does():
    operands = _operands(sequence_length=_SEQUENCE, heads=_HEADS)
    features = _HEADS * _HEAD_DIM
    output_weight = torch.randint(-127, 128, (_WIDTH, features), device="cuda", dtype=torch.int8)
    output_scale = torch.rand((_WIDTH, 1), device="cuda").mul_(0.01).add_(0.001)
    gate = torch.ones((1, _WIDTH), device="cuda", dtype=torch.bfloat16)
    gate_indices = torch.zeros(_SEQUENCE, device="cuda", dtype=torch.int64)
    input_mean = int8_ops.dequantized_input_mean(operands.input_qdata, operands.input_scale)
    value_format = int(require_attention_backend(operands.input_qdata).value_format)
    attention = SparsePiperAttention(_RATIOS)

    def add(base: torch.Tensor, rows: slice, **position: int) -> None:
        shared = (operands.cos, operands.sin, 1e-5)
        context = (
            *key._project_key_op(
                operands.input_qdata,
                operands.input_scale,
                operands.key_weight[rows],
                operands.key_weight_scale[rows],
                operands.key_norm,
                *shared,
                attention._routing_mode,
            ),
            *value._project_value_op(
                operands.input_qdata,
                operands.input_scale,
                input_mean,
                operands.value_weight[rows],
                operands.value_weight_scale[rows],
                value_format=value_format,
            ),
        )
        output_fusion._projected_query_attention_residual_add_op(
            base,
            gate,
            gate_indices,
            True,
            operands.input_qdata,
            operands.input_scale,
            operands.query_weight[rows],
            operands.query_weight_scale[rows],
            operands.query_norm,
            *shared,
            _HEAD_DIM**-0.5,
            *context,
            attention._ratio_profile,
            SparsePiperKeyLayout.prefix(_KEY_BLOCKS, _KEY_BLOCKS),
            _SEQUENCE,
            attention._routing_mode,
            output_weight[:, rows].contiguous(),
            output_scale,
            None,
            256,
            **position,
        )

    whole = torch.zeros(1, _SEQUENCE, _WIDTH, device="cuda", dtype=torch.bfloat16)
    sliced = torch.zeros_like(whole)
    with torch.no_grad():
        add(whole, slice(None))
        for index in range(2):
            rows = slice(index * features // 2, (index + 1) * features // 2)
            add(sliced, rows, head_slice_index=index, num_head_slices=2)

    # Each slice quantizes its own output-projection input and rounds base once, which
    # leaves 0.75% here; slices that resolved their own keep budgets differ by over 40%.
    assert _relative_l2(sliced, whole) < 0.02
