"""A copy-back residual block updates its input in place after attention and FFN fusion."""

import pytest
import torch
from _compile_capture import TargetCapturePass
from torch._dynamo.utils import ReinplaceCounters
from torch.nn import functional as F  # noqa: N812

from piper_kernels import piper_attention
from piper_kernels.fusions.convrot_int8_piper import convrot_int8_piper_compile_options
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    convrot_int8_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_int8_swiglu_ffn import convrot_int8_swiglu_ffn_compile_options
from piper_kernels.fusions.residual_updates import _compile as residual_compile
from piper_kernels.fusions.residual_updates import residual_updates_compile_options
from piper_kernels.linear.convrot.int8 import _ops

from ..convrot_int8_piper.test_compile import _available
from ..convrot_int8_sparse_piper._helpers import output_available as _sparse_output_available
from ..convrot_int8_sparse_piper.test_compile import _SparseProjectionAttentionOutput
from ..convrot_int8_swiglu_ffn._helpers import SwiGluFfn

_kernels = torch.ops.piper_kernels
_ATTENTION_OUTPUT = _kernels.convrot_int8_piper_projected_query_attention_output.default
_ATTENTION_RESIDUAL_ADD = (
    _kernels.convrot_int8_piper_projected_query_attention_residual_add_.default
)
_SPARSE_ATTENTION_OUTPUT = (
    _kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
)
_SPARSE_ATTENTION_RESIDUAL_ADD = (
    _kernels.convrot_int8_sparse_piper_projected_query_attention_residual_add_.default
)
_FFN_UPDATES = _kernels.convrot_int8_swiglu_ffn_gated_updates_.default
_FFN_RESIDUAL_ADD = _kernels.convrot_int8_swiglu_ffn_residual_add_.default
_RESIDUAL_ADD = _kernels.residual_add_.default
_HEAD_GROUPED_ATTENTION_ADD = (
    _kernels.convrot_int8_piper_head_grouped_attention_residual_add_.default
)
_DENSE_KEY = _kernels.convrot_int8_piper_project_key.default
_DENSE_VALUE = _kernels.convrot_int8_piper_project_value.default


class _Ffn(SwiGluFfn):
    output_features = 256


class _FunctionalCapture(TargetCapturePass):
    """Record each functional call as the mutating operator it wraps."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        super().__call__(graph, is_inference)
        self.targets = [
            node.args[0]
            if node.target is torch.ops.higher_order.auto_functionalized
            else node.target
            for node in graph.nodes
            if node.op == "call_function"
        ]


class _ResidualBlock(torch.nn.Module):
    """Gated self-attention, then a gated SwiGLU FFN, over one hidden stream.

    A nonzero ``input_skip`` also adds that multiple of the original input to the FFN
    input, read after the attention update through the row view attention normalization
    read.
    """

    width = 256
    heads = 4
    kv_heads = 2
    head_dim = 64
    rotary_dim = 48

    def __init__(self, *, input_skip: float = 0.0) -> None:
        super().__init__()
        self.input_skip = input_skip
        torch.manual_seed(1201)
        for name, rows, columns in (
            ("query", self.heads * self.head_dim, self.width),
            ("key", self.kv_heads * self.head_dim, self.width),
            ("value", self.kv_heads * self.head_dim, self.width),
            ("output", self.width, self.heads * self.head_dim),
        ):
            self.register_buffer(
                f"{name}_weight",
                torch.randint(-127, 128, (rows, columns), device="cuda", dtype=torch.int8),
            )
            self.register_buffer(
                f"{name}_scale",
                torch.rand((rows, 1), device="cuda").mul_(0.002).add_(0.0005),
            )
        self.register_buffer(
            "norm", torch.rand(self.head_dim, device="cuda", dtype=torch.bfloat16).add_(0.5)
        )
        modulation = torch.randn(3, 6 * self.width, device="cuda", dtype=torch.bfloat16)
        # Keep the large random FFN from dominating, so the attention update matters.
        modulation[:, 5 * self.width :].mul_(0.01)
        self.register_buffer("modulation", modulation.mul_(0.1))
        self.ffn = _Ffn(promote_gate=True, reverse_multiply=True, bias_dtype=None)

    def _heads(self, projected, heads, cos, sin):
        shaped = projected.view(projected.shape[0], projected.shape[1], heads, self.head_dim)
        normalized = F.rms_norm(shaped, (self.head_dim,), self.norm, 1e-5)
        rotary = normalized[..., : self.rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        rotary = (
            rotary * cos.to(projected.dtype)[None, :, None, :]
            + torch.cat((-second, first), dim=-1) * sin.to(projected.dtype)[None, :, None, :]
        )
        return torch.cat((rotary, normalized[..., self.rotary_dim :]), dim=-1).transpose(1, 2)

    def forward(self, hidden, cos, sin, gate_indices):
        shift, scale, attention_gate, ffn_shift, ffn_scale, ffn_gate = self.modulation.split(
            self.width, dim=-1
        )
        rows = hidden.view(-1, self.width)
        normalized = F.rms_norm(rows, (self.width,), eps=1e-6).view(hidden.shape)
        normalized = normalized * (1 + scale[gate_indices]) + shift[gate_indices]
        query = self._heads(
            _ops.linear(normalized, self.query_weight, self.query_scale, None, 256, None, None),
            self.heads,
            cos,
            sin,
        )
        key = self._heads(
            _ops.linear(normalized, self.key_weight, self.key_scale, None, 256, None, None),
            self.kv_heads,
            cos,
            sin,
        )
        value = _ops.linear(normalized, self.value_weight, self.value_scale, None, 256, None, None)
        value = value.view(hidden.shape[0], hidden.shape[1], self.kv_heads, self.head_dim)
        attended = piper_attention(query, key, value.transpose(1, 2), scale=0.125)
        merged = attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        attention = _ops.linear(merged, self.output_weight, self.output_scale, None, 64, None, None)
        residual = hidden + attention_gate[gate_indices] * attention
        ffn_source = residual
        if self.input_skip:
            ffn_source = residual + (rows * self.input_skip).view(residual.shape)
        ffn_input = F.rms_norm(ffn_source, (self.width,), eps=1e-6)
        ffn_input = ffn_input * (1 + ffn_scale[gate_indices]) + ffn_shift[gate_indices]
        return residual + ffn_gate[gate_indices] * self.ffn(ffn_input)


class _SparseAttention(_SparseProjectionAttentionOutput):
    """Sparse Piper attention whose output projection returns to the residual width."""

    output_features = 256


class _SparseResidualBlock(torch.nn.Module):
    """The gated residual block with sparse Piper attention over a fixed 192-token layout."""

    width = _SparseAttention.input_features
    sequence_length = _SparseAttention.sequence_length

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1301)
        self.attention = _SparseAttention()
        modulation = torch.randn(3, 6 * self.width, device="cuda", dtype=torch.bfloat16)
        modulation[:, 5 * self.width :].mul_(0.01)
        self.register_buffer("modulation", modulation.mul_(0.1))
        self.ffn = _Ffn(promote_gate=True, reverse_multiply=True, bias_dtype=None)

    def forward(self, hidden, gate_indices):
        shift, scale, attention_gate, ffn_shift, ffn_scale, ffn_gate = self.modulation.split(
            self.width, dim=-1
        )
        normalized = F.rms_norm(hidden, (self.width,), eps=1e-6)
        normalized = normalized * (1 + scale[gate_indices]) + shift[gate_indices]
        residual = hidden + attention_gate[gate_indices] * self.attention(normalized)
        ffn_input = F.rms_norm(residual, (self.width,), eps=1e-6)
        ffn_input = ffn_input * (1 + ffn_scale[gate_indices]) + ffn_shift[gate_indices]
        return residual + ffn_gate[gate_indices] * self.ffn(ffn_input)


def _options(capture, *, fuse_output=True, attention_head_groups=1):
    options = convrot_int8_piper_compile_options(
        convrot_int8_swiglu_ffn_compile_options(), fuse_output=fuse_output
    )
    return capture.wrap_options(
        residual_updates_compile_options(options, attention_head_groups=attention_head_groups)
    )


def _inputs(model, sequence):
    hidden = torch.randn(1, sequence, model.width, device="cuda", dtype=torch.bfloat16)
    angles = torch.rand(sequence, model.rotary_dim, device="cuda") * (2 * torch.pi)
    gate_indices = torch.randint(0, 3, (sequence,), device="cuda")
    return hidden, angles.cos(), angles.sin(), gate_indices


def _compile_copy_back(model, options):
    def copy_back(hidden, cos, sin, gate_indices):
        return hidden.copy_(model(hidden, cos, sin, gate_indices))

    torch._dynamo.reset()
    return torch.compile(copy_back, fullgraph=True, dynamic=True, options=options)


def _relative_l2(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _assert_updates_input_in_place(model, compiled):
    ReinplaceCounters.clear()
    with torch.no_grad():
        for sequence in (65, 300):
            hidden, cos, sin, gate_indices = _inputs(model, sequence)
            expected = model(hidden.clone(), cos, sin, gate_indices)
            result = compiled(hidden, cos, sin, gate_indices)

            assert result.data_ptr() == hidden.data_ptr()
            assert _relative_l2(hidden, expected) < 0.01
    # Inductor added every update into the input instead of copying it first.
    assert ReinplaceCounters.get_total_missed() == 0


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_copy_back_block_updates_its_input_in_place():
    model = _ResidualBlock()
    capture = _FunctionalCapture()
    compiled = _compile_copy_back(model, _options(capture))
    _assert_updates_input_in_place(model, compiled)

    assert capture.targets.count(_ATTENTION_RESIDUAL_ADD) == 1
    assert capture.targets.count(_FFN_RESIDUAL_ADD) == 1
    assert _ATTENTION_OUTPUT not in capture.targets
    assert _FFN_UPDATES not in capture.targets
    assert _RESIDUAL_ADD not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _sparse_output_available(), reason="requires fused sparse output support")
def test_sparse_attention_output_is_computed_inside_its_residual_add():
    model = _SparseResidualBlock()
    capture = _FunctionalCapture()
    options = convrot_int8_sparse_piper_compile_options(convrot_int8_swiglu_ffn_compile_options())
    hidden = torch.randn(1, model.sequence_length, model.width, device="cuda", dtype=torch.bfloat16)
    gate_indices = torch.randint(0, 3, (model.sequence_length,), device="cuda")
    torch._dynamo.reset()
    with torch.no_grad():
        # The sparse fusions alone differ from eager by about 2% here, so compare
        # against them without the rewrite.
        expected = torch.compile(model, fullgraph=True, options=options)(
            hidden.clone(), gate_indices
        )
    torch._dynamo.reset()

    def copy_back(hidden, gate_indices):
        return hidden.copy_(model(hidden, gate_indices))

    compiled = torch.compile(
        copy_back,
        fullgraph=True,
        options=capture.wrap_options(residual_updates_compile_options(options)),
    )
    ReinplaceCounters.clear()
    with torch.no_grad():
        result = compiled(hidden, gate_indices)

    assert result.data_ptr() == hidden.data_ptr()
    assert _relative_l2(hidden, expected) < 0.01
    assert ReinplaceCounters.get_total_missed() == 0
    assert capture.targets.count(_SPARSE_ATTENTION_RESIDUAL_ADD) == 1
    assert capture.targets.count(_FFN_RESIDUAL_ADD) == 1
    assert _SPARSE_ATTENTION_OUTPUT not in capture.targets
    assert _RESIDUAL_ADD not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_unfused_attention_output_is_added_in_place_by_the_ffn_rewrite():
    model = _ResidualBlock()
    capture = _FunctionalCapture()
    compiled = _compile_copy_back(model, _options(capture, fuse_output=False))

    _assert_updates_input_in_place(model, compiled)

    assert capture.targets.count(_RESIDUAL_ADD) == 1
    assert capture.targets.count(_FFN_RESIDUAL_ADD) == 1
    assert _ATTENTION_RESIDUAL_ADD not in capture.targets
    assert _FFN_UPDATES not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_input_read_after_an_update_is_copied_instead_of_overwritten():
    model = _ResidualBlock(input_skip=16.0)
    # Let the FFN, whose input mostly reads the original input, shape the result.
    model.modulation[:, 5 * model.width :].mul_(100)
    capture = _FunctionalCapture()
    compiled = _compile_copy_back(model, _options(capture))
    ReinplaceCounters.clear()
    with torch.no_grad():
        hidden, cos, sin, gate_indices = _inputs(model, 300)
        expected = model(hidden.clone(), cos, sin, gate_indices)
        result = compiled(hidden, cos, sin, gate_indices)

    assert result.data_ptr() == hidden.data_ptr()
    # The compiled block is within 2% of eager; reading the updated input would be 6% off.
    assert _relative_l2(hidden, expected) < 0.03
    # Inductor copies the input before the attention add, then adds into that copy.
    assert ReinplaceCounters.get_total_missed() == 1
    assert capture.targets.count(_ATTENTION_RESIDUAL_ADD) == 1
    assert capture.targets.count(_FFN_RESIDUAL_ADD) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_block_without_copy_back_keeps_its_input():
    model = _ResidualBlock()
    torch._dynamo.reset()
    capture = TargetCapturePass()
    compiled = torch.compile(model, fullgraph=True, dynamic=True, options=_options(capture))
    with torch.no_grad():
        hidden, cos, sin, gate_indices = _inputs(model, 65)
        original = hidden.clone()
        compiled(hidden, cos, sin, gate_indices)

    torch.testing.assert_close(hidden, original, atol=0, rtol=0)
    assert _ATTENTION_RESIDUAL_ADD not in capture.targets
    assert _ATTENTION_OUTPUT in capture.targets
    assert _FFN_UPDATES in capture.targets


def test_compile_options_append_the_rewrite_and_disable_buffer_reuse():
    options = convrot_int8_piper_compile_options(fuse_output=True)
    combined = residual_updates_compile_options(options)

    passes = combined["post_grad_custom_pre_pass"]
    assert isinstance(passes, tuple)
    assert passes[:-1] == options["post_grad_custom_pre_pass"]
    assert combined["allow_buffer_reuse"] is False
    # Applying the rewrite again keeps one copy of it, in place.
    again = residual_updates_compile_options(combined)["post_grad_custom_pre_pass"]
    assert again[:-1] == passes[:-1]
    assert again[-1].uuid() == passes[-1].uuid()


def test_ffn_residual_adds_take_the_two_update_arguments_without_the_reusable_update():
    for two_updates, residual_add in residual_compile._FFN_RESIDUAL_ADDS.items():
        assert isinstance(two_updates, torch._ops.OpOverload)
        expected = [
            argument.name
            for argument in two_updates._schema.arguments
            if argument.name not in residual_compile._REUSABLE_UPDATE_ARGUMENTS
        ]
        assert [argument.name for argument in residual_add._schema.arguments] == expected


def test_attention_residual_adds_take_the_add_then_the_output_arguments():
    for attention_output, residual_add in residual_compile._ATTENTION_RESIDUAL_ADDS.items():
        assert isinstance(attention_output, torch._ops.OpOverload)
        output_names = [
            argument.name
            for argument in attention_output._schema.arguments
            if argument.name != "output_dtype"
        ]
        assert [argument.name for argument in residual_add._schema.arguments] == [
            "base",
            "gate",
            "gate_indices",
            "python_indexing",
            *output_names,
        ]


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_head_grouped_attention_projects_each_groups_context_inside_its_add():
    # Two groups of two query heads and one K/V head each.
    model = _ResidualBlock()
    capture = _FunctionalCapture()
    compiled = _compile_copy_back(model, _options(capture, attention_head_groups=2))
    _assert_updates_input_in_place(model, compiled)

    assert capture.targets.count(_HEAD_GROUPED_ATTENTION_ADD) == 1
    assert _ATTENTION_RESIDUAL_ADD not in capture.targets
    # The grouped add projects K and V itself, so no sequence-wide K or V remains.
    assert _DENSE_KEY not in capture.targets
    assert _DENSE_VALUE not in capture.targets
    assert capture.targets.count(_FFN_RESIDUAL_ADD) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_head_groups_apply_to_options_that_already_install_the_rewrite():
    model = _ResidualBlock()
    capture = _FunctionalCapture()
    options = residual_updates_compile_options(
        convrot_int8_piper_compile_options(
            convrot_int8_swiglu_ffn_compile_options(), fuse_output=True
        )
    )
    options = residual_updates_compile_options(options, attention_head_groups=2)
    compiled = _compile_copy_back(model, capture.wrap_options(options))
    _assert_updates_input_in_place(model, compiled)

    assert capture.targets.count(_HEAD_GROUPED_ATTENTION_ADD) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_head_groups_that_do_not_divide_the_heads_leave_attention_ungrouped(caplog):
    model = _ResidualBlock()
    capture = _FunctionalCapture()
    compiled = _compile_copy_back(model, _options(capture, attention_head_groups=3))
    with caplog.at_level("WARNING", logger=residual_compile.__name__):
        _assert_updates_input_in_place(model, compiled)

    assert "attention keeps one group" in caplog.text
    assert capture.targets.count(_ATTENTION_RESIDUAL_ADD) == 1
    assert _HEAD_GROUPED_ATTENTION_ADD not in capture.targets


def test_a_later_head_group_setting_replaces_the_earlier_rewrite_in_place():
    options = convrot_int8_piper_compile_options(fuse_output=True)
    ungrouped = residual_updates_compile_options(options)
    grouped = residual_updates_compile_options(ungrouped, attention_head_groups=2)

    passes = grouped["post_grad_custom_pre_pass"]
    assert passes[:-1] == options["post_grad_custom_pre_pass"]
    expected = residual_updates_compile_options(attention_head_groups=2)
    assert passes[-1].uuid() == expected["post_grad_custom_pre_pass"][-1].uuid()
    assert passes[-1].uuid() != ungrouped["post_grad_custom_pre_pass"][-1].uuid()


@pytest.mark.parametrize(
    ("attention_head_groups", "error", "message"),
    [
        (0, ValueError, "must be positive"),
        (True, TypeError, "integer"),
        (2.0, TypeError, "integer"),
    ],
)
def test_head_group_counts_must_be_positive_integers(attention_head_groups, error, message):
    with pytest.raises(error, match=message):
        residual_updates_compile_options(attention_head_groups=attention_head_groups)
