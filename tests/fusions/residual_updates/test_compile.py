"""A copy-back residual block updates its input in place after attention and FFN fusion."""

import pytest
import torch
from _compile_capture import TargetCapturePass
from torch.nn import functional as F  # noqa: N812

from piper_kernels import piper_attention
from piper_kernels.fusions.convrot_int8_piper import convrot_int8_piper_compile_options
from piper_kernels.fusions.convrot_int8_swiglu_ffn import convrot_int8_swiglu_ffn_compile_options
from piper_kernels.fusions.residual_updates import _compile as residual_compile
from piper_kernels.fusions.residual_updates import residual_updates_compile_options
from piper_kernels.linear.convrot.int8 import _ops

from ..convrot_int8_piper.test_compile import _available
from ..convrot_int8_swiglu_ffn._helpers import SwiGluFfn

_kernels = torch.ops.piper_kernels
_ATTENTION_OUTPUT = _kernels.convrot_int8_piper_projected_query_attention_output.default
_ATTENTION_RESIDUAL_ADD = (
    _kernels.convrot_int8_piper_projected_query_attention_residual_add_.default
)
_FFN_UPDATES = _kernels.convrot_int8_swiglu_ffn_gated_updates_.default
_FFN_RESIDUAL_ADD = _kernels.convrot_int8_swiglu_ffn_residual_add_.default
_RESIDUAL_ADD = _kernels.residual_add_.default


class _Ffn(SwiGluFfn):
    output_features = 256


class _ResidualBlock(torch.nn.Module):
    """Gated self-attention, then a gated SwiGLU FFN, over one hidden stream."""

    width = 256
    heads = 4
    kv_heads = 2
    head_dim = 64
    rotary_dim = 48

    def __init__(self) -> None:
        super().__init__()
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
        normalized = F.rms_norm(hidden, (self.width,), eps=1e-6)
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
        ffn_input = F.rms_norm(residual, (self.width,), eps=1e-6)
        ffn_input = ffn_input * (1 + ffn_scale[gate_indices]) + ffn_shift[gate_indices]
        return residual + ffn_gate[gate_indices] * self.ffn(ffn_input)


def _options(capture, *, fuse_output=True):
    options = convrot_int8_piper_compile_options(
        convrot_int8_swiglu_ffn_compile_options(), fuse_output=fuse_output
    )
    return capture.wrap_options(residual_updates_compile_options(options))


def _inputs(model, sequence):
    hidden = torch.randn(1, sequence, model.width, device="cuda", dtype=torch.bfloat16)
    angles = torch.rand(sequence, model.rotary_dim, device="cuda") * (2 * torch.pi)
    gate_indices = torch.randint(0, 3, (sequence,), device="cuda")
    return hidden, angles.cos(), angles.sin(), gate_indices


def _compiled_copy_back(model, capture, **options):
    def copy_back(hidden, cos, sin, gate_indices):
        return hidden.copy_(model(hidden, cos, sin, gate_indices))

    torch._dynamo.reset()
    return torch.compile(
        copy_back, fullgraph=True, dynamic=True, options=_options(capture, **options)
    )


def _assert_updates_input_in_place(model, compiled):
    with torch.no_grad():
        for sequence in (65, 300):
            hidden, cos, sin, gate_indices = _inputs(model, sequence)
            expected = model(hidden.clone(), cos, sin, gate_indices)
            result = compiled(hidden, cos, sin, gate_indices)

            assert result.data_ptr() == hidden.data_ptr()
            relative_l2 = (hidden.float() - expected.float()).norm() / expected.float().norm()
            assert relative_l2 < 0.01


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_copy_back_block_updates_its_input_in_place():
    model = _ResidualBlock()
    capture = TargetCapturePass()
    compiled = _compiled_copy_back(model, capture)
    _assert_updates_input_in_place(model, compiled)

    assert capture.targets.count(_ATTENTION_RESIDUAL_ADD) == 1
    assert capture.targets.count(_FFN_RESIDUAL_ADD) == 1
    assert _ATTENTION_OUTPUT not in capture.targets
    assert _FFN_UPDATES not in capture.targets
    assert _RESIDUAL_ADD not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
def test_unfused_attention_output_is_added_in_place_by_the_ffn_rewrite():
    model = _ResidualBlock()
    capture = TargetCapturePass()
    compiled = _compiled_copy_back(model, capture, fuse_output=False)

    _assert_updates_input_in_place(model, compiled)

    assert capture.targets.count(_RESIDUAL_ADD) == 1
    assert capture.targets.count(_FFN_RESIDUAL_ADD) == 1
    assert _ATTENTION_RESIDUAL_ADD not in capture.targets
    assert _FFN_UPDATES not in capture.targets


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
    assert residual_updates_compile_options(options)["post_grad_custom_pre_pass"][-1] is passes[-1]


def test_ffn_residual_adds_take_the_two_update_arguments_without_the_first_update():
    for two_updates, residual_add in residual_compile._FFN_RESIDUAL_ADDS.items():
        assert isinstance(two_updates, torch._ops.OpOverload)
        expected = [
            argument.name
            for argument in two_updates._schema.arguments
            if argument.name not in residual_compile._FIRST_UPDATE_ARGUMENTS
        ]
        assert [argument.name for argument in residual_add._schema.arguments] == expected
