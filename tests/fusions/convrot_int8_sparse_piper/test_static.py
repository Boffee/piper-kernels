"""Independent static input scales survive complete sparse-attention fusion."""

import pytest
import torch

from piper_kernels.fusions.convrot_int8_sparse_piper import (
    convrot_int8_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import output as output_fusion

from .._accuracy import assert_fusion_output_close
from ._helpers import output_available
from .test_compile import (
    _POST_GRAD_PRE_PASS,
    _CoarseSparseProjectionAttention,
    _ProjectedGateCoarseSparseAttentionOutput,
    _run_explicit_attention_output,
    _SparseProjectionAttentionOutput,
    _TargetCapturePass,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not output_available(), reason="requires fused sparse output support"),
]


def _set_scales(model, mode):
    shared = torch.tensor(0.03, device="cuda")
    if mode == "shared":
        scales = (shared,) * 4
        preparations = 1
    elif mode == "distinct":
        scales = tuple(torch.tensor(value, device="cuda") for value in (0.02, 0.03, 0.04, 0.05))
        preparations = 4
    elif mode == "mixed":
        scales = (shared, None, torch.tensor(0.04, device="cuda"), None)
        preparations = 3
    elif mode == "static-output":
        scales = (None,) * 4
        preparations = 1
    else:
        raise AssertionError(mode)
    for projection, scale in zip(
        (model.query, model.key, model.value, model.gate), scales, strict=True
    ):
        projection.weight.act_per_tensor_scale = scale
    model.output.weight.act_per_tensor_scale = torch.tensor(0.04, device="cuda")
    return preparations


def _options(capture):
    options = convrot_int8_sparse_piper_compile_options()
    options[_POST_GRAD_PRE_PASS] = (*options[_POST_GRAD_PRE_PASS], capture)
    return options


@pytest.mark.parametrize("mode", ["shared", "distinct", "mixed", "static-output"])
@pytest.mark.parametrize("routing", ["mean", "minmax"])
@pytest.mark.parametrize("chunk_rows", [64, 4096])
def test_static_sparse_attention_preserves_independent_projection_scales(
    monkeypatch, mode, routing, chunk_rows
):
    torch._dynamo.reset()
    torch.manual_seed(1081)
    monkeypatch.setattr(output_fusion, "_DEFAULT_QUERY_CHUNK_ROWS", chunk_rows)
    monkeypatch.setattr(_ProjectedGateCoarseSparseAttentionOutput, "sequence_length", 320)
    model = _ProjectedGateCoarseSparseAttentionOutput(routing=routing).eval()
    model.set_projection_bias(torch.bfloat16)
    preparations = _set_scales(model, mode)
    hidden_states = torch.randn(1, 320, model.input_features, device="cuda", dtype=torch.bfloat16)
    block_lengths = torch.tensor([64, 17, 51, 64, 1], device="cuda", dtype=torch.int32)
    capture = _TargetCapturePass()
    with torch.inference_mode():
        coarse_gate = model.gate(hidden_states).view(1, 320, model.heads, model.head_dim)
        expected, _ = _run_explicit_attention_output(
            model,
            hidden_states,
            model.cos,
            model.sin,
            model.sparse_key_blocks,
            coarse_gate=coarse_gate,
            coarse_scale=model.coarse_scale,
            coarse_key_blocks=model.coarse_key_blocks,
            block_lengths=block_lengths,
            sparse_query_blocks=2,
        )
        actual = torch.compile(model, fullgraph=True, options=_options(capture))(
            hidden_states, block_lengths, 2
        )
    assert_fusion_output_close(actual, expected)
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default)
        == preparations
    )
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets
    assert torch.ops.piper_kernels.convrot_int8_linear_prepared.default not in capture.targets


@pytest.mark.parametrize("projection", ["query", "key", "value", "output"])
def test_static_sparse_scales_can_change_without_recompilation(monkeypatch, projection):
    torch._dynamo.reset()
    torch.manual_seed(1082)
    monkeypatch.setattr(_SparseProjectionAttentionOutput, "sequence_length", 257)
    model = _SparseProjectionAttentionOutput().eval()
    for name, value in zip(
        ("query", "key", "value", "output"), (0.03, 0.04, 0.05, 0.06), strict=True
    ):
        getattr(model, name).weight.act_per_tensor_scale = torch.tensor(value, device="cuda")
    hidden_states = torch.randn(
        1, model.sequence_length, model.input_features, device="cuda", dtype=torch.bfloat16
    )
    capture = _TargetCapturePass()
    compiled = torch.compile(model, fullgraph=True, options=_options(capture))
    with torch.inference_mode():
        first = compiled(hidden_states)
        getattr(model, projection).weight.act_per_tensor_scale.mul_(0.5)
        expected, _ = _run_explicit_attention_output(
            model, hidden_states, model.cos, model.sin, model.sparse_key_blocks
        )
        second = compiled(hidden_states)
    assert_fusion_output_close(second, expected)
    assert not torch.equal(first, second)
    assert capture.calls == 1


def test_static_coarse_gate_keeps_valid_metadata_when_attention_escapes():
    class EscapingAttention(_ProjectedGateCoarseSparseAttentionOutput):
        def forward(self, hidden_states):
            gate = self.gate(hidden_states).view(1, self.sequence_length, self.heads, self.head_dim)
            attention = _CoarseSparseProjectionAttention.forward(self, hidden_states, gate)
            return self.output(attention.flatten(2)), attention

    torch._dynamo.reset()
    torch.manual_seed(1083)
    model = EscapingAttention(routing="minmax").eval()
    _set_scales(model, "distinct")
    hidden = torch.randn(
        1, model.sequence_length, model.input_features, device="cuda", dtype=torch.bfloat16
    )
    capture = _TargetCapturePass()
    with torch.inference_mode():
        gate = model.gate(hidden).view(1, model.sequence_length, model.heads, model.head_dim)
        expected = _run_explicit_attention_output(
            model,
            hidden,
            model.cos,
            model.sin,
            model.sparse_key_blocks,
            coarse_gate=gate,
            coarse_scale=model.coarse_scale,
            coarse_key_blocks=model.coarse_key_blocks,
        )
        actual = torch.compile(model, fullgraph=True, options=_options(capture))(hidden)
    for result, reference in zip(actual, expected, strict=True):
        assert_fusion_output_close(result, reference)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default) == 4
    assert (
        torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
        not in capture.targets
    )
