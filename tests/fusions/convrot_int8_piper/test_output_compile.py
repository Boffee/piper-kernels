"""Compiler integration for complete dense Q/K/V and output projection fusion."""

import pytest
import torch

from piper_kernels.fusions.convrot_int8_piper import _backend, _compile, _output_compile
from piper_kernels.linear.convrot.int8 import _ops

from .test_compile import (
    _available,
    _compiled_model,
    _explicit_qkv,
    _qkv_inputs,
    _qkv_semantic_graph,
    _QKVAttention,
)


class _QKVOutput(_QKVAttention):
    def __init__(self, *, output_static=False, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(
            "output_weight",
            torch.randint(-100, 100, (80, 4 * self.head_dim), device="cuda", dtype=torch.int8),
        )
        self.register_buffer("output_weight_scale", torch.full((80, 1), 0.001, device="cuda"))
        self.register_buffer(
            "output_bias", torch.randn(80, device="cuda", dtype=self.weight_scale.dtype)
        )
        self.register_buffer(
            "output_scale", torch.tensor(0.03, device="cuda") if output_static else None
        )

    def forward(self, hidden, context, cos, sin, key_cos, key_sin):
        attended = super().forward(hidden, context, cos, sin, key_cos, key_sin)
        merged = attended.transpose(1, 2).reshape(
            hidden.shape[0], hidden.shape[1], 4 * self.head_dim
        )
        return _ops.linear(
            merged,
            self.output_weight,
            self.output_weight_scale,
            self.output_bias,
            64,
            None,
            self.output_scale,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
@pytest.mark.parametrize(
    ("head_dim", "causal", "dtype", "static"),
    [(64, False, torch.bfloat16, False), (128, True, torch.float16, True)],
)
def test_compiler_folds_complete_dense_attention_output(head_dim, causal, dtype, static):
    torch._dynamo.reset()
    model = _QKVOutput(
        head_dim=head_dim,
        causal=causal,
        dtype=dtype,
        affine=True,
        rotary_dim=head_dim * 3 // 4,
        bias=True,
        static=static,
        output_static=static,
    )
    compiled, capture = _compiled_model(model, dynamic=True, fuse_output=True)
    with torch.no_grad():
        for sequence in (65, 4353):
            args = _qkv_inputs(model, sequence, sequence if causal else sequence + 32, dtype)
            actual = compiled(*args)
            attended = _explicit_qkv(model, args).transpose(1, 2).reshape(2, sequence, -1)
            expected = _ops.linear(
                attended,
                model.output_weight,
                model.output_weight_scale,
                model.output_bias,
                64,
                None,
                model.output_scale,
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert capture.calls == 1
    assert (
        torch.ops.piper_kernels.convrot_int8_piper_projected_query_attention_output.default
        in capture.targets
    )
    assert torch.ops.piper_kernels.convrot_int8_piper_project_query.default not in capture.targets
    assert torch.ops.piper_kernels.piper_attention_from_quantized.default not in capture.targets


def _semantic_output_graph():
    graph = _qkv_semantic_graph()
    old_output = next(node for node in graph.nodes if node.op == "output")
    attention = old_output.args[0]
    with graph.inserting_before(attention):
        weight = graph.placeholder("output_weight")
        scale = graph.placeholder("output_scale")
        bias = graph.placeholder("output_bias")
    with attention.meta["val"].fake_mode:
        weight.meta["val"] = torch.empty(80, 256, dtype=torch.int8)
        scale.meta["val"] = torch.empty(80, 1)
        bias.meta["val"] = torch.empty(80, dtype=torch.bfloat16)
        with graph.inserting_before(old_output):
            transposed = graph.call_function(torch.ops.aten.transpose.int, args=(attention, 1, 2))
            transposed.meta["val"] = attention.meta["val"].transpose(1, 2)
            cloned = graph.call_function(torch.ops.aten.clone.default, args=(transposed,))
            cloned.meta["val"] = transposed.meta["val"].contiguous()
            shaped = graph.call_function(
                torch.ops.aten.reshape.default, args=(cloned, (2, 65, 256))
            )
            shaped.meta["val"] = cloned.meta["val"].reshape(2, 65, 256)
            linear = graph.call_function(
                torch.ops.piper_kernels.convrot_int8_linear.default,
                args=(shaped, weight, scale, bias, 64, None, None),
            )
            linear.meta["val"] = torch.empty(2, 65, 80, dtype=torch.bfloat16)
    old_output.args = (linear,)
    graph.lint()
    return graph, attention, transposed, linear, old_output


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "attention_escape",
        "query_escape",
        "layout",
        "activation",
        "weight",
        "metadata",
        "backend",
    ],
)
def test_output_rewrite_preserves_unsupported_or_escaping_regions(monkeypatch, unsupported):
    graph, attention, transposed, linear, old_output = _semantic_output_graph()
    monkeypatch.setattr(_backend, "select_projection_backend", lambda *a, **kw: object())
    monkeypatch.setattr(
        _output_compile.linear_backend,
        "select_linear_backend",
        lambda *a, **kw: None if unsupported == "backend" else object(),
    )
    if unsupported == "attention_escape":
        old_output.args = ((linear, attention),)
    elif unsupported == "query_escape":
        old_output.args = ((linear, attention.args[0]),)
    elif unsupported == "layout":
        transposed.args = (attention, 0, 1)
    elif unsupported == "activation":
        linear.args = (*linear.args[:5], "silu", None)
    elif unsupported == "weight":
        linear.args[1].meta["val"] = torch.empty(80, 128, dtype=torch.int8)
    elif unsupported == "metadata":
        linear.args[2].meta["val"] = None
    _compile.compile_pass(graph, is_inference=True)
    _compile.output_compile_pass(graph, is_inference=True)
    graph.lint()
    targets = [node.target for node in graph.nodes]
    assert (
        torch.ops.piper_kernels.convrot_int8_piper_projected_query_attention_output.default
        in targets
    ) is (unsupported is None)
    if unsupported is not None:
        assert torch.ops.piper_kernels.convrot_int8_linear.default in targets


def test_output_rewrite_keeps_a_different_logical_query_extent(monkeypatch):
    graph, _, transposed, linear, _ = _semantic_output_graph()
    monkeypatch.setattr(_backend, "select_projection_backend", lambda *a, **kw: object())
    monkeypatch.setattr(
        _output_compile.linear_backend, "select_linear_backend", lambda *a, **kw: object()
    )
    _compile.compile_pass(graph, is_inference=True)
    attention = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.piper_kernels.piper_attention_from_quantized.default
    )
    # Q storage can describe a different logical extent within its final Q64 tile.
    # The output fusion must retain the extent actually consumed by attention.
    attention.args = (*attention.args[:8], 96, *attention.args[9:])
    reshaped = linear.args[0]
    cloned = reshaped.args[0]
    with attention.meta["val"].fake_mode:
        attention.meta["val"] = torch.empty(2, 4, 96, 64, dtype=torch.bfloat16)
        transposed.meta["val"] = attention.meta["val"].transpose(1, 2)
        cloned.meta["val"] = transposed.meta["val"].contiguous()
        reshaped.args = (cloned, (2, 96, 256))
        reshaped.meta["val"] = cloned.meta["val"].reshape(2, 96, 256)
        linear.meta["val"] = torch.empty(2, 96, 80, dtype=torch.bfloat16)
    graph.lint()
    assert not _output_compile.fold_attention_output(graph)
