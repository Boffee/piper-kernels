"""Compiler folding supports both Q-only and complete Q/K/V preparation."""

import operator

import pytest
import torch
from _compile_capture import TargetCapturePass
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.node import map_arg
from torch.nn import functional as F  # noqa: N812

from piper_kernels import piper_attention
from piper_kernels.attention.piper_attention import _quantized_dispatch
from piper_kernels.attention.piper_attention._nvidia import _plan as attention_plan
from piper_kernels.fusions.convrot_int8_piper import (
    _backend,
    _compile,
    convrot_int8_piper_compile_options,
    query,
)
from piper_kernels.fusions.convrot_int8_piper import (
    key as key_projection,
)
from piper_kernels.fusions.convrot_int8_piper import (
    value as value_projection,
)
from piper_kernels.linear.convrot.int8 import _compile as linear_compile
from piper_kernels.linear.convrot.int8 import _ops


def _semantic_graph(*, dtype=torch.bfloat16, affine=True, permute=True):
    graph = torch.fx.Graph()

    def placeholder(name, shape, tensor_dtype=dtype):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=tensor_dtype)
        return node

    def call(target, *args):
        node = graph.call_function(target, args)
        node.meta["val"] = target(*map_arg(args, lambda argument: argument.meta["val"]))
        return node

    with FakeTensorMode():
        hidden = placeholder("hidden", (2, 65, 256))
        weight = placeholder("weight", (256, 256), torch.int8)
        weight_scale = placeholder("weight_scale", (256, 1), torch.float32)
        bias = placeholder("bias", (256,), torch.float32)
        input_scale = placeholder("input_scale", (), torch.float32)
        norm = placeholder("norm", (64,)) if affine else None
        cos = placeholder("cos", (65, 48), torch.float32)
        sin = placeholder("sin", (65, 48), torch.float32)
        key = placeholder("key", (2, 2, 97, 64))
        value = placeholder("value", (2, 2, 97, 64))
        projected = call(
            torch.ops.piper_kernels.convrot_int8_linear.default,
            hidden,
            weight,
            weight_scale,
            bias,
            256,
            None,
            input_scale,
        )
        shaped = call(torch.ops.aten.reshape.default, projected, (2, 65, 4, 64))
        promoted = call(torch.ops.prims.convert_element_type.default, shaped, torch.float32)
        squared = call(torch.ops.aten.pow.Tensor_Scalar, promoted, 2)
        mean = call(torch.ops.aten.mean.dim, squared, [3], True)
        variance = call(torch.ops.aten.add.Scalar, mean, 1e-5)
        inverse_rms = call(torch.ops.aten.rsqrt.default, variance)
        normalized = call(torch.ops.aten.mul.Tensor, promoted, inverse_rms)
        scaled = call(torch.ops.aten.mul.Tensor, normalized, norm) if affine else normalized
        rounded = call(torch.ops.prims.convert_element_type.default, scaled, dtype)
        rotary = call(torch.ops.aten.slice.Tensor, rounded, 3, 0, 48)
        split = call(torch.ops.aten.split.Tensor, rotary, 24, -1)
        first, second = (call(operator.getitem, split, index) for index in (0, 1))
        rotated = call(
            torch.ops.aten.cat.default, [call(torch.ops.aten.neg.default, second), first], -1
        )

        def table_view(table):
            converted = call(torch.ops.prims.convert_element_type.default, table, dtype)
            return call(
                torch.ops.aten.unsqueeze.default,
                call(torch.ops.aten.unsqueeze.default, converted, 0),
                2,
            )

        direct = call(torch.ops.aten.mul.Tensor, rotary, table_view(cos))
        rotated = call(torch.ops.aten.mul.Tensor, rotated, table_view(sin))
        rotary_output = call(torch.ops.aten.add.Tensor, direct, rotated)
        tail = call(torch.ops.aten.slice.Tensor, rounded, 3, 48, torch.iinfo(torch.int64).max)
        transformed = call(torch.ops.aten.cat.default, [rotary_output, tail], -1)
        query_value = (
            call(torch.ops.aten.permute.default, transformed, [0, 2, 1, 3])
            if permute
            else call(torch.ops.aten.transpose.int, transformed, 1, 2)
        )
        output = call(
            torch.ops.piper_kernels.piper_attention.default, query_value, key, value, 0.125, False
        )
        graph.output(output)
    return torch.fx.GraphModule({}, graph).graph


def _targets(graph):
    return [node.target for node in graph.nodes if node.op == "call_function"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("permute", [False, True])
def test_fake_cpu_rewrite_preserves_floating_key_value_and_padded_query(
    monkeypatch, dtype, affine, permute
):
    graph = _semantic_graph(dtype=dtype, affine=affine, permute=permute)
    key = next(node for node in graph.nodes if node.target == "key")
    value = next(node for node in graph.nodes if node.target == "value")
    monkeypatch.setattr(_backend, "select_projection_backend", lambda *_args, **_kwargs: object())

    _compile.compile_pass(graph, is_inference=True)

    targets = _targets(graph)
    assert targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default) == 1
    assert targets.count(torch.ops.piper_kernels.convrot_int8_piper_project_query.default) == 1
    assert targets.count(torch.ops.piper_kernels.piper_attention_from_quantized_query.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in targets
    assert torch.ops.piper_kernels.piper_attention.default not in targets
    attention = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.piper_kernels.piper_attention_from_quantized_query.default
    )
    assert attention.args[2] is key
    assert attention.args[3] is value
    assert attention.args[0].meta["val"].shape == (2, 4, 128, 64)
    assert attention.args[1].meta["val"].shape == (2, 4, 4)
    assert attention.meta["val"].shape == (2, 4, 65, 64)
    assert attention.meta["val"].dtype is dtype


@pytest.mark.parametrize(
    "invalid",
    [
        "target",
        "training",
        "cos_stride",
        "norm",
        "input_scale",
        "input_features",
        "bias",
        "group",
        "scale",
        "causal",
        "escape",
        "rotation",
    ],
)
def test_unrecognized_or_unsupported_region_stays_unchanged(monkeypatch, invalid):
    graph = _semantic_graph()
    nodes = {node.name: node for node in graph.nodes}
    if invalid != "target":
        monkeypatch.setattr(
            _backend, "select_projection_backend", lambda *_args, **_kwargs: object()
        )
    with FakeTensorMode():
        if invalid == "cos_stride":
            nodes["cos"].meta["val"] = torch.empty((65, 96))[:, ::2]
        elif invalid == "norm":
            nodes["norm"].meta["val"] = torch.empty(32, dtype=torch.bfloat16)
        elif invalid == "input_scale":
            nodes["input_scale"].meta["val"] = torch.empty(1)
        elif invalid == "input_features":
            nodes["hidden"].meta["val"] = torch.empty((2, 65, 0), dtype=torch.bfloat16)
            nodes["weight"].meta["val"] = torch.empty((256, 0), dtype=torch.int8)
        elif invalid == "bias":
            nodes["bias"].meta["val"] = torch.empty(256, dtype=torch.int8)
    linear = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.piper_kernels.convrot_int8_linear.default
    )
    attention = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.piper_kernels.piper_attention.default
    )
    if invalid == "group":
        linear.args = (*linear.args[:4], 32, *linear.args[5:])
    elif invalid == "scale":
        attention.args = (*attention.args[:3], -0.125, attention.args[4])
    elif invalid == "causal":
        attention.args = (*attention.args[:4], True)
    elif invalid == "escape":
        output = next(node for node in graph.nodes if node.op == "output")
        output.args = ((attention, attention.args[0]),)
    elif invalid == "rotation":
        negated = next(node for node in graph.nodes if node.target is torch.ops.aten.neg.default)
        negated.target = torch.ops.aten.clone.default
    before = str(graph)

    _compile.compile_pass(graph, is_inference=invalid != "training")

    assert str(graph) == before


def test_compile_options_preserve_external_order_and_are_idempotent():
    before, after = object(), object()
    original = {
        "max_autotune": True,
        "post_grad_custom_pre_pass": [before, linear_compile.compile_pass, after],
    }
    options = convrot_int8_piper_compile_options(original)
    assert options["post_grad_custom_pre_pass"] == (
        before,
        _compile.compile_pass,
        linear_compile.compile_pass,
        after,
    )
    assert options["max_autotune"] is True
    assert original["post_grad_custom_pre_pass"] == [before, linear_compile.compile_pass, after]
    assert convrot_int8_piper_compile_options(options) == options
    assert _compile.compile_pass.uuid() == _compile.compile_pass.uuid()
    assert attention_plan.__file__ in _compile._source_files()


def test_output_pipeline_is_explicit_and_preserves_pass_order():
    before, after = object(), object()
    original = {"post_grad_custom_pre_pass": [before, linear_compile.compile_pass, after]}
    default = convrot_int8_piper_compile_options(original)
    assert _compile.output_compile_pass not in default["post_grad_custom_pre_pass"]
    enabled = convrot_int8_piper_compile_options(original, fuse_output=True)
    assert enabled["post_grad_custom_pre_pass"] == (
        before,
        _compile.compile_pass,
        _compile.output_compile_pass,
        linear_compile.compile_pass,
        after,
    )
    assert convrot_int8_piper_compile_options(enabled, fuse_output=True) == enabled
    assert original["post_grad_custom_pre_pass"] == [before, linear_compile.compile_pass, after]
    assert _compile.output_compile_pass.uuid() != _compile.compile_pass.uuid()
    with pytest.raises(TypeError, match="boolean"):
        convrot_int8_piper_compile_options(fuse_output=1)


def _available():
    return (
        torch.cuda.is_available()
        and _backend.select_projection_backend(torch.empty((), device="cuda")) is not None
    )


class _QueryAttention(torch.nn.Module):
    def __init__(self, *, head_dim, dtype, affine, rotary_dim, bias, causal, static):
        super().__init__()
        self.head_dim = head_dim
        self.heads = 4
        self.rotary_dim = rotary_dim
        self.causal = causal
        self.register_buffer(
            "weight", torch.randint(-127, 128, (4 * head_dim, 256), device="cuda", dtype=torch.int8)
        )
        self.register_buffer(
            "weight_scale", torch.rand((4 * head_dim, 1), device="cuda").mul_(0.01).add_(0.001)
        )
        self.register_parameter(
            "norm",
            torch.nn.Parameter(torch.rand(head_dim, device="cuda", dtype=dtype).add_(0.5))
            if affine
            else None,
        )
        self.register_parameter(
            "bias", torch.nn.Parameter(torch.randn(4 * head_dim, device="cuda")) if bias else None
        )
        self.register_buffer("input_scale", torch.tensor(0.08, device="cuda") if static else None)

    def forward(self, hidden, key, value, cos, sin):
        projected = _ops.linear(
            hidden, self.weight, self.weight_scale, self.bias, 256, None, self.input_scale
        )
        shaped = projected.view(hidden.shape[0], hidden.shape[1], self.heads, self.head_dim)
        normalized = F.rms_norm(shaped, (self.head_dim,), self.norm, 1e-5)
        rotary = normalized[..., : self.rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        rotary = (
            rotary * cos.to(hidden.dtype)[None, :, None, :]
            + rotated * sin.to(hidden.dtype)[None, :, None, :]
        )
        transformed = torch.cat((rotary, normalized[..., self.rotary_dim :]), dim=-1)
        return piper_attention(
            transformed.transpose(1, 2), key, value, scale=0.125, is_causal=self.causal
        )


def _inputs(model, *, sequence=65, key_length=97, dtype=torch.bfloat16):
    hidden = torch.randn(2, sequence, 256, device="cuda", dtype=dtype)
    key = torch.randn(2, 2, key_length, model.head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    angles = torch.rand(sequence, model.rotary_dim, device="cuda") * (2 * torch.pi)
    return hidden, key, value, angles.cos(), angles.sin()


def _explicit_fused(model, arguments):
    hidden, key, value, cos, sin = arguments
    qdata, scales = _ops.prepare_input(hidden, 256, None, model.input_scale)
    prepared_query, query_scales = query._project_query_op(
        qdata,
        scales,
        model.weight,
        model.weight_scale,
        model.norm,
        cos,
        sin,
        1e-5,
        0.125,
        model.bias,
        head_dim=model.head_dim,
    )
    return _quantized_dispatch._piper_attention_from_quantized_query_op(
        prepared_query, query_scales, key, value, hidden.shape[1], model.causal
    )


def _compiled_model(model, *, dynamic=False, fuse_output=False):
    torch._dynamo.reset()
    capture = TargetCapturePass()
    options = convrot_int8_piper_compile_options(fuse_output=fuse_output)
    options["post_grad_custom_pre_pass"] = (*options["post_grad_custom_pre_pass"], capture)
    return torch.compile(model, fullgraph=True, dynamic=dynamic, options=options), capture


def _assert_rewritten(capture):
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_int8_piper_project_query.default) == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.piper_attention_from_quantized_query.default)
        == 1
    )
    assert torch.ops.piper_kernels.piper_attention.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused Q projection support")
@pytest.mark.parametrize(
    ("head_dim", "dtype", "affine", "rotary_dim", "bias", "causal", "static"),
    [
        (64, torch.bfloat16, True, 48, False, False, False),
        (128, torch.bfloat16, False, 128, True, True, False),
        (64, torch.float16, False, 64, True, False, True),
        (128, torch.float16, True, 96, False, True, True),
    ],
)
def test_compiled_projection_matches_explicit_fusion_and_preserves_attention_quality(
    head_dim, dtype, affine, rotary_dim, bias, causal, static
):
    torch.manual_seed(603)
    model = _QueryAttention(
        head_dim=head_dim,
        dtype=dtype,
        affine=affine,
        rotary_dim=rotary_dim,
        bias=bias,
        causal=causal,
        static=static,
    )
    arguments = _inputs(model, key_length=65 if causal else 97, dtype=dtype)
    compiled, capture = _compiled_model(model)
    with torch.no_grad():
        reference = torch.compile(model, fullgraph=True)(*arguments)
        expected = _explicit_fused(model, arguments)
        actual = compiled(*arguments)
    _assert_rewritten(capture)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative_l2 < 0.025


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused Q projection support")
def test_dynamic_fullgraph_reuses_the_rewrite_and_captures_live_inputs():
    torch.manual_seed(604)
    model = _QueryAttention(
        head_dim=64,
        dtype=torch.bfloat16,
        affine=True,
        rotary_dim=48,
        bias=True,
        causal=False,
        static=False,
    )
    compiled, capture = _compiled_model(model, dynamic=True)
    with torch.no_grad():
        for sequence, key_length in ((65, 97), (129, 161)):
            arguments = _inputs(model, sequence=sequence, key_length=key_length)
            actual = compiled(*arguments)
            torch.testing.assert_close(actual, _explicit_fused(model, arguments), atol=0, rtol=0)
        _assert_rewritten(capture)
        assert capture.calls == 1
        compiled(*arguments)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(*arguments)
        arguments[0].mul_(0.5)
        expected = _explicit_fused(model, arguments)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


class _QKVAttention(_QueryAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("key_weight", self.weight[: 2 * self.head_dim].clone())
        self.register_buffer("key_weight_scale", self.weight_scale[: 2 * self.head_dim].clone())
        self.register_buffer("value_weight", self.key_weight.clone().neg_())
        self.register_buffer("value_weight_scale", self.key_weight_scale.clone())

    def forward(self, hidden, context, cos, sin, key_cos, key_sin):
        key = _ops.linear(
            context, self.key_weight, self.key_weight_scale, None, 256, None, self.input_scale
        )
        key = key.view(context.shape[0], context.shape[1], 2, self.head_dim)
        key = F.rms_norm(key, (self.head_dim,), self.norm, 1e-5)
        rotary = key[..., : self.rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        rotary = (
            rotary * key_cos.to(context.dtype)[None, :, None]
            + torch.cat((-second, first), -1) * key_sin.to(context.dtype)[None, :, None]
        )
        key = torch.cat((rotary, key[..., self.rotary_dim :]), -1).transpose(1, 2)
        value = _ops.linear(
            context, self.value_weight, self.value_weight_scale, None, 256, None, self.input_scale
        )
        value = value.view(context.shape[0], context.shape[1], 2, self.head_dim).transpose(1, 2)
        return super().forward(hidden, key, value, cos, sin)


def _qkv_inputs(model, sequence, key_length, dtype):
    hidden, _, _, cos, sin = _inputs(model, sequence=sequence, key_length=key_length, dtype=dtype)
    context = torch.randn(2, key_length, 256, device="cuda", dtype=dtype)
    angle = torch.rand(key_length, model.rotary_dim, device="cuda") * (2 * torch.pi)
    return hidden, context, cos, sin, angle.cos(), angle.sin()


def _explicit_qkv(model, arguments):
    hidden, context, cos, sin, key_cos, key_sin = arguments
    qdata, qs = _ops.prepare_input(hidden, 256, None, model.input_scale)
    data, scales = _ops.prepare_input(context, 256, None, model.input_scale)
    q, qscale = query._project_query_op(
        qdata,
        qs,
        model.weight,
        model.weight_scale,
        model.norm,
        cos,
        sin,
        1e-5,
        0.125,
        model.bias,
        head_dim=model.head_dim,
    )
    k, kscale = key_projection._project_key_op(
        data,
        scales,
        model.key_weight,
        model.key_weight_scale,
        model.norm,
        key_cos,
        key_sin,
        1e-5,
        head_dim=model.head_dim,
    )
    v, mult, logs, mean = value_projection._project_value_op(
        data,
        scales,
        model.value_weight,
        model.value_weight_scale,
        head_dim=model.head_dim,
        is_causal=model.causal,
    )
    return _quantized_dispatch._piper_attention_from_quantized_op(
        q,
        qscale,
        k,
        kscale,
        v,
        mult,
        logs,
        mean,
        hidden.shape[1],
        context.shape[1],
        model.causal,
        hidden.dtype,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
@pytest.mark.parametrize(
    ("head_dim", "causal", "dtype", "affine", "rotary_dim", "static"),
    [
        (64, False, torch.bfloat16, True, 48, False),
        (128, True, torch.bfloat16, False, 128, False),
        (64, True, torch.float16, False, 64, True),
        (128, False, torch.float16, True, 96, True),
    ],
)
def test_compiler_fuses_qkv_and_preserves_attention_quality(
    head_dim, causal, dtype, affine, rotary_dim, static
):
    torch._dynamo.reset()
    torch.manual_seed(923)
    model = _QKVAttention(
        head_dim=head_dim,
        causal=causal,
        dtype=dtype,
        affine=affine,
        rotary_dim=rotary_dim,
        bias=True,
        static=static,
    )
    arguments = _qkv_inputs(model, 65, 65 if causal else 97, dtype)
    compiled, capture = _compiled_model(model)
    with torch.no_grad():
        reference = torch.compile(model, fullgraph=True)(*arguments)
        actual = compiled(*arguments)
        expected = _explicit_qkv(model, arguments)
    assert torch.ops.piper_kernels.convrot_int8_piper_project_key.default in capture.targets
    assert torch.ops.piper_kernels.convrot_int8_piper_project_value.default in capture.targets
    assert torch.ops.piper_kernels.piper_attention_from_quantized.default in capture.targets
    assert (
        torch.ops.piper_kernels.piper_attention_from_quantized_query.default not in capture.targets
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    relative = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative < 0.025


@pytest.mark.gpu
@pytest.mark.skipif(not _available(), reason="requires dense fused projection support")
@pytest.mark.parametrize("causal", [False, True])
def test_qkv_dynamic_fullgraph_and_live_graph_capture(causal):
    torch._dynamo.reset()
    torch.manual_seed(924)
    model = _QKVAttention(
        head_dim=64,
        causal=causal,
        dtype=torch.bfloat16,
        affine=True,
        rotary_dim=48,
        bias=True,
        static=False,
    )
    compiled, capture = _compiled_model(model, dynamic=True)
    with torch.no_grad():
        for sequence in (65, 129):
            arguments = _qkv_inputs(
                model, sequence, sequence if causal else sequence + 32, torch.bfloat16
            )
            actual = compiled(*arguments)
            torch.testing.assert_close(actual, _explicit_qkv(model, arguments), atol=0, rtol=0)
        assert capture.calls == 1
        assert torch.ops.piper_kernels.piper_attention_from_quantized.default in capture.targets
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(*arguments)
        arguments[0].mul_(0.5)
        arguments[1].mul_(0.25)
        arguments[-1].neg_()
        expected = _explicit_qkv(model, arguments)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


def _qkv_semantic_graph():
    graph = _semantic_graph()
    attention = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.piper_kernels.piper_attention.default
    )
    transformed = attention.args[0]
    linear = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.piper_kernels.convrot_int8_linear.default
    )
    original_nodes = list(graph.nodes)
    copies = {node: node for node in original_nodes if node.op == "placeholder"}
    with graph.inserting_before(attention):
        for node in original_nodes:
            if node is attention:
                break
            if node.op == "call_function":
                copies[node] = graph.node_copy(node, lambda argument: copies[argument])
        projected_value = graph.node_copy(linear, lambda argument: argument)
        shaped = graph.call_function(
            torch.ops.aten.reshape.default, args=(projected_value, (2, 65, 4, 64))
        )
        value = graph.call_function(torch.ops.aten.permute.default, args=(shaped, [0, 2, 1, 3]))
        with FakeTensorMode():
            shaped.meta["val"] = torch.empty((2, 65, 4, 64), dtype=torch.bfloat16)
            value.meta["val"] = torch.empty((2, 4, 65, 64), dtype=torch.bfloat16)
    attention.args = (transformed, copies[transformed], value, 0.125, False)
    graph.lint()
    return graph


def test_qkv_fake_rewrite_shares_preparation_without_hardware_execution(monkeypatch):
    graph = _qkv_semantic_graph()
    monkeypatch.setattr(_backend, "select_projection_backend", lambda *_args, **_kwargs: object())
    _compile.compile_pass(graph, is_inference=True)
    targets = _targets(graph)
    assert targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default) == 1
    for operation in ("query", "key", "value"):
        assert (
            targets.count(
                getattr(torch.ops.piper_kernels, f"convrot_int8_piper_project_{operation}").default
            )
            == 1
        )
    assert targets.count(torch.ops.piper_kernels.piper_attention_from_quantized.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in targets
    graph.lint()


def test_qkv_preparation_reuse_keeps_late_tuple_users_after_attention(monkeypatch):
    graph = _qkv_semantic_graph()
    linear = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.piper_kernels.convrot_int8_linear.default
    )
    attention = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.piper_kernels.piper_attention.default
    )
    with graph.inserting_before(linear):
        data, scales, _ = _compile._compile_fx.emit_prepared_input(
            graph, linear.args[0], 256, None, (2, 65, 256), linear.args[6]
        )
    attention.append(data)
    data.append(scales)
    output = next(node for node in graph.nodes if node.op == "output")
    output.args = ((attention, data, scales),)
    graph.lint()
    monkeypatch.setattr(_backend, "select_projection_backend", lambda *_args, **_kwargs: object())

    _compile.compile_pass(graph, is_inference=True)

    graph.lint()
    targets = _targets(graph)
    assert torch.ops.piper_kernels.piper_attention_from_quantized.default in targets
    assert targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default) == 2
    assert output.args[0][1:] == (data, scales)


@pytest.mark.parametrize("escaped_operand", [1, 2])
def test_context_escape_preserves_the_query_only_rewrite(monkeypatch, escaped_operand):
    graph = _qkv_semantic_graph()
    attention = next(
        node
        for node in graph.nodes
        if node.target is torch.ops.piper_kernels.piper_attention.default
    )
    output = next(node for node in graph.nodes if node.op == "output")
    output.args = ((attention, attention.args[escaped_operand]),)
    monkeypatch.setattr(_backend, "select_projection_backend", lambda *_args, **_kwargs: object())
    _compile.compile_pass(graph, is_inference=True)
    targets = _targets(graph)
    assert torch.ops.piper_kernels.piper_attention_from_quantized_query.default in targets
    assert torch.ops.piper_kernels.piper_attention_from_quantized.default not in targets
    graph.lint()
