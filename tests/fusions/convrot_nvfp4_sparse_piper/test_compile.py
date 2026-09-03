"""Tests for automatic ConvRot NVFP4 sparse-Piper graph fusion."""

from __future__ import annotations

import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._routes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.fusions.convrot_nvfp4_sparse_piper import (
    convrot_nvfp4_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_nvfp4_sparse_piper._compile import (
    compile_pass as fusion_compile_pass,
)
from piper_kernels.fusions.convrot_nvfp4_sparse_piper.output import (
    _attention_output_op,
)
from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn import (
    convrot_nvfp4_swiglu_ffn_compile_options,
)
from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn._compile import (
    compile_pass as ffn_compile_pass,
)
from piper_kernels.fusions.nvfp4_sparse_piper import key as fused_key
from piper_kernels.fusions.nvfp4_sparse_piper import query as fused_query
from piper_kernels.fusions.nvfp4_sparse_piper import value as fused_value
from piper_kernels.linear.convrot._rotation import rotate_groups
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.linear.convrot.nvfp4 import _ops as convrot_nvfp4_ops
from piper_kernels.linear.convrot.nvfp4._compile import (
    compile_pass as convrot_nvfp4_compile_pass,
)
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4.triton import linear_mean

from ..nvfp4_sparse_piper.test_compile import (
    _ProjectedGateCoarseSparseAttentionOutput,
    _quantized_attention_output_graph,
    _semantic_attention_graph,
    _SparseProjectionAttentionOutput,
)

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"
_GROUP_SIZE = 16


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _convrot_graph(
    *,
    dynamic: bool,
    output_dynamic: bool | None = None,
    routing_mode: int = _MINMAX_ROUTING,
    with_block_lengths: bool = False,
    with_coarse: bool = False,
    with_sparse_query_blocks: bool = False,
) -> torch.fx.Graph:
    graph = _semantic_attention_graph(
        dynamic=dynamic,
        output_dynamic=output_dynamic,
        routing_mode=routing_mode,
        with_block_lengths=with_block_lengths,
        with_coarse=with_coarse,
        with_sparse_query_blocks=with_sparse_query_blocks,
    )
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.piper_kernels.nvfp4_linear.default
        ):
            node.target = torch.ops.piper_kernels.convrot_nvfp4_linear.default
            node.args = (*node.args, _GROUP_SIZE)
    graph.lint()
    return graph


def _run_passes(graph: torch.fx.Graph, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        AcceleratorTarget,
        "from_device",
        classmethod(lambda _cls, _device: AcceleratorTarget("cuda", "sm120")),
    )
    convrot_nvfp4_compile_pass(graph, is_inference=True)
    fusion_compile_pass(graph, is_inference=True)


def test_compile_options_install_versioned_idempotent_passes() -> None:
    options = convrot_nvfp4_sparse_piper_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (
        convrot_nvfp4_compile_pass,
        fusion_compile_pass,
    )
    assert convrot_nvfp4_sparse_piper_compile_options(options) == options
    assert fusion_compile_pass.uuid() == fusion_compile_pass.uuid()
    assert fusion_compile_pass.uuid()


@pytest.mark.parametrize("sparse_first", [False, True])
def test_compile_options_compose_with_convrot_nvfp4_ffn(sparse_first: bool) -> None:
    if sparse_first:
        options = convrot_nvfp4_swiglu_ffn_compile_options(
            convrot_nvfp4_sparse_piper_compile_options()
        )
    else:
        options = convrot_nvfp4_sparse_piper_compile_options(
            convrot_nvfp4_swiglu_ffn_compile_options()
        )

    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    assert passes[0] is convrot_nvfp4_compile_pass
    assert set(passes[1:]) == {fusion_compile_pass, ffn_compile_pass}


@pytest.mark.parametrize(("dynamic", "preparation_count"), [(False, 3), (True, 1)])
@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
def test_convrot_preparation_reuses_nvfp4_sparse_projection_kernels(
    monkeypatch: pytest.MonkeyPatch,
    dynamic: bool,
    preparation_count: int,
    routing_mode: int,
) -> None:
    graph = _convrot_graph(dynamic=dynamic, routing_mode=routing_mode)

    _run_passes(graph, monkeypatch)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert (
        targets.count(torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default)
        == preparation_count
    )
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_linear_mean.default) == 1
    assert targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default) == 1
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in targets
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in targets
    graph.lint()


def test_static_convrot_output_fuses_after_sparse_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _convrot_graph(dynamic=True, output_dynamic=False)

    _run_passes(graph, monkeypatch)

    call_nodes = [node for node in graph.nodes if node.op == "call_function"]
    targets = [node.target for node in call_nodes]
    fused = torch.ops.piper_kernels.convrot_nvfp4_sparse_piper_attention_output.default
    assert targets.count(fused) == 1
    assert torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    output_node = next(node for node in call_nodes if node.target is fused)
    assert output_node.args[19:21] == (_GROUP_SIZE, 8_192)
    graph.lint()


@pytest.mark.parametrize("with_block_lengths", [False, True])
@pytest.mark.parametrize("with_coarse", [False, True])
@pytest.mark.parametrize("with_sparse_query_blocks", [False, True])
def test_complete_convrot_fold_supports_every_bounded_attention_variant(
    monkeypatch: pytest.MonkeyPatch,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
) -> None:
    graph = _convrot_graph(
        dynamic=False,
        output_dynamic=False,
        with_block_lengths=with_block_lengths,
        with_coarse=with_coarse,
        with_sparse_query_blocks=with_sparse_query_blocks,
    )

    _run_passes(graph, monkeypatch)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    expected_value = (
        torch.ops.piper_kernels.nvfp4_sparse_piper_project_value_with_block_means.default
        if with_coarse
        else torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default
    )
    assert targets.count(expected_value) == 1
    assert (
        targets.count(torch.ops.piper_kernels.convrot_nvfp4_sparse_piper_attention_output.default)
        == 1
    )
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in targets
    assert torch.ops.piper_kernels.sparse_piper_coarse_residual.default not in targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    graph.lint()


@pytest.mark.parametrize("with_block_lengths", [False, True])
@pytest.mark.parametrize("with_coarse", [False, True])
@pytest.mark.parametrize("with_sparse_query_blocks", [False, True])
def test_convrot_output_fold_supports_every_bounded_attention_variant(
    monkeypatch: pytest.MonkeyPatch,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
) -> None:
    graph = _quantized_attention_output_graph(
        with_block_lengths=with_block_lengths,
        with_coarse=with_coarse,
        with_sparse_query_blocks=with_sparse_query_blocks,
    )
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.piper_kernels.nvfp4_linear.default
        ):
            node.target = torch.ops.piper_kernels.convrot_nvfp4_linear.default
            node.args = (*node.args, _GROUP_SIZE)

    _run_passes(graph, monkeypatch)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    fused = torch.ops.piper_kernels.convrot_nvfp4_sparse_piper_attention_output.default
    assert targets.count(fused) == 1
    assert torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in targets
    assert (
        torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
        not in targets
    )
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    graph.lint()


def test_dynamic_convrot_output_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _convrot_graph(dynamic=False, output_dynamic=True)

    _run_passes(graph, monkeypatch)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert (
        torch.ops.piper_kernels.convrot_nvfp4_sparse_piper_attention_output.default not in targets
    )
    assert targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default) == 1
    assert targets.count(torch.ops.piper_kernels.convrot_nvfp4_linear.default) == 1
    graph.lint()


def _convert_weight(weight: torch.Tensor) -> ConvRotNVFP4Tensor:
    assert isinstance(weight, TorchAONVFP4Tensor)
    dense = weight.dequantize(torch.bfloat16)
    rotated = rotate_groups(dense, _GROUP_SIZE)
    converted = TorchAONVFP4Tensor.to_nvfp4(
        rotated,
        per_tensor_scale=per_tensor_amax_to_scale(rotated.abs().amax()),
        act_per_tensor_scale=weight.act_per_tensor_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=weight.act_quant_kwargs,
    )
    return ConvRotNVFP4Tensor.from_torchao(converted, group_size=_GROUP_SIZE)


class _ConvRotSparseProjectionAttentionOutput(_SparseProjectionAttentionOutput):
    def __init__(self, *, dynamic: bool, routing: str = "minmax") -> None:
        super().__init__(dynamic=dynamic, routing=routing)
        for projection in (self.query, self.key, self.value, self.output):
            projection.weight = torch.nn.Parameter(
                _convert_weight(projection.weight),
                requires_grad=False,
            )


class _ConvRotProjectedGateCoarseAttentionOutput(_ProjectedGateCoarseSparseAttentionOutput):
    """Coarse sparse attention with ConvRot NVFP4 Q/K/V/gate/output linears."""

    def __init__(self, *, dynamic: bool, routing: str) -> None:
        super().__init__(dynamic=dynamic, routing=routing)
        for projection in (self.query, self.key, self.value, self.gate, self.output):
            projection.weight = torch.nn.Parameter(
                _convert_weight(projection.weight),
                requires_grad=False,
            )


def _prepared_input(
    input: torch.Tensor,  # noqa: A002
    projection: torch.nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = projection.weight
    assert isinstance(weight, ConvRotNVFP4Tensor)
    quantization = weight.act_quant_kwargs
    assert quantization is not None
    return convrot_nvfp4_ops.prepare_input(
        input,
        weight.act_per_tensor_scale,
        quantization.use_dynamic_per_tensor_scale,
        weight.group_size,
    )


def _explicit_fused(
    model: _ConvRotSparseProjectionAttentionOutput,
    input: torch.Tensor,  # noqa: A002
) -> torch.Tensor:
    prepared = []
    for projection in (model.query, model.key, model.value):
        weight = projection.weight
        assert isinstance(weight, ConvRotNVFP4Tensor)
        quantization = weight.act_quant_kwargs
        assert quantization is not None
        prepared.append(
            convrot_nvfp4_ops.prepare_input(
                input,
                weight.act_per_tensor_scale,
                quantization.use_dynamic_per_tensor_scale,
                weight.group_size,
            )
        )
    q_weight = model.query.weight
    k_weight = model.key.weight
    v_weight = model.value.weight
    assert isinstance(q_weight, ConvRotNVFP4Tensor)
    assert isinstance(k_weight, ConvRotNVFP4Tensor)
    assert isinstance(v_weight, ConvRotNVFP4Tensor)
    query = fused_query.project_query(
        *prepared[0],
        q_weight.qdata,
        q_weight.scale,
        q_weight.per_tensor_scale,
        model.query.bias,
        model.query_norm,
        model.cos,
        model.sin,
        1e-5,
        model.head_dim**-0.5,
        4_096,
        model.sparse_attention._routing_mode,
    )
    key = fused_key.project_key(
        *prepared[1],
        k_weight.qdata,
        k_weight.scale,
        k_weight.per_tensor_scale,
        model.key.bias,
        model.key_norm,
        model.cos,
        model.sin,
        1e-5,
        4_096,
        model.sparse_attention._routing_mode,
    )
    value_mean = linear_mean(
        *prepared[2],
        v_weight.qdata,
        v_weight.scale,
        v_weight.per_tensor_scale,
        model.value.bias,
        model.batch,
        model.sequence_length,
    ).view(model.batch, model.heads, model.head_dim)
    value = fused_value.project_value(
        *prepared[2],
        v_weight.qdata,
        v_weight.scale,
        v_weight.per_tensor_scale,
        model.value.bias,
        value_mean,
        4_096,
    )
    output_weight = model.output.weight
    assert isinstance(output_weight, ConvRotNVFP4Tensor)
    return _attention_output_op(
        *query,
        *key,
        *value,
        value_mean,
        list(model.sparse_attention._head_keep_ratio_units),
        model.sparse_key_blocks,
        model.sequence_length,
        model.sparse_attention._routing_mode,
        output_weight.qdata,
        output_weight.scale,
        output_weight.per_tensor_scale,
        output_weight.act_per_tensor_scale,
        model.output.bias,
        output_weight.group_size,
        8_192,
    )


def _explicit_fused_projected_gate(
    model: _ConvRotProjectedGateCoarseAttentionOutput,
    input: torch.Tensor,  # noqa: A002
    block_lengths: torch.Tensor,
    sparse_query_blocks: int,
) -> torch.Tensor:
    q_input, k_input, v_input, gate_input = (
        _prepared_input(input, projection)
        for projection in (model.query, model.key, model.value, model.gate)
    )
    q_weight = model.query.weight
    k_weight = model.key.weight
    v_weight = model.value.weight
    gate_weight = model.gate.weight
    output_weight = model.output.weight
    assert isinstance(q_weight, ConvRotNVFP4Tensor)
    assert isinstance(k_weight, ConvRotNVFP4Tensor)
    assert isinstance(v_weight, ConvRotNVFP4Tensor)
    assert isinstance(gate_weight, ConvRotNVFP4Tensor)
    assert isinstance(output_weight, ConvRotNVFP4Tensor)
    query = fused_query.project_query(
        *q_input,
        q_weight.qdata,
        q_weight.scale,
        q_weight.per_tensor_scale,
        model.query.bias,
        model.query_norm,
        model.cos,
        model.sin,
        1e-5,
        model.head_dim**-0.5,
        4_096,
        model.sparse_attention._routing_mode,
        block_lengths,
    )
    key = fused_key.project_key(
        *k_input,
        k_weight.qdata,
        k_weight.scale,
        k_weight.per_tensor_scale,
        model.key.bias,
        model.key_norm,
        model.cos,
        model.sin,
        1e-5,
        4_096,
        model.sparse_attention._routing_mode,
        block_lengths,
    )
    value_mean = linear_mean(
        *v_input,
        v_weight.qdata,
        v_weight.scale,
        v_weight.per_tensor_scale,
        model.value.bias,
        model.batch,
        model.sequence_length,
        block_lengths,
    ).view(model.batch, model.heads, model.head_dim)
    value = fused_value.project_value_with_block_means(
        *v_input,
        v_weight.qdata,
        v_weight.scale,
        v_weight.per_tensor_scale,
        model.value.bias,
        value_mean,
        4_096,
        block_lengths,
    )
    coarse_gate = nvfp4_ops.linear_prepared(
        *gate_input,
        gate_weight.qdata,
        gate_weight.scale,
        gate_weight.per_tensor_scale,
        model.gate.bias,
        torch.bfloat16,
    ).view(model.batch, model.sequence_length, model.heads, model.head_dim)
    return _attention_output_op(
        *query,
        *key,
        *value[:2],
        value_mean,
        list(model.sparse_attention._head_keep_ratio_units),
        model.sparse_key_blocks,
        model.sequence_length,
        model.sparse_attention._routing_mode,
        output_weight.qdata,
        output_weight.scale,
        output_weight.per_tensor_scale,
        output_weight.act_per_tensor_scale,
        model.output.bias,
        output_weight.group_size,
        8_192,
        block_lengths,
        value[2],
        coarse_gate,
        model.coarse_scale,
        model.coarse_key_blocks,
        sparse_query_blocks,
    )


class _TargetCapturePass(CustomInferenceAwareGraphPass):
    def __init__(self) -> None:
        self.targets: list[object] = []
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("dynamic", "routing"),
    [(False, "minmax"), (True, "minmax"), (False, "mean")],
)
def test_cuda_compile_fuses_complete_convrot_nvfp4_sparse_attention(
    dynamic: bool,
    routing: str,
) -> None:
    torch.manual_seed(967 + dynamic)
    model = _ConvRotSparseProjectionAttentionOutput(dynamic=dynamic, routing=routing).eval()
    input = torch.randn(  # noqa: A001
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_nvfp4_sparse_piper_compile_options()
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*passes, capture)
    with torch.no_grad():
        expected = _explicit_fused(model, input)
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=options)(input)

    assert torch.equal(actual, expected)
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_nvfp4_sparse_piper_attention_output.default
        )
        == 1
    )
    for target in (
        torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default,
        torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default,
        torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default,
    ):
        assert capture.targets.count(target) == 1
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("dynamic", "routing", "preparation_count"),
    [(False, "minmax", 4), (False, "mean", 4), (True, "minmax", 1)],
)
def test_cuda_compile_lifetime_chunks_a_convrot_nvfp4_gate(
    dynamic: bool,
    routing: str,
    preparation_count: int,
) -> None:
    torch.manual_seed(977 + dynamic)
    model = _ConvRotProjectedGateCoarseAttentionOutput(
        dynamic=dynamic,
        routing=routing,
    ).eval()
    input = torch.randn(  # noqa: A001
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
    valid_rows = (torch.arange(64, device="cuda")[None, :] < block_lengths[:, None]).flatten()
    capture = _TargetCapturePass()
    options = convrot_nvfp4_sparse_piper_compile_options()
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*passes, capture)
    with torch.no_grad():
        expected = _explicit_fused_projected_gate(model, input, block_lengths, 2)
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=options)(
            input,
            block_lengths,
            2,
        )

    torch.testing.assert_close(
        actual[:, valid_rows],
        expected[:, valid_rows],
        atol=0,
        rtol=0,
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default)
        == preparation_count
    )
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_nvfp4_sparse_piper_attention_output.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in capture.targets
