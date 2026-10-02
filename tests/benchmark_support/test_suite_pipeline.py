"""Pipeline comparisons preserve operation identity, graph evidence, and bounded storage."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import pytest
import torch
from lib import suite_pipeline as pipeline
from lib.cases import PipelineCase

from piper_kernels.fusions.convrot_int8_piper import output as dense_output
from piper_kernels.fusions.convrot_int8_sparse_piper import key, query, value
from piper_kernels.fusions.convrot_int8_sparse_piper import output as sparse_output
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.linear.convrot.int8 import _ops


def _case(**overrides):
    return replace(
        PipelineCase(
            id="test",
            sequence=65,
            heads=2,
            kv_heads=1,
            head_dim=64,
            width=128,
            rotary_dim=32,
        ),
        **overrides,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_projection_preserves_dtype_and_reproducible_packed_storage(dtype):
    kwargs = {"dtype": dtype, "device": torch.device("cpu"), "seed": 14}
    first = pipeline._projection(128, 64, **kwargs)
    second = pipeline._projection(128, 64, **kwargs)
    assert first.weight.dtype is dtype
    assert first.weight.qdata.dtype is torch.int8
    assert first.weight.group_size == 64
    torch.testing.assert_close(first.weight.qdata, second.weight.qdata, atol=0, rtol=0)
    torch.testing.assert_close(first.weight.scale, second.weight.scale, atol=0, rtol=0)


@pytest.mark.parametrize("keep_ratio", [None, 0.5])
def test_pipeline_uses_case_dimensions_and_query_head_budget(keep_ratio):
    case = _case(keep_ratio=keep_ratio)
    model = pipeline.Pipeline(case, torch.device("cpu"))
    hidden = torch.randn(1, case.sequence, case.width, dtype=torch.bfloat16)
    angles = torch.randn(case.sequence, case.rotary_dim)
    with torch.no_grad():
        result = model(hidden, angles.cos(), angles.sin())
    assert result.shape == hidden.shape
    assert result.dtype == hidden.dtype
    assert model.key.weight.shape == (case.kv_heads * case.head_dim, case.width)
    assert torch.isfinite(result).all()


def test_unsupported_device_does_not_allocate_inputs_or_models(monkeypatch):
    unexpected = Mock(side_effect=AssertionError("allocated before support check"))
    monkeypatch.setattr(pipeline, "Pipeline", unexpected)
    monkeypatch.setattr(torch, "empty", unexpected)
    implementations = pipeline.implementations(_case(), torch.device("cpu"))
    assert len(implementations) == 2
    assert all(item.unsupported_reason for item in implementations)
    unexpected.assert_not_called()


def test_all_family_discovery_reports_missing_torchao_without_importing_it():
    script = """
import sys
sys.path.insert(0, sys.argv[1])
# A None module entry blocks imports and makes find_spec report the missing package.
sys.modules["torchao"] = None
import torch
from lib.cases import PipelineCase, diagnostic_cases
from lib.suite import implementations

for case in diagnostic_cases():
    providers = implementations(case, torch.device("cpu"))
    assert providers
    if isinstance(case, PipelineCase):
        assert all("TorchAO" in provider.unsupported_reason for provider in providers)
assert sys.modules["torchao"] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2] / "benchmarks")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_sparse_gqa_fusion_is_not_advertised_as_native(monkeypatch):
    monkeypatch.setattr(pipeline, "_unsupported_reason", lambda *args: None)
    implementations = pipeline.implementations(_case(keep_ratio=0.5), torch.device("cuda"))
    assert implementations[0].unsupported_reason is None
    assert "equal Q/KV heads" in implementations[1].unsupported_reason


def test_materialized_releases_preparation_and_projections_before_next_stage(monkeypatch):
    prepared, projections = [], []

    def temporary(references):
        tensor = torch.empty(1)
        references.append(ref(tensor))
        return tensor

    monkeypatch.setattr(
        _ops, "prepare_input", lambda *args: (temporary(prepared), temporary(prepared))
    )
    monkeypatch.setattr(_ops, "dequantized_input_mean", lambda *args: temporary(prepared))
    for module, name, count in [
        (query, "_project_query_op", 3),
        (key, "_project_key_op", 4),
        (value, "_project_value_op", 3),
    ]:
        monkeypatch.setattr(
            module,
            name,
            lambda *args, count=count, **kwargs: tuple(
                temporary(projections) for _ in range(count)
            ),
        )

    def attention(*args, **kwargs):
        assert all(reference() is None for reference in prepared)
        assert len(projections) == 10
        assert all(reference() is not None for reference in projections)
        assert args[10] == [500_000, 500_000]
        assert kwargs["output_dtype"] == torch.float16
        return torch.empty(1, 64, 2, 64)

    monkeypatch.setattr(pipeline, "_sparse_piper_attention_from_quantized_op", attention)

    def linear(*args):
        assert all(reference() is None for reference in (*prepared, *projections))
        return torch.empty(1)

    monkeypatch.setattr(
        linear_backend,
        "require_linear_backend",
        lambda tensor: SimpleNamespace(linear=linear),
    )
    layer = SimpleNamespace(weight=SimpleNamespace(qdata=torch.empty(1), scale=torch.empty(1)))
    model = SimpleNamespace(
        query=layer,
        key=layer,
        value=layer,
        output=layer,
        query_norm=torch.empty(64),
        key_norm=torch.empty(64),
        keep_ratio=0.5,
        heads=2,
        head_dim=64,
        dtype=torch.float16,
    )
    pipeline.Pipeline.materialized(
        model, torch.empty(1, 64, 128), torch.empty(64, 32), torch.empty(64, 32)
    )


def _fused_graph(sparse, argument_style):
    module = sparse_output if sparse else dense_output
    target = module._projected_query_attention_output_op._opoverload
    index = next(
        i
        for i, argument in enumerate(target._schema.arguments)
        if argument.name == "query_chunk_rows"
    )
    graph = torch.fx.Graph()
    operand = graph.placeholder("operand")
    args = (operand,) * index
    kwargs = {}
    if argument_style == "positional":
        args += (4096,)
    elif argument_style == "keyword":
        kwargs["query_chunk_rows"] = 4096
    node = graph.call_function(target, args=args, kwargs=kwargs)
    graph.output(node)
    return graph, node


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("argument_style", ["positional", "keyword", "default"])
def test_capture_reports_actual_window_without_rewriting_graph(sparse, argument_style):
    graph, node = _fused_graph(sparse, argument_style)
    args, kwargs = node.args, node.kwargs.copy()
    capture = pipeline.CaptureFusion(sparse=sparse)
    capture(graph, True)
    assert capture.query_chunk_rows == (8192 if argument_style == "default" else 4096)
    assert node.args == args
    assert node.kwargs == kwargs
    # Projection producers are required too; a lone output node is insufficient evidence.
    with pytest.raises(AssertionError):
        capture.check()
    capture.targets.extend(f"{capture.prefix}_project_{name}.default" for name in ("key", "value"))
    capture.check()
    capture(graph, True)
    with pytest.raises(AssertionError, match="one compiled graph"):
        capture.check()


def test_build_samples_reference_then_releases_full_output(monkeypatch):
    outputs = []

    def materialized(hidden, cos, sin):
        tensor = torch.ones_like(hidden)
        outputs.append(ref(tensor))
        return tensor

    model = SimpleNamespace(
        dtype=torch.bfloat16,
        materialized=materialized,
        eval=lambda: model,
    )
    monkeypatch.setattr(pipeline, "Pipeline", lambda *args: model)
    case = _case(sequence=8193)
    operation = pipeline._build(case, torch.device("cpu"), fused=False)
    assert outputs[0]() is None
    actual = operation.run()
    quality = operation.check(actual)
    assert quality.sample_count == 64 * case.width
    assert quality.total_count == case.sequence * case.width
    assert quality.metrics.relative_l2_error == 0
    actual[:, -1] += 1
    assert operation.check(actual).metrics.relative_l2_error > 0
    actual[:, -1] = float("nan")
    assert operation.check(actual).metrics.actual_nonfinite_count == case.width
