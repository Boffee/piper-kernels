"""Standalone fusion benchmarks must verify fusion and bounded comparisons."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import benchmark_sparse_piper_fusion as benchmark
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels.fusions.convrot_int8_sparse_piper import output as output_fusion


@pytest.mark.parametrize(
    "arguments",
    [
        ["--sequence", "63"],
        ["--samples", "0"],
        ["--device", "-1"],
        ["--sequence", str(65537 * 64)],
        ["--query-chunk-rows", "0"],
        ["--query-chunk-rows", "-64"],
        ["--query-chunk-rows", "63"],
        ["--query-chunk-rows", "65"],
        ["--query-chunk-rows", "4096", "4096"],
    ],
)
def test_invalid_arguments_are_rejected(arguments):
    with pytest.raises(SystemExit):
        benchmark._parse_args(arguments)


def test_defaults_cover_h3_through_150k():
    args = benchmark._parse_args([])
    assert args.sequence == [8192, 32768, 100000, 150000]
    assert args.query_chunk_rows == [4096]
    assert args.samples == 11


def test_query_window_sweep_preserves_requested_order():
    args = benchmark._parse_args(["--query-chunk-rows", "16384", "4096", "8192"])
    assert args.query_chunk_rows == [16384, 4096, 8192]


@pytest.mark.parametrize("format_name", ["json", "jsonl"])
def test_output_arguments_select_the_shared_writer(format_name, tmp_path):
    path = tmp_path / f"fusion.{format_name}"
    args = benchmark._parse_args([f"--{format_name}", str(path)])
    target = benchmark.output_target(args)
    assert target.path == path
    assert target.format.value == format_name
    with pytest.raises(SystemExit):
        benchmark._parse_args(["--json", str(path), "--jsonl", str(path)])


def test_projection_uses_seeded_int8_weights():
    first = benchmark._projection(256, 128, torch.Generator().manual_seed(14))
    second = benchmark._projection(256, 128, torch.Generator().manual_seed(14))
    assert isinstance(first.weight, benchmark.ConvRotInt8Tensor)
    assert first.weight.group_size == 256
    assert first.bias is None
    assert first.weight.device.type == "cpu"
    torch.testing.assert_close(first.weight.qdata, second.weight.qdata, atol=0, rtol=0)
    torch.testing.assert_close(first.weight.scale, second.weight.scale, atol=0, rtol=0)


def test_materialized_releases_qkv_before_output_projection(monkeypatch):
    prepared, projections = [], []

    def temporary(references):
        tensor = torch.empty(1)
        references.append(ref(tensor))
        return tensor

    monkeypatch.setattr(
        benchmark._ops, "prepare_input", lambda *args: (temporary(prepared), temporary(prepared))
    )
    monkeypatch.setattr(benchmark._ops, "dequantized_input_mean", lambda *args: temporary(prepared))
    for module, name, count in [
        (benchmark.query, "_project_query_op", 3),
        (benchmark.key, "_project_key_op", 4),
        (benchmark.value, "_project_value_op", 3),
    ]:
        monkeypatch.setattr(
            module,
            name,
            lambda *args, count=count: tuple(temporary(projections) for _ in range(count)),
        )

    def attention(*args):
        assert len(prepared) == 3
        assert all(reference() is None for reference in prepared)
        assert len(projections) == 10
        assert all(reference() is not None for reference in projections)
        return torch.empty(1, 64, 56, 128)

    monkeypatch.setattr(
        benchmark,
        "_sparse_piper_attention_from_quantized_op",
        attention,
    )

    def linear(*args):
        assert all(reference() is None for reference in (*prepared, *projections))
        return torch.empty(1)

    monkeypatch.setattr(
        benchmark.linear_backend,
        "require_linear_backend",
        lambda tensor: SimpleNamespace(linear=linear),
    )
    layer = SimpleNamespace(weight=SimpleNamespace(qdata=torch.empty(1), scale=torch.empty(1)))
    model = SimpleNamespace(
        query=layer,
        key=layer,
        value=layer,
        output=layer,
        query_norm=torch.empty(128),
        key_norm=torch.empty(128),
    )
    benchmark._H3Attention.materialized(
        model,
        torch.empty(1, 64, 5376),
        torch.empty(64, 96),
        torch.empty(64, 96),
        1,
    )


def test_comparison_checks_every_chunk_including_the_tail():
    expected = torch.ones(1, 8193, 4, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[:, -1] += 1
    error = benchmark._relative_l2(actual, expected)
    assert error == pytest.approx(8193**-0.5)
    actual[:, 4096] = float("nan")
    with pytest.raises(AssertionError):
        benchmark._relative_l2(actual, expected)


def test_comparison_transfers_only_bounded_slices(monkeypatch):
    transferred_rows = []
    copy_to_cpu = torch.Tensor.cpu

    def record_transfer(tensor):
        transferred_rows.append(tensor.shape[1])
        return copy_to_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", record_transfer)
    expected = torch.ones(1, 8193, 4, dtype=torch.bfloat16)
    assert benchmark._relative_l2(expected.clone(), expected) == 0.0
    assert max(transferred_rows) <= 256
    assert sum(transferred_rows) == 2 * 8193
    assert transferred_rows[-2:] == [1, 1]


def test_capture_requires_the_complete_fusion_and_a_single_graph():
    capture = benchmark._CaptureFusion()
    graph = torch.fx.Graph()
    capture(graph, True)
    with pytest.raises(AssertionError):
        capture.check()
    capture.targets = [benchmark._FUSED_OUTPUT]
    capture.check()
    capture(graph, True)
    with pytest.raises(AssertionError, match="one dynamic graph"):
        capture.check()


def _fused_output_graph(argument_style="positional"):
    target = output_fusion._projected_query_attention_output_op._opoverload
    index = next(
        index
        for index, argument in enumerate(target._schema.arguments)
        if argument.name == "query_chunk_rows"
    )
    graph = torch.fx.Graph()
    placeholder = graph.placeholder("operand")
    arguments = (placeholder,) * index
    keywords = {"output_dtype": torch.bfloat16}
    if argument_style == "positional":
        arguments += (4096,)
    elif argument_style == "keyword":
        keywords["query_chunk_rows"] = 4096
    node = graph.call_function(target, args=arguments, kwargs=keywords)
    graph.output(node)
    return graph, node, index


@pytest.mark.parametrize("argument_style", ["positional", "keyword", "default"])
def test_capture_rewrites_the_actual_fused_operator_window(argument_style):
    graph, node, index = _fused_output_graph(argument_style)
    other_arguments = node.args[:index]
    capture = benchmark._CaptureFusion(16384)
    capture(graph, True)
    capture.check()
    graph.lint()
    actual = node.args[index] if len(node.args) > index else node.kwargs["query_chunk_rows"]
    assert actual == 16384
    assert capture.requested_query_chunk_rows == 16384
    assert capture.actual_query_chunk_rows == 16384
    assert node.args[:index] == other_arguments
    assert node.kwargs["output_dtype"] == torch.bfloat16


def test_capture_only_reports_the_existing_operator_window():
    graph, node, index = _fused_output_graph()
    capture = benchmark._CaptureFusion()
    capture(graph, True)
    capture.check()
    assert node.args[index] == 4096
    assert capture.requested_query_chunk_rows is None
    assert capture.actual_query_chunk_rows == 4096


@pytest.mark.parametrize("windows", [[8192], [4096, 8192, 16384]])
def test_window_variants_compile_with_independent_capture_and_cache_identity(monkeypatch, windows):
    compile_model = Mock(side_effect=lambda *args, **kwargs: Mock())
    monkeypatch.setattr(torch, "compile", compile_model)
    model = torch.nn.Identity()
    variants = benchmark._compile_variants(model, windows)
    expected_names = (
        ["compiled_fused"] if len(windows) == 1 else [f"compiled_fused_q{rows}" for rows in windows]
    )
    assert list(variants) == expected_names
    assert compile_model.call_count == len(windows)
    captures = [capture for _compiled, capture in variants.values()]
    assert len({capture.uuid() for capture in captures}) == len(windows)
    for rows, (compiled, capture), call in zip(
        windows, variants.values(), compile_model.call_args_list, strict=True
    ):
        assert callable(compiled)
        assert call.args == (model,)
        assert call.kwargs["dynamic"] is True
        assert call.kwargs["fullgraph"] is True
        assert call.kwargs["options"]["post_grad_custom_pre_pass"][-1] is capture
        graph, node, index = _fused_output_graph()
        capture(graph, True)
        capture.check()
        assert node.args[index] == rows


def test_every_window_variant_is_compared_against_the_complete_reference():
    expected = torch.ones(1, 8193, 4, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[:, -1] += 1
    captures = {name: Mock() for name in ("compiled_fused_q4096", "compiled_fused_q8192")}
    functions = {
        "materialized": Mock(return_value=expected),
        "compiled_fused_q4096": Mock(return_value=expected),
        "compiled_fused_q8192": Mock(return_value=actual),
    }
    errors = benchmark._compare_variants(functions, captures)
    assert errors["compiled_fused_q4096"] == 0.0
    assert errors["compiled_fused_q8192"] == pytest.approx(8193**-0.5)
    for function in functions.values():
        function.assert_called_once_with()
    for capture in captures.values():
        capture.check.assert_called_once_with()


def test_comparison_releases_reference_device_output_and_each_candidate():
    outputs = []

    class MaterializedOutput:
        def cpu(self):
            return torch.ones(1, 257, 4, dtype=torch.bfloat16)

    def materialized():
        output = MaterializedOutput()
        outputs.append(ref(output))
        return output

    def candidate():
        assert all(reference() is None for reference in outputs)
        output = torch.ones(1, 257, 4, dtype=torch.bfloat16)
        outputs.append(ref(output))
        return output

    captures = {name: Mock() for name in ("compiled_fused_q4096", "compiled_fused_q8192")}
    errors = benchmark._compare_variants(
        {"materialized": materialized, **dict.fromkeys(captures, candidate)}, captures
    )
    assert errors == dict.fromkeys(("materialized", *captures), 0.0)
    assert len(outputs) == 3
    assert all(reference() is None for reference in outputs)


def test_unsupported_backend_rejects_before_model_or_input_allocations(monkeypatch):
    select = Mock(side_effect=ValueError("no projection backend"))
    monkeypatch.setattr(benchmark._backend, "require_projection_backend", select)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "device_context", lambda device: nullcontext())
    monkeypatch.setattr(
        benchmark, "_H3Attention", Mock(side_effect=AssertionError("created model"))
    )
    with FakeTensorMode(), pytest.raises(ValueError, match="no projection backend"):
        benchmark.main(["--device", "1"])
    select.assert_called_once()
    assert select.call_args.args[0].device == torch.device("cuda:1")
    assert select.call_args.args[0].numel() == 0


@pytest.mark.parametrize(
    "providers",
    [
        ("materialized", "compiled_fused"),
        ("materialized", "compiled_fused_q4096", "compiled_fused_q8192", "compiled_fused_q16384"),
    ],
)
def test_paired_measurement_reports_every_sample_and_peak_extra_bytes(monkeypatch, providers):
    functions = {name: Mock(return_value=torch.empty(1)) for name in providers}
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 300)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", Mock())
    timer = Mock(side_effect=lambda fn, **kwargs: (fn(), 2.0))
    monkeypatch.setattr(benchmark, "time_first_call", timer)
    result = benchmark._measure(functions, benchmark._parse_args(["--samples", "3"]))
    for name, function in functions.items():
        assert function.call_count == 4  # One warmup plus three measured calls.
        timings, peak = result[name]
        assert timings.samples_ms == (2.0,) * 3
        assert timings.warmup_calls == 1
        assert timings.operator_end_to_end.median_ms == 2.0
        assert timings.operator_end_to_end.clock == "synchronized_wall"
        assert peak == 200
    count = len(providers)
    assert timer.call_count == 3 * count
    assert all(
        {call.args[0] for call in timer.call_args_list[start : start + count]}
        == set(functions.values())
        for start in range(0, 3 * count, count)
    )
    assert all(
        call.kwargs["synchronize"] is torch.cuda.synchronize for call in timer.call_args_list
    )
