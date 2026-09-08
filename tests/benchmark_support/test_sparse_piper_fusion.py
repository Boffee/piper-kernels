"""Standalone fusion benchmarks must verify fusion and bounded comparisons."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import benchmark_sparse_piper_fusion as benchmark
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize(
    "arguments",
    [
        ["--sequence", "63"],
        ["--samples", "0"],
        ["--device", "-1"],
        ["--sequence", str(65537 * 64)],
    ],
)
def test_invalid_arguments_are_rejected(arguments):
    with pytest.raises(SystemExit):
        benchmark._parse_args(arguments)


def test_defaults_cover_h3_through_100k():
    args = benchmark._parse_args([])
    assert args.sequence == [8192, 32768, 100000]
    assert args.samples == 11


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
    temporaries = []

    def temporary():
        tensor = torch.empty(1)
        temporaries.append(ref(tensor))
        return tensor

    monkeypatch.setattr(benchmark._ops, "prepare_input", lambda *args: (temporary(), temporary()))
    monkeypatch.setattr(benchmark._ops, "dequantized_input_mean", lambda *args: temporary())
    for module, name, count in [
        (benchmark.query, "_project_query_op", 3),
        (benchmark.key, "_project_key_op", 4),
        (benchmark.value, "_project_value_op", 3),
    ]:
        monkeypatch.setattr(
            module, name, lambda *args, count=count: tuple(temporary() for _ in range(count))
        )
    monkeypatch.setattr(
        benchmark,
        "_sparse_piper_attention_from_quantized_op",
        lambda *args: torch.empty(1, 64, 56, 128),
    )

    def linear(*args):
        assert temporaries
        assert all(reference() is None for reference in temporaries)
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


def test_paired_measurement_reports_every_sample_and_peak_extra_bytes(monkeypatch):
    functions = {
        name: Mock(return_value=torch.empty(1)) for name in ("materialized", "compiled_fused")
    }
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
    assert timer.call_count == 6
    assert all(
        {call.args[0] for call in timer.call_args_list[start : start + 2]}
        == set(functions.values())
        for start in range(0, 6, 2)
    )
    assert all(
        call.kwargs["synchronize"] is torch.cuda.synchronize for call in timer.call_args_list
    )
