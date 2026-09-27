"""Dense fusion measurements must include the real tail and both timed providers."""

from unittest.mock import Mock

import benchmark_piper_fusion as benchmark
import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_projection_wrapper_matches_activation_dtype(dtype):
    layer = benchmark._projection(256, 128, 256, dtype, torch.Generator().manual_seed(14))
    assert layer.weight.dtype is dtype
    assert layer.weight.qdata.dtype is torch.int8


def test_exact_comparison_checks_tail_shape_and_nonfinite_values():
    expected = torch.ones(1, 8193, 4, dtype=torch.bfloat16)
    benchmark._check_equal(expected.clone(), expected)
    actual = expected.clone()
    actual[:, -1] += 1
    with pytest.raises(AssertionError):
        benchmark._check_equal(actual, expected)
    with pytest.raises(AssertionError):
        benchmark._check_equal(expected[:, :-1], expected)
    expected[:, 4096] = float("nan")
    with pytest.raises(AssertionError):
        benchmark._check_equal(expected, expected)


def test_wall_measurement_keeps_clock_counts_peaks_and_paired_order(monkeypatch):
    functions = {name: Mock(return_value=torch.empty(1)) for name in ("qkv_only", "qkv_output")}
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 300)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", Mock())
    timer = Mock(side_effect=lambda fn, **kwargs: (fn(), 2.0))
    monkeypatch.setattr(benchmark, "time_first_call", timer)
    monkeypatch.setattr(
        benchmark, "_measure_graphs", Mock(side_effect=AssertionError("unexpected graph capture"))
    )
    args = benchmark._parse_args(["--samples", "3", "--no-cuda-graph"])
    results = benchmark._measure(functions, args)
    for name, function in functions.items():
        assert function.call_count == 4
        timings, peak, graph = results[name]
        assert timings.samples_ms == (2.0,) * 3
        assert timings.warmup_calls == 1
        assert timings.operator_end_to_end.clock == "synchronized_wall"
        assert peak == 200
        assert graph is None
    assert timer.call_count == 6
    assert all(
        {call.args[0] for call in timer.call_args_list[start : start + 2]}
        == set(functions.values())
        for start in range(0, 6, 2)
    )
