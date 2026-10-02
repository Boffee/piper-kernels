"""The score benchmark preserves real chunk sizes and the FP32 baseline."""

import benchmark_sparse_piper_scores as benchmark
import pytest
import torch
from lib.timing import ClockDomain, Timing


@pytest.mark.parametrize(
    "arguments",
    [
        ["--sequence", "63"],
        ["--samples", "0"],
        ["--rep-ms", "0"],
        ["--device", "-1"],
    ],
)
def test_invalid_arguments_are_rejected(arguments):
    with pytest.raises(SystemExit):
        benchmark._parse_args(arguments)


@pytest.mark.parametrize(
    ("sequence", "chunks"),
    [
        (64, [1]),
        (4096, [64]),
        (8193, [128, 1]),
        (8192, [128]),
        (32768, [128]),
        (100000, [128, 27]),
        (150000, [128, 40]),
    ],
)
def test_scoring_uses_actual_full_and_tail_chunk_sizes(sequence, chunks):
    assert benchmark._query_chunks(sequence) == chunks


def test_defaults_use_the_shared_lower_video_case():
    args = benchmark._parse_args([])
    assert args.sequence == [20480]
    assert args.case == "sparse-attention-video-low-half"
    assert args.samples == 7


def test_catalog_case_preserves_image_gqa_and_rejects_shape_overrides():
    args = benchmark._parse_args(["--case", "attention-image-high"])
    assert (args.sequence, args.heads, args.kv_heads, args.head_dim) == ([16896], 48, 12, 128)
    with pytest.raises(SystemExit):
        benchmark._parse_args(["--case", "attention-image-high", "--sequence", "65"])
    with pytest.raises(SystemExit):
        benchmark._parse_args(["--case", "linear-small"])


@pytest.mark.parametrize("format_name", ["json", "jsonl"])
def test_output_arguments_select_the_shared_writer(format_name, tmp_path):
    path = tmp_path / f"scores.{format_name}"
    args = benchmark._parse_args([f"--{format_name}", str(path)])
    target = benchmark.output_target(args)
    assert target.path == path
    assert target.format.value == format_name
    with pytest.raises(SystemExit):
        benchmark._parse_args(["--json", str(path), "--jsonl", str(path)])


def test_torch_baseline_matches_exact_integer_products_with_prefix_views():
    generator = torch.Generator().manual_seed(13)
    query = torch.randint(-4, 5, (2, 3, 8, 128), generator=generator).float()[:, :, ::2]
    keys = [
        torch.randint(-4, 5, (2, 3, 7, 128), generator=generator).float()[:, :, :5]
        for _ in range(2)
    ]
    expected = torch.maximum(query @ keys[0].transpose(-1, -2), query @ keys[1].transpose(-1, -2))
    actual = benchmark._torch_scores(query, *keys)
    assert actual.is_contiguous()
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_torch_baseline_preserves_grouped_head_mapping():
    query = torch.randn(2, 6, 3, 64)
    primary, auxiliary = torch.randn(2, 2, 7, 64), torch.randn(2, 2, 7, 64)
    expected = torch.maximum(
        query @ primary.repeat_interleave(3, dim=1).transpose(-1, -2),
        query @ auxiliary.repeat_interleave(3, dim=1).transpose(-1, -2),
    )
    torch.testing.assert_close(benchmark._torch_scores(query, primary, auxiliary), expected)


def test_no_gpu_rejects_before_allocating_inputs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="requires a CUDA or ROCm GPU"):
        benchmark.main([])


def test_records_preserve_paired_panels_and_label_the_aggregate(monkeypatch, tmp_path):
    normal_tensor = benchmark.normal_tensor
    monkeypatch.setattr(
        benchmark,
        "normal_tensor",
        lambda shape, **kwargs: normal_tensor(shape, **(kwargs | {"device": torch.device("cpu")})),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(benchmark._backend, "select_minmax_scores", lambda *args: None)
    monkeypatch.setattr(
        benchmark,
        "routing_scores",
        lambda query, primary, auxiliary, mode: benchmark._torch_scores(query, primary, auxiliary),
    )
    calls = []
    medians = {}

    def measure(function, warmup_ms, measurement_time_ms):
        calls.append(function)
        assert warmup_ms == 20
        assert measurement_time_ms == 7
        samples = medians.setdefault(function, [])
        value = 1.0 + 2 * len(samples)
        samples.append(value)
        return Timing(value, value - 0.5, value + 0.5, ClockDomain.DEVICE_EVENT)

    monkeypatch.setattr(benchmark, "triton_benchmark", measure)
    args = benchmark._parse_args(["--samples", "3", "--rep-ms", "7"])
    environment = benchmark.capture_environment(tmp_path)
    records = benchmark._benchmark(args, 65, 2, environment)
    assert len(records) == 2
    assert len(calls) == 6
    assert all(set(calls[start : start + 2]) == set(calls) for start in range(0, 6, 2))
    for record in records:
        value = record.as_dict()
        assert value["schema_version"] == 1
        assert value["environment"] == environment.as_dict()
        assert value["shape"]["score_shape_bhqk"] == [1, 56, 2, 1]
        assert value["shape"]["key_storage_blocks"] == 2
        assert value["configuration"]["key_stride"] == (14336, 256, 128, 1)
        assert value["configuration"]["implementation"] == "torch"
        assert value["configuration"]["panel_count"] == 3
        assert value["configuration"]["timing_summary"] == "quantiles_of_panel_medians"
        assert value["timings"]["warmup_ms"] == 20
        assert value["timings"]["measurement_time_ms"] == 7
        assert value["timings"]["prepared_execution"] == {
            "median_ms": 3.0,
            "p20_ms": 1.8,
            "p80_ms": 4.2,
            "clock": "device_event",
        }
        assert [panel["median_ms"] for panel in value["extra"]["panels"]] == [1.0, 3.0, 5.0]
        assert all(panel["clock"] == "device_event" for panel in value["extra"]["panels"])
        assert value["extra"]["relative_l2_vs_fp64"] < 2e-5
