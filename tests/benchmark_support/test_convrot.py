"""Tests for ConvRot benchmark presets, records, and reference helpers."""

import json

import pytest
import torch
from benchmark_convrot import (
    BenchmarkShape,
    _apply_input_activation,
    _benchmark_shapes,
    _comfy_input,
    _parse_args,
    _quality_row_indices,
    _raw_input_features,
    _selected_input_preparation,
    _skip_reference_timing,
    _validate_args,
)
from benchmark_convrot_preparation import (
    PreparationPhaseResult,
    _compiler_requested,
    _inspection_provider,
    _minimum_global_bytes,
    _preparation_records,
)
from benchmark_convrot_preparation import (
    _parse_args as _parse_preparation_args,
)
from lib import ClockDomain, EnvironmentInfo, Timing, output_target, write_records


def _environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        captured_at_utc="2026-08-08T00:00:00+00:00",
        python_version="3.14.0",
        platform="test",
        torch_version="2.12.0",
        triton_version="3.7.1",
        accelerator_backend="cuda",
        accelerator_runtime_version="13.0",
        accelerator_driver_version="580.0",
        gpu_name="test GPU",
        gpu_architecture="SM120",
        gpu_index=0,
        git_revision="a" * 40,
        git_dirty=False,
    )


def test_custom_shapes_expand_requested_rows() -> None:
    arguments = _parse_args(["--rows", "3", "7", "--out-features", "96", "--in-features", "512"])

    shapes = _benchmark_shapes(arguments)

    actual = [(shape.name, shape.rows, shape.out_features, shape.in_features) for shape in shapes]
    assert actual == [
        ("custom", 3, 96, 512),
        ("custom", 7, 96, 512),
    ]


def test_minimax_h3_five_second_preset_uses_principal_linears() -> None:
    arguments = _parse_args(["--preset", "minimax-h3-5s"])

    shapes = _benchmark_shapes(arguments)

    actual = [(shape.name, shape.rows, shape.out_features, shape.in_features) for shape in shapes]
    assert actual == [
        ("qkv", 37_710, 21_504, 5_376),
        ("attention-out", 37_710, 5_376, 7_168),
        ("mlp-fc1", 37_710, 28_672, 5_376),
        ("mlp-fc2", 37_710, 5_376, 14_336),
    ]
    assert [shape.input_activation for shape in shapes] == [None, None, None, "swiglu"]
    assert not any(shape.has_bias for shape in shapes)
    assert not _skip_reference_timing(arguments)


def test_minimax_h3_128k_preset_uses_sampled_reference_mode() -> None:
    arguments = _parse_args(["--preset", "minimax-h3-128k"])

    shapes = _benchmark_shapes(arguments)

    assert [(shape.rows, shape.out_features, shape.in_features) for shape in shapes] == [
        (131_072, 21_504, 5_376),
        (131_072, 5_376, 7_168),
        (131_072, 28_672, 5_376),
        (131_072, 5_376, 14_336),
    ]
    assert _skip_reference_timing(arguments)


def test_custom_shape_can_include_swiglu_without_bias() -> None:
    arguments = _parse_args(
        [
            "--rows",
            "7",
            "--out-features",
            "96",
            "--in-features",
            "512",
            "--input-activation",
            "swiglu",
            "--no-bias",
        ]
    )

    (shape,) = _benchmark_shapes(arguments)

    assert shape.input_activation == "swiglu"
    assert not shape.has_bias
    assert _raw_input_features(shape) == 1024


def test_cli_none_input_activation_is_python_none() -> None:
    assert _parse_args([]).input_activation is None
    assert _parse_args(["--input-activation", "none"]).input_activation is None
    assert _parse_preparation_args([]).input_activation is None
    assert _parse_preparation_args(["--input-activation", "none"]).input_activation is None


@pytest.mark.parametrize(
    "custom_option",
    [
        ["--rows", "1"],
        ["--out-features", "96"],
        ["--in-features", "512"],
        ["--input-activation", "none"],
        ["--no-bias"],
    ],
)
def test_named_preset_rejects_custom_shape_options(custom_option: list[str]) -> None:
    arguments = _parse_args(["--preset", "minimax-h3-5s", *custom_option])

    with pytest.raises(SystemExit, match="cannot be combined"):
        _validate_args(arguments, _benchmark_shapes(arguments))


def test_selected_input_preparation_matches_public_dispatch(monkeypatch) -> None:
    activation = torch.empty(2, 1024)
    qdata = torch.empty(96, 512, dtype=torch.int8)
    monkeypatch.setattr(
        "benchmark_convrot.convrot_dispatch._can_use_triton_swiglu",
        lambda *_args: True,
    )

    assert _selected_input_preparation(activation, qdata, 256, "swiglu") == "fused"

    monkeypatch.setattr(
        "benchmark_convrot.convrot_dispatch._can_use_triton_swiglu",
        lambda *_args: False,
    )
    assert _selected_input_preparation(activation, qdata, 256, "swiglu") == "materialized"
    assert _selected_input_preparation(activation, qdata, 256, None) is None


def test_swiglu_reference_uses_up_gate_order() -> None:
    raw = torch.tensor([[2.0, -3.0, 0.5, 1.5]])

    actual = _apply_input_activation(raw, "swiglu")

    up, gate = raw.chunk(2, dim=-1)
    torch.testing.assert_close(actual, up * torch.nn.functional.silu(gate))


def test_comfy_adapter_changes_up_gate_to_gate_up() -> None:
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    adapted = _comfy_input(raw, "swiglu")

    assert adapted.tolist() == [[3.0, 4.0, 1.0, 2.0]]


def test_quality_rows_include_large_projection_output_boundaries() -> None:
    for name, out_features in (("qkv", 21_504), ("mlp-fc1", 28_672)):
        shape = BenchmarkShape(name, 131_072, out_features, 5_376, has_bias=False)

        rows = _quality_row_indices(shape)

        output_boundary = ((1 << 31) + shape.out_features - 1) // shape.out_features
        assert len(rows) == 256
        assert rows[0] == 0
        assert rows[-1] == shape.rows - 1
        assert {output_boundary - 1, output_boundary, output_boundary + 1} <= set(rows)


def test_quality_rows_include_raw_swiglu_input_address_boundary() -> None:
    shape = BenchmarkShape("mlp-fc2", 131_072, 5_376, 14_336, "swiglu", False)

    rows = _quality_row_indices(shape)

    raw_width = 2 * shape.in_features
    input_boundary = ((1 << 31) + raw_width - 1) // raw_width
    assert {input_boundary - 1, input_boundary, input_boundary + 1} <= set(rows)


def test_preparation_minimum_global_traffic_accounts_for_split_intermediate() -> None:
    rows, in_features, element_size = 3, 512, 2

    rotate = _minimum_global_bytes("rotate", rows, in_features, element_size)
    quantize = _minimum_global_bytes("quantize", rows, in_features, element_size)
    split = _minimum_global_bytes("split", rows, in_features, element_size)
    fused = _minimum_global_bytes("fused", rows, in_features, element_size)
    fused_swiglu = _minimum_global_bytes(
        "fused",
        rows,
        in_features,
        element_size,
        "swiglu",
    )

    assert rotate == 4 * rows * in_features
    assert quantize == 3 * rows * in_features + 4 * rows
    assert split == rotate + quantize
    assert fused == quantize
    assert fused_swiglu == 5 * rows * in_features + 4 * rows


def test_preparation_cli_exposes_current_compiler_reporting(tmp_path) -> None:
    compiler_path = tmp_path / "convrot-compiler.json"
    benchmark_path = tmp_path / "convrot-preparation.json"
    arguments = _parse_preparation_args(
        [
            "--in-features",
            "5376",
            "--json",
            str(benchmark_path),
            "--compiler-report",
            "--compiler-json",
            str(compiler_path),
            "--no-sass",
        ]
    )

    assert _compiler_requested(arguments)
    assert arguments.json == benchmark_path
    assert arguments.compiler_json == compiler_path
    assert arguments.sass is False


def test_preparation_cli_and_records_expose_phase_timings(tmp_path) -> None:
    output_path = tmp_path / "preparation.jsonl"
    arguments = _parse_preparation_args(["--jsonl", str(output_path)])
    phase = PreparationPhaseResult(
        phase="fused",
        provider="piper-triton",
        operation_provenance=(
            "piper_kernels.convrot.int8.triton._fused_rotate_quantize_activations"
        ),
        timing=Timing(1.0, 0.8, 1.2, ClockDomain.DEVICE_EVENT),
        minimum_global_bytes=1024,
        effective_minimum_tbps=2.0,
        baseline_phase="fused",
        speedup_vs_baseline=1.0,
    )

    (record,) = _preparation_records(
        rows=3,
        in_features=512,
        dtype_name="bfloat16",
        input_activation="swiglu",
        seed=0,
        warmup_ms=100,
        measurement_time_ms=300,
        results=[phase],
        environment=_environment(),
    )
    value = record.as_dict()

    assert arguments.jsonl == output_path
    assert value["benchmark"] == "convrot-preparation"
    assert value["shape"] == {
        "rows": 3,
        "in_features": 512,
        "raw_input_features": 1024,
    }
    assert value["configuration"]["input_activation"] == "swiglu"
    assert value["configuration"]["baseline_provider"] == "piper-triton"
    assert value["configuration"]["baseline_phase"] == "fused"
    assert value["configuration"]["operation_provenance"] == phase.operation_provenance
    assert value["timings"]["prepared_execution"]["clock"] == "device_event"
    assert value["extra"]["speedup_vs_baseline"] == 1.0

    write_records([record], output_target(arguments))
    written = json.loads(output_path.read_text())
    assert written["benchmark"] == "convrot-preparation"
    assert written["shape"]["raw_input_features"] == 1024


def test_preparation_compiler_report_only_inspects_launched_phases() -> None:
    ordinary = _inspection_provider(_parse_preparation_args(["--in-features", "512"]))
    swiglu = _inspection_provider(
        _parse_preparation_args(["--in-features", "512", "--input-activation", "swiglu"])
    )

    assert set(ordinary.triton_jit_functions) == {"rotate", "quantize", "fused"}
    assert set(swiglu.triton_jit_functions) == {"fused"}
