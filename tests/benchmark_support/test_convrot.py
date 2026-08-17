"""Tests for ConvRot benchmark records and reference helpers."""

import json
from pathlib import Path
from types import ModuleType

import pytest
import torch
from benchmark_convrot import (
    Result,
    _benchmark_shapes,
    _comfy_provider_configuration,
    _parse_args,
    _records_for_result,
    _validate_args,
)
from benchmark_convrot_preparation import (
    COMFY_KITCHEN_ADAPTER_CONTRACT_VERSION,
    PIPER_TRITON_PROVIDER,
    PreparationPhaseResult,
    _compiler_requested,
    _inspection_provider,
    _load_comfy_kitchen_cuda,
    _minimum_global_bytes,
    _output_targets,
    _preparation_records,
    _PreparationConfiguration,
)
from benchmark_convrot_preparation import (
    _parse_args as _parse_preparation_args,
)
from benchmark_convrot_preparation import (
    _validate_args as _validate_preparation_args,
)
from lib.convrot import (
    ConvRotConfig,
    ConvRotShape,
    comfy_convrot_input,
    make_convrot_inputs,
    raw_input_features,
)
from lib.convrot_providers import (
    make_convrot_workload,
    make_public_convrot_provider,
)
from lib.environment import EnvironmentInfo
from lib.providers import ProviderMeasurement
from lib.quality import measure_quality
from lib.reporting import output_target, write_records
from lib.timing import ClockDomain, PhaseTimings, Timing

from piper_kernels.linear._input_activations import apply_input_activation


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


def _preparation_configuration() -> _PreparationConfiguration:
    return {
        "fuse_rotation_quantization": True,
        "fused_num_warps": 8,
        "rotation_num_warps": 4,
        "quantization_num_warps": 8,
    }


def _phase_timings() -> PhaseTimings:
    return PhaseTimings(
        warmup_ms=100,
        measurement_time_ms=300,
        first_call_ms=2.0,
        preparation=None,
        prepared_execution=Timing(1.0, 0.8, 1.2, ClockDomain.DEVICE_EVENT),
        operator_end_to_end=None,
    )


def test_shapes_expand_requested_dimension_product() -> None:
    arguments = _parse_args(
        [
            "--rows",
            "3",
            "7",
            "--out-features",
            "96",
            "128",
            "--in-features",
            "512",
            "768",
        ]
    )

    shapes = _benchmark_shapes(arguments)

    actual = [(shape.name, shape.rows, shape.out_features, shape.in_features) for shape in shapes]
    assert actual == [
        ("linear", 3, 96, 512),
        ("linear", 3, 96, 768),
        ("linear", 3, 128, 512),
        ("linear", 3, 128, 768),
        ("linear", 7, 96, 512),
        ("linear", 7, 96, 768),
        ("linear", 7, 128, 512),
        ("linear", 7, 128, 768),
    ]


def test_default_shapes_use_lower_width_linear_anchors() -> None:
    arguments = _parse_args([])
    preparation_arguments = _parse_preparation_args([])

    shapes = _benchmark_shapes(arguments)
    assert [(shape.rows, shape.out_features, shape.in_features) for shape in shapes] == [
        (8192, 4096, 6144),
        (32768, 4096, 6144),
    ]
    assert all(not shape.has_bias for shape in shapes)
    assert preparation_arguments.rows == 8192
    assert preparation_arguments.in_features == [6144]


def test_shape_can_include_swiglu_without_bias() -> None:
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
    assert raw_input_features(shape.in_features, shape.input_activation) == 1024


def test_bias_is_enabled_only_explicitly() -> None:
    assert not ConvRotShape("linear", 3, 96, 512).has_bias
    assert not _benchmark_shapes(_parse_args([]))[0].has_bias
    assert _benchmark_shapes(_parse_args(["--bias"]))[0].has_bias
    assert not _benchmark_shapes(_parse_args(["--no-bias"]))[0].has_bias


def test_omitted_cli_input_activation_is_python_none() -> None:
    assert _parse_args([]).input_activation is None
    assert _parse_preparation_args([]).input_activation is None


def test_cli_accepts_gelu_tanh_input_activation() -> None:
    arguments = ["--input-activation", "gelu_tanh"]
    assert _parse_args(arguments).input_activation == "gelu_tanh"
    assert _parse_preparation_args(arguments).input_activation == "gelu_tanh"


def test_cli_input_activation_rejects_none_spelling() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--input-activation", "none"])
    with pytest.raises(SystemExit):
        _parse_preparation_args(["--input-activation", "none"])


def test_shared_shape_and_inputs_define_one_reproducible_workload() -> None:
    shape = ConvRotShape("mlp-fc2", 3, 96, 512, "swiglu", False)
    config = ConvRotConfig(torch.float32, 256, 7)

    first = make_convrot_inputs(shape, config, device=torch.device("cpu"))
    second = make_convrot_inputs(shape, config, device=torch.device("cpu"))

    assert shape.as_dict() == {
        "case": "mlp-fc2",
        "rows": 3,
        "out_features": 96,
        "in_features": 512,
        "raw_input_features": 1024,
    }
    assert first[0].shape == (3, 1024)
    assert first[1].shape == (96, 512)
    assert first[2].shape == (96, 1)
    assert first[3] is None
    for first_tensor, second_tensor in zip(first[:3], second[:3], strict=True):
        torch.testing.assert_close(first_tensor, second_tensor)


def test_shared_public_provider_and_reference_use_the_same_workload() -> None:
    workload = make_convrot_workload(
        ConvRotShape("custom", 2, 3, 256, has_bias=False),
        ConvRotConfig(torch.float32, 256, 7),
        device=torch.device("cpu"),
    )
    public = make_public_convrot_provider(workload)
    public_output = public.run(workload.inputs)

    assert public.prepare() is workload.inputs
    torch.testing.assert_close(
        public_output,
        workload.reference(),
    )
    assert workload.production_plan.as_dict().items() <= public.configuration.items()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--rows", "0"],
        ["--out-features", "96", "0"],
        ["--in-features", "512", "0"],
    ],
)
def test_shape_cli_rejects_nonpositive_dimensions(arguments: list[str]) -> None:
    parsed = _parse_args(arguments)

    with pytest.raises(SystemExit, match="must all be positive"):
        _validate_args(parsed)


def test_shape_cli_validates_every_input_width_group() -> None:
    parsed = _parse_args(["--in-features", "512", "640"])

    with pytest.raises(SystemExit, match="every in_features value"):
        _validate_args(parsed)


def test_swiglu_reference_uses_up_gate_order() -> None:
    raw = torch.tensor([[2.0, -3.0, 0.5, 1.5]])

    actual = apply_input_activation(raw, "swiglu")

    up, gate = raw.chunk(2, dim=-1)
    torch.testing.assert_close(actual, up * torch.nn.functional.silu(gate))


def test_comfy_adapter_changes_up_gate_to_gate_up() -> None:
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    adapted = comfy_convrot_input(raw, "swiglu")

    assert adapted.tolist() == [[3.0, 4.0, 1.0, 2.0]]


def test_main_provider_configuration_distinguishes_logical_and_provider_layouts() -> None:
    shape = ConvRotShape("mlp-fc2", 3, 96, 512, "swiglu", False)
    config = ConvRotConfig(torch.bfloat16, 256, 7)
    workload = make_convrot_workload(shape, config, device=torch.device("cpu"))

    common = workload.common_configuration()
    comfy = _comfy_provider_configuration(common, shape, "0.2.28")

    assert common["logical_input_layout"] == "up_gate"
    assert common["provider_input_layout"] == "up_gate"
    assert comfy["logical_input_layout"] == "up_gate"
    assert comfy["provider_input_layout"] == "gate_up"
    assert comfy["input_layout_adapter"] is True
    assert comfy["installed_version"] == "0.2.28"
    assert "version" not in comfy


def test_main_record_shape_contains_only_case_and_dimensions() -> None:
    shape = ConvRotShape("mlp-fc2", 3, 96, 512, "swiglu", False)
    output = torch.ones((1, 1))
    quality = measure_quality(output, output)
    workload = make_convrot_workload(
        shape,
        ConvRotConfig(torch.bfloat16, 256, 7),
        device=torch.device("cpu"),
    )
    configuration = {
        **workload.common_configuration(),
        "operation_entrypoint": "piper_kernels.linear.convrot.convrot_int8_linear",
        "input_preparation": "fused",
    }
    measurement = ProviderMeasurement(
        provider="piper-convrot",
        output=output,
        timings=_phase_timings(),
        configuration=configuration,
    )
    result = Result(
        input_preparation="fused",
        piper=measurement,
        quality=quality,
    )

    (record,) = _records_for_result(shape, result, _environment())
    value = record.as_dict()

    assert value["shape"] == {
        "case": "mlp-fc2",
        "rows": 3,
        "out_features": 96,
        "in_features": 512,
        "raw_input_features": 1024,
    }
    assert value["configuration"]["input_activation"] == "swiglu"
    assert value["configuration"]["has_bias"] is False
    assert value["configuration"]["logical_input_layout"] == "up_gate"
    assert value["configuration"]["provider_input_layout"] == "up_gate"


def test_main_comfy_record_uses_installed_version_and_provider_layout() -> None:
    shape = ConvRotShape("mlp-fc2", 3, 96, 512, "swiglu", False)
    output = torch.ones((1, 1))
    quality = measure_quality(output, output)
    workload = make_convrot_workload(
        shape,
        ConvRotConfig(torch.bfloat16, 256, 7),
        device=torch.device("cpu"),
    )
    common = workload.common_configuration()
    piper = ProviderMeasurement(
        provider="piper-convrot",
        output=output,
        timings=_phase_timings(),
        configuration={
            **common,
            "operation_entrypoint": "piper_kernels.linear.convrot.convrot_int8_linear",
            "input_preparation": "fused",
        },
    )
    comfy = ProviderMeasurement(
        provider="comfy-kitchen",
        output=output,
        timings=_phase_timings(),
        configuration=_comfy_provider_configuration(common, shape, "0.2.28"),
    )
    result = Result(
        input_preparation="fused",
        piper=piper,
        quality=quality,
        comfy_kitchen=comfy,
        comfy_kitchen_quality=quality,
    )

    records = _records_for_result(shape, result, _environment())
    comfy_record = next(record for record in records if record.provider == "comfy-kitchen")
    configuration = comfy_record.as_dict()["configuration"]

    assert configuration["installed_version"] == "0.2.28"
    assert "version" not in configuration
    assert configuration["logical_input_layout"] == "up_gate"
    assert configuration["provider_input_layout"] == "gate_up"


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


def test_preparation_compiler_reporting_rejects_multiple_widths() -> None:
    arguments = _parse_preparation_args(["--in-features", "512", "768", "--compiler-report"])

    with pytest.raises(SystemExit, match="exactly one --in-features"):
        _validate_preparation_args(arguments)


@pytest.mark.parametrize(
    ("benchmark_option", "compiler_option"),
    [
        ("--json", "--compiler-json"),
        ("--json", "--compiler-jsonl"),
        ("--jsonl", "--compiler-json"),
        ("--jsonl", "--compiler-jsonl"),
    ],
)
def test_preparation_benchmark_and_compiler_outputs_must_not_collide(
    tmp_path: Path,
    benchmark_option: str,
    compiler_option: str,
) -> None:
    path = tmp_path / "records.json"
    arguments = _parse_preparation_args(
        [
            "--in-features",
            "5376",
            benchmark_option,
            str(path),
            compiler_option,
            str(path),
        ]
    )

    with pytest.raises(SystemExit, match="must be different"):
        _output_targets(arguments)


def _comfy_cuda_module() -> ModuleType:
    module = ModuleType("comfy_kitchen.backends.cuda")
    module._C = object()
    module._wrap_for_dlpack = lambda tensor: tensor
    return module


def test_private_comfy_preparation_adapter_accepts_its_declared_contract(
    monkeypatch,
) -> None:
    module = _comfy_cuda_module()
    monkeypatch.setattr(
        "benchmark_convrot_preparation.importlib.metadata.version",
        lambda _name: "0.2.28",
    )
    monkeypatch.setattr(
        "benchmark_convrot_preparation.importlib.import_module",
        lambda _name: module,
    )

    adapter = _load_comfy_kitchen_cuda()

    assert COMFY_KITCHEN_ADAPTER_CONTRACT_VERSION == "0.2.28"
    assert adapter.cuda is module
    assert adapter.installed_version == "0.2.28"
    assert adapter.adapter_contract_version == "0.2.28"


def test_private_comfy_preparation_adapter_rejects_other_versions(monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmark_convrot_preparation.importlib.metadata.version",
        lambda _name: "0.2.29",
    )
    monkeypatch.setattr(
        "benchmark_convrot_preparation.importlib.import_module",
        lambda _name: pytest.fail("wrong-version package backend must not be imported"),
    )

    with pytest.raises(SystemExit, match=r"supports comfy-kitchen==0\.2\.28; found 0\.2\.29"):
        _load_comfy_kitchen_cuda()


def test_preparation_cli_and_records_expose_phase_timings(tmp_path) -> None:
    output_path = tmp_path / "preparation.jsonl"
    arguments = _parse_preparation_args(["--jsonl", str(output_path)])
    phase = PreparationPhaseResult(
        phase="fused",
        provider=PIPER_TRITON_PROVIDER,
        operation_provenance=(
            "piper_kernels.linear.convrot.int8.triton.fused_rotate_quantize_input"
        ),
        timing=Timing(1.0, 0.8, 1.2, ClockDomain.DEVICE_EVENT),
        minimum_global_bytes=1024,
        effective_minimum_tbps=2.0,
        baseline_phase="fused",
        speedup_vs_baseline=1.0,
        provider_configuration={
            "fuse_rotation_quantization": True,
            "fused_num_warps": 8,
            "rotation_num_warps": 4,
            "quantization_num_warps": 8,
        },
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
    assert value["configuration"]["logical_input_layout"] == "up_gate"
    assert value["configuration"]["provider_input_layout"] == "up_gate"
    assert value["configuration"]["baseline_provider"] == PIPER_TRITON_PROVIDER
    assert value["configuration"]["baseline_phase"] == "fused"
    assert value["configuration"]["operation_provenance"] == phase.operation_provenance
    assert value["configuration"]["fuse_rotation_quantization"] is True
    assert value["configuration"]["fused_num_warps"] == 8
    assert value["configuration"]["rotation_num_warps"] == 4
    assert value["configuration"]["quantization_num_warps"] == 8
    assert "installed_version" not in value["configuration"]
    assert "adapter_contract_version" not in value["configuration"]
    assert value["timings"]["prepared_execution"]["clock"] == "device_event"
    assert value["extra"]["speedup_vs_baseline"] == 1.0

    write_records([record], output_target(arguments))
    written = json.loads(output_path.read_text())
    assert written["benchmark"] == "convrot-preparation"
    assert written["shape"]["raw_input_features"] == 1024


def test_comfy_preparation_record_includes_installed_and_contract_versions() -> None:
    phase = PreparationPhaseResult(
        phase="comfy-kitchen",
        provider="comfy-kitchen",
        operation_provenance=("comfy_kitchen.backends.cuda._C.quantize_int8_rowwise_convrot64"),
        timing=Timing(1.0, 0.8, 1.2, ClockDomain.DEVICE_EVENT),
        minimum_global_bytes=1024,
        effective_minimum_tbps=2.0,
        baseline_phase="fused",
        speedup_vs_baseline=1.1,
        provider_configuration={
            "installed_version": "0.2.28",
            "adapter_contract_version": COMFY_KITCHEN_ADAPTER_CONTRACT_VERSION,
        },
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

    configuration = record.as_dict()["configuration"]
    assert configuration["installed_version"] == "0.2.28"
    assert configuration["adapter_contract_version"] == "0.2.28"
    assert configuration["operation_provenance"] == phase.operation_provenance
    assert configuration["logical_input_layout"] == "up_gate"
    assert configuration["provider_input_layout"] == "gate_up"


def test_preparation_compiler_report_only_inspects_launched_phases() -> None:
    configuration = _preparation_configuration()
    ordinary = _inspection_provider(
        _parse_preparation_args(["--in-features", "512"]),
        configuration,
    )
    swiglu = _inspection_provider(
        _parse_preparation_args(["--in-features", "512", "--input-activation", "swiglu"]),
        configuration,
    )

    assert ordinary.name == PIPER_TRITON_PROVIDER
    assert swiglu.name == PIPER_TRITON_PROVIDER
    assert set(ordinary.triton_jit_functions) == {"rotate", "quantize", "fused"}
    assert set(swiglu.triton_jit_functions) == {"fused"}
    assert swiglu.configuration["fused_num_warps"] == 8
    assert swiglu.configuration["rotation_num_warps"] == 4
    assert swiglu.configuration["quantization_num_warps"] == 8
    assert swiglu.configuration["fuse_rotation_quantization"] is True
    assert swiglu.configuration["logical_input_layout"] == "up_gate"
    assert swiglu.configuration["provider_input_layout"] == "up_gate"
