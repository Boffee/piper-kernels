"""Tests for the offline ConvRot INT8 linear execution-plan tuner."""

import pytest
import torch
from lib.convrot import ConvRotConfig, ConvRotShape
from lib.convrot_providers import make_convrot_workload
from lib.providers import ProviderPhase
from tune_convrot_int8_linear import (
    _candidate_plans,
    _make_candidate,
    _parse_args,
    _validate_args,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.convrot.int8 import triton as convrot_backend
from piper_kernels.convrot.int8._policy import select_execution_plan

_SM120 = AcceleratorTarget(backend="cuda", architecture="sm120")


def _production_plan(*, rows: int = 512):
    return select_execution_plan(
        _SM120,
        rows=rows,
        out_features=4096,
        in_features=4096,
        group_size=256,
        dtype=torch.bfloat16,
    )


def _workload(*, rows: int = 2, out_features: int = 96, in_features: int = 512):
    return make_convrot_workload(
        ConvRotShape("custom", rows, out_features, in_features),
        ConvRotConfig(torch.bfloat16, 256, 0),
        device=torch.device("cpu"),
        target=_SM120,
    )


def test_tuner_defaults_to_complete_production_device_path() -> None:
    arguments = _parse_args([])

    assert arguments.phase is ProviderPhase.PREPARED_EXECUTION
    assert arguments.fuse_rotation_quantization is None
    assert arguments.fused_num_warps is None
    assert arguments.rotation_num_warps is None
    assert arguments.quantization_num_warps is None
    assert arguments.matmul_block_m is None
    assert arguments.matmul_block_n is None
    assert arguments.matmul_block_k is None
    assert arguments.matmul_num_warps is None
    assert arguments.matmul_num_stages is None
    assert arguments.minimum_sqnr_db == 20.0
    assert arguments.quality_rows == 256


def test_tuner_input_activation_is_enabled_only_explicitly() -> None:
    assert _parse_args([]).input_activation is None
    assert _parse_args(["--input-activation", "swiglu"]).input_activation == "swiglu"
    with pytest.raises(SystemExit):
        _parse_args(["--input-activation", "none"])


def test_omitted_axes_measure_only_the_production_plan() -> None:
    production_plan = _production_plan()

    assert _candidate_plans(_parse_args([]), production_plan) == (production_plan,)


def test_explicit_axes_form_a_deduplicated_cartesian_search() -> None:
    arguments = _parse_args(
        [
            "--fuse-rotation-quantization",
            "--fused-num-warps",
            "4",
            "8",
            "--matmul-block-m",
            "32",
            "64",
            "64",
            "--matmul-block-n",
            "64",
            "128",
            "--matmul-num-stages",
            "2",
            "3",
        ]
    )

    plans = _candidate_plans(arguments, _production_plan())

    assert len(plans) == 16
    assert len({tuple(plan.as_dict().items()) for plan in plans}) == 16


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("--fuse-rotation-quantization", True),
        ("--no-fuse-rotation-quantization", False),
    ],
)
def test_fusion_boolean_override(option: str, expected: bool) -> None:
    plan = _candidate_plans(_parse_args([option]), _production_plan())[0]

    assert plan.fuse_rotation_quantization is expected


def test_fused_warp_search_requires_fused_candidate() -> None:
    arguments = _parse_args(
        [
            "--no-fuse-rotation-quantization",
            "--fused-num-warps",
            "4",
            "8",
        ]
    )

    with pytest.raises(SystemExit, match="requires --fuse-rotation-quantization"):
        _candidate_plans(arguments, _production_plan())


def test_split_warp_axes_require_split_candidate() -> None:
    arguments = _parse_args(
        [
            "--fuse-rotation-quantization",
            "--rotation-num-warps",
            "2",
            "4",
        ]
    )

    with pytest.raises(SystemExit, match="require --no-fuse-rotation-quantization"):
        _candidate_plans(arguments, _production_plan())


def test_split_warp_axes_form_candidate_search() -> None:
    arguments = _parse_args(
        [
            "--no-fuse-rotation-quantization",
            "--rotation-num-warps",
            "2",
            "4",
            "--quantization-num-warps",
            "4",
            "8",
        ]
    )

    plans = _candidate_plans(arguments, _production_plan())

    assert len(plans) == 4
    assert {(plan.rotation_num_warps, plan.quantization_num_warps) for plan in plans} == {
        (2, 4),
        (2, 8),
        (4, 4),
        (4, 8),
    }


def test_candidate_limit_prevents_accidental_compile_explosion() -> None:
    arguments = _parse_args(
        [
            "--matmul-block-m",
            "32",
            "64",
            "--matmul-block-n",
            "64",
            "128",
            "--max-candidates",
            "3",
        ]
    )

    with pytest.raises(SystemExit, match="search expands to 4 candidates"):
        _candidate_plans(arguments, _production_plan())


def test_candidate_configuration_contains_flat_execution_plan_fields() -> None:
    plan = _production_plan()
    candidate = _make_candidate(plan, _workload())

    assert plan.as_dict().items() <= candidate.configuration.items()
    assert "preparation" not in candidate.configuration
    assert candidate.configuration["dtype"] == "bfloat16"
    assert candidate.configuration["quality_row_indices"] == (0, 1)


def test_candidate_provider_injects_plan_into_complete_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _production_plan()
    workload = _workload()
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.empty((2, 96), device="meta")

    monkeypatch.setattr(convrot_backend, "_run_convrot_int8_linear", fake_run)
    provider = _make_candidate(plan, workload).make_provider()

    prepared = provider.prepare()
    output = provider.run(prepared)

    assert prepared is workload.inputs
    assert output.shape == (2, 96)
    assert calls[0][1]["execution_plan"] is plan
    assert calls[0][1]["apply_swiglu"] is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--rows", "0"], "must all be positive"),
        (["--in-features", "65", "--group-size", "64"], "must be divisible"),
        (["--quality-rows", "0"], "quality rows must be positive"),
        (["--minimum-sqnr-db=nan"], "minimum SQNR must be finite"),
        (["--max-candidates", "0"], "maximum candidate count must be positive"),
    ],
)
def test_tuner_rejects_invalid_arguments(arguments: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        _validate_args(_parse_args(arguments))
