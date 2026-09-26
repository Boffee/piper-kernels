"""Tests for the representative ConvRot INT8 benchmark cases and records."""

import pytest
import torch
from benchmark_convrot_int8_realistic import (
    Case,
    _cases,
    _parse_args,
    _run_case,
    _summaries,
)

from piper_kernels._triton.targets import AcceleratorTarget


def test_default_cases_cover_h3_blocks_vae_and_short_projections() -> None:
    cases = _cases(_parse_args([]))

    blocks = [case for case in cases if case.group == "h3_block"]
    assert sorted({case.rows for case in blocks}) == [8192, 32768, 131072, 131073]
    assert {(case.stage, case.in_features, case.out_features) for case in blocks} == {
        ("qkv", 5376, 7168),
        ("out", 7168, 5376),
        ("ffn_up", 5376, 14336),
        ("ffn_down", 14336, 5376),
    }
    qkv = next(case for case in blocks if case.stage == "qkv")
    down = next(case for case in blocks if case.stage == "ffn_down")
    assert qkv.projections == 3
    assert down.activation == "gelu_tanh"
    assert {case.rows for case in cases if case.group == "h3_vae"} == {1797, 7188}
    assert min(case.rows for case in cases) == 1
    assert {
        (case.rows, case.in_features, case.out_features) for case in cases if case.group == "anchor"
    } == {(rows, k, n) for rows in (8192, 32768) for k in (6144, 14336) for n in (4096, 16384)}
    assert len(cases) == 59


def test_operations_count_every_projection_sharing_one_preparation() -> None:
    case = Case("h3_block", "qkv", 8, 5376, 7168, projections=3)

    assert case.operations == 2 * 8 * 5376 * 7168 * 3


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA CUDA GPU",
)
@pytest.mark.parametrize("bias", [False, True])
def test_small_run_agrees_with_original_plan_and_summarizes(bias: bool) -> None:
    arguments = ["--rounds", "2", "--sample-ms", "1", "--warmup-s", "0"]
    args = _parse_args(arguments + (["--bias"] if bias else []))
    cases = [
        Case("h3_block", "ffn_down", 257, 1024, 512, activation="gelu_tanh"),
        Case("h3_block", "qkv", 257, 512, 256, projections=3),
        Case("projection_mix", "256->96", 3, 256, 96),
    ]
    target = AcceleratorTarget.from_device(torch.device("cuda"))

    with torch.inference_mode():
        records = [_run_case(case, args, target) for case in cases]
    summaries = _summaries(records, cases)

    assert all(record["exact_vs_original"] for record in records)
    assert all(len(record["samples_us"]["production"]) == 2 for record in records)
    assert [(summary["group"], summary["rows"], summary["cases"]) for summary in summaries] == [
        ("h3_block", 257, 2),
        ("projection_mix", 3, 1),
    ]
    block = summaries[0]
    assert block["total_us"]["production"] == pytest.approx(
        sum(record["median_us"]["production"] for record in records[:2])
    )
    assert block["speedup_vs_original"] > 0
