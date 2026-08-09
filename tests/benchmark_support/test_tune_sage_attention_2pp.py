import pytest
from lib.tuning import UnsupportedTuningCandidateError
from tune_sage_attention_2pp import (
    _candidate_choices,
    _parse_args,
    _resolve_plan,
    _validate_args,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sage_attention_2pp.triton import (
    _select_sage_attention_2pp_execution_plan,
)

_SM120 = AcceleratorTarget(backend="cuda", architecture="sm120")


def _production_plan():
    return _select_sage_attention_2pp_execution_plan(
        _SM120,
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
        head_dim=128,
        is_causal=False,
    )


def test_omitted_axes_measure_only_the_production_plan() -> None:
    choices = _candidate_choices(_parse_args([]), _production_plan())

    assert len(choices) == 1
    choice = choices[0]
    assert choice.block_m == 128
    assert choice.num_warps == 4
    assert choice.num_stages == 3
    assert choice.use_tensor_descriptors
    assert choice.fuse_kv_quantization
    assert choice.fuse_query_quantization
    assert choice.use_packed_probability_conversion


def test_explicit_axes_form_a_deduplicated_cartesian_search() -> None:
    arguments = _parse_args(
        [
            "--block-m",
            "64",
            "128",
            "128",
            "--num-stages",
            "2",
            "3",
            "--probability-conversion",
            "stock",
            "packed",
        ]
    )

    choices = _candidate_choices(arguments, _production_plan())

    assert len(choices) == 8
    assert len({choice.name for choice in choices}) == 8


def test_candidate_limit_prevents_accidental_compile_explosion() -> None:
    arguments = _parse_args(
        [
            "--block-m",
            "64",
            "128",
            "--num-stages",
            "2",
            "3",
            "--max-candidates",
            "3",
        ]
    )

    with pytest.raises(SystemExit, match="search expands to 4 candidates"):
        _candidate_choices(arguments, _production_plan())


def test_noncausal_search_rejects_reverse_block_order() -> None:
    arguments = _parse_args(["--causal-block-order", "reverse"])

    with pytest.raises(SystemExit, match="requires causal attention"):
        _validate_args(arguments)


def test_plan_resolution_records_unsupported_descriptor_shapes() -> None:
    production_plan = _production_plan()
    choice = _candidate_choices(_parse_args(["--block-m", "64"]), production_plan)[0]

    with pytest.raises(UnsupportedTuningCandidateError, match="D128 M128"):
        _resolve_plan(
            choice,
            production_plan,
            target=_SM120,
            head_dim=128,
            is_causal=False,
        )


def test_plan_resolution_applies_valid_schedule_overrides() -> None:
    production_plan = _production_plan()
    choice = _candidate_choices(
        _parse_args(
            [
                "--block-m",
                "64",
                "--load-path",
                "pointer",
                "--num-warps",
                "8",
                "--num-stages",
                "2",
            ]
        ),
        production_plan,
    )[0]

    plan = _resolve_plan(
        choice,
        production_plan,
        target=_SM120,
        head_dim=128,
        is_causal=False,
    )

    assert plan.block_m == 64
    assert plan.num_warps == 8
    assert plan.num_stages == 2
    assert not plan.use_tensor_descriptors
