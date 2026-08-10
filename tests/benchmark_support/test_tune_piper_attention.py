import pytest
import torch
from lib.tuning import TuningPhase
from tune_piper_attention import (
    _candidate_plans,
    _make_candidate,
    _parse_args,
    _validate_args,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention.triton import (
    _select_piper_attention_execution_plan,
)

_SM120 = AcceleratorTarget(backend="cuda", architecture="sm120")


def _production_plan(*, is_causal: bool = False):
    return _select_piper_attention_execution_plan(
        _SM120,
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
        head_dim=128,
        is_causal=is_causal,
    )


def test_tuner_defaults_to_production_plan() -> None:
    arguments = _parse_args([])

    assert arguments.use_tensor_descriptors is None
    assert arguments.phase is TuningPhase.PREPARED_EXECUTION
    assert arguments.minimum_sqnr_db == 20.0
    assert arguments.block_m is None
    assert arguments.num_warps is None
    assert arguments.num_stages is None
    assert arguments.use_packed_probability_conversion is None


def test_omitted_axes_measure_only_the_production_plan() -> None:
    production_plan = _production_plan()
    plans = _candidate_plans(_parse_args([]), production_plan)

    assert plans == (production_plan,)


def test_explicit_axes_form_a_deduplicated_cartesian_search() -> None:
    arguments = _parse_args(
        [
            "--no-use-tensor-descriptors",
            "--block-m",
            "64",
            "128",
            "128",
            "--num-warps",
            "2",
            "4",
            "--num-stages",
            "2",
            "3",
            "--loop-num-stages",
            "none",
            "2",
            "--use-packed-probability-conversion",
        ]
    )

    plans = _candidate_plans(arguments, _production_plan())

    assert len(plans) == 16
    assert len({tuple(plan.as_dict().items()) for plan in plans}) == 16


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("--no-use-packed-probability-conversion", False),
        ("--use-packed-probability-conversion", True),
    ],
)
def test_probability_conversion_boolean_override(option: str, expected: bool) -> None:
    arguments = _parse_args([option])

    plans = _candidate_plans(arguments, _production_plan())

    assert [plan.use_packed_probability_conversion for plan in plans] == [expected]


def test_candidate_configuration_uses_raw_execution_plan_fields() -> None:
    plan = _production_plan()
    tensor = torch.empty((1, 1, 8, 128), device="meta")

    candidate = _make_candidate(
        plan,
        (tensor, tensor, tensor),
        scale=128**-0.5,
        is_causal=False,
        target=_SM120,
        common_configuration={"dtype": "float16"},
    )

    assert plan.as_dict().items() <= candidate.configuration.items()
    assert "load_path" not in candidate.configuration


def test_candidate_limit_prevents_accidental_compile_explosion() -> None:
    arguments = _parse_args(
        [
            "--no-use-tensor-descriptors",
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
        _candidate_plans(arguments, _production_plan())


def test_tuner_accepts_causal_native_loop_controls() -> None:
    arguments = _parse_args(
        [
            "--causal",
            "--reverse-causal-blocks",
            "--loop-num-stages",
            "3",
            "--loop-licm",
        ]
    )

    _validate_args(arguments)
    plans = _candidate_plans(arguments, _production_plan(is_causal=True))

    assert all(plan.reverse_causal_blocks for plan in plans)
    assert all(plan.loop_num_stages == 3 for plan in plans)
    assert all(plan.loop_licm for plan in plans)


def test_tuner_rejects_causal_cross_attention() -> None:
    arguments = _parse_args(["--causal", "--sequence", "128", "--kv-sequence", "256"])

    with pytest.raises(SystemExit, match="equal query and key/value lengths"):
        _validate_args(arguments)


def test_tuner_rejects_reverse_order_for_noncausal_attention() -> None:
    arguments = _parse_args(["--reverse-causal-blocks"])

    with pytest.raises(SystemExit, match="requires causal attention"):
        _validate_args(arguments)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_tuner_rejects_nonfinite_minimum_sqnr(value: str) -> None:
    arguments = _parse_args([f"--minimum-sqnr-db={value}"])

    with pytest.raises(SystemExit, match="minimum SQNR must be finite"):
        _validate_args(arguments)
