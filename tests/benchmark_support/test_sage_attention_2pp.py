"""Tests for the SageAttention2++ benchmark provider wiring."""

from pathlib import Path

import pytest
import torch
from benchmark_sage_attention_2pp import (
    _CANONICAL_SAGE2,
    _CANONICAL_SAGE2PP,
    _PURE_TRITON,
    _SDPA,
    _canonical_qk_granularity,
    _make_providers,
    _output_targets,
    _parse_args,
    _resolved_provider_names,
    _validate_args,
)
from lib import AttentionConfig


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 9), "per_thread"), ((12, 0), "per_warp"), ((12, 1), "per_warp")],
)
def test_canonical_qk_granularity_matches_architecture(
    capability: tuple[int, int],
    expected: str,
) -> None:
    assert _canonical_qk_granularity(capability) == expected


def test_default_providers_use_common_provider_contract() -> None:
    tensor = torch.empty((1, 1, 8, 64), dtype=torch.float16)
    providers = _make_providers(
        (tensor, tensor, tensor),
        provider_names=(_PURE_TRITON, _SDPA),
        config=AttentionConfig(dtype="float16"),
        capability=(8, 9),
    )

    assert tuple(providers) == (_PURE_TRITON, _SDPA)
    assert providers[_PURE_TRITON].configuration["algorithm"] == "sage_attention_2pp"
    assert providers[_PURE_TRITON].configuration["qk_quantization"] == "per_thread"
    assert "attention" in providers[_PURE_TRITON].triton_jit_functions
    assert providers[_SDPA].configuration["algorithm"] == "scaled_dot_product_attention"
    assert not providers[_SDPA].triton_jit_functions


def test_canonical_flag_adds_both_pinned_providers() -> None:
    arguments = _parse_args(["--canonical", "--sequence", "128"])

    assert _resolved_provider_names(arguments) == (
        _PURE_TRITON,
        _SDPA,
        _CANONICAL_SAGE2PP,
        _CANONICAL_SAGE2,
    )


def test_causal_cross_attention_is_rejected() -> None:
    arguments = _parse_args(
        ["--causal", "--sequence", "128", "--kv-sequence", "256"]
    )
    names = _resolved_provider_names(arguments)

    with pytest.raises(SystemExit, match="equal query and key/value lengths"):
        _validate_args(arguments, names)


def test_compiler_inspection_requires_one_shape() -> None:
    arguments = _parse_args(
        ["--sequence", "128", "256", "--compiler-report"]
    )
    names = _resolved_provider_names(arguments)

    with pytest.raises(SystemExit, match="exactly one query length"):
        _validate_args(arguments, names)


@pytest.mark.parametrize(
    ("benchmark_option", "compiler_option"),
    [
        ("--json", "--compiler-json"),
        ("--json", "--compiler-jsonl"),
        ("--jsonl", "--compiler-json"),
        ("--jsonl", "--compiler-jsonl"),
    ],
)
def test_benchmark_and_compiler_outputs_must_not_collide(
    tmp_path: Path,
    benchmark_option: str,
    compiler_option: str,
) -> None:
    path = tmp_path / "records.json"
    arguments = _parse_args(
        [
            "--sequence",
            "128",
            benchmark_option,
            str(path),
            compiler_option,
            str(path),
        ]
    )

    with pytest.raises(SystemExit, match="must be different"):
        _output_targets(arguments)
