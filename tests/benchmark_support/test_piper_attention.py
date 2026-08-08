"""Tests for the Piper Attention benchmark provider wiring."""

from pathlib import Path

import pytest
import torch
from benchmark_piper_attention import (
    _CANONICAL_SAGE2,
    _CANONICAL_SAGE2PP,
    _PIPER,
    _PIPER_AFFINE,
    _PIPER_CENTERED,
    _PIPER_UNCENTERED,
    _PURE_TRITON_SAGE2PP,
    _SDPA,
    _canonical_qk_granularity,
    _make_providers,
    _output_targets,
    _parse_args,
    _resolved_provider_names,
    _validate_args,
    _validate_provider_support,
)
from lib import AttentionConfig


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 9), "per_thread"), ((12, 0), "per_warp"), ((12, 1), "per_warp")],
)
def test_qk_granularity_matches_architecture(
    capability: tuple[int, int],
    expected: str,
) -> None:
    assert _canonical_qk_granularity(capability) == expected


def test_provider_metadata_distinguishes_piper_controls() -> None:
    tensor = torch.empty((1, 1, 8, 64), dtype=torch.float16)
    providers = _make_providers(
        (tensor, tensor, tensor),
        provider_names=(_PIPER_CENTERED, _PIPER_UNCENTERED, _PIPER_AFFINE),
        config=AttentionConfig(dtype="float16"),
        capability=(12, 0),
    )

    assert providers[_PIPER_CENTERED].configuration["center_value"] is True
    assert providers[_PIPER_UNCENTERED].configuration["center_value"] is False
    assert providers[_PIPER_AFFINE].configuration["mixed_sign_mma"] == "affine_proxy"
    assert "attention" in providers[_PIPER_CENTERED].triton_jit_functions


def test_default_providers_cover_piper_sage_and_sdpa() -> None:
    arguments = _parse_args([])

    assert _resolved_provider_names(arguments, fp8_supported=True) == (
        _PIPER,
        _PIPER_UNCENTERED,
        _PURE_TRITON_SAGE2PP,
        _SDPA,
    )


def test_default_providers_omit_sage2pp_without_fp8() -> None:
    arguments = _parse_args([])

    assert _resolved_provider_names(arguments, fp8_supported=False) == (
        _PIPER,
        _PIPER_UNCENTERED,
        _SDPA,
    )


def test_canonical_flag_adds_both_pinned_providers() -> None:
    arguments = _parse_args(["--canonical", "--sequence", "128"])

    assert _resolved_provider_names(arguments, fp8_supported=True) == (
        _PIPER,
        _PIPER_UNCENTERED,
        _PURE_TRITON_SAGE2PP,
        _SDPA,
        _CANONICAL_SAGE2PP,
        _CANONICAL_SAGE2,
    )


def test_explicit_sage2pp_provider_requires_fp8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "benchmark_piper_attention.supports_uint8_int8_mma",
        lambda _device: True,
    )
    monkeypatch.setattr(
        "benchmark_piper_attention.supports_fp8_fp16_mma",
        lambda _device: False,
    )

    with pytest.raises(SystemExit, match="different FP16-PV algorithm"):
        _validate_provider_support(
            (_PIPER, _PURE_TRITON_SAGE2PP),
            torch.device("cuda"),
        )


def test_piper_provider_requires_supported_mmav2_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmark_piper_attention.supports_uint8_int8_mma",
        lambda _device: False,
    )

    with pytest.raises(SystemExit, match="SM8x or consumer Blackwell SM12x"):
        _validate_provider_support((_PIPER,), torch.device("cuda"))


def test_causal_cross_attention_is_rejected() -> None:
    arguments = _parse_args(
        ["--causal", "--sequence", "128", "--kv-sequence", "256"]
    )
    names = _resolved_provider_names(arguments, fp8_supported=True)

    with pytest.raises(SystemExit, match="equal query and key/value lengths"):
        _validate_args(arguments, names)


def test_compiler_inspection_requires_one_shape() -> None:
    arguments = _parse_args(
        ["--providers", "piper", "--sequence", "128", "256", "--compiler-report"]
    )
    names = _resolved_provider_names(arguments, fp8_supported=True)

    with pytest.raises(SystemExit, match="exactly one query length"):
        _validate_args(arguments, names)


def test_compiler_inspection_requires_only_piper() -> None:
    arguments = _parse_args(["--sequence", "128", "--compiler-report"])
    names = _resolved_provider_names(arguments, fp8_supported=True)

    with pytest.raises(SystemExit, match="only the default Piper provider"):
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
