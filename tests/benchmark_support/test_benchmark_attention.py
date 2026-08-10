"""Tests for the unified full-attention benchmark."""

from pathlib import Path

import pytest
import torch
from benchmark_attention import (
    _compiler_provider_name,
    _effective_tflops,
    _output_targets,
    _parse_args,
    _profile_provider_name,
    _validate_args,
)
from lib.attention import AttentionConfig, AttentionShape
from lib.attention_providers import (
    CANONICAL_CUDA_SAGE_ATTENTION_2,
    CANONICAL_CUDA_SAGE_ATTENTION_2PP,
    PIPER_ATTENTION,
    PIPER_ATTENTION_AFFINE,
    PYTORCH_SDPA,
    SAGE_ATTENTION_2PP,
    make_attention_providers,
    qk_quantization_granularity,
    resolve_provider_names,
    validate_provider_support,
)

from piper_kernels._triton.targets import AcceleratorTarget

_SM120 = AcceleratorTarget(backend="cuda", architecture="sm120")


def test_provider_ids_use_canonical_attention_names() -> None:
    assert PIPER_ATTENTION == "piper_attention"
    assert PIPER_ATTENTION_AFFINE == "piper_attention_affine"
    assert SAGE_ATTENTION_2PP == "sage_attention_2pp"
    assert CANONICAL_CUDA_SAGE_ATTENTION_2PP == "canonical_cuda_sage_attention_2pp"


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 9), "per_thread"), ((12, 0), "per_warp"), ((12, 1), "per_warp")],
)
def test_qk_granularity_matches_architecture(
    capability: tuple[int, int],
    expected: str,
) -> None:
    target = AcceleratorTarget(
        backend="cuda",
        architecture=f"sm{capability[0]}{capability[1]}",
    )
    assert qk_quantization_granularity(target) == expected


def test_provider_metadata_distinguishes_algorithms_and_controls() -> None:
    tensor = torch.empty((1, 1, 8, 64), dtype=torch.float16)
    providers = make_attention_providers(
        (tensor, tensor, tensor),
        provider_names=(
            PIPER_ATTENTION,
            PIPER_ATTENTION_AFFINE,
            SAGE_ATTENTION_2PP,
            PYTORCH_SDPA,
        ),
        config=AttentionConfig(dtype="float16"),
        target=_SM120,
    )

    assert providers[PIPER_ATTENTION_AFFINE].configuration["native_uint8"] is False
    assert not providers[PIPER_ATTENTION_AFFINE].configuration[
        "use_packed_probability_conversion"
    ]
    assert "mixed_sign_mma" not in providers[PIPER_ATTENTION_AFFINE].configuration
    assert providers[PIPER_ATTENTION].configuration["use_packed_probability_conversion"]
    assert "attention" in providers[PIPER_ATTENTION].triton_jit_functions
    assert providers[SAGE_ATTENTION_2PP].configuration["algorithm"] == ("sage_attention_2pp")
    assert providers[SAGE_ATTENTION_2PP].configuration["block_n"] == 64
    assert providers[SAGE_ATTENTION_2PP].configuration["use_packed_probability_conversion"]
    assert providers[SAGE_ATTENTION_2PP].configuration["fuse_kv_quantization"]
    assert "attention" in providers[SAGE_ATTENTION_2PP].triton_jit_functions
    assert "quantize-key-value-per-block" in providers[SAGE_ATTENTION_2PP].triton_jit_functions
    assert "quantize-query-per-warp" not in providers[SAGE_ATTENTION_2PP].triton_jit_functions
    assert providers[PYTORCH_SDPA].configuration["algorithm"] == ("scaled_dot_product_attention")
    assert not providers[PYTORCH_SDPA].triton_jit_functions


def test_short_causal_sage_attention_2pp_provider_registers_query_quantization() -> None:
    # D128 isolates the short-sequence policy: causal D64 always uses external
    # Q quantization regardless of sequence length.
    tensor = torch.empty((1, 1, 8, 128), dtype=torch.float16)
    providers = make_attention_providers(
        (tensor, tensor, tensor),
        provider_names=(SAGE_ATTENTION_2PP,),
        config=AttentionConfig(dtype="float16", is_causal=True),
        target=_SM120,
    )

    jit_functions = providers[SAGE_ATTENTION_2PP].triton_jit_functions
    assert "quantize-query-per-warp" in jit_functions
    assert "quantize-key-value-per-block" in jit_functions


def test_long_causal_sage_attention_2pp_provider_omits_query_quantization() -> None:
    tensor = torch.empty((1, 1, 32 * 1024, 128), device="meta", dtype=torch.float16)
    providers = make_attention_providers(
        (tensor, tensor, tensor),
        provider_names=(SAGE_ATTENTION_2PP,),
        config=AttentionConfig(dtype="float16", is_causal=True),
        target=_SM120,
    )

    jit_functions = providers[SAGE_ATTENTION_2PP].triton_jit_functions
    assert "quantize-query-per-warp" not in jit_functions


def test_other_sm12x_sage_attention_2pp_provider_uses_grouped_quantization() -> None:
    tensor = torch.empty((1, 1, 8, 128), dtype=torch.float16)
    providers = make_attention_providers(
        (tensor, tensor, tensor),
        provider_names=(SAGE_ATTENTION_2PP,),
        config=AttentionConfig(dtype="float16"),
        target=AcceleratorTarget(backend="cuda", architecture="sm121"),
    )

    jit_functions = providers[SAGE_ATTENTION_2PP].triton_jit_functions
    assert "quantize-query-per-warp" in jit_functions
    assert "quantize-key-per-block" in jit_functions
    assert "quantize-value-per-channel" in jit_functions
    assert "quantize-key-value-per-block" not in jit_functions


def test_default_providers_use_hardware_aware_comparison_set() -> None:
    assert resolve_provider_names(
        None,
        include_canonical=False,
        piper_attention_supported=True,
        sage_attention_2pp_supported=True,
    ) == (PIPER_ATTENTION, SAGE_ATTENTION_2PP, PYTORCH_SDPA)
    assert resolve_provider_names(
        None,
        include_canonical=False,
        piper_attention_supported=True,
        sage_attention_2pp_supported=False,
    ) == (PIPER_ATTENTION, PYTORCH_SDPA)
    assert resolve_provider_names(
        None,
        include_canonical=False,
        piper_attention_supported=False,
        sage_attention_2pp_supported=False,
    ) == (PYTORCH_SDPA,)


def test_canonical_flag_adds_both_pinned_providers() -> None:
    assert resolve_provider_names(
        (PYTORCH_SDPA,),
        include_canonical=True,
        piper_attention_supported=True,
        sage_attention_2pp_supported=True,
    ) == (
        PYTORCH_SDPA,
        CANONICAL_CUDA_SAGE_ATTENTION_2PP,
        CANONICAL_CUDA_SAGE_ATTENTION_2,
    )


def test_explicit_sage_attention_2pp_provider_requires_fp8() -> None:
    with pytest.raises(SystemExit, match="different FP16-PV algorithm"):
        validate_provider_support(
            (SAGE_ATTENTION_2PP,),
            AcceleratorTarget(backend="cuda", architecture="sm80"),
        )


def test_piper_attention_provider_requires_supported_mmav2_lowering() -> None:
    with pytest.raises(SystemExit, match="SM8x or consumer Blackwell SM12x"):
        validate_provider_support(
            (PIPER_ATTENTION,),
            AcceleratorTarget(backend="cuda", architecture="sm90"),
        )


def test_profile_provider_defaults_to_first_selected_provider() -> None:
    arguments = _parse_args(["--profile", "--sequence", "128"])
    names = (PYTORCH_SDPA, SAGE_ATTENTION_2PP)

    assert _profile_provider_name(arguments, names) == PYTORCH_SDPA


def test_compiler_provider_is_inferred_when_unambiguous() -> None:
    arguments = _parse_args(
        [
            "--providers",
            SAGE_ATTENTION_2PP,
            PYTORCH_SDPA,
            "--sequence",
            "128",
            "--compiler-report",
        ]
    )
    names = (SAGE_ATTENTION_2PP, PYTORCH_SDPA)

    assert _compiler_provider_name(arguments, names) == SAGE_ATTENTION_2PP


def test_compiler_provider_must_be_explicit_when_ambiguous() -> None:
    arguments = _parse_args(["--sequence", "128", "--compiler-report"])
    names = (PIPER_ATTENTION, SAGE_ATTENTION_2PP, PYTORCH_SDPA)

    with pytest.raises(SystemExit, match="--compiler-provider"):
        _compiler_provider_name(arguments, names)


def test_compiler_inspection_requires_a_triton_provider() -> None:
    arguments = _parse_args(["--sequence", "128", "--compiler-report"])

    with pytest.raises(SystemExit, match="selected Triton provider"):
        _compiler_provider_name(arguments, (PYTORCH_SDPA,))


def test_causal_cross_attention_is_rejected() -> None:
    arguments = _parse_args(["--causal", "--sequence", "128", "--kv-sequence", "256"])

    with pytest.raises(SystemExit, match="equal query and key/value lengths"):
        _validate_args(arguments, (PYTORCH_SDPA,))


def test_compiler_inspection_requires_one_shape() -> None:
    arguments = _parse_args(["--sequence", "128", "256", "--compiler-report"])

    with pytest.raises(SystemExit, match="exactly one query length"):
        _validate_args(arguments, (PIPER_ATTENTION,))


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


def test_effective_tflops_accounts_for_causal_triangle() -> None:
    shape = AttentionShape(1, 2, 4, 4, 8)
    noncausal = _effective_tflops(shape, AttentionConfig("float16"), 1.0)
    causal = _effective_tflops(
        shape,
        AttentionConfig("float16", is_causal=True),
        1.0,
    )

    assert noncausal == 4 * 1 * 2 * 4 * 4 * 8 / 1e9
    assert causal == 4 * 1 * 2 * (4 * 5 // 2) * 8 / 1e9
