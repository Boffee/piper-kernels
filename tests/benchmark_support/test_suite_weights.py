"""Independent checks for shared weight-family workloads and sampled references."""

import builtins

import pytest
import torch
from lib.cases import Conv3DCase, FFNCase, LinearCase, named_case
from lib.suite_weights import _conv3d, _convolution_patches, _linear, implementations
from torch.nn import functional as F  # noqa: N812


def test_enumeration_does_not_construct_or_substitute_workloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("enumeration must not allocate workload tensors")

    monkeypatch.setattr("lib.suite_weights.normal_tensor", forbidden)
    case = LinearCase(id="small", rows=7, in_features=64, out_features=96)
    choices = implementations(case, torch.device("cpu"))
    assert [choice.name for choice in choices] == [
        "torch",
        "convrot_int8",
        "nvfp4",
        "convrot_nvfp4",
    ]
    assert choices[0].unsupported_reason is None
    assert all(choice.unsupported_reason for choice in choices[1:])
    assert case.rows == 7


def test_torch_baselines_do_not_import_optional_weight_or_kernel_packages(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("torchao", "triton", "piper_kernels.weights", "piper_kernels.fusions")):
            raise AssertionError(f"torch baseline imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for identity in ("linear-small", "ffn-small-gelu", "conv3d-small"):
        case = named_case(identity)
        operation = implementations(case, torch.device("cpu"))[0].build()
        with torch.inference_mode():
            assert operation.check(operation.run()).metrics.relative_l2_error < 0.02


@pytest.mark.parametrize("bias", [False, True])
def test_linear_sampled_reference_detects_changed_values(bias):
    case = LinearCase(id="linear", rows=71, in_features=64, out_features=96, bias=bias)
    operation = implementations(case, torch.device("cpu"))[0].build()
    with torch.inference_mode():
        output = operation.run()
    quality = operation.check(output)
    assert quality.metrics.relative_l2_error < 0.01
    assert quality.sample_count == 64 * 96
    assert quality.total_count == 71 * 96
    output = output.clone()
    output[0] += 10
    assert operation.check(output).metrics.relative_l2_error > quality.relative_l2_limit


def test_int8_linear_reports_quantized_correctness_separately_from_fp_quality():
    case = LinearCase(id="quantized", rows=7, in_features=64, out_features=96, bias=True)
    operation = _linear(case, torch.device("cpu"), "convrot_int8")
    with torch.inference_mode():
        quality = operation.check(operation.run())
    assert quality.metrics.relative_l2_error > 0
    assert quality.comparisons["portable_quantized_linear"].relative_l2_error == 0
    with torch.inference_mode():
        corrupted = operation.run() + 1
    with pytest.raises(ValueError, match="quantized reference"):
        operation.check(corrupted)


@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_ffn_baseline_matches_independent_sampled_reference(activation):
    case = FFNCase(id="ffn", rows=71, width=64, intermediate=128, activation=activation)
    operation = implementations(case, torch.device("cpu"))[0].build()
    with torch.inference_mode():
        output = operation.run()
    quality = operation.check(output)
    assert output.shape == (71, 64)
    assert quality.metrics.relative_l2_error < 0.02
    assert quality.sample_count == 64 * 64


def test_convolution_neighborhoods_match_full_causal_reflection_padding():
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(2, 4, 3, 4, 5, generator=generator, dtype=torch.float64)
    weight = torch.randn(7, 4, 3, 3, 3, generator=generator, dtype=torch.float64)
    padded = F.pad(F.pad(source, (1, 1, 1, 1, 0, 0), mode="reflect"), (0, 0, 0, 0, 2, 0))
    expected = F.conv3d(padded, weight).permute(0, 2, 3, 4, 1).reshape(-1, 7)
    selected = torch.tensor([0, 4, 19, 20, 59, 60, 83, 119])
    patches = _convolution_patches(source, selected)
    actual = patches.flatten(1) @ weight.permute(0, 2, 3, 4, 1).flatten(1).T
    torch.testing.assert_close(actual, expected[selected], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("group_norm", [False, True])
@pytest.mark.parametrize("format_name", ["torch", "convrot_int8"])
def test_convolution_quality_reference_covers_quantization_and_framewise_norm(
    group_norm, format_name
):
    case = Conv3DCase(
        id="conv",
        dtype="float16",
        batch=2,
        channels=64,
        frames=2,
        height=3,
        width=4,
        out_channels=8,
        group_norm_silu=group_norm,
    )
    operation = _conv3d(case, torch.device("cpu"), format_name)
    with torch.inference_mode():
        output = operation.run()
    quality = operation.check(output)
    assert output.shape == (2, 8, 2, 3, 4)
    assert quality.sample_count == quality.total_count
    assert quality.metrics.relative_l2_error < quality.relative_l2_limit
    if format_name == "convrot_int8":
        assert quality.comparisons["portable_quantized_convolution"].relative_l2_error < 0.001


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator")
@pytest.mark.parametrize(
    "case_id",
    ["linear-tail", "ffn-small-gelu", "ffn-small-swiglu", "conv3d-small", "conv3d-small-norm-silu"],
)
def test_native_weight_families_use_the_shared_workload_and_quality_checks(case_id):
    case = named_case(case_id)
    assert isinstance(case, (LinearCase, FFNCase, Conv3DCase))
    for implementation in implementations(case, torch.device("cuda")):
        if implementation.unsupported_reason:
            continue
        with torch.inference_mode():
            operation = implementation.build()
            output = operation.run()
            quality = operation.check(output)
        assert torch.isfinite(output).all()
        assert quality.metrics.relative_l2_error < quality.relative_l2_limit
        assert all(metric.relative_l2_error < 0.02 for metric in quality.comparisons.values())
        if isinstance(case, FFNCase) and implementation.name == "convrot_int8":
            assert "portable_quantized_ffn" in quality.comparisons
            # An 8% scale error fits the floating-weight allowance but must fail
            # agreement with the quantized algorithm, for both GELU and SwiGLU.
            with pytest.raises(ValueError, match="quantized reference"):
                operation.check(output * 1.08)
        del output, operation
