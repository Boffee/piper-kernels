import pytest
import torch
from lib import suite_attention
from lib.cases import AttentionCase
from lib.inputs import sample_indices
from lib.suite_attention import implementations, make_inputs, sampled_dense_reference

from piper_kernels._triton.targets import AcceleratorTarget


def _case(**changes):
    values = {"id": "test-attention", "sequence": 193, "heads": 4, "kv_heads": 2, "head_dim": 64}
    return AttentionCase(**(values | changes))


@pytest.mark.parametrize("causal", [False, True])
def test_sampled_reference_keeps_full_context_and_original_causal_positions(causal):
    case = _case(causal=causal)
    inputs = make_inputs(case, torch.device("cpu"))
    query, key, value = (tensor.float() for tensor in inputs)
    rows = torch.tensor([0, 93, 192])
    reference = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=causal, enable_gqa=True
    ).index_select(2, rows)
    torch.testing.assert_close(sampled_dense_reference(inputs, rows, causal=causal), reference)


def test_sdpa_quality_reports_sampled_coverage_and_detects_bad_output():
    case = _case()
    operation = implementations(case, torch.device("cpu"))[0].build()
    output = operation.run()
    check = operation.check(output)
    assert check.sample_count == case.batch * case.heads * 64 * case.head_dim
    assert check.total_count == case.batch * case.heads * case.sequence * case.head_dim
    assert check.metrics.relative_l2_error < check.relative_l2_limit
    bad = operation.check(torch.zeros_like(output))
    assert bad.metrics.relative_l2_error > bad.relative_l2_limit


def test_native_providers_are_explicitly_unsupported_on_cpu():
    providers = implementations(_case(), torch.device("cpu"))
    assert providers[0].unsupported_reason is None
    assert providers[1].unsupported_reason
    assert "equal" in providers[2].unsupported_reason
    sparse = implementations(_case(keep_ratio=0.5), torch.device("cpu"))
    assert sparse[0].unsupported_reason


def test_sage_gqa_is_unsupported_even_on_capable_hardware(monkeypatch):
    monkeypatch.setattr(suite_attention, "supports_sage", lambda target: True)
    providers = implementations(_case(), torch.device("cpu"))
    assert "equal" in providers[2].unsupported_reason


@pytest.mark.parametrize(("backend", "architecture"), [("cuda", "sm120"), ("hip", "gfx1201")])
def test_provider_listing_uses_native_dispatch_capability_without_changing_case(
    monkeypatch, backend, architecture
):
    target = AcceleratorTarget(backend, architecture)
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda _: target)
    case = _case(kv_heads=4)
    providers = implementations(case, torch.device("cpu"))
    assert providers[1].unsupported_reason is None
    assert (providers[2].unsupported_reason is None) == (backend == "cuda")
    assert case.sequence == 193


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")
def test_inputs_are_identical_across_cpu_and_gpu():
    case = _case(sequence=65)
    cpu = make_inputs(case, torch.device("cpu"))
    gpu = make_inputs(case, torch.device("cuda"))
    for left, right in zip(cpu, gpu, strict=True):
        torch.testing.assert_close(left, right.cpu(), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")
@pytest.mark.parametrize("keep_ratio", [None, 0.5, 1.0])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_native_attention_suite_small_case(keep_ratio, kv_heads):
    case = _case(keep_ratio=keep_ratio, kv_heads=kv_heads)
    providers = implementations(case, torch.device("cuda"))
    for provider in providers:
        if provider.unsupported_reason:
            continue
        with torch.inference_mode():
            operation = provider.build()
            output = operation.run()
            check = operation.check(output)
        assert check.metrics.actual_nonfinite_count == 0
        assert check.metrics.relative_l2_error < check.relative_l2_limit
        if keep_ratio is not None:
            assert "fp32_same_sparse_routes" in check.comparisons
            assert check.sample_count <= check.total_count
        del output, operation


def test_sample_indices_cover_first_and_last_rows():
    indices = sample_indices(110592, device=torch.device("cpu"))
    assert indices.shape == (64,)
    assert int(indices[0]) == 0
    assert int(indices[-1]) == 110591
