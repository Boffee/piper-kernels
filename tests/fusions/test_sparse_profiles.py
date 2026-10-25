"""Keep-ratio changes must preserve the complete sparse projection fusions."""

import pytest
import torch
from _compile_capture import TargetCapturePass

from piper_kernels import SparsePiperAttention
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    convrot_int8_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_nvfp4_sparse_piper import (
    convrot_nvfp4_sparse_piper_compile_options,
)
from piper_kernels.fusions.nvfp4_sparse_piper import nvfp4_sparse_piper_compile_options

from ._accuracy import assert_fusion_output_close
from .convrot_int8_sparse_piper import test_compile as int8_cases
from .convrot_nvfp4_sparse_piper import test_compile as convrot_cases
from .nvfp4_sparse_piper import test_compile as nvfp4_cases


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="all three sparse projection families require NVIDIA SM120",
)
@pytest.mark.parametrize("cudagraphs", [False, True])
@pytest.mark.parametrize("weight_format", ["convrot_int8", "nvfp4", "convrot_nvfp4"])
def test_sparse_projection_and_output_reuse_graph_across_profiles(
    weight_format: str, cudagraphs: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(300)
    attention_kernel = int8_cases._reset_attention_kernel_cache()
    if weight_format == "convrot_int8":
        model_type = int8_cases._SparseProjectionAttentionOutput
        options = convrot_int8_sparse_piper_compile_options()
    elif weight_format == "nvfp4":
        model_type = nvfp4_cases._SparseProjectionAttentionOutput
        options = nvfp4_sparse_piper_compile_options()
    else:
        model_type = convrot_cases._ConvRotSparseProjectionAttentionOutput
        options = convrot_nvfp4_sparse_piper_compile_options()
    monkeypatch.setattr(model_type, "sequence_length", 1024)
    monkeypatch.setattr(model_type, "sparse_key_blocks", 12)
    model = (model_type(dynamic=False) if weight_format == "convrot_nvfp4" else model_type()).eval()
    inputs = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    options["triton.cudagraphs"] = cudagraphs
    capture = TargetCapturePass()
    torch._dynamo.reset()
    compiled = torch.compile(
        model, dynamic=False, fullgraph=True, options=capture.wrap_options(options)
    )
    profiles = [(index / 16, (index + 3) / 16) for index in range(1, 13)]
    # Exceed Dynamo's default eight variants, include a full physical budget,
    # then revisit earlier profiles to detect stale configuration in fused ops.
    profiles += [(1.0, 1.0), profiles[0], profiles[7]]
    specialization_count = None
    with torch.no_grad():
        for profile in profiles:
            model.sparse_attention = SparsePiperAttention(profile, routing="mean")
            if weight_format == "convrot_int8":
                expected, _attention = int8_cases._run_explicit_attention_output(
                    model, inputs, model.cos, model.sin, model.sparse_key_blocks
                )
            elif weight_format == "nvfp4":
                expected = nvfp4_cases._run_explicit_attention_output(model, inputs)
            else:
                expected = convrot_cases._run_explicit_attention_output(model, inputs)
            actual = compiled(inputs)
            assert_fusion_output_close(actual, expected)
            current_count = int8_cases._compiled_specialization_count(attention_kernel)
            if specialization_count is None:
                specialization_count = current_count
                assert specialization_count > 0
            assert current_count == specialization_count
    assert capture.calls == 1
    target = getattr(
        torch.ops.piper_kernels,
        f"{weight_format}_sparse_piper_projected_query_attention_output",
    ).default
    assert capture.targets.count(target) == 1
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in capture.targets
