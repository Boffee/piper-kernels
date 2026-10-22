"""Implicit unit scales across both sparse-attention NVFP4 output formats."""

from __future__ import annotations

import pytest
import torch
from _compile_capture import TargetCapturePass

from piper_kernels.fusions.convrot_nvfp4_sparse_piper import (
    convrot_nvfp4_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_nvfp4_sparse_piper import output as convrot_output
from piper_kernels.fusions.nvfp4_sparse_piper import (
    _output,
    nvfp4_sparse_piper_compile_options,
    output,
)

from .._accuracy import assert_fusion_output_close
from ..convrot_nvfp4_sparse_piper.test_compile import (
    _ConvRotSparseProjectionAttentionOutput,
)
from ..convrot_nvfp4_sparse_piper.test_compile import (
    _run_explicit_attention_output as convrot_reference,
)
from ..convrot_nvfp4_sparse_piper.test_output import _arguments as convrot_arguments
from ._helpers import exact_sm120_available
from .test_compile import _run_explicit_attention_output, _SparseProjectionAttentionOutput
from .test_output import _arguments

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120"),
]


@pytest.mark.parametrize("rotated", [False, True], ids=["ordinary", "convrot"])
def test_unit_output_matches_explicit_scale_and_replays(rotated: bool) -> None:
    torch.manual_seed(1031)
    if rotated:
        arguments, _ = convrot_arguments(sequence_length=385)
        op = convrot_output._attention_output_op
    else:
        arguments, _ = _arguments(sequence_length=385, bias=True)
        op = output._attention_output_op
    # Four chunks, including a partial final chunk, exercise the consumer stream.
    unit = torch.ones((), device="cuda", dtype=torch.float32)
    explicit = (*arguments[:17], unit, *arguments[18:], 128)
    implicit = (*arguments[:17], None, *arguments[18:], 128)
    value_mean = arguments[9]
    assert isinstance(value_mean, torch.Tensor)

    with torch.no_grad():
        expected = op(*explicit)
        actual = op(*implicit)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(*implicit)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = op(*implicit)
        # Both replays must use the current input while retaining the unit scalar.
        for _ in range(2):
            value_mean.add_(0.25)
            graph.replay()
            updated = op(*explicit)
            assert not torch.equal(updated, expected)
            torch.testing.assert_close(replayed, updated, atol=0, rtol=0)


@pytest.mark.parametrize("rotated", [False, True], ids=["ordinary", "convrot"])
def test_compile_fuses_unit_output(rotated: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(1033)
    model_type = (
        _ConvRotSparseProjectionAttentionOutput if rotated else _SparseProjectionAttentionOutput
    )
    monkeypatch.setattr(model_type, "sequence_length", 384)
    monkeypatch.setattr(_output, "DEFAULT_QUERY_CHUNK_ROWS", 128)
    model = model_type(dynamic=True, output_dynamic=False).eval()
    hidden_states = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    model.output.weight.act_per_tensor_scale = torch.ones((), device="cuda", dtype=torch.float32)
    reference = convrot_reference if rotated else _run_explicit_attention_output
    options = (
        convrot_nvfp4_sparse_piper_compile_options()
        if rotated
        else nvfp4_sparse_piper_compile_options()
    )
    capture = TargetCapturePass()
    with torch.no_grad():
        expected = reference(model, hidden_states)
        model.output.weight.act_per_tensor_scale = None
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=capture.wrap_options(options))(
            hidden_states
        )
    assert_fusion_output_close(actual, expected)
    prefix = "convrot_nvfp4" if rotated else "nvfp4"
    fused = getattr(
        torch.ops.piper_kernels, f"{prefix}_sparse_piper_projected_query_attention_output"
    ).default
    assert capture.targets.count(fused) == 1
    assert getattr(torch.ops.piper_kernels, f"{prefix}_linear").default not in capture.targets
