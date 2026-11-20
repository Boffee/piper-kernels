"""Implicit unit scales across both sparse-attention NVFP4 output formats."""

from __future__ import annotations

import pytest
import torch
from _accelerators import exact_sm120_available
from _compile_capture import TargetCapturePass

from piper_kernels import compile_options
from piper_kernels.fusions.nvfp4_sparse_piper import _output, output

from .._accuracy import assert_fusion_output_close
from .test_compile import _run_explicit_attention_output, _SparseProjectionAttentionOutput
from .test_output import _arguments

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120"),
]


@pytest.mark.parametrize("group_size", [None, 64], ids=["ordinary", "convrot"])
def test_unit_output_matches_explicit_scale_and_replays(group_size: int | None) -> None:
    torch.manual_seed(1031)
    arguments, _ = _arguments(sequence_length=385, bias=True, group_size=group_size)
    op = output._attention_output_op
    # Four chunks, including a partial final chunk, exercise the consumer stream.
    unit = torch.ones((), device="cuda", dtype=torch.float32)
    explicit = (*arguments[:19], unit, *arguments[20:], 128)
    implicit = (*arguments[:19], None, *arguments[20:], 128)
    value_mean = arguments[11]
    assert isinstance(value_mean, torch.Tensor)

    with torch.no_grad():
        expected = op(*explicit, group_size=group_size)
        actual = op(*implicit, group_size=group_size)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(*implicit, group_size=group_size)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = op(*implicit, group_size=group_size)
        # Both replays must use the current input while retaining the unit scalar.
        for _ in range(2):
            value_mean.add_(0.25)
            graph.replay()
            updated = op(*explicit, group_size=group_size)
            assert not torch.equal(updated, expected)
            torch.testing.assert_close(replayed, updated, atol=0, rtol=0)


@pytest.mark.parametrize("group_size", [None, 64], ids=["ordinary", "convrot"])
def test_compile_fuses_unit_output(
    group_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1033)
    monkeypatch.setattr(_SparseProjectionAttentionOutput, "sequence_length", 384)
    monkeypatch.setattr(_output, "DEFAULT_QUERY_CHUNK_ROWS", 128)
    model = _SparseProjectionAttentionOutput(
        dynamic=True,
        output_dynamic=False,
        group_size=group_size,
    ).eval()
    hidden_states = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    model.output.weight.act_per_tensor_scale = torch.ones((), device="cuda", dtype=torch.float32)
    options = compile_options(attention=True)
    capture = TargetCapturePass()
    with torch.no_grad():
        expected = _run_explicit_attention_output(model, hidden_states)
        model.output.weight.act_per_tensor_scale = None
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=capture.wrap_options(options))(
            hidden_states
        )
    assert_fusion_output_close(actual, expected)
    fused = torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
    assert capture.targets.count(fused) == 1
    assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in capture.targets
