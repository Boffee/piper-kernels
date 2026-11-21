"""SM120 numerical coverage for forced routing through both NVFP4 output fusions."""

import pytest
import torch
from _accelerators import exact_sm120_available
from _compile_capture import TargetCapturePass

from piper_kernels import compile_options
from piper_kernels.fusions.nvfp4_sparse_piper import _output

from ._accuracy import assert_fusion_output_close
from .nvfp4_sparse_piper import test_compile as nvfp4


# Run this on SM120 for PR validation; portable graph tests do not check numerics.
@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", [None, 64], ids=["ordinary", "convrot"])
def test_forced_nvfp4_output_matches_materialized_attention(monkeypatch, group_size):
    torch.manual_seed(1093)
    monkeypatch.setattr(_output, "DEFAULT_QUERY_CHUNK_ROWS", 128)
    model = nvfp4._SparseProjectionAttentionOutput(dynamic=True, group_size=group_size).eval()
    options = compile_options(attention=True)
    hidden = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    num_query_blocks = (model.sequence_length + 63) // 64
    ids = torch.zeros((num_query_blocks, 1), device="cuda", dtype=torch.int32)
    counts = torch.zeros(num_query_blocks, device="cuda", dtype=torch.int32)
    capture = TargetCapturePass()
    compiled = torch.compile(model, fullgraph=True, options=capture.wrap_options(options))
    with torch.no_grad():
        for active, tile in ((0, 0), (1, 1), (1, 0)):
            ids.fill_(tile)
            counts.fill_(active)
            force_tiles = {"force_tile_ids": ids, "force_tile_counts": counts}
            expected = nvfp4._run_explicit_attention_output(model, hidden, **force_tiles)
            actual = compiled(hidden, **force_tiles)
            assert_fusion_output_close(actual, expected)
            assert capture.calls == 1
    target = torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
    assert capture.targets.count(target) == 1
