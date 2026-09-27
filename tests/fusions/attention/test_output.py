"""Numerical and stream coverage for the attention-independent output pipeline."""

from __future__ import annotations

import math

import pytest
import torch

from piper_kernels.fusions.attention._output import run_chunked_attention_output


@pytest.mark.parametrize("project_auxiliary", [False, True])
def test_global_scale_projection_uses_all_ragged_windows(project_auxiliary: bool) -> None:
    shape = (2, 23, 2, 3)
    source = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).to(torch.bfloat16)
    auxiliary = source.mul(0.25)
    bias = torch.arange(5, dtype=torch.bfloat16)

    def launch_chunk(buffer, start, rows, auxiliary_chunk):
        assert auxiliary_chunk is not None
        torch.add(source[:, start : start + rows], auxiliary_chunk, out=buffer)

    def project_auxiliary_chunk(buffer, start, rows):
        buffer.copy_(auxiliary[:, start : start + rows])

    def project_attention(attention):
        # A separate scale per query window would produce different rounded values.
        values = attention.float().flatten(2)
        scale = values.abs().amax() / 127
        quantized = torch.round(values / scale) * scale
        return (quantized.sum(-1, keepdim=True) + bias).to(torch.bfloat16)

    actual = run_chunked_attention_output(
        shape,
        source.device,
        bias.numel(),
        7,
        launch_chunk,
        None,
        (bias,),
        auxiliary_input=None if project_auxiliary else auxiliary,
        project_auxiliary_chunk=project_auxiliary_chunk if project_auxiliary else None,
        project_attention=project_attention,
    )

    torch.testing.assert_close(actual, project_attention(source + auxiliary), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA streams")
@pytest.mark.parametrize("auxiliary_mode", ["none", "materialized", "projected"])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("output_features", [5, 12, 17])
@pytest.mark.parametrize("chunk_rows", [7, 128])
@pytest.mark.parametrize("reuse_output", [False, True])
def test_chunk_output_on_nondefault_stream_preserves_caller_storage(
    auxiliary_mode: str,
    capture: bool,
    output_features: int,
    chunk_rows: int,
    reuse_output: bool,
) -> None:
    shape = (2, 67, 3, 4)
    source = (
        torch.arange(math.prod(shape), device="cuda", dtype=torch.float32)
        .remainder(31)
        .reshape(shape)
        .to(torch.bfloat16)
    )
    auxiliary = source.mul(0.25)
    bias = torch.arange(output_features, device="cuda", dtype=torch.bfloat16)
    backing = torch.full((4, shape[1], bias.numel()), -999, device="cuda", dtype=source.dtype)
    out = backing[1:3]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch_chunk(buffer, start, rows, auxiliary_chunk):
        shares_output = buffer.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
        assert shares_output == (reuse_output and output_features >= shape[2] * shape[3])
        assert buffer[0].is_contiguous()
        current = source[:, start : start + rows]
        if auxiliary_chunk is None:
            buffer.copy_(current)
        else:
            torch.add(current, auxiliary_chunk, out=buffer)

    def project_auxiliary_chunk(buffer, start, rows):
        buffer.copy_(auxiliary[:, start : start + rows])

    def project_chunk(attention, output, start, rows):
        output[:, start : start + rows].copy_(attention.flatten(2).sum(-1, keepdim=True) + bias)

    def run():
        return run_chunked_attention_output(
            shape,
            source.device,
            bias.numel(),
            chunk_rows,
            launch_chunk,
            project_chunk,
            (bias,),
            auxiliary_input=auxiliary if auxiliary_mode == "materialized" else None,
            project_auxiliary_chunk=(
                project_auxiliary_chunk if auxiliary_mode == "projected" else None
            ),
            out=out,
            reuse_output_for_attention=reuse_output,
        )

    with torch.cuda.stream(stream):
        actual = run()
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = run()
            source.add_(1)
            graph.replay()
        attention = source if auxiliary_mode == "none" else source + auxiliary
        expected = attention.flatten(2).sum(-1, keepdim=True) + bias
    stream.synchronize()

    assert actual is out
    assert actual.storage_offset() == out.storage_offset() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(backing[0], torch.full_like(backing[0], -999), rtol=0, atol=0)
    torch.testing.assert_close(backing[-1], torch.full_like(backing[-1], -999), rtol=0, atol=0)
