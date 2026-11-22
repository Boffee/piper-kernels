"""Numerical and stream coverage for the attention-independent output pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


@dataclass(slots=True)
class _SlotTrace:
    source: torch.Tensor
    auxiliary: torch.Tensor
    bias: torch.Tensor
    attention_ptrs: set[int] = field(default_factory=set)
    auxiliary_ptrs: set[int] = field(default_factory=set)
    attention_nbytes: set[int] = field(default_factory=set)
    auxiliary_nbytes: set[int] = field(default_factory=set)
    launch_streams: list[int] = field(default_factory=list)
    aux_streams: list[int] = field(default_factory=list)
    project_streams: list[int] = field(default_factory=list)

    def clear(self) -> None:
        self.attention_ptrs.clear()
        self.auxiliary_ptrs.clear()
        self.attention_nbytes.clear()
        self.auxiliary_nbytes.clear()
        self.launch_streams.clear()
        self.aux_streams.clear()
        self.project_streams.clear()

    def launch_chunk(self, buffer, start, rows, auxiliary_chunk) -> None:
        assert auxiliary_chunk is not None
        if self.source.device.type == "cuda":
            self.launch_streams.append(torch.cuda.current_stream().cuda_stream)
        self.attention_ptrs.add(buffer.data_ptr())
        self.attention_nbytes.add(buffer.untyped_storage().nbytes())
        torch.add(self.source[:, start : start + rows], auxiliary_chunk, out=buffer)

    def project_auxiliary_chunk(self, buffer, start, rows) -> None:
        if self.source.device.type == "cuda":
            self.aux_streams.append(torch.cuda.current_stream().cuda_stream)
        self.auxiliary_ptrs.add(buffer.data_ptr())
        self.auxiliary_nbytes.add(buffer.untyped_storage().nbytes())
        buffer.copy_(self.auxiliary[:, start : start + rows])

    def project_chunk(self, attention, output, start, rows) -> None:
        if self.source.device.type == "cuda":
            self.project_streams.append(torch.cuda.current_stream().cuda_stream)
        projected = attention.flatten(2).sum(-1, keepdim=True) + self.bias
        output[:, start : start + rows].copy_(projected)


def _record_chunked_slot_contract(
    device: str | torch.device,
    *,
    capture: bool = False,
    overlap_streams: bool | None = None,
) -> None:
    shape = (2, 67, 3, 4)
    chunk_rows = 7
    output_features = 5
    source = (
        torch.arange(math.prod(shape), device=device, dtype=torch.float32)
        .remainder(31)
        .reshape(shape)
        .to(torch.bfloat16)
    )
    bias = torch.arange(output_features, device=device, dtype=torch.bfloat16)
    backing = torch.full((4, shape[1], output_features), -999, device=device, dtype=source.dtype)
    out = backing[1:3]
    trace = _SlotTrace(source=source, auxiliary=source.mul(0.25), bias=bias)
    single_chunk_bytes = (
        math.prod((shape[0], chunk_rows, shape[2], shape[3])) * source.element_size()
    )
    is_gpu = source.device.type == "cuda"
    expect_overlap = (
        False
        if not is_gpu
        else ((torch.version.hip is None) if overlap_streams is None else overlap_streams)
    )

    def run():
        return run_chunked_attention_output(
            shape,
            source.device,
            output_features,
            chunk_rows,
            trace.launch_chunk,
            trace.project_chunk,
            (bias,),
            project_auxiliary_chunk=trace.project_auxiliary_chunk,
            out=out,
            reuse_output_for_attention=False,
            overlap_streams=overlap_streams,
        )

    if is_gpu:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = run()
            if capture:
                trace.clear()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = run()
                source.add_(1)
                graph.replay()
            expected = (source + trace.auxiliary).flatten(2).sum(-1, keepdim=True) + bias
        stream.synchronize()
        caller_stream = stream.cuda_stream
        expected_chunks = math.ceil(shape[1] / chunk_rows)
        assert len(trace.launch_streams) == expected_chunks
        assert len(trace.project_streams) == expected_chunks
        assert len(trace.aux_streams) == expected_chunks
        assert all(recorded == caller_stream for recorded in trace.launch_streams)
        if expect_overlap:
            assert all(recorded != caller_stream for recorded in trace.project_streams)
            assert all(recorded != caller_stream for recorded in trace.aux_streams)
        else:
            assert all(recorded == caller_stream for recorded in trace.project_streams)
            assert all(recorded == caller_stream for recorded in trace.aux_streams)
    else:
        actual = run()
        expected = (source + trace.auxiliary).flatten(2).sum(-1, keepdim=True) + bias

    slots = 2 if expect_overlap else 1
    assert actual is out
    assert trace.attention_nbytes == {slots * single_chunk_bytes}
    assert trace.auxiliary_nbytes == {slots * single_chunk_bytes}
    assert len(trace.attention_ptrs) == slots
    assert len(trace.auxiliary_ptrs) == slots
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(backing[0], torch.full_like(backing[0], -999), rtol=0, atol=0)
    torch.testing.assert_close(backing[-1], torch.full_like(backing[-1], -999), rtol=0, atol=0)


def test_multichunk_cpu_pipeline_uses_single_slot_buffers() -> None:
    _record_chunked_slot_contract("cpu")


def test_overlap_streams_rejects_cpu_device_before_allocating() -> None:
    def fail(*_args, **_kwargs):
        pytest.fail("validated stream overlap should fail before running chunk callbacks")

    with pytest.raises(ValueError, match="CUDA or HIP device"):
        run_chunked_attention_output(
            (1, 14, 2, 4),
            torch.device("cpu"),
            5,
            7,
            fail,
            fail,
            (),
            project_auxiliary_chunk=fail,
            overlap_streams=True,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA streams")
@pytest.mark.parametrize("overlap_streams", [None, False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_chunk_output_stream_overlap_and_buffer_slot_contracts(
    overlap_streams: bool | None,
    capture: bool,
) -> None:
    _record_chunked_slot_contract("cuda", capture=capture, overlap_streams=overlap_streams)
