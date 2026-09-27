"""Dense output fusion preserves global attention semantics with bounded query storage."""

import pytest
import torch

from piper_kernels.attention.piper_attention import _quantized_dispatch as attention
from piper_kernels.fusions.convrot_int8_piper import _backend, key, output, query, value
from piper_kernels.linear.convrot.int8 import _ops

from .test_query import _NATIVE, _operands


def _arguments(
    *,
    head_dim=64,
    sequence=257,
    causal=False,
    dtype=torch.bfloat16,
    device="cuda",
    output_features=80,
):
    q = _operands(device, sequence=sequence, head_dim=head_dim)
    k = _operands(device, sequence=sequence if causal else sequence + 17, head_dim=head_dim)
    if device == "meta":
        context = (
            *key._project_key_op_fake(*k, 1e-6, head_dim=head_dim),
            *value._project_value_op_fake(*k[:4], head_dim=head_dim, is_causal=causal),
        )
    else:
        context = (
            *key._project_key_op(*k, 1e-6, head_dim=head_dim),
            *value._project_value_op(*k[:4], head_dim=head_dim, is_causal=causal),
        )
    weight = torch.randint(
        -100, 100, (output_features, 3 * head_dim), device=device, dtype=torch.int8
    )
    scales = torch.full((output_features, 1), 0.001, device=device)
    bias = torch.randn(output_features, device=device, dtype=dtype)
    return (
        *q,
        1e-6,
        head_dim**-0.5,
        None,
        *context,
        k[0].shape[1],
        causal,
        dtype,
        weight,
        scales,
        bias,
        64,
        None,
    )


def _reference(args, head_dim):
    q, qs = query._project_query_op(*args[:10], head_dim=head_dim)
    attended = attention._piper_attention_from_quantized_op(
        q, qs, *args[10:16], args[0].shape[1], *args[16:19]
    )
    rows = attended.transpose(1, 2).reshape(args[0].shape[0], args[0].shape[1], -1)
    return _ops.linear(rows, *args[19:23], None, args[23])


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sequence", [65, 257])
@pytest.mark.parametrize("output_expansion", [None, 0, 64])
def test_chunked_output_matches_materialized_fused_attention(
    head_dim, causal, dtype, sequence, output_expansion
):
    args = _arguments(
        head_dim=head_dim,
        sequence=sequence,
        causal=causal,
        dtype=dtype,
        output_features=80 if output_expansion is None else 3 * head_dim + output_expansion,
    )
    actual = output._projected_query_attention_output_op(
        *args, head_dim=head_dim, query_chunk_rows=128
    )
    expected = _reference(args, head_dim)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("head_dim", [64, 128])
def test_projected_query_windows_keep_global_rope_positions(head_dim):
    args = _operands("cuda", sequence=321, head_dim=head_dim)
    full = query._project_query_op(*args, 1e-6, head_dim**-0.5, head_dim=head_dim)
    backend = _backend.select_projection_backend(args[0], head_dim=head_dim)
    for start, rows in ((64, 128), (192, 129)):
        result = query._new_outputs(args[0], (2, rows, 3, head_dim))
        backend.project_query(
            *args, 1e-6, head_dim**-0.5, chunk_start=start, chunk_rows=rows, out=result
        )
        torch.testing.assert_close(
            result[0][:, :, :rows], full[0][:, :, start : start + rows], atol=0, rtol=0
        )
        torch.testing.assert_close(
            result[1][:, :, : (rows + 31) // 32],
            full[1][:, :, start // 32 : (start + rows + 31) // 32],
            atol=0,
            rtol=0,
        )


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("chunk_rows", [128, 512])
@pytest.mark.parametrize("output_features", [80, 192, 256])
def test_output_graph_capture_reads_live_input_and_static_scale(chunk_rows, output_features):
    args = list(_arguments(sequence=1025, causal=True, output_features=output_features))
    args[-1] = torch.tensor(0.03, device="cuda")
    for _ in range(2):
        output._projected_query_attention_output_op(*args, head_dim=64, query_chunk_rows=chunk_rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = output._projected_query_attention_output_op(
            *args, head_dim=64, query_chunk_rows=chunk_rows
        )
    args[0].zero_()
    args[-1].mul_(1.5)
    expected = _reference(args, 64)
    graph.replay()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.gpu
@_NATIVE
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("chunk_rows", "sequence"),
    [
        (None, 2 * output.DEFAULT_QUERY_CHUNK_ROWS + 1),
        (128, 385),
        (512, 1025),
    ],
)
def test_output_reuses_bounded_query_storage(monkeypatch, chunk_rows, sequence, causal):
    args = _arguments(sequence=sequence, causal=causal)
    backend = _backend.select_projection_backend(args[0], head_dim=64)
    project = backend.project_query
    calls = []

    def record(*operands, **kwargs):
        calls.append((kwargs["chunk_start"], kwargs["chunk_rows"], kwargs["out"]))
        project(*operands, **kwargs)

    monkeypatch.setattr(backend, "project_query", record)
    kwargs = {} if chunk_rows is None else {"query_chunk_rows": chunk_rows}
    actual = output._projected_query_attention_output_op(*args, head_dim=64, **kwargs)
    capacity = calls[0][2][0].shape[2]
    assert capacity <= (chunk_rows or output.DEFAULT_QUERY_CHUNK_ROWS)
    next_start = 0
    for start, rows, _ in calls:
        assert start == next_start
        assert start % 128 == 0
        assert 0 < rows <= capacity
        next_start += rows
    assert next_start == sequence
    assert all(buffers[0].shape[2] == capacity for _, _, buffers in calls)
    assert len({buffers[0].data_ptr() for _, _, buffers in calls}) == 1
    assert len({buffers[1].data_ptr() for _, _, buffers in calls}) == 1
    monkeypatch.setattr(backend, "project_query", project)
    torch.testing.assert_close(actual, _reference(args, 64), atol=0, rtol=0)
