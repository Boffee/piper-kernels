"""Dense NVIDIA query windows preserve full attention and caller output views."""

from dataclasses import replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._nvidia import triton as backend


def _piper_gpu_available():
    return (
        torch.cuda.is_available()
        and AcceleratorTarget.from_device(torch.device("cuda")).supports_uint8_int8_mma
    )


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _piper_gpu_available(),
        reason="requires NVIDIA SM8x or consumer Blackwell SM12x mixed-sign MMAv2",
    ),
]


def _operands(head_dim, causal, descriptors, *, optimize_causal=None):
    if descriptors and torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("tensor descriptor window coverage requires SM120")
    block_m = 64 if causal else 128
    sequence = 2 * block_m + 17
    torch.manual_seed(529)
    query = torch.randn(2, sequence, 6, head_dim, device="cuda", dtype=torch.bfloat16)
    query = query.transpose(1, 2)
    key = torch.randn(
        2, sequence if causal else sequence + 31, 2, head_dim, device="cuda", dtype=query.dtype
    ).transpose(1, 2)
    value = torch.randn_like(key)
    plan = backend._default_piper_attention_execution_plan(query, causal)
    if plan.use_gluon_kernel:
        # The Gluon kernel always stops causal rows at the diagonal.
        if optimize_causal:
            pytest.skip("the Gluon kernel has a single causal traversal")
        return query, key, value, replace(plan, block_m=block_m)
    plan = replace(
        plan,
        block_m=block_m,
        use_tensor_descriptors=descriptors,
        optimize_causal_traversal=causal if optimize_causal is None else optimize_causal,
    )
    return query, key, value, plan


def _guarded_output(rows, head_dim, layout):
    if layout == "compact":
        elements = 2 * 6 * rows * head_dim
        backing = torch.full((elements + 32,), 7, device="cuda", dtype=torch.bfloat16)
        guards = torch.ones_like(backing, dtype=torch.bool)
        output = backing[16:-16].view(2, 6, rows, head_dim)
        guards[16:-16] = False
    elif layout == "bhsd":
        backing = torch.full((2, 6, rows + 8, head_dim), 7, device="cuda", dtype=torch.bfloat16)
        guards = torch.ones_like(backing, dtype=torch.bool)
        output = backing[:, :, 4 : rows + 4]
        guards[:, :, 4 : rows + 4] = False
    elif layout == "bshd":
        backing = torch.full((2, rows + 8, 6, head_dim), 7, device="cuda", dtype=torch.bfloat16)
        guards = torch.ones_like(backing, dtype=torch.bool)
        output = backing[:, 4 : rows + 4].transpose(1, 2)
        guards[:, 4 : rows + 4] = False
    else:
        backing = torch.full((2, 2, rows + 8, 6, head_dim), 7, device="cuda", dtype=torch.bfloat16)
        guards = torch.ones_like(backing, dtype=torch.bool)
        output = backing[1, :, :rows].transpose(1, 2)
        guards[1, :, :rows] = False
    output.fill_(float("nan"))
    return output, backing, guards


@pytest.mark.parametrize(
    ("head_dim", "causal", "descriptors", "optimize_causal"),
    [
        (64, False, False, False),
        (64, False, True, False),
        (64, True, False, False),
        (64, True, False, True),
        (64, True, True, True),
        (128, False, False, False),
        (128, False, True, False),
        (128, True, True, True),
    ],
)
def test_query_windows_and_independent_chunks_match_full_attention(
    head_dim, causal, descriptors, optimize_causal
):
    query, key, value, plan = _operands(
        head_dim, causal, descriptors, optimize_causal=optimize_causal
    )
    block_m = plan.block_m
    with torch.no_grad():
        prepared = backend._prepare_piper_attention(
            query, key, value, head_dim**-0.5, causal, execution_plan=plan
        )
        expected = backend._launch_piper_attention(prepared).clone()
        windows = (
            (0, block_m + 13, "bhsd"),
            (block_m, 5, "bshd"),
            (block_m, block_m + 17, "pingpong"),
            (block_m, block_m + 17, "compact"),
        )
        for start, rows, layout in windows:
            output, backing, guards = _guarded_output(rows, head_dim, layout)
            actual = backend._launch_piper_attention_into(
                prepared.context, prepared.query, output, query_start=start, query_rows=rows
            )
            assert actual is output
            torch.testing.assert_close(actual, expected[:, :, start : start + rows], atol=0, rtol=0)
            assert torch.all(backing[guards] == 7)

        # Each independent chunk retains complete Q scale groups, except the
        # original sequence tail, so it has the same quantization as full Q.
        for start in range(0, query.shape[2], block_m):
            rows = min(block_m, query.shape[2] - start)
            query_chunk = backend._prepare_piper_query(
                query[:, :, start : start + rows],
                head_dim**-0.5,
                execution_plan=plan,
                global_row_offset=start,
            )
            output, backing, guards = _guarded_output(rows, head_dim, "pingpong")
            actual = backend._launch_piper_attention_into(prepared.context, query_chunk, output)
            torch.testing.assert_close(actual, expected[:, :, start : start + rows], atol=0, rtol=0)
            assert torch.all(backing[guards] == 7)

        query_suffix = backend._prepare_piper_query(
            query[:, :, block_m:],
            head_dim**-0.5,
            execution_plan=plan,
            global_row_offset=block_m,
        )
        output, backing, guards = _guarded_output(17, head_dim, "bshd")
        actual = backend._launch_piper_attention_into(
            prepared.context, query_suffix, output, query_start=block_m
        )
        torch.testing.assert_close(actual, expected[:, :, 2 * block_m :], atol=0, rtol=0)
        assert torch.all(backing[guards] == 7)


@pytest.mark.parametrize(
    ("head_dim", "causal", "descriptors"),
    [(64, True, False), (64, False, True), (64, True, True), (128, False, True)],
)
def test_window_graph_replay_uses_live_query_storage(head_dim, causal, descriptors):
    query, key, value, plan = _operands(head_dim, causal, descriptors)
    context = backend._prepare_piper_context(key, value, is_causal=causal, execution_plan=plan)
    query_suffix = backend._prepare_piper_query(
        query[:, :, plan.block_m :],
        head_dim**-0.5,
        execution_plan=plan,
        global_row_offset=plan.block_m,
    )
    output, backing, guards = _guarded_output(17, head_dim, "pingpong")

    def launch():
        return backend._launch_piper_attention_into(
            context, query_suffix, output, query_start=plan.block_m
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    query_suffix.data.zero_()
    expected = launch().clone()
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert torch.all(backing[guards] == 7)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="causal strided schedule is calibrated on SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("batch", "heads", "kv_heads"), [(1, 6, 2), (2, 4, 1), (1, 16, 16)])
def test_grouped_causal_windows_cover_partial_groups_and_replay_live_queries(
    dtype, batch, heads, kv_heads
):
    torch.manual_seed(991)
    sequence, head_dim = 4097, 64
    query = torch.randn(batch, heads, sequence, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(batch, kv_heads, sequence, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    plan = backend._default_piper_attention_execution_plan(query, True)
    assert plan.strided_output_query_group == 8
    assert plan.ragged_strided_output_maxnreg == 168
    prepared = backend._prepare_piper_attention(
        query, key, value, head_dim**-0.5, True, execution_plan=plan
    )
    expected = backend._launch_piper_attention(prepared).clone()

    # Include a nonzero causal origin, an identity grouping, a partial group,
    # complete groups, and the actual final query tail. Batch storage is padded.
    # Independently quantized interior windows retain complete Q32 scale groups.
    for start, rows in ((128, 416), (64, 672), (512, 1024), (3072, 1025)):
        local_query = backend._prepare_piper_query(
            query[:, :, start : start + rows],
            head_dim**-0.5,
            execution_plan=plan,
            global_row_offset=start,
        )
        storage = torch.full((batch, rows + 8, heads, head_dim), 7, device="cuda", dtype=dtype)
        output = storage[:, 4 : rows + 4].transpose(1, 2)
        backend._launch_piper_attention_into(prepared.context, local_query, output)
        torch.testing.assert_close(output, expected[:, :, start : start + rows], atol=0, rtol=0)
        assert torch.all(storage[:, :4] == 7)
        assert torch.all(storage[:, rows + 4 :] == 7)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend._launch_piper_attention_into(prepared.context, local_query, output)
    local_query.data.zero_()
    reference = torch.empty_like(output, memory_format=torch.contiguous_format)
    backend._launch_piper_attention_into(prepared.context, local_query, reference)
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    assert torch.all(storage[:, :4] == 7)
    assert torch.all(storage[:, rows + 4 :] == 7)
