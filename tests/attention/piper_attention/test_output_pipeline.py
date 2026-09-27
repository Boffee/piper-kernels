"""Dense query windows feed the shared bounded attention-output pipeline."""

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._nvidia import triton as backend
from piper_kernels.fusions.attention._output import run_chunked_attention_output

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not AcceleratorTarget.from_device(torch.device("cuda")).supports_uint8_int8_mma,
        reason="requires NVIDIA mixed-sign MMA",
    ),
]


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("prepare_chunks", [False, True])
def test_dense_windows_feed_reusable_output_pipeline(is_causal, prepare_chunks):
    torch.manual_seed(427)
    batch, heads, sequence, head_dim = 2, 4, 385, 128
    query = torch.randn(
        batch, sequence, heads, head_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    key = torch.randn(batch, 2, sequence, head_dim, device=query.device, dtype=query.dtype)
    value = torch.randn_like(key)
    scale = head_dim**-0.5
    plan = backend._default_piper_attention_execution_plan(query, is_causal)
    prepared = backend._prepare_piper_attention(
        query, key, value, scale, is_causal, execution_plan=plan
    )
    expected = backend._launch_piper_attention(prepared).transpose(1, 2).flatten(2)

    def launch_chunk(buffer, start, rows, auxiliary):
        assert auxiliary is None
        if prepare_chunks:
            prepared_query = backend._prepare_piper_query(
                query[:, :, start : start + rows],
                scale,
                execution_plan=plan,
                global_row_offset=start,
            )
            query_start = 0
        else:
            prepared_query = prepared.query
            query_start = start
        backend._launch_piper_attention_into(
            prepared.context,
            prepared_query,
            buffer.transpose(1, 2),
            query_start=query_start,
            query_rows=rows,
        )

    def project_chunk(attention, output, start, rows):
        # Identity projection isolates storage layout and producer/consumer ordering.
        output[:, start : start + rows].copy_(attention.flatten(2))

    actual = run_chunked_attention_output(
        (batch, sequence, heads, head_dim),
        query.device,
        heads * head_dim,
        128,
        launch_chunk,
        project_chunk,
        (),
        output_dtype=query.dtype,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
