"""Integer accumulation checks for the SM120 sparse kernel."""

import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._nvidia.gluon import (
    _piper_pv_pair,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not AcceleratorTarget.from_device(torch.device("cuda")).is_cuda_capability(12, 0),
        reason="requires exact NVIDIA SM120",
    ),
]


@gluon.jit
def _paired_pv_kernel(
    probability_0_ptr,
    probability_1_ptr,
    value_0_ptr,
    value_1_ptr,
    old_weight_ptr,
    output_ptr,
    head_dim: gl.constexpr,
    query_rows: gl.constexpr,
    mma_warps: gl.constexpr,
):
    blocked: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [mma_warps, 1], [1, 0])
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[mma_warps, 1], instr_shape=[16, 8]
    )
    probability_layout: gl.constexpr = gl.DotOperandLayout(0, mma, k_width=4)
    value_layout: gl.constexpr = gl.DotOperandLayout(1, mma, k_width=4)
    shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([head_dim, 64], gl.int8)
    m = gl.arange(0, query_rows, gl.SliceLayout(1, blocked))
    k = gl.arange(0, 64, gl.SliceLayout(0, blocked))
    d = gl.arange(0, head_dim, gl.SliceLayout(1, blocked))
    probability_offsets = m[:, None] * 64 + k[None, :]
    probability_0 = gl.load(probability_0_ptr + probability_offsets)
    probability_1 = gl.load(probability_1_ptr + probability_offsets)
    probabilities = gl.reshape(
        gl.permute(gl.join(probability_0, probability_1), [0, 2, 1]), [query_rows, 128]
    )
    probabilities = gl.convert_layout(probabilities, probability_layout)
    value_offsets = d[:, None] + k[None, :] * head_dim
    values = gl.allocate_shared_memory(gl.int8, [2, head_dim, 64], shared_layout)
    values.index(0).store(gl.load(value_0_ptr + value_offsets))
    values.index(1).store(gl.load(value_1_ptr + value_offsets))
    gl.barrier()
    output_m = gl.arange(0, query_rows, gl.SliceLayout(1, mma))
    output = _piper_pv_pair(
        probabilities,
        values,
        gl.full([query_rows, head_dim], 0.25, gl.float32, mma),
        gl.load(old_weight_ptr + output_m),
        gl.full([query_rows], 1.0 / 1024, gl.float32, gl.SliceLayout(1, mma)),
        mma,
        value_layout,
    )
    output_d = gl.arange(0, head_dim, gl.SliceLayout(0, mma))
    gl.store(output_ptr + output_m[:, None] * head_dim + output_d[None, :], output)


@pytest.mark.parametrize(
    "value_extremes",
    [
        pytest.param(None, id="random"),
        pytest.param((127, 127), id="positive_limit"),
        pytest.param((-128, -128), id="negative_limit"),
        pytest.param((127, -128), id="opposing_limits"),
    ],
)
@pytest.mark.parametrize("weight_pattern", ["changed", "unchanged", "mixed"])
@pytest.mark.parametrize(
    ("head_dim", "query_rows", "warps"),
    [(64, 64, 4), (128, 64, 4), (64, 64, 2), (64, 128, 4)],
)
def test_paired_pv_accumulation_matches_int64_products(
    value_extremes, weight_pattern, head_dim, query_rows, warps
):
    generator = torch.Generator().manual_seed(631)
    probabilities = [
        torch.randint(0, 256, (query_rows, 64), dtype=torch.uint8, generator=generator)
        for _ in range(2)
    ]
    values = [
        torch.randint(-128, 128, (64, head_dim), dtype=torch.int8, generator=generator)
        for _ in range(2)
    ]
    if value_extremes is not None:
        for probability in probabilities:
            probability.fill_(255)
        for value, extreme in zip(values, value_extremes, strict=True):
            value.fill_(extreme)
    old_weight = torch.full((query_rows,), 0.5)
    if weight_pattern == "unchanged":
        old_weight.fill_(1)
    elif weight_pattern == "mixed":
        old_weight[::3] = 1
    expected = (
        probabilities[0].long() @ values[0].long() + probabilities[1].long() @ values[1].long()
    ).float() / 1024 + 0.25 * old_weight[:, None]
    operands = [tensor.cuda() for tensor in (*probabilities, *values, old_weight)]
    actual = torch.empty((query_rows, head_dim), dtype=torch.float32, device="cuda")
    install_uint8_int8_dot_hook()
    _paired_pv_kernel[(1,)](*operands, actual, head_dim, query_rows, warps, num_warps=warps)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
