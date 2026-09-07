"""Integer accumulation and final-rounding checks for the SM120 sparse kernel."""

import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.mixed_int8 import install_uint8_int8_dot_hook
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.sparse_piper.layout import QUERY_SCALE_ROWS
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._nvidia.gluon import (
    _launch_sparse_piper_attention,
    _piper_pv_pair,
)
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
)
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MINMAX_ROUTING,
)
from piper_kernels.attention.sparse_piper_attention.triton import _prepare_sparse_piper_attention

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
    probability_0_ptr, probability_1_ptr, value_0_ptr, value_1_ptr, old_weight_ptr, output_ptr
):
    blocked: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]
    )
    probability_layout: gl.constexpr = gl.DotOperandLayout(0, mma, k_width=4)
    value_layout: gl.constexpr = gl.DotOperandLayout(1, mma, k_width=4)
    shared_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([128, 64], gl.int8)
    m = gl.arange(0, 64, gl.SliceLayout(1, blocked))
    k = gl.arange(0, 64, gl.SliceLayout(0, blocked))
    d = gl.arange(0, 128, gl.SliceLayout(1, blocked))
    probability_offsets = m[:, None] * 64 + k[None, :]
    probability_0 = gl.load(probability_0_ptr + probability_offsets)
    probability_1 = gl.load(probability_1_ptr + probability_offsets)
    probabilities = gl.reshape(
        gl.permute(gl.join(probability_0, probability_1), [0, 2, 1]), [64, 128]
    )
    probabilities = gl.convert_layout(probabilities, probability_layout)
    value_offsets = d[:, None] + k[None, :] * 128
    values = gl.allocate_shared_memory(gl.int8, [2, 128, 64], shared_layout)
    values.index(0).store(gl.load(value_0_ptr + value_offsets))
    values.index(1).store(gl.load(value_1_ptr + value_offsets))
    gl.barrier()
    output_m = gl.arange(0, 64, gl.SliceLayout(1, mma))
    output = _piper_pv_pair(
        probabilities,
        values,
        gl.full([64, 128], 0.25, gl.float32, mma),
        gl.load(old_weight_ptr + output_m),
        gl.full([64], 1.0 / 1024, gl.float32, gl.SliceLayout(1, mma)),
        mma,
        value_layout,
    )
    output_d = gl.arange(0, 128, gl.SliceLayout(0, mma))
    gl.store(output_ptr + output_m[:, None] * 128 + output_d[None, :], output)


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
def test_paired_pv_accumulation_matches_int64_products(value_extremes, weight_pattern):
    generator = torch.Generator().manual_seed(631)
    probabilities = [
        torch.randint(0, 256, (64, 64), dtype=torch.uint8, generator=generator) for _ in range(2)
    ]
    values = [
        torch.randint(-128, 128, (64, 128), dtype=torch.int8, generator=generator) for _ in range(2)
    ]
    if value_extremes is not None:
        for probability in probabilities:
            probability.fill_(255)
        for value, extreme in zip(values, value_extremes, strict=True):
            value.fill_(extreme)
    old_weight = torch.full((64,), 0.5)
    if weight_pattern == "unchanged":
        old_weight.fill_(1)
    elif weight_pattern == "mixed":
        old_weight[::3] = 1
    expected = (
        probabilities[0].long() @ values[0].long() + probabilities[1].long() @ values[1].long()
    ).float() / 1024 + 0.25 * old_weight[:, None]
    operands = [tensor.cuda() for tensor in (*probabilities, *values, old_weight)]
    actual = torch.empty((64, 128), dtype=torch.float32, device="cuda")
    install_uint8_int8_dot_hook()
    _paired_pv_kernel[(1,)](*operands, actual, num_warps=4)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("sequence_length", [129, 193, 257])
@pytest.mark.parametrize("tail_position", [0, 1, -1])
def test_routed_ragged_tile_ignores_padding_at_every_route_position(sequence_length, tail_position):
    tiles = (sequence_length + 63) // 64
    storage_length = tiles * 64
    query = torch.zeros((1, 1, storage_length, 128), device="cuda", dtype=torch.int8)
    key = torch.zeros_like(query)
    value = torch.full((1, 1, 128, storage_length), 64, device="cuda", dtype=torch.int8)
    value[..., sequence_length:] = 0
    context = _prepare_sparse_piper_context_from_quantized(
        key,
        torch.ones((1, 1, tiles), device="cuda"),
        value,
        torch.ones((1, 1, tiles, 1), device="cuda"),
        torch.zeros((1, 1, 128), device="cuda"),
        torch.tensor([tiles], device="cuda", dtype=torch.int32),
        torch.tensor([0, tiles], device="cuda", dtype=torch.int32),
        sparse_key_blocks=tiles,
        routes_per_query=tiles,
        logical_sequence_length=sequence_length,
    )
    order = list(range(tiles - 1))
    order.insert(tiles - 1 if tail_position == -1 else tail_position, tiles - 1)
    routes = (
        torch.tensor(order, device="cuda", dtype=torch.uint16)[None, None, :]
        .expand(1, tiles, tiles)
        .contiguous()
    )
    query_state = _prepare_sparse_piper_query_from_quantized(
        query,
        torch.ones((1, 1, storage_length // QUERY_SCALE_ROWS), device="cuda"),
        routes,
        context,
    )
    prepared = _PreparedSparsePiperAttention(context, query_state)
    expected = torch.empty((1, 1, sequence_length, 128), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    _launch_sparse_piper_attention(prepared, expected)
    # Padding is not part of the attention sequence, even when its physical
    # block occurs in the middle of a caller-supplied sparse route.
    value[..., sequence_length:] = -128
    _launch_sparse_piper_attention(prepared, actual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("sequence_length", [128, 193, 320])
def test_coarse_epilogue_rounds_only_after_combining_fine_and_residual(sequence_length):
    query = torch.zeros((1, 1, sequence_length, 128), dtype=torch.bfloat16, device="cuda")
    key = torch.zeros_like(query)
    value = torch.full_like(query, 1.0 / 256)
    # In the first half of D, averaging adjacent BF16 values produces a fine
    # result between BF16 values. In the second half, the coarse term supplies
    # that extra precision. Prematurely rounding either term loses the update.
    value[..., :64] = 1.0
    value[:, :, 1::2, :64] += 1.0 / 128
    blocks = sequence_length // 64
    layout = _resolve_route_layout(_normalize_head_keep_ratios((1.0,)), blocks, query.device)
    routes = packed_routes_from_sequences(query, key[:, :, : blocks * 64], layout, _MINMAX_ROUTING)
    prepared = _prepare_sparse_piper_attention(
        query,
        routes.indices,
        routes.head_keep_blocks,
        128**-0.5,
        sparse_key_blocks=blocks,
        route_head_offsets=routes.route_head_offsets,
        combined_key=key,
        combined_value=value,
    )
    coarse = torch.full(
        (1, 1, (sequence_length + 63) // 64, 128),
        1.0 / 256,
        device="cuda",
        dtype=torch.float32,
    )
    coarse[..., 64:] += 1.0
    gate = torch.ones((1, sequence_length, 1, 128), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty_like(query)
    _launch_sparse_piper_attention(prepared, actual, coarse_output=coarse, coarse_gate=gate)
    torch.testing.assert_close(actual, torch.full_like(actual, 1.0 + 1.0 / 128), atol=0, rtol=0)
