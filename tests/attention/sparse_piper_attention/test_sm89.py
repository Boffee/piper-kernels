"""Numerical, schedule, range, and addressing checks for the SM89 ``cp.async`` kernel."""

import math
from dataclasses import replace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention import _backend
from piper_kernels.attention.sparse_piper_attention._budget import (
    _normalize_head_keep_ratios,
    _resolve_route_layout,
)
from piper_kernels.attention.sparse_piper_attention._nvidia import policy
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
)
from piper_kernels.attention.sparse_piper_attention._routing import packed_routes_from_sequences
from piper_kernels.attention.sparse_piper_attention._routing_modes import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not policy.uses_async_copies(AcceleratorTarget.from_device(torch.device("cuda"))),
        reason="requires exact NVIDIA SM89",
    ),
]

if _backend.nvidia_async_copy_gluon is not None:
    from piper_kernels.attention.sparse_piper_attention._nvidia import gluon_async_copy as native
    from piper_kernels.attention.sparse_piper_attention._nvidia import gluon_tma as sm120
    from piper_kernels.attention.sparse_piper_attention.triton import (
        _prepare_sparse_piper_operands,
    )


def _prepare(
    head_dim,
    sequence,
    ratios,
    *,
    kv_heads=None,
    routing=_MINMAX_ROUTING,
    dtype=torch.bfloat16,
    block_lengths=None,
    sparse_query_blocks=None,
    skip_dense_routing=False,
    seed=0,
):
    heads = len(ratios)
    kv_heads = heads if kv_heads is None else kv_heads
    generator = torch.Generator(device="cuda").manual_seed(8900 + seed)
    query = torch.randn(
        1, sequence, heads, head_dim, device="cuda", dtype=dtype, generator=generator
    )
    key, value = (
        torch.randn(
            1, sequence, kv_heads, head_dim, device="cuda", dtype=dtype, generator=generator
        )
        for _ in range(2)
    )
    blocks = sequence // 64
    head_major = [tensor.transpose(1, 2) for tensor in (query, key, value)]
    layout = _resolve_route_layout(_normalize_head_keep_ratios(ratios), blocks, query.device)
    routes = packed_routes_from_sequences(
        head_major[0],
        head_major[1][:, :, : blocks * 64],
        layout,
        routing,
        block_lengths,
        skip_dense_routing=skip_dense_routing,
    )
    prepared = _prepare_sparse_piper_operands(
        head_major[0],
        head_dim**-0.5,
        sparse_key_blocks=blocks,
        combined_key=head_major[1],
        combined_value=head_major[2],
        block_lengths=block_lengths,
        sparse_query_blocks=sparse_query_blocks,
    ).with_routes(routes.indices, routes.head_keep_blocks, routes.route_head_offsets)
    return (query, key, value), prepared


def _launch(prepared, output, uncapped=False, **kwargs):
    """Launch the SM89 kernel, optionally without its D64 register cap."""
    if not uncapped:
        native._launch_sparse_piper_attention(prepared, output, **kwargs)
        return
    original = policy.sm89_max_registers
    policy.sm89_max_registers = lambda head_dim: None
    try:
        native._launch_sparse_piper_attention(prepared, output, **kwargs)
    finally:
        policy.sm89_max_registers = original


def _fp64_query_block(prepared, head, query_block):
    """Independent FP64 paired attention on the kernel's own quantized operands."""
    query, context = prepared.query, prepared.context
    device = query.data.device
    kv_head = head // (query.data.shape[1] // context.key.shape[1])
    global_block = query.global_block_offset + query_block
    stored_blocks = context.key.shape[2] // 64
    use_routes = context.sparse_query_blocks is None or global_block < context.sparse_query_blocks
    if use_routes and context.routes_per_query != 0:
        start, stop = context.route_head_offsets[head : head + 2].tolist()
        tiles = query.routes[0, query_block, start:stop].long()
    else:
        tiles = torch.arange(context.sparse_key_blocks, device=device)
    tiles = torch.cat(
        (tiles, torch.arange(context.sparse_key_blocks, stored_blocks, device=device))
    )
    tile_count = tiles.numel()
    if tile_count % 2:
        tiles = torch.cat((tiles, tiles[-1:]))
    offsets = torch.arange(64, device=device)
    indices = (tiles[:, None] * 64 + offsets).flatten()
    if context.block_lengths is None:
        valid = indices < context.logical_sequence_length
        rows = min(64, context.logical_sequence_length - global_block * 64)
    else:
        valid = (offsets < context.block_lengths[tiles, None]).flatten()
        rows = 64
    valid &= torch.arange(indices.numel(), device=device) < tile_count * 64
    pairs = tiles.numel() // 2
    q = query.data[0, head, query_block * 64 : query_block * 64 + rows].double()
    q_scale = query.scale[0, head, query_block * 2 : query_block * 2 + 2].repeat_interleave(32)
    k = context.key[0, kv_head].index_select(0, indices).double()
    k_scale = context.key_scale[0, kv_head, tiles].repeat_interleave(64).double()
    multiplier = context.value_scale_multiplier[0, kv_head, tiles, 0].repeat_interleave(64)
    scores = (q @ k.T) * q_scale[:rows, None].double() * k_scale[None, :]
    scores = scores.masked_fill(~valid[None, :], -torch.inf)
    scores = scores.reshape(rows, pairs, 128).transpose(0, 1)
    multipliers = multiplier.double().reshape(pairs, 1, 128)
    # The kernel shifts by a conservative log2(multiplier / 255) bound read from
    # the FP32 bits; mirroring it keeps both on the same UINT8 code grid.
    log_bound = multiplier.view(torch.int32).double() / 2**23 - (127 + math.log2(255) - 0.086085)
    pair_max = (scores + log_bound.reshape(pairs, 1, 128)).amax(dim=-1)
    probabilities = torch.exp2(scores - pair_max[:, :, None])
    codes = (probabilities * multipliers + 0.5).floor().clamp(0, 255)
    values = context.value[0, kv_head].index_select(1, indices).T.reshape(pairs, 128, -1)
    products = torch.bmm(codes, values.double())
    weights = torch.exp2(pair_max - pair_max.amax(dim=0))
    numerator = (products * weights[:, :, None]).sum(dim=0)
    denominator = (probabilities.sum(dim=-1) * weights).sum(dim=0)
    output = numerator / (denominator.clamp_min(1e-30)[:, None] * 255)
    return output + context.value_mean[0, kv_head].double()


def _assert_matches_fp64(prepared, output, coarse=None, limit=0.005, blocks=None):
    """Check every (or each listed) query block of every head against FP64."""
    heads, sequence = output.shape[1], output.shape[2]
    lengths = prepared.context.block_lengths
    for head in range(heads):
        for block in range((sequence + 63) // 64) if blocks is None else blocks:
            expected = _fp64_query_block(prepared, head, block)
            rows = expected.shape[0]
            if coarse is not None:
                gate, coarse_output = coarse
                expected = expected + (
                    gate[0, block * 64 : block * 64 + rows, head].double()
                    * coarse_output[0, head, block].double()
                )
            actual = output[0, head, block * 64 : block * 64 + rows]
            if lengths is not None:
                valid = int(lengths[block])
                actual, expected = actual[:valid], expected[:valid]
            assert torch.isfinite(actual).all()
            error = (actual.double() - expected).norm() / expected.norm().clamp_min(1e-30)
            assert error < limit, (head, block, error.item())


_FEATURE_CASES = {
    "routed": {"sequence": 320, "ratios": (0.5, 0.5)},
    "mixed_ratios_gqa": {"sequence": 1061, "ratios": (0.25, 1.0, 0.5, 0.75), "kv_heads": 2},
    "multi_query": {"sequence": 193, "ratios": (0.5, 0.5, 1.0), "kv_heads": 1},
    "ragged_tail": {"sequence": 193, "ratios": (0.5, 1.0)},
    "single_block": {"sequence": 64, "ratios": (1.0, 1.0)},
    "partial_single_block": {"sequence": 100, "ratios": (1.0,)},
    "full_keep_routes": {"sequence": 448, "ratios": (1.0, 1.0)},
    "padded_blocks": {"sequence": 448, "ratios": (0.5, 1.0), "padded": True},
    "dense_suffix": {"sequence": 705, "ratios": (0.25, 0.5), "sparse_query_blocks": 5},
    "padded_dense_suffix": {
        "sequence": 512,
        "ratios": (0.5, 0.5),
        "padded": True,
        "sparse_query_blocks": 3,
    },
}


@pytest.mark.parametrize("case", sorted(_FEATURE_CASES))
@pytest.mark.parametrize("routing", [_MEAN_ROUTING, _MINMAX_ROUTING], ids=["mean", "minmax"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_matches_fp64_attention_on_prepared_operands(case, routing, head_dim):
    options = dict(_FEATURE_CASES[case])
    sequence = options.pop("sequence")
    ratios = options.pop("ratios")
    block_lengths = None
    if options.pop("padded", False):
        lengths = torch.full((sequence // 64,), 64, dtype=torch.int32)
        lengths[1::3] = 17
        lengths[2::5] = 1
        block_lengths = lengths.cuda()
    _operands, prepared = _prepare(
        head_dim,
        sequence,
        ratios,
        routing=routing,
        block_lengths=block_lengths,
        seed=len(case),
        **options,
    )
    output = torch.full(
        (1, len(ratios), sequence, head_dim), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    _launch(prepared, output)
    _assert_matches_fp64(prepared, output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_supported_dtypes_match_fp64_attention(dtype, head_dim):
    _operands, prepared = _prepare(head_dim, 257, (0.5, 1.0), dtype=dtype, seed=3)
    output = torch.empty(1, 2, 257, head_dim, device="cuda", dtype=dtype)
    _launch(prepared, output)
    _assert_matches_fp64(prepared, output)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_coarse_residual_is_added_before_rounding(head_dim):
    sequence = 449
    _operands, prepared = _prepare(head_dim, sequence, (0.5, 1.0), seed=5)
    coarse_output = torch.randn(1, 2, 8, head_dim, device="cuda")
    gate = torch.randn(1, sequence, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, 2, sequence, head_dim, device="cuda", dtype=torch.bfloat16)
    _launch(prepared, output, coarse_output=coarse_output, coarse_gate=gate)
    _assert_matches_fp64(prepared, output, coarse=(gate, coarse_output))


@pytest.mark.parametrize(
    ("head_dim", "ratios", "skip_dense_routing"),
    [(64, (0.25, 1.0), False), (64, (1.0, 1.0), True), (128, (0.25, 1.0), False)],
)
def test_long_sequences_match_fp64_samples(head_dim, ratios, skip_dense_routing):
    """128k rows exercise 2,048 K64 tiles per head, long pair loops, and a ragged tail.

    Two heads keep each operand near 64 MB. First, middle, and last query blocks of
    every head are checked, since a full FP64 sweep would take minutes.
    """
    sequence = 131072 + 17
    _operands, prepared = _prepare(
        head_dim, sequence, ratios, skip_dense_routing=skip_dense_routing, seed=17
    )
    output = torch.full(
        (1, len(ratios), sequence, head_dim), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    _launch(prepared, output)
    last = (sequence + 63) // 64 - 1
    _assert_matches_fp64(prepared, output, blocks=(0, last // 2, last))


@pytest.mark.parametrize("sequence", [64, 129, 1797, 8193])
@pytest.mark.parametrize(
    ("head_dim", "ratios", "skip_dense_routing"),
    [
        (64, (0.25, 1.0, 0.5), False),
        (64, (1.0, 1.0), True),
    ],
)
def test_register_cap_changes_no_output_bits(sequence, head_dim, ratios, skip_dense_routing):
    """The D64 register cap changes register allocation, never the arithmetic."""
    _operands, prepared = _prepare(
        head_dim, sequence, ratios, skip_dense_routing=skip_dense_routing, seed=sequence
    )
    if skip_dense_routing:
        assert prepared.context.routes_per_query == 0
    shape = (1, len(ratios), sequence, head_dim)
    expected = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    _launch(prepared, expected)
    _assert_matches_fp64(prepared, expected)
    actual = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    _launch(prepared, actual, uncapped=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("coarse", [False, True])
def test_dense_suffix_and_coarse_residual_match_fp64(coarse):
    sequence = 8193
    _operands, prepared = _prepare(64, sequence, (0.25, 0.5, 1.0), sparse_query_blocks=80)
    kwargs = {}
    if coarse:
        kwargs = {
            "coarse_output": torch.randn(1, 3, 129, 64, device="cuda"),
            "coarse_gate": torch.randn(1, sequence, 3, 64, device="cuda", dtype=torch.bfloat16),
        }
    expected = torch.empty(1, 3, sequence, 64, device="cuda", dtype=torch.bfloat16)
    _launch(prepared, expected, **kwargs)
    coarse_terms = (kwargs["coarse_gate"], kwargs["coarse_output"]) if coarse else None
    _assert_matches_fp64(prepared, expected, coarse=coarse_terms)
    actual = torch.empty_like(expected)
    _launch(prepared, actual, uncapped=True, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("skip_dense_routing", [False, True])
def test_query_ranges_local_storage_and_graph_replay(skip_dense_routing):
    sequence = 4161
    ratios = (1.0, 1.0) if skip_dense_routing else (0.5, 1.0)
    _operands, prepared = _prepare(64, sequence, ratios, skip_dense_routing=skip_dense_routing)
    expected = torch.empty(1, 2, sequence, 64, device="cuda", dtype=torch.bfloat16)
    _launch(prepared, expected, uncapped=True)

    rows = sequence - 64 * 3
    guarded = torch.full((1, 2, rows + 2, 64), 123.0, device="cuda", dtype=torch.bfloat16)
    _launch(prepared, guarded[:, :, 1:-1], query_block_offset=3)
    torch.testing.assert_close(guarded[:, :, 1:-1], expected[:, :, 192:], atol=0, rtol=0)
    assert torch.all(guarded[:, :, (0, -1)] == 123.0)

    local = replace(
        prepared,
        query=_prepare_sparse_piper_query_from_quantized(
            prepared.query.data[:, :, 192:].contiguous(),
            prepared.query.scale[:, :, 6:].contiguous(),
            prepared.query.routes[:, 3:].contiguous(),
            prepared.context,
            global_block_offset=3,
        ),
    )
    guarded.fill_(123.0)
    _launch(local, guarded[:, :, 1:-1])
    torch.testing.assert_close(guarded[:, :, 1:-1], expected[:, :, 192:], atol=0, rtol=0)
    assert torch.all(guarded[:, :, (0, -1)] == 123.0)

    replayed = torch.empty_like(expected)
    _launch(prepared, replayed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(prepared, replayed)
    replayed.zero_()
    graph.replay()
    torch.testing.assert_close(replayed, expected, atol=0, rtol=0)


def _jit_arguments(module, prepared, output, **kwargs):
    """Return a launcher's grid, kernel arguments, and options without launching."""
    captured = {}

    class _Capture:
        def __getitem__(self, grid):
            def run(*args, **options):
                captured.update(grid=grid, args=args, options=options)

            return run

    kernel = module._sparse_piper_attention_kernel
    module._sparse_piper_attention_kernel = _Capture()
    try:
        module._launch_sparse_piper_attention(prepared, output, **kwargs)
    finally:
        module._sparse_piper_attention_kernel = kernel
    return captured


@pytest.mark.parametrize(
    ("head_dim", "sequence", "heads", "ratio", "query_block_offset", "coarse"),
    [
        (64, 1000, 8, 0.25, 0, False),
        (64, 4096, 8, 0.25, 0, False),
        (64, 8193, 2, 0.25, 0, False),
        (64, 8193, 2, 0.25, 70, False),
        (64, 8193, 2, 0.25, 0, True),
        (64, 64, 1, 1.0, 0, False),
        (64, 1000, 8, 1.0, 0, False),
        (64, 4160, 8, 1.0, 0, False),
        (64, 16384, 1, 1.0, 3, False),
        (64, 32768, 2, 1.0, 0, False),
        (64, 32769, 2, 1.0, 0, False),
        (64, 4096, 2, 1.0, 0, True),
        (128, 4096, 8, 0.25, 0, False),
        (128, 16384, 4, 1.0, 0, False),
    ],
)
def test_launch_specializes_no_more_finely_than_sm120(
    head_dim, sequence, heads, ratio, query_block_offset, coarse
):
    """Two calls that share an SM120 specialization must share an SM89 one.

    SM120 passes TMA descriptors and SM89 passes 16-byte-aligned pointers, one
    specialization each. SM89 has no query-tile or warp-count arguments because it
    always runs Q64 with four warps and caps D64 registers; it masks its output
    tail exactly when the final Q64 tile is ragged. Every other argument it shares
    with SM120 must be SM120's.
    """
    full = ratio == 1.0 and head_dim == 64
    _operands, prepared = _prepare(head_dim, sequence, (ratio,) * heads, skip_dense_routing=full)
    rows = sequence - 64 * query_block_offset
    output = torch.empty(1, heads, rows, head_dim, device="cuda", dtype=torch.bfloat16)
    kwargs = {"query_block_offset": query_block_offset}
    if coarse:
        kwargs["coarse_output"] = torch.randn(
            1, heads, (sequence + 63) // 64, head_dim, device="cuda"
        )
        kwargs["coarse_gate"] = torch.randn(
            1, rows, heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
    sm89 = _jit_arguments(native, prepared, output, **kwargs)
    reference = _jit_arguments(sm120, prepared, output, **kwargs)
    sm89_names = native._sparse_piper_attention_kernel.arg_names
    sm120_arguments = dict(
        zip(sm120._sparse_piper_attention_kernel.arg_names, reference["args"], strict=True)
    )
    assert set(sm120_arguments) - set(sm89_names) == {
        "query_desc",
        "key_desc",
        "value_desc",
        "block_m",
        "mma_warps",
    }
    assert sm89["grid"] == ((rows + 63) // 64, *reference["grid"][1:])
    options = {**reference["options"], "num_warps": 4}
    if head_dim == 64:
        options["maxnreg"] = 168
    assert sm89["options"] == options
    sm89_arguments = dict(zip(sm89_names, sm89["args"], strict=True))
    assert sm89_arguments["mask_output_tail"] is (rows % 64 != 0)
    for name, actual in sm89_arguments.items():
        if name in ("query_ptr", "key_ptr", "value_ptr", "mask_output_tail"):
            continue
        expected = sm120_arguments[name]
        if isinstance(expected, torch.Tensor):
            assert actual is expected, name
        else:
            assert type(actual) is type(expected), name
            assert actual == expected, name


@pytest.mark.usefixtures("large_device_memory")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_last_head_addresses_beyond_signed_32_bit_offsets(head_dim):
    """Pointer-based copies must address K/V storage past 2**31 bytes."""
    # Many short heads keep the allocation near 2**31 bytes per operand and the
    # tile count within UINT16 routes, while the last head starts past 2**31.
    kv_heads = 16
    storage = ((1 << 31) // (head_dim * (kv_heads - 1)) + 64 + 63) // 64 * 64
    required = 2 * kv_heads * storage * head_dim + 512 * 1024**2
    if torch.cuda.mem_get_info()[0] < required:
        pytest.skip("not enough free device memory for the large-offset regression")
    tiles = storage // 64
    generator = torch.Generator(device="cuda").manual_seed(31)
    key = torch.empty((1, kv_heads, storage, head_dim), device="cuda", dtype=torch.int8)
    value = torch.empty((1, kv_heads, head_dim, storage), device="cuda", dtype=torch.int8)
    # Only the final K64 tile is ever read; place it beyond the 32-bit limit.
    last = slice(storage - 64, storage)
    key[0, -1, last] = torch.randint(
        -127, 128, (64, head_dim), device="cuda", dtype=torch.int8, generator=generator
    )
    value[0, -1, :, last] = torch.randint(
        -127, 128, (head_dim, 64), device="cuda", dtype=torch.int8, generator=generator
    )
    assert (kv_heads - 1) * storage * head_dim + (storage - 64) * head_dim > 1 << 31
    context = _prepare_sparse_piper_context_from_quantized(
        key,
        torch.full((1, kv_heads, tiles), 0.01, device="cuda"),
        value,
        torch.full((1, kv_heads, tiles, 1), 255.0, device="cuda"),
        torch.zeros((1, kv_heads, head_dim), device="cuda"),
        torch.ones(kv_heads, device="cuda", dtype=torch.int32),
        torch.arange(kv_heads + 1, device="cuda", dtype=torch.int32),
        sparse_key_blocks=tiles,
        routes_per_query=kv_heads,
        logical_sequence_length=storage,
    )
    query = _prepare_sparse_piper_query_from_quantized(
        torch.randint(
            -127,
            128,
            (1, kv_heads, 64, head_dim),
            device="cuda",
            dtype=torch.int8,
            generator=generator,
        ),
        torch.full((1, kv_heads, 2), 0.01, device="cuda"),
        torch.full((1, 1, kv_heads), tiles - 1, device="cuda", dtype=torch.uint16),
        context,
        global_block_offset=tiles - 1,
    )
    prepared = _PreparedSparsePiperAttention(context, query)
    output = torch.empty(1, kv_heads, 64, head_dim, device="cuda", dtype=torch.bfloat16)
    _launch(prepared, output)
    expected = _fp64_query_block(prepared, kv_heads - 1, 0)
    actual = output[0, -1]
    error = (actual.double() - expected).norm() / expected.norm()
    assert error < 0.005
