"""Exact RDNA4 fragment arithmetic and explicit packed-context lifetime."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._amd import gluon as backend
from piper_kernels.attention.sparse_piper_attention._amd._fragments import (
    MMA_LAYOUT,
    pack_probabilities,
    pv_pair,
    qk_pair,
    query_fragments,
    rescale_numerator,
)
from piper_kernels.attention.sparse_piper_attention._amd._packing import _pack_values
from piper_kernels.attention.sparse_piper_attention._amd.gluon import bind_context
from piper_kernels.attention.sparse_piper_attention._prepared import (
    _prepare_sparse_piper_context_from_quantized,
    _prepare_sparse_piper_query_from_quantized,
    _PreparedSparsePiperAttention,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not AcceleratorTarget.from_device(torch.device("cuda")).is_amd_hip
        or not AcceleratorTarget.from_device(torch.device("cuda")).is_architecture(
            "gfx1200", "gfx1201"
        ),
        reason="requires an RDNA4 ROCm GPU",
    ),
]


@gluon.jit
def _pack_probability_kernel(probability_ptr, output_ptr, columns: gl.constexpr):
    rows = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, MMA_LAYOUT)))
    cols = gl.arange(0, columns, gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT)))
    codes = gl.load(probability_ptr + rows[None, :, None] * columns + cols[None, None, :])
    words = pack_probabilities(codes.to(gl.float32))
    packed_rows = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, words.type.layout)))
    packed_cols = gl.arange(
        0, columns // 8, gl.SliceLayout(0, gl.SliceLayout(1, words.type.layout))
    )
    gl.store(
        output_ptr + packed_rows[None, :, None] * (columns // 8) + packed_cols[None, None, :], words
    )


@pytest.mark.parametrize("columns", [32, 64, 128])
def test_probability_packing_preserves_every_byte(columns):
    codes = torch.arange(64 * columns, device="cuda").reshape(64, columns).to(torch.uint8)
    output = torch.empty((64, columns // 8), dtype=torch.uint64, device="cuda")
    _pack_probability_kernel[(1,)](codes, output, columns, num_warps=4)
    torch.testing.assert_close(output.view(torch.uint8), codes, rtol=0, atol=0)


def test_probability_packing_rounds_fp32_ties_to_even_and_saturates():
    boundaries = torch.arange(255, dtype=torch.float32) + 0.5
    below = torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf))
    above = torch.nextafter(boundaries, torch.full_like(boundaries, torch.inf))
    limits = torch.tensor(
        [
            -torch.finfo(torch.float32).max,
            -256,
            -1,
            -0.0,
            0,
            255,
            255.5,
            256,
            torch.finfo(torch.float32).max,
        ],
        dtype=torch.float32,
    )
    values = torch.cat((below, boundaries, above, limits))
    # Exercise every boundary in every packed byte position. This checks the
    # AMD conversion instruction, not cross-backend bit-exact attention.
    values = values.repeat_interleave(8)
    codes = values.repeat((64 * 128 + values.numel() - 1) // values.numel())[: 64 * 128]
    codes = codes.reshape(64, 128)
    expected = codes.double().round().clamp(0, 255).to(torch.uint8)
    output = torch.empty((64, 16), dtype=torch.uint64, device="cuda")
    _pack_probability_kernel[(1,)](codes.cuda(), output, 128, num_warps=4)
    torch.testing.assert_close(output.view(torch.uint8).cpu(), expected, rtol=0, atol=0)


@gluon.jit
def _paired_pv_kernel(
    probability_ptr, value_ptr, numerator_ptr, weight_ptr, output_ptr, start_0, start_1
):
    block_m: gl.constexpr = 64
    rows = gl.arange(0, block_m, gl.SliceLayout(0, gl.SliceLayout(2, MMA_LAYOUT)))
    columns = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT)))
    offsets = rows[None, :, None] * 128 + columns[None, None, :]
    probability = pack_probabilities(gl.load(probability_ptr + offsets).to(gl.float32))
    block_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, MMA_LAYOUT))
    result = pv_pair(
        probability,
        value_ptr,
        gl.full([1], start_0 // 64, gl.int32, block_layout),
        gl.full([1], start_1 // 64, gl.int32, block_layout),
        gl.load(numerator_ptr + offsets),
        gl.load(weight_ptr + rows[None, :]),
        256,
    )
    gl.store(output_ptr + offsets, result)


@pytest.mark.parametrize("starts", [(0, 64), (192, 64), (128, 128)])
@pytest.mark.parametrize("recurrence", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        "random",
        "positive_limit",
        "negative_limit",
        "opposing_limits",
        "signed_boundary",
        "all_bytes",
    ],
)
def test_paired_pv_matches_exact_reference(starts, values, recurrence):
    generator = torch.Generator().manual_seed(952)
    block_m = 64
    probability = torch.randint(0, 256, (block_m, 128), dtype=torch.uint8, generator=generator)
    value = torch.randint(-128, 128, (256, 128), dtype=torch.int8, generator=generator)
    if values in ("positive_limit", "negative_limit", "opposing_limits"):
        probability.fill_(255)
        value.fill_(-128 if values == "negative_limit" else 127)
        if values == "opposing_limits":
            value[starts[1] : starts[1] + 64] = -128
    elif values == "signed_boundary":
        probability.copy_(torch.tensor([0, 127, 128, 255], dtype=torch.uint8).repeat(block_m, 32))
        value.copy_(torch.tensor([-128, -1, 0, 127], dtype=torch.int8).repeat(256, 32))
    elif values == "all_bytes":
        probability.copy_(torch.arange(block_m * 128).reshape(block_m, 128).to(torch.uint8))
        value.zero_()
        value[starts[0] + torch.arange(64), torch.arange(64)] = 1
        value[starts[1] + torch.arange(64), 64 + torch.arange(64)] = 1
    expected = (
        probability[:, :64].long() @ value[starts[0] : starts[0] + 64].long()
        + probability[:, 64:].long() @ value[starts[1] : starts[1] + 64].long()
    )
    numerator = torch.zeros((block_m, 128), dtype=torch.float32)
    weight = torch.ones(block_m)
    if recurrence:
        numerator.copy_(torch.randint(-1000, 1001, numerator.shape, generator=generator))
        weight.copy_(torch.arange(block_m).remainder(3) * 0.5)
    # INT64 dot products followed by independent FP64 recurrence arithmetic.
    expected = (expected.double() * weight[:, None].double() + numerator.double()).float()
    storage = torch.full((block_m * 128 + 16,), -12345, dtype=torch.float32, device="cuda")
    output = storage[8:-8].view(block_m, 128)
    packed_value = torch.empty((1, 1, 4, 128, 64), dtype=torch.int8, device="cuda")
    _pack_values[(4, 1)](value.T.contiguous().cuda(), packed_value, 256, num_warps=4)
    _paired_pv_kernel[(1,)](
        probability.cuda(),
        packed_value,
        numerator.cuda(),
        weight.cuda(),
        output,
        *starts,
        num_warps=4,
    )
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
    assert (storage[:8] == -12345).all()
    assert (storage[-8:] == -12345).all()


@gluon.jit
def _qk_probe(query_ptr, key_ptr, output_ptr, tile_0, tile_1):
    block_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, MMA_LAYOUT))
    query = query_fragments(query_ptr, gl.full([1], 0, gl.int32, block_layout), 64)
    result = qk_pair(
        query,
        key_ptr,
        gl.full([1], tile_0, gl.int32, block_layout),
        gl.full([1], tile_1, gl.int32, block_layout),
        256,
    )
    rows = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, MMA_LAYOUT)))
    cols = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT)))
    gl.store(output_ptr + rows[None, :, None] * 128 + cols[None, None, :], result)


@pytest.mark.parametrize("tiles", [(0, 1), (3, 1), (2, 2)])
@pytest.mark.parametrize("values", ["random", "minimum", "maximum", "opposing", "all_bytes"])
def test_qk_bias_recovers_exact_full_range_int8_products(tiles, values):
    generator = torch.Generator().manual_seed(751)
    query = torch.randint(-128, 128, (64, 128), dtype=torch.int8, generator=generator)
    key = torch.randint(-128, 128, (256, 128), dtype=torch.int8, generator=generator)
    if values != "random":
        query.fill_(-128 if values == "minimum" else 127)
        key.fill_(-128 if values in ("minimum", "opposing") else 127)
        if values == "all_bytes":
            query.copy_(torch.arange(query.numel()).view_as(query).to(torch.int8))
            key.zero_()
            key[:128].copy_(torch.eye(128, dtype=torch.int8))
            key[128:].copy_(torch.eye(128, dtype=torch.int8))
    selected = torch.cat([key[tile * 64 : (tile + 1) * 64] for tile in tiles])
    expected = query.long() @ selected.long().T
    actual = torch.empty((64, 128), dtype=torch.float32, device="cuda")
    _qk_probe[(1,)](query.cuda(), key.cuda(), actual, *tiles, num_warps=4)
    torch.testing.assert_close(actual.cpu().double(), expected.double(), rtol=0, atol=0)


@gluon.jit
def _rescale_probe(numerator_ptr, weight_ptr, output_ptr):
    rows = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, MMA_LAYOUT)))
    cols = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT)))
    offsets = rows[None, :, None] * 128 + cols[None, None, :]
    numerator = gl.load(numerator_ptr + offsets)
    weight = gl.load(weight_ptr + rows[None, :])
    gl.store(output_ptr + offsets, rescale_numerator(numerator, weight))


def test_rescale_is_exact_for_every_row_predicate_and_layout_mapping():
    generator = torch.Generator(device="cuda").manual_seed(1945)
    numerator = torch.randn((64, 128), device="cuda", generator=generator) * 1024
    numerator[:, ::7] = 0
    actual = torch.empty_like(numerator)
    cases = [torch.full((64,), value, device="cuda") for value in (0.0, 0.5, 1.0)]
    cases += [torch.nextafter(cases[-1], cases[0]), torch.arange(1, 65, device="cuda").float()]
    cases += [torch.arange(64, device="cuda").remainder(2).float() * 0.5 + 0.5]
    for row in range(64):
        changed = torch.ones(64, device="cuda")
        changed[row] = 0.5
        unchanged = torch.full((64,), 0.5, device="cuda")
        unchanged[row] = 1.0
        cases.extend((changed, unchanged))
    for weight in cases:
        _rescale_probe[(1,)](numerator, weight, actual, num_warps=4)
        torch.testing.assert_close(actual, numerator * weight[:, None], rtol=0, atol=0)


def _constant_context(batch, heads, sequence):
    storage = (sequence + 63) // 64 * 64
    # One allocation backs both INT8 K and V. Query=0 makes scores uniform;
    # distinct per-head V constants exercise global addressing independently.
    value = torch.empty((batch, heads, 128, storage), dtype=torch.int8, device="cuda")
    constants = torch.arange(batch * heads, device="cuda").remainder(127).to(torch.int8)
    value.copy_(constants.view(batch, heads, 1, 1))
    context = _prepare_sparse_piper_context_from_quantized(
        value.view(batch, heads, storage, 128),
        torch.ones((batch, heads, storage // 64), device="cuda"),
        value,
        torch.full((batch, heads, storage // 64, 1), 255.0, device="cuda"),
        torch.zeros((batch, heads, 128), device="cuda"),
        torch.ones(heads, dtype=torch.int32, device="cuda"),
        torch.arange(heads + 1, dtype=torch.int32, device="cuda"),
        sparse_key_blocks=storage // 64,
        routes_per_query=heads,
        logical_sequence_length=sequence,
    )
    query = _prepare_sparse_piper_query_from_quantized(
        torch.zeros((batch, heads, 64, 128), dtype=torch.int8, device="cuda"),
        torch.ones((batch, heads, 2), device="cuda"),
        torch.full((batch, 1, heads), storage // 64 - 1, dtype=torch.uint16, device="cuda"),
        context,
    )
    return _PreparedSparsePiperAttention(context, query), constants.view(batch, heads, 1, 1)


@pytest.mark.parametrize("bound", [False, True])
def test_launch_validates_once_and_only_packs_unbound_contexts(monkeypatch, bound):
    prepared, constants = _constant_context(2, 3, 193)
    launch = bind_context(prepared.context) if bound else backend._launch_sparse_piper_attention
    validate = Mock(wraps=backend.validate_attention_launch)
    pack = Mock(wraps=backend.pack_context)
    monkeypatch.setattr(backend, "validate_attention_launch", validate)
    monkeypatch.setattr(backend, "pack_context", pack)
    output = torch.empty((2, 3, 64, 128), dtype=torch.bfloat16, device="cuda")
    launch(prepared, output)
    validate.assert_called_once_with(prepared, output, 0, None, None, None)
    if bound:
        pack.assert_not_called()
    else:
        pack.assert_called_once_with(prepared.context)
    torch.testing.assert_close(output, constants.expand_as(output).to(output.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("invalid", ["output", "range", "coarse"])
def test_invalid_launch_is_rejected_before_packing_or_execution(monkeypatch, bound, invalid):
    prepared, _ = _constant_context(1, 1, 64)
    launch = bind_context(prepared.context) if bound else backend._launch_sparse_piper_attention
    monkeypatch.setattr(backend, "pack_context", Mock(side_effect=AssertionError("packed")))
    monkeypatch.setattr(
        backend,
        "_sparse_piper_attention_kernel",
        Mock(side_effect=AssertionError("launched")),
    )
    output = torch.empty((1, 1, 64, 128), dtype=torch.bfloat16, device="cuda")
    if invalid == "output":
        with pytest.raises(ValueError, match="output must match"):
            launch(prepared, output.float())
    elif invalid == "range":
        with pytest.raises(ValueError, match="query block range"):
            launch(prepared, output, query_block_count=2)
    else:
        with pytest.raises(ValueError, match="supplied together"):
            launch(prepared, output, coarse_output=output.float())


def test_bound_context_reuses_packing_and_rebinding_observes_writes(monkeypatch):
    prepared, constants = _constant_context(2, 3, 193)
    launch = bind_context(prepared.context)
    output = torch.empty((2, 3, 64, 128), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(backend, "pack_context", lambda _: pytest.fail("packed again"))
    for _ in range(2):
        launch(prepared, output)
        torch.testing.assert_close(
            output, constants.expand_as(output).to(output.dtype), rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="original context"):
        launch(replace(prepared, context=replace(prepared.context)), output)
    monkeypatch.undo()
    prepared.context.value.fill_(7)
    bind_context(prepared.context)(prepared, output)
    torch.testing.assert_close(output, torch.full_like(output, 7), rtol=0, atol=0)


@pytest.mark.parametrize("operand", ["query_scale", "key_scale"])
@pytest.mark.parametrize("mixed", [False, True])
def test_zero_scales_with_a_routed_partial_tile_are_finite(operand, mixed):
    prepared, constants = _constant_context(2, 3, 193)
    scale = prepared.query.scale if operand == "query_scale" else prepared.context.key_scale
    if mixed:
        scale[:, 0] = 0
    else:
        scale.zero_()
    output = torch.empty((2, 3, 64, 128), dtype=torch.bfloat16, device="cuda")
    bind_context(prepared.context)(prepared, output)
    torch.testing.assert_close(output, constants.expand_as(output).to(output.dtype), rtol=0, atol=0)


def test_large_batch_local_query_uses_offsets_beyond_signed_int32():
    batch, heads, sequence = 4, 56, 100000
    bytes_per_tensor = batch * heads * 128 * 100032
    assert bytes_per_tensor > 2**31
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 2 * bytes_per_tensor + (1 << 30):
        pytest.skip("requires memory for original and packed >2-GiB INT8 tensors")
    prepared, constants = _constant_context(batch, heads, sequence)
    output = torch.empty((batch, heads, 64, 128), dtype=torch.bfloat16, device="cuda")
    bind_context(prepared.context)(prepared, output)
    torch.testing.assert_close(output, constants.expand_as(output).to(output.dtype), rtol=0, atol=0)
