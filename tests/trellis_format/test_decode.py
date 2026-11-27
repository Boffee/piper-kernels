"""Tests for decoding trellis-coded blocks."""

import pytest
import torch

from piper_kernels.trellis import TrellisLayout
from piper_kernels.trellis._reference import decode_reference, decode_variable_rows_reference
from trellis_format._vectors import RESEARCH_VECTORS, ResearchVector

_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")


def _vector_tensors(
    vector: ResearchVector,
) -> tuple[TrellisLayout, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = TrellisLayout(vector.block_size, vector.block_bits)
    bitstreams = torch.frombuffer(bytearray.fromhex(vector.bitstream_hex), dtype=torch.uint8)
    scales = torch.tensor([vector.scale_code], dtype=torch.uint8).view(torch.float8_e4m3fn)
    values = torch.frombuffer(bytearray.fromhex(vector.values_hex), dtype=torch.int8)
    return layout, bitstreams[None], scales, values[None]


def _random_storage(
    layout: TrellisLayout, num_blocks: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arbitrary streams (every bit pattern is a valid walk) and finite E4M3 scales."""
    generator = torch.Generator().manual_seed(seed)
    shape = (num_blocks, layout.bytes_per_block)
    bitstreams = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
    codes = torch.randint(0, 256, (num_blocks,), dtype=torch.uint8, generator=generator)
    codes = torch.where(codes & 0x7F == 0x7F, codes - 1, codes)  # 0x7F and 0xFF are NaN
    return bitstreams, codes.view(torch.float8_e4m3fn)


@pytest.mark.parametrize("vector", RESEARCH_VECTORS, ids=lambda v: f"{v.block_bits}")
def test_reference_matches_research_encodings(vector: ResearchVector) -> None:
    layout, bitstreams, scales, expected = _vector_tensors(vector)

    assert torch.equal(decode_reference(bitstreams, scales, layout), expected)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("vector", RESEARCH_VECTORS, ids=lambda v: f"{v.block_bits}")
def test_triton_matches_research_encodings(vector: ResearchVector) -> None:
    from piper_kernels.trellis.triton import decode_int8_out  # noqa: PLC0415

    layout, bitstreams, scales, expected = _vector_tensors(vector)
    values = torch.empty(1, layout.block_size, dtype=torch.int8, device="cuda")

    decode_int8_out(bitstreams.cuda(), scales.view(torch.uint8).cuda(), layout, values)

    assert torch.equal(values.cpu(), expected)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("num_blocks", [1036, 1037], ids=["u32-aligned", "tail"])
@pytest.mark.parametrize(
    "layout",
    [
        TrellisLayout(256, 512),
        TrellisLayout(256, 632),
        TrellisLayout(256, 768),
        TrellisLayout(256, 1144),
        TrellisLayout(224, 1000),
        TrellisLayout(256, 1400),
        TrellisLayout(256, 1536),
        TrellisLayout(16, 32),
        TrellisLayout(16, 56),  # every window near the wrap
        TrellisLayout(200, 1088),  # a partial step tile
    ],
    ids=str,
)
def test_triton_matches_reference_on_arbitrary_streams(
    layout: TrellisLayout, num_blocks: int
) -> None:
    from piper_kernels.trellis.triton import decode_fp32_out, decode_int8_out  # noqa: PLC0415

    bitstreams, scales = _random_storage(layout, num_blocks, seed=layout.block_bits)
    values = torch.empty(num_blocks, layout.block_size, dtype=torch.int8, device="cuda")
    fp32_values = torch.empty(num_blocks, layout.block_size, dtype=torch.float32, device="cuda")

    decode_int8_out(bitstreams.cuda(), scales.view(torch.uint8).cuda(), layout, values)
    decode_fp32_out(bitstreams.cuda(), scales.view(torch.uint8).cuda(), layout, fp32_values)

    assert torch.equal(values.cpu(), decode_reference(bitstreams, scales, layout))
    torch.testing.assert_close(
        fp32_values.cpu(),
        decode_reference(bitstreams, scales, layout, round_int8=False),
        rtol=0,
        atol=0,
    )


def test_little_endian_wrap_guard_bytes_semantics() -> None:
    layout = TrellisLayout(16, 32)  # 4 code bytes + 2 trailing wrap guard bytes
    raw = torch.tensor([[0x34, 0x12, 0x78, 0x56]], dtype=torch.uint8)
    valid = torch.cat((raw, raw[:, :2]), dim=1)
    corrupted_guard = valid.clone()
    corrupted_guard[0, 4] ^= 0xFF
    scales = torch.tensor([0x38], dtype=torch.uint8).view(torch.float8_e4m3fn)  # scale = 1.0

    ref_valid = decode_reference(valid, scales, layout, round_int8=False)
    ref_bad = decode_reference(corrupted_guard, scales, layout, round_int8=False)

    # First 9 steps lie strictly inside the 4 code bytes; steps 9..15 cross into guard byte 4.
    assert torch.equal(ref_valid[:, :9], ref_bad[:, :9])
    assert not torch.equal(ref_valid[:, 9:], ref_bad[:, 9:])

    # Explicit LE check: step t reads bits [2*t .. 2*t + 15] across code bytes + wrap guard bytes.
    stream_u64 = 0
    for idx, byte in enumerate(valid[0].tolist()):
        stream_u64 |= byte << (8 * idx)
    states_expected = [(stream_u64 >> (2 * t)) & 0xFFFF for t in range(16)]
    expected_vals = []
    for st in states_expected:
        x = (st * 34038481 + 76625530) & 0xFFFFFFFF
        num = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24) - 510
        expected_vals.append(num / 147.800537109375)
    torch.testing.assert_close(
        ref_valid[0], torch.tensor(expected_vals, dtype=torch.float32), rtol=1e-6, atol=1e-6
    )


def _random_variable_storage(
    block_size: int, widths: list[int], blocks_per_row: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    rows = len(widths)
    row_widths = torch.tensor(widths, dtype=torch.uint8)
    total_bytes = blocks_per_row * sum(widths)
    bitstreams = torch.randint(0, 256, (total_bytes,), dtype=torch.uint8, generator=generator)
    codes = torch.randint(0, 256, (rows, blocks_per_row), dtype=torch.uint8, generator=generator)
    codes = torch.where(codes & 0x7F == 0x7F, codes - 1, codes)
    # Validate row widths through TrellisLayout on host to catch fixture typos.
    for width in widths:
        TrellisLayout.from_storage(block_size, width)
    return bitstreams, codes.view(torch.float8_e4m3fn), row_widths


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize(
    ("block_size", "widths", "blocks_per_row"),
    [
        # 39 blocks (% 8 != 0)
        (256, [66, 81, 98, 114, 130, 146, 162, 178, 194, 143, 145, 66, 194], 3),
        # Total bytes divisible by 4 with odd per-row widths mixing base alignments (0..3).
        (256, [66, 81, 98, 143, 145, 194, 113, 144], 4),
        (224, [58, 72, 86, 100, 114, 128, 142, 156, 170], 3),
        (200, [52, 77, 102, 138, 152, 101], 3),
        (200, [52, 77, 102, 138, 152, 101], 4),
        (16, [6, 7, 8, 9, 10, 11, 12, 13, 14], 3),
        (256, [144], 5),
        (256, [], 2),
    ],
    ids=lambda val: str(val) if not isinstance(val, list) else f"rows{len(val)}",
)
def test_triton_variable_rows_matches_reference(
    block_size: int, widths: list[int], blocks_per_row: int
) -> None:
    from piper_kernels.trellis.triton import (  # noqa: PLC0415
        decode_variable_rows_fp32_out,
        decode_variable_rows_int8_out,
    )

    bitstreams, scales, row_widths = _random_variable_storage(
        block_size, widths, blocks_per_row, seed=block_size + len(widths)
    )
    values = torch.empty(len(widths), blocks_per_row * block_size, dtype=torch.int8, device="cuda")
    fp32_values = torch.empty(
        len(widths), blocks_per_row * block_size, dtype=torch.float32, device="cuda"
    )

    decode_variable_rows_int8_out(
        bitstreams.cuda(),
        scales.view(torch.uint8).cuda(),
        row_widths.cuda(),
        block_size,
        values,
    )
    decode_variable_rows_fp32_out(
        bitstreams.cuda(),
        scales.view(torch.uint8).cuda(),
        row_widths.cuda(),
        block_size,
        fp32_values,
    )

    expected = decode_variable_rows_reference(bitstreams, scales, row_widths, block_size)
    assert torch.equal(values.cpu(), expected)
    expected_fp32 = decode_variable_rows_reference(
        bitstreams, scales, row_widths, block_size, round_int8=False
    )
    torch.testing.assert_close(fp32_values.cpu(), expected_fp32, rtol=0, atol=0)


def test_variable_rows_reference_handles_strided_and_rejects_bad_shapes() -> None:
    bitstreams, scales, row_widths = _random_variable_storage(256, [66, 144, 194], 2, seed=17)
    strided = torch.stack((bitstreams, bitstreams), dim=1)[:, 0]
    expected = decode_variable_rows_reference(bitstreams, scales, row_widths, 256)
    assert torch.equal(decode_variable_rows_reference(strided, scales, row_widths, 256), expected)

    pad = 13
    packed = torch.cat(
        (
            torch.full((pad,), 0xA5, dtype=torch.uint8),
            bitstreams,
            torch.full((5,), 0x5A, dtype=torch.uint8),
        )
    )
    assert torch.equal(
        decode_variable_rows_reference(
            packed, scales, row_widths, 256, base_offset=torch.tensor(pad, dtype=torch.int64)
        ),
        expected,
    )
    zero_expected = decode_variable_rows_reference(
        torch.empty((0,), dtype=torch.uint8), scales, row_widths, 256
    )
    assert torch.equal(
        decode_variable_rows_reference(
            bitstreams, scales, row_widths, 256, base_offset=torch.tensor(-5, dtype=torch.int64)
        ),
        zero_expected,
    )
    assert torch.equal(
        decode_variable_rows_reference(
            bitstreams,
            scales,
            row_widths,
            256,
            base_offset=torch.tensor(bitstreams.numel() + 7, dtype=torch.int64),
        ),
        zero_expected,
    )
    with pytest.raises(ValueError, match="invalid variable-row trellis reference input shapes"):
        decode_variable_rows_reference(bitstreams, scales, row_widths[:1], 256)
    with pytest.raises(ValueError, match="base_offset must be a 0-D tensor"):
        decode_variable_rows_reference(
            bitstreams, scales, row_widths, 256, base_offset=torch.tensor([0], dtype=torch.int64)
        )


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize(
    ("block_size", "width", "num_bytes"),
    [(16, 8, 0), (16, 8, 1), (64, 26, 4)],
    ids=["bs16-0B", "bs16-1B", "bs64-4B"],
)
def test_triton_variable_rows_bounds_clamp_handles_truncated_bitstreams(
    block_size: int, width: int, num_bytes: int
) -> None:
    from piper_kernels.trellis.triton import decode_variable_rows_int8_out  # noqa: PLC0415

    bitstreams = torch.arange(1, 1 + num_bytes, dtype=torch.uint8, device="cuda")
    scales = torch.full((1, 1), 0x3C, dtype=torch.uint8, device="cuda")
    row_widths = torch.tensor([width], dtype=torch.uint8, device="cuda")
    values = torch.empty(1, block_size, dtype=torch.int8, device="cuda")

    decode_variable_rows_int8_out(bitstreams, scales, row_widths, block_size, values)
    expected = decode_variable_rows_reference(
        bitstreams.cpu(), scales.view(torch.float8_e4m3fn).cpu(), row_widths.cpu(), block_size
    )
    assert torch.equal(values.cpu(), expected)
