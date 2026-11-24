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
@pytest.mark.parametrize(
    "layout",
    [
        TrellisLayout(256, 768),
        TrellisLayout(256, 1144),
        TrellisLayout(224, 1000),
        TrellisLayout(256, 1400),
        TrellisLayout(256, 1536),
        TrellisLayout(16, 56),  # every window near the wrap
        TrellisLayout(200, 1088),  # a partial step tile
    ],
    ids=str,
)
def test_triton_matches_reference_on_arbitrary_streams(layout: TrellisLayout) -> None:
    from piper_kernels.trellis.triton import decode_int8_out  # noqa: PLC0415

    num_blocks = 1037  # not a multiple of the blocks per program
    bitstreams, scales = _random_storage(layout, num_blocks, seed=layout.block_bits)
    values = torch.empty(num_blocks, layout.block_size, dtype=torch.int8, device="cuda")

    decode_int8_out(bitstreams.cuda(), scales.view(torch.uint8).cuda(), layout, values)

    assert torch.equal(values.cpu(), decode_reference(bitstreams, scales, layout))


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
        (256, [96, 112, 128, 144, 160, 176, 192, 143, 145, 96, 192], 3),  # 33 blocks (% 8 != 0)
        (224, [84, 98, 112, 126, 140, 154, 168], 3),
        (200, [75, 100, 136, 150, 99], 3),
        (16, [6, 7, 8, 9, 10, 11, 12], 3),
        (256, [144], 5),
        (256, [], 2),
    ],
    ids=lambda val: str(val) if not isinstance(val, list) else f"rows{len(val)}",
)
def test_triton_variable_rows_matches_reference(
    block_size: int, widths: list[int], blocks_per_row: int
) -> None:
    from piper_kernels.trellis.triton import decode_variable_rows_int8_out  # noqa: PLC0415

    bitstreams, scales, row_widths = _random_variable_storage(
        block_size, widths, blocks_per_row, seed=block_size + len(widths)
    )
    values = torch.empty(len(widths), blocks_per_row * block_size, dtype=torch.int8, device="cuda")

    decode_variable_rows_int8_out(
        bitstreams.cuda(),
        scales.view(torch.uint8).cuda(),
        row_widths.cuda(),
        block_size,
        values,
    )

    expected = decode_variable_rows_reference(bitstreams, scales, row_widths, block_size)
    assert torch.equal(values.cpu(), expected)


def test_variable_rows_reference_handles_strided_and_rejects_bad_shapes() -> None:
    bitstreams, scales, row_widths = _random_variable_storage(256, [96, 144], 2, seed=17)
    strided = torch.stack((bitstreams, bitstreams), dim=1)[:, 0]
    assert torch.equal(
        decode_variable_rows_reference(strided, scales, row_widths, 256),
        decode_variable_rows_reference(bitstreams, scales, row_widths, 256),
    )
    with pytest.raises(ValueError, match="invalid variable-row trellis reference input shapes"):
        decode_variable_rows_reference(bitstreams, scales, row_widths[:1], 256)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("num_bytes", [0, 1])
def test_triton_variable_rows_bounds_clamp_handles_truncated_bitstreams(num_bytes: int) -> None:
    from piper_kernels.trellis.triton import decode_variable_rows_int8_out  # noqa: PLC0415

    bitstreams = torch.zeros(num_bytes, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(1, 1, dtype=torch.uint8, device="cuda")
    row_widths = torch.tensor([6], dtype=torch.uint8, device="cuda")
    values = torch.empty(1, 16, dtype=torch.int8, device="cuda")

    decode_variable_rows_int8_out(bitstreams, scales, row_widths, 16, values)
    torch.cuda.synchronize()
    assert values.shape == (1, 16)
