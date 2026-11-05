"""Tests for decoding trellis-coded blocks."""

import pytest
import torch

from piper_kernels.trellis._layout import TrellisLayout
from piper_kernels.trellis._reference import decode_reference
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
