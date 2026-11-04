"""Long-sequence means retain bounded tiles and accurate represented sums."""

import pytest
import torch
import triton

from piper_kernels._triton.reductions import mean_reduction_tile
from piper_kernels._triton.runtime import device_context
from piper_kernels.fusions.convrot_int8_centered_projection import triton as centered


@pytest.mark.parametrize("chunks", [1, 128, 129, 770, 4029, 8192, 8193, 32769])
def test_mean_tile_bounds_working_set(chunks):
    rows, features = mean_reduction_tile(chunks)
    assert 1 <= rows <= 8192
    assert 1 <= features <= 64
    assert rows * features <= 8192
    assert rows & (rows - 1) == 0
    assert features & (features - 1) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    ("chunks", "features"), [(3, 17), (770, 128), (4029, 128), (8193, 65), (32769, 128)]
)
def test_long_mean_is_accurate_without_register_spills(chunks, features):
    generator = torch.Generator(device="cuda").manual_seed(624)
    partials = torch.randn(1, 2, chunks, features, generator=generator, device="cuda")
    sequence = chunks * 64 - 17
    output = torch.empty(1, 2, features, device="cuda")
    centered.finalize_mean(partials, sequence, out=output)
    expected = (partials.double().sum(2) / sequence).float()
    torch.testing.assert_close(output, expected, atol=1e-8, rtol=1e-5)
    # Resource pressure is per program; a few groups reproduce the H3 failure.
    block_chunks, block_d = mean_reduction_tile(chunks)
    with device_context(partials.device):
        kernel = centered._kernels._mean_finalize_kernel[(2, triton.cdiv(features, block_d))](
            partials,
            output,
            sequence,
            chunks,
            features=features,
            block_chunks=block_chunks,
            block_d=block_d,
            num_warps=4,
        )
    # The memory regression was measured on SM120; other targets still check accuracy.
    if not torch.version.hip and torch.cuda.get_device_capability() == (12, 0):
        assert kernel.n_spills == 0
