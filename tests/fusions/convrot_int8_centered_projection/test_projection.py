"""Projection centering primitives compose without Q/K transforms or K64 encoding."""

import pytest
import torch
import triton
import triton.language as tl

from piper_kernels.fusions.convrot_int8_centered_projection import triton as centered
from piper_kernels.fusions.convrot_int8_centered_projection._kernels import store_projection_tile
from piper_kernels.fusions.convrot_int8_projection.triton import project_tile


@triton.jit
def _plain_projection_kernel(  # noqa: PLR0913, PLR0917
    x,
    xs,
    w,
    ws,
    bias,
    stored,
    partials,
    sequence,
    storage,
    batch: tl.constexpr,
    groups: tl.constexpr,
    features: tl.constexpr,
    input_features: tl.constexpr,
    tile_rows: tl.constexpr,
):
    block_m: tl.constexpr = 128
    groups_per_program: tl.constexpr = 2
    row_block, group_block, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = row_block * block_m + tl.arange(0, block_m)
    group_offsets = group_block * groups_per_program + tl.arange(0, groups_per_program)
    columns = group_block * groups_per_program * features + tl.arange(
        0, groups_per_program * features
    )
    projection = project_tile(
        x,
        xs,
        w,
        ws,
        batch_id * sequence + rows,
        columns,
        batch * sequence,
        input_features,
        groups * features,
        False,
        block_m,
        groups_per_program * features,
        64,
        bias_ptr=bias,
    )
    projection = tl.reshape(projection, (block_m, groups_per_program, features))
    # The caller decides what internal padding means, independently of attention.
    valid = (rows < sequence) & (rows % tile_rows < tile_rows - 7)
    projection = tl.where(valid[:, None, None], projection, 0.0)
    grouped = tl.reshape(
        tl.permute(projection, (1, 0, 2)),
        (groups_per_program, block_m // tile_rows, tile_rows, features),
    )
    tiles = row_block * (block_m // tile_rows) + tl.arange(0, block_m // tile_rows)
    store_projection_tile(
        grouped,
        stored,
        partials,
        batch_id,
        group_offsets,
        rows,
        tiles,
        storage,
        groups,
        groups_per_program,
        features,
        block_m,
        tile_rows,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    ("sequence", "features", "tile_rows"), [(65, 32, 32), (193, 64, 64), (256, 128, 128)]
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_plain_projection_stores_bf16_and_reduces_its_represented_mean(
    sequence, features, tile_rows, with_bias
):
    torch.manual_seed(482)
    batch, groups, input_features = 2, 3, 80
    x = torch.randint(-16, 17, (batch, sequence, input_features), device="cuda", dtype=torch.int8)
    w = torch.randint(-16, 17, (groups * features, input_features), device="cuda", dtype=torch.int8)
    xs = torch.full((batch, sequence), 0.125, device="cuda")
    ws = torch.full((groups * features, 1), 0.0625, device="cuda")
    bias = torch.full((groups * features,), 0.125, device="cuda") if with_bias else None
    storage = triton.cdiv(sequence, tile_rows) * tile_rows
    stored, partials, mean = centered.allocate_workspace(
        x, (batch, groups, storage, features), tile_rows=tile_rows
    )
    _plain_projection_kernel[(triton.cdiv(sequence, 128), triton.cdiv(groups, 2), batch)](
        x,
        xs,
        w,
        ws,
        bias,
        stored,
        partials,
        sequence,
        storage,
        batch,
        groups,
        features,
        input_features,
        tile_rows,
    )
    centered.finalize_mean(partials, sequence, out=mean)
    reference = (x.double() @ w.double().T) * xs.double()[..., None] * ws.double()[:, 0]
    if bias is not None:
        reference += bias.double()
    reference = reference.reshape(batch, sequence, groups, features).transpose(1, 2).bfloat16()
    reference[:, :, torch.arange(sequence, device="cuda") % tile_rows >= tile_rows - 7] = 0
    assert torch.equal(stored[:, :, :sequence], reference)
    assert torch.count_nonzero(stored[:, :, sequence:]) == 0
    torch.testing.assert_close(mean, reference.float().mean(2), atol=1e-7, rtol=1e-6)
