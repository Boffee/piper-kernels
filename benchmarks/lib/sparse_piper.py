"""Bounded-memory checks and useful-operation accounting for sparse Piper."""

import torch

from piper_kernels.attention.sparse_piper_attention._prepared import _PreparedSparsePiperAttention


def useful_integer_operations(sequence: int, keep_blocks: list[int], batch: int = 1) -> int:
    """Count selected QK/PV work, including the valid dense ragged suffix."""
    selected_rows = 64 * sum(keep_blocks) + len(keep_blocks) * (sequence % 64)
    return 4 * batch * sequence * 128 * selected_rows


def assert_equal_finite(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare complete outputs without allocating full-sized FP32 temporaries."""
    assert actual.shape == expected.shape
    for left, right in zip(
        actual.flatten().split(1 << 20), expected.flatten().split(1 << 20), strict=True
    ):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def reference_prepared_query(
    prepared: _PreparedSparsePiperAttention, batch: int, head: int, query_block: int
) -> torch.Tensor:
    """Independent FP64 paired attention for one query block and one head.

    Quantized operands are shared with the kernel, but matrix products and
    probability quantization use PyTorch FP64. Pair results are combined with
    a single global normalization, not the kernel's online FP32 recurrence.
    Memory scales with one selected sequence, never the full attention matrix.
    """
    query, context = prepared.query, prepared.context
    device = query.data.device
    global_block = query.global_block_offset + query_block
    stored_blocks = context.key.shape[2] // 64
    use_routes = context.sparse_query_blocks is None or global_block < context.sparse_query_blocks
    if use_routes:
        start, stop = context.route_head_offsets[head : head + 2].tolist()
        tiles = query.routes[batch, query_block, start:stop].long()
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
    q = query.data[batch, head, query_block * 64 : query_block * 64 + rows].double()
    q_scale = query.scale[batch, head, query_block * 2 : query_block * 2 + 2].repeat_interleave(32)
    k = context.key[batch, head].index_select(0, indices).double()
    k_scale = context.key_scale[batch, head, tiles].repeat_interleave(64)
    multiplier = (
        context.value_scale_multiplier[batch, head, tiles, 0].repeat_interleave(64).double()
    )
    scores = (q @ k.T) * q_scale[:rows, None].double() * k_scale[None, :].double()
    scores = (
        scores.masked_fill(~valid[None, :], -torch.inf).reshape(rows, pairs, 128).transpose(0, 1)
    )
    multipliers = multiplier.reshape(pairs, 1, 128)
    pair_max = (scores + torch.log2(multipliers / 255)).amax(dim=-1)
    probabilities = torch.exp2(scores - pair_max[:, :, None])
    codes = (probabilities * multipliers + 0.5).floor().clamp(0, 255)
    values = context.value[batch, head].index_select(1, indices).T.reshape(pairs, 128, 128).double()
    products = torch.bmm(codes, values)
    weights = torch.exp2(pair_max - pair_max.amax(dim=0))
    numerator = (products * weights[:, :, None]).sum(dim=0)
    denominator = (probabilities.sum(dim=-1) * weights).sum(dim=0)
    output = numerator / (denominator.clamp_min(1e-30)[:, None] * 255)
    return (output + context.value_mean[batch, head].double()).to(torch.bfloat16)


def check_query_samples(
    prepared: _PreparedSparsePiperAttention, output: torch.Tensor
) -> list[dict[str, int | float]]:
    """Check first/middle/last query blocks and heads against bounded FP64 math."""
    batch, sequence, heads, _ = output.shape
    blocks = (sequence + 63) // 64
    checks = []
    for b in range(batch):
        for head in sorted({0, heads // 2, heads - 1}):
            for block in sorted({0, blocks // 2, blocks - 1}):
                expected = reference_prepared_query(prepared, b, head, block)
                actual = output[b, block * 64 : block * 64 + expected.shape[0], head]
                error = float(
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-30)
                )
                assert torch.isfinite(actual).all()
                assert error < 0.015, (b, head, block, error)
                checks.append(
                    {"batch": b, "head": head, "query_block": block, "relative_l2": error}
                )
    return checks
