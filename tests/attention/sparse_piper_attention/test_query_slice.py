"""Query-row slices of sparse Piper attention reproduce the whole-sequence rows."""

import itertools
from collections.abc import Callable

import pytest
import torch

from piper_kernels import SparsePiperAttention, sparse_piper_coarse_residual
from piper_kernels.attention.sparse_piper_attention._backend import select_attention_backend

_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]

type _SliceCall = Callable[[torch.Tensor, int], torch.Tensor]


def _require_backend(device: str, head_dim: int = 128) -> None:
    if device == "cuda" and (
        not torch.cuda.is_available()
        or select_attention_backend(torch.empty(1, 64, 1, head_dim, device=device)) is None
    ):
        pytest.skip("requires a native sparse-attention backend")


def _operands(
    device: str,
    sequence_length: int,
    *,
    query_heads: int = 2,
    kv_heads: int = 2,
    head_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(1203)

    def sample(heads: int) -> torch.Tensor:
        shape = (1, sequence_length, heads, head_dim)
        return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)

    return sample(query_heads), sample(kv_heads), sample(kv_heads)


def _call_in_slices(
    call: _SliceCall, query: torch.Tensor, boundaries: tuple[int, ...]
) -> torch.Tensor:
    """Concatenate one call per query slice ``[boundaries[i], boundaries[i + 1])``."""
    with torch.no_grad():
        return torch.cat(
            [call(query[:, start:end], start) for start, end in itertools.pairwise(boundaries)],
            dim=1,
        )


def _forced_tiles(
    attention: SparsePiperAttention, sparse_key_blocks: int, query_blocks: int, device: str
) -> dict[str, torch.Tensor]:
    """Force a different tile pair into each global query block, within every budget."""
    count = min(2, *attention.resolve_head_keep_blocks(sparse_key_blocks))
    blocks = torch.arange(query_blocks, device=device, dtype=torch.int32)
    ids = torch.stack((blocks, blocks + 1), dim=1).remainder_(sparse_key_blocks)
    counts = blocks.remainder(count + 1).to(torch.int32)
    return {"force_tile_ids": ids.contiguous(), "force_tile_counts": counts}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("routing", ["mean", "minmax"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_compact_query_slices_reproduce_whole_sequence_rows(
    device: str, routing: str, head_dim: int
) -> None:
    _require_backend(device, head_dim)
    # GQA, a ragged tail in the last of four slices, and a routed query scope that
    # ends inside the third slice.
    query, key, value = _operands(device, 500, query_heads=4, kv_heads=2, head_dim=head_dim)
    attention = SparsePiperAttention((0.25, 0.5, 0.75, 1.0), routing=routing)
    options = {"sparse_key_blocks": 6, "sparse_query_blocks": 5}

    with torch.no_grad():
        expected = attention(query, key, value, **options)
    actual = _call_in_slices(
        lambda rows, start: attention(rows, key, value, global_row_offset=start, **options),
        query,
        (0, 128, 256, 448, 500),
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_padded_forced_query_slices_reproduce_whole_sequence_rows(
    device: str, routing: str
) -> None:
    _require_backend(device)
    block_lengths = torch.tensor([64, 17, 51, 64, 3, 64, 40, 64], device=device, dtype=torch.int32)
    query, key, value = _operands(device, 8 * 64)
    attention = SparsePiperAttention((0.5, 0.75), routing=routing)
    options = {
        "sparse_key_blocks": 7,
        "sparse_query_blocks": 3,
        "block_lengths": block_lengths,
        **_forced_tiles(attention, 7, 8, device),
    }

    with torch.no_grad():
        expected = attention(query, key, value, **options)
    actual = _call_in_slices(
        lambda rows, start: attention(rows, key, value, global_row_offset=start, **options),
        query,
        (0, 128, 384, 512),
    )

    valid_rows = (torch.arange(512, device=device) % 64) < block_lengths.repeat_interleave(64)
    assert torch.equal(actual[:, valid_rows], expected[:, valid_rows])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("routing", ["mean", "minmax"])
@pytest.mark.parametrize("padded", [False, True])
def test_coarse_residual_query_slices_reproduce_whole_sequence_rows(
    device: str, routing: str, padded: bool
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    sequence_length = 512 if padded else 449
    query, key, value = _operands(device, sequence_length)
    gate = _operands(device, sequence_length)[0].sigmoid()
    block_lengths = (
        torch.tensor([64, 17, 51, 64, 3, 64, 40, 64], device=device, dtype=torch.int32)
        if padded
        else None
    )
    options = {"routing": routing, "coarse_scale": 0.125, "block_lengths": block_lengths}

    with torch.no_grad():
        expected = sparse_piper_coarse_residual(query, key, value, gate, **options)
    actual = _call_in_slices(
        lambda rows, start: sparse_piper_coarse_residual(
            rows,
            key,
            value,
            gate[:, start : start + rows.shape[1]],
            global_row_offset=start,
            **options,
        ),
        query,
        (0, 192, sequence_length),
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("sequence_length", "global_row_offset", "query_rows", "padded", "error", "message"),
    [
        (256, 32, 64, False, ValueError, "multiple of 64"),
        (256, -64, 64, False, ValueError, "must fit"),
        (256, 192, 128, False, ValueError, "must fit"),
        (256, 256, 64, False, ValueError, "must fit"),
        (256, 64, 100, True, ValueError, "whole K64 blocks"),
        (300, 0, 100, False, ValueError, "whole K64 blocks"),
        (256, True, 64, False, TypeError, "integer"),
    ],
    ids=["misaligned", "negative", "past-end", "at-end", "padded-ragged", "ragged-middle", "bool"],
)
def test_query_slice_rejects_rows_outside_whole_global_blocks(
    sequence_length: int,
    global_row_offset: int,
    query_rows: int,
    padded: bool,
    error: type[Exception],
    message: str,
) -> None:
    query, key, value = _operands("cpu", sequence_length)
    block_lengths = torch.full((sequence_length // 64,), 64, dtype=torch.int32) if padded else None
    attention = SparsePiperAttention((0.5, 1.0))
    rows = query[:, :query_rows]

    with pytest.raises(error, match=message):
        attention(
            rows,
            key,
            value,
            sparse_key_blocks=2,
            block_lengths=block_lengths,
            global_row_offset=global_row_offset,
        )
    with pytest.raises(error, match=message):
        sparse_piper_coarse_residual(
            rows,
            key,
            value,
            rows,
            routing="minmax",
            coarse_scale=0.125,
            block_lengths=block_lengths,
            global_row_offset=global_row_offset,
        )


def test_query_slices_still_require_global_force_tile_rows() -> None:
    query, key, value = _operands("cpu", 256)
    attention = SparsePiperAttention((0.5, 1.0))
    local_rows = _forced_tiles(attention, 2, 2, "cpu")

    with pytest.raises(ValueError, match="forced tiles"):
        attention(
            query[:, 128:], key, value, sparse_key_blocks=2, global_row_offset=128, **local_rows
        )


def test_compiled_query_slices_trace_symbolic_rows() -> None:
    query, key, value = _operands("cpu", 448)
    attention = SparsePiperAttention((0.5, 1.0))

    def run(rows: torch.Tensor, start: int) -> torch.Tensor:
        fine = attention(
            rows, key, value, sparse_key_blocks=3, scale=128**-0.5, global_row_offset=start
        )
        coarse = sparse_piper_coarse_residual(
            rows, key, value, rows, routing="minmax", coarse_scale=0.125, global_row_offset=start
        )
        return fine + coarse

    torch._dynamo.reset()
    compiled = torch.compile(run, backend="eager", fullgraph=True, dynamic=True)
    with torch.no_grad():
        for start, end in ((0, 192), (192, 448)):
            rows = query[:, start:end]
            torch.testing.assert_close(compiled(rows, start), run(rows, start), atol=0, rtol=0)
        with pytest.raises(RuntimeError, match="multiple of 64"):
            compiled(query[:, 32:96], 32)


@pytest.mark.gpu
@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_native_query_slice_passes_opcheck_and_full_compile(routing: str) -> None:
    from piper_kernels.attention.sparse_piper_attention.dispatch import (  # noqa: PLC0415
        _sparse_piper_attention_op,
    )

    _require_backend("cuda")
    query, key, value = _operands("cuda", 449)
    attention = SparsePiperAttention((0.5, 1.0), routing=routing)
    rows = query[:, 192:]
    result = torch.library.opcheck(
        _sparse_piper_attention_op,
        (rows, key, value, attention._ratio_profile, 7, 128**-0.5, attention._routing_mode),
        {"global_row_offset": 192},
    )
    assert set(result.values()) == {"SUCCESS"}

    def run(rows: torch.Tensor) -> torch.Tensor:
        return attention(rows, key, value, sparse_key_blocks=7, global_row_offset=192)

    torch._dynamo.reset()
    with torch.no_grad():
        actual = torch.compile(run, fullgraph=True)(rows)
        expected = attention(query, key, value, sparse_key_blocks=7)[:, 192:]
    assert actual.shape == rows.shape
    assert torch.equal(actual, expected)
