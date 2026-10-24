"""H3 dimensions used in the forced-route overhead benchmark."""

import pytest
import torch
from benchmark_sparse_piper_forced import _force_inputs, h3_shape


@pytest.mark.parametrize(
    ("seconds", "frames", "latent_frames", "frame_blocks", "cube_blocks"),
    [(5, 124, 37, 555, 600), (10, 243, 72, 1080, 1080), (15, 362, 107, 1605, 1620)],
)
def test_h3_shape_uses_valid_duration_and_padded_cube_tiles(
    seconds, frames, latent_frames, frame_blocks, cube_blocks
):
    for tiling, blocks, forced in (("frame", frame_blocks, 15), ("cube", cube_blocks, 60)):
        shape = h3_shape(seconds, tiling)
        assert shape["pixel_frames"] == frames
        assert shape["latent_frames"] == latent_frames
        assert shape["video_tokens"] == latent_frames * 24 * 40
        assert shape["sparse_key_blocks"] == blocks
        assert shape["forced_tiles_per_video_query"] == forced
        ids, counts, lengths = _force_inputs(shape, torch.device("cpu"))
        assert lengths.sum().item() == shape["video_tokens"] + shape["dense_rows"]
        assert counts[:blocks].tolist() == [forced] * blocks
        assert counts[blocks:].tolist() == [0] * (shape["query_blocks"] - blocks)
        for q in (0, blocks // 2, blocks - 1):
            assert ids[q].tolist() == list(range(q // forced * forced, (q // forced + 1) * forced))
