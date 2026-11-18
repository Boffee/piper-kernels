"""Coverage for the long SwiGLU weight prefix and the first output row it borrows."""

import pytest

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_ffn._amd.policy import packed_prefix_rows


@pytest.mark.parametrize(
    ("rows", "expected"),
    [(54_271, 0), (54_272, 32_768), (100_000, 77_824)],
)
def test_packed_prefix_ends_before_weight_storage(rows, expected):
    assert (
        packed_prefix_rows(AcceleratorTarget("hip", "gfx1201"), rows, 5376, 14336, 5376, 4096)
        == expected
    )


@pytest.mark.parametrize(
    ("arch", "width", "hidden", "output", "chunk"),
    [
        ("gfx1200", 5376, 14336, 5376, 4096),
        ("gfx1201", 8192, 14336, 5376, 4096),
        ("gfx1201", 5376, 16384, 5376, 4096),
        ("gfx1201", 5376, 14336, 5380, 4096),
        ("gfx1201", 5376, 14336, 5376, 8192),
        ("gfx1201", 5376, 14336, 262144, 4096),
    ],
)
def test_packed_prefix_preserves_other_schedules(arch, width, hidden, output, chunk):
    assert (
        packed_prefix_rows(AcceleratorTarget("hip", arch), 100_000, width, hidden, output, chunk)
        == 0
    )
