"""Tests for the trellis block layout."""

import itertools

import pytest

from piper_kernels.trellis._layout import TrellisLayout

LAYOUTS = [
    TrellisLayout(256, 768),
    TrellisLayout(256, 888),
    TrellisLayout(256, 1144),
    TrellisLayout(224, 1000),
    TrellisLayout(256, 1400),
    TrellisLayout(256, 1536),
    TrellisLayout(200, 1088),
    TrellisLayout(16, 56),
]


@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
def test_step_widths_spread_wide_steps_evenly(layout: TrellisLayout) -> None:
    widths = layout.compute_step_widths()

    assert len(widths) == layout.block_size
    assert sum(widths) == layout.block_bits
    assert set(widths) <= {layout.step_bits, layout.step_bits + 1}
    assert 3 <= min(widths) <= max(widths) <= 6
    wide = [width == layout.step_bits + 1 for width in widths]
    for t in range(layout.block_size + 1):
        assert sum(wide[:t]) == t * layout.num_wide_steps // layout.block_size


@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
def test_state_offsets_accumulate_later_step_widths(layout: TrellisLayout) -> None:
    widths = layout.compute_step_widths()

    assert layout.compute_state_offsets() == (0, *itertools.accumulate(widths[1:]))


def test_rate_follows_from_the_stored_bytes() -> None:
    layout = TrellisLayout.from_storage(256, 143)

    assert layout == TrellisLayout(256, 1144)
    assert (layout.step_bits, layout.num_wide_steps, layout.bytes_per_block) == (4, 120, 143)
    assert layout.compute_step_widths().count(5) == 120
    assert TrellisLayout(256, 1536).step_bits == 6
    assert TrellisLayout(256, 1536).num_wide_steps == 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((15, 48), "hold 16 to 256 weights, got 15"),
        ((257, 1032), "hold 16 to 256 weights, got 257"),
        ((256, 760), "take 96 to 192 whole bytes"),
        ((256, 1544), "take 96 to 192 whole bytes"),
        ((256, 1145), "take 96 to 192 whole bytes"),
    ],
)
def test_layout_rejects_unsupported_rates(arguments: tuple[int, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrellisLayout(*arguments)


def test_layout_requires_integers() -> None:
    with pytest.raises(TypeError, match="block_size must be an integer"):
        TrellisLayout(True, 1024)  # type: ignore[arg-type]
