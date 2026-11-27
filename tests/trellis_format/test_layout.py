"""Tests for the trellis block layout."""

import itertools

import pytest

from piper_kernels.trellis import TrellisLayout, resolve_block_size, resolve_layout

LAYOUTS = [
    TrellisLayout(256, 512),
    TrellisLayout(256, 632),
    TrellisLayout(256, 768),
    TrellisLayout(256, 888),
    TrellisLayout(256, 1144),
    TrellisLayout(224, 1000),
    TrellisLayout(256, 1400),
    TrellisLayout(256, 1536),
    TrellisLayout(200, 1088),
    TrellisLayout(16, 32),
    TrellisLayout(16, 56),
]


@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
def test_step_widths_spread_wide_steps_evenly(layout: TrellisLayout) -> None:
    widths = layout.compute_step_widths()

    assert len(widths) == layout.block_size
    assert sum(widths) == layout.block_bits
    assert set(widths) <= {layout.step_bits, layout.step_bits + 1}
    assert 2 <= min(widths) <= max(widths) <= 6
    wide = [width == layout.step_bits + 1 for width in widths]
    for t in range(layout.block_size + 1):
        assert sum(wide[:t]) == t * layout.num_wide_steps // layout.block_size


@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
def test_state_offsets_accumulate_later_step_widths(layout: TrellisLayout) -> None:
    widths = layout.compute_step_widths()

    assert layout.compute_state_offsets() == (0, *itertools.accumulate(widths[1:]))


def test_rate_follows_from_the_stored_bytes() -> None:
    layout = TrellisLayout.from_storage(256, 143)

    assert layout == TrellisLayout(256, 1128)
    assert (layout.step_bits, layout.num_wide_steps, layout.code_bytes, layout.bytes_per_block) == (
        4,
        104,
        141,
        143,
    )
    assert layout.compute_step_widths().count(5) == 104
    assert TrellisLayout.from_storage(256, 66) == TrellisLayout(256, 512)
    assert TrellisLayout.from_storage(256, 194) == TrellisLayout(256, 1536)
    assert TrellisLayout(256, 1536).step_bits == 6
    assert TrellisLayout(256, 1536).num_wide_steps == 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((15, 48), "hold 16 to 256 weights, got 15"),
        ((257, 1032), "hold 16 to 256 weights, got 257"),
        ((256, 504), "take 64 to 192 whole code bytes"),
        ((256, 1544), "take 64 to 192 whole code bytes"),
        ((256, 1145), "take 64 to 192 whole code bytes"),
    ],
)
def test_layout_rejects_unsupported_rates(arguments: tuple[int, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrellisLayout(*arguments)
    with pytest.raises(ValueError, match="take 64 to 192 whole code bytes"):
        TrellisLayout.from_storage(256, 65)
    with pytest.raises(ValueError, match="take 64 to 192 whole code bytes"):
        TrellisLayout.from_storage(256, 195)


def test_layout_requires_integers() -> None:
    with pytest.raises(TypeError, match="block_size must be an integer"):
        TrellisLayout(True, 1024)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes_per_block must be an integer"):
        TrellisLayout.from_storage(256, True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes_per_block must be an integer"):
        TrellisLayout.from_storage(256, 143.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_width", [False, 256.0])
def test_resolve_block_size_rejects_non_integers(bad_width: object) -> None:
    with pytest.raises(TypeError, match="in_features must be an integer"):
        resolve_block_size(bad_width)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_width", [0, -256])
def test_resolve_block_size_rejects_non_positive_widths(bad_width: int) -> None:
    with pytest.raises(ValueError, match="positive multiple of 16 wide"):
        resolve_block_size(bad_width)


@pytest.mark.parametrize("bad_rate", [False, "4.5"])
def test_resolve_layout_rejects_non_real_rates(bad_rate: object) -> None:
    with pytest.raises(TypeError, match="bits_per_weight must be a real number"):
        resolve_layout(256, bad_rate)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_rate", [float("inf"), float("nan")])
def test_resolve_layout_rejects_non_finite_rates(bad_rate: float) -> None:
    with pytest.raises(ValueError, match="bits_per_weight must be finite"):
        resolve_layout(256, bad_rate)
