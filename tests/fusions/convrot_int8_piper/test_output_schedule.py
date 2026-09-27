"""Dense query windows retain tile alignment and bounded storage at any length."""

from types import SimpleNamespace

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.convrot_int8_piper import _schedule
from piper_kernels.fusions.convrot_int8_piper._schedule import _balanced_query_chunk_rows


@pytest.mark.parametrize(
    ("sequence", "maximum", "expected"),
    [
        (65, 16384, 128),
        (16384, 16384, 16384),
        (16385, 16384, 8320),
        (32769, 16384, 11008),
        (100001, 16384, 14336),
        (1025, 512, 384),
        (385, 128, 128),
    ],
)
def test_query_windows_distribute_rows_within_requested_cap(sequence, maximum, expected):
    assert _balanced_query_chunk_rows(sequence, maximum) == expected


@pytest.mark.parametrize("maximum", [128, 384, 4096, 16384, 32768])
def test_query_windows_preserve_alignment_cap_and_minimum_chunk_count(maximum):
    # Include lengths far beyond benchmark anchors without allocating tensors.
    for sequence in (
        1,
        127,
        128,
        maximum - 1,
        maximum,
        maximum + 1,
        2 * maximum + 1,
        17 * maximum + 13,
        129 * maximum + 1,
        2**32 + 1,
    ):
        rows = _balanced_query_chunk_rows(sequence, maximum)
        assert 128 <= rows <= maximum
        assert rows % 128 == 0
        chunks = (sequence + rows - 1) // rows
        assert chunks == (sequence + maximum - 1) // maximum
        assert 0 < sequence - (chunks - 1) * rows <= rows
        # No smaller aligned uniform window can preserve this chunk count.
        assert chunks * (rows - 128) < sequence


@pytest.mark.parametrize(
    ("sequence", "parallel_heads", "concurrent_blocks", "expected"),
    [
        (8192, 56, 340, 8192),
        (16385, 4, 340, 8320),  # Same wave count: retain balanced buffers.
        (32769, 4, 340, 10880),  # Avoid two almost-empty scheduling waves.
        (100001, 56, 340, 16256),
        (100001, 56, 264, 16256),  # Different SM count, same cap.
        (32769, 1, 340, 11008),  # No full wave fits under the cap.
    ],
)
def test_query_windows_account_for_device_waves(
    sequence, parallel_heads, concurrent_blocks, expected
):
    assert (
        _schedule._wave_query_chunk_rows(
            sequence,
            16384,
            block_rows=128,
            parallel_heads=parallel_heads,
            concurrent_blocks=concurrent_blocks,
        )
        == expected
    )


@pytest.mark.parametrize("maximum", [128, 384, 4096, 16384, 32768])
@pytest.mark.parametrize("parallel_heads", [1, 3, 4, 32, 56, 384])
def test_wave_windows_stay_bounded_and_never_increase_predicted_waves(maximum, parallel_heads):
    def waves(sequence, rows):
        return sum(
            (((min(rows, sequence - start) + 127) // 128 * parallel_heads) + 339) // 340
            for start in range(0, sequence, rows)
        )

    for sequence in (maximum - 1, maximum + 1, 2 * maximum + 1, 129 * maximum + 1):
        rows = _schedule._wave_query_chunk_rows(
            sequence,
            maximum,
            block_rows=128,
            parallel_heads=parallel_heads,
            concurrent_blocks=340,
        )
        balanced = _balanced_query_chunk_rows(sequence, maximum)
        assert 128 <= rows <= maximum
        assert rows % 128 == 0
        if rows != balanced:
            assert waves(sequence, rows) < waves(sequence, balanced)


@pytest.mark.parametrize("architecture", ["sm89", "sm120", "sm121", "gfx1201"])
def test_output_schedule_reads_only_applicable_device_metadata(monkeypatch, architecture):
    device = torch.device("cuda:3")
    target = AcceleratorTarget(
        backend="hip" if architecture.startswith("gfx") else "cuda", architecture=architecture
    )
    monkeypatch.setattr(AcceleratorTarget, "from_device", lambda actual: target)

    def properties(actual):
        assert actual == device
        assert architecture == "sm120", "unmeasured target queried CUDA occupancy metadata"
        return SimpleNamespace(multi_processor_count=170)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    actual = _schedule.select_query_chunk_rows((1, 4, 32769, 64), device, 16384, is_causal=False)
    assert actual == (10880 if architecture == "sm120" else 11008)


@pytest.mark.parametrize(("sequence", "causal"), [(8192, False), (32769, True)])
def test_single_window_and_causal_schedules_do_not_probe_hardware(monkeypatch, sequence, causal):
    def forbidden(*args, **kwargs):
        raise AssertionError("unnecessary hardware probe")

    monkeypatch.setattr(AcceleratorTarget, "from_device", forbidden)
    actual = _schedule.select_query_chunk_rows(
        (2, 8, sequence, 128), torch.device("meta"), 16384, is_causal=causal
    )
    assert actual == _balanced_query_chunk_rows(sequence, 16384)
