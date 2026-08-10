import pytest
import torch
from benchmark_integer_pv_dot import (
    _benchmark_output,
    _configure_variant_runtime,
    _parse_args,
    _reference_output,
)
from lib.providers import ProviderPhase


def test_reference_output_validates_every_tile() -> None:
    probability = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ],
        dtype=torch.int8,
    )
    value = torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[1, 2], [3, 4]],
            [[-1, 1], [2, -2]],
        ],
        dtype=torch.int8,
    )

    actual = _reference_output(probability, value, tile_batch=2)

    expected = torch.stack(
        [probability[index].to(torch.int32) @ value[index].to(torch.int32) for index in range(3)]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tile_batch", [0, -1])
def test_reference_output_requires_positive_tile_batch(tile_batch: int) -> None:
    values = torch.ones((1, 2, 2), dtype=torch.int8)

    with pytest.raises(ValueError, match="tile batch must be positive"):
        _reference_output(values, values, tile_batch=tile_batch)


def test_cli_exposes_generic_compiler_and_profile_controls(tmp_path) -> None:
    compiler_path = tmp_path / "compiler.json"

    arguments = _parse_args(
        [
            "s8-s8",
            "--compiler-report",
            "--compiler-json",
            str(compiler_path),
            "--no-sass",
            "--nvdisasm",
            "/opt/cuda/bin/nvdisasm",
            "--profile",
            "--profile-phase",
            "prepared_execution",
            "--profile-range-name",
            "integer-pv",
            "--profile-include-setup",
        ]
    )

    assert arguments.compiler_report
    assert arguments.compiler_json == compiler_path
    assert arguments.sass is False
    assert arguments.nvdisasm.as_posix() == "/opt/cuda/bin/nvdisasm"
    assert arguments.profile
    assert arguments.profile_phase is ProviderPhase.PREPARED_EXECUTION
    assert arguments.profile_range_name == "integer-pv"
    assert arguments.profile_include_setup


def test_profile_mode_rejects_benchmark_result_output(tmp_path) -> None:
    arguments = _parse_args(
        ["s8-s8", "--profile", "--json", str(tmp_path / "benchmark.json")]
    )

    with pytest.raises(SystemExit, match="cannot produce benchmark"):
        _benchmark_output(arguments)


def test_native_uint8_selection_installs_the_compiler_extension(monkeypatch) -> None:
    enabled = False

    def enable() -> None:
        nonlocal enabled
        enabled = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("benchmark_integer_pv_dot.install_uint8_int8_dot_hook", enable)

    _configure_variant_runtime("u8-s8-native")

    assert enabled


def test_signed_selection_does_not_install_the_compiler_extension(monkeypatch) -> None:
    def unexpected_enable() -> None:
        pytest.fail("signed dot should not install the mixed-sign compiler extension")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "benchmark_integer_pv_dot.install_uint8_int8_dot_hook",
        unexpected_enable,
    )

    _configure_variant_runtime("s8-s8")


@pytest.mark.parametrize(
    ("benchmark_option", "compiler_option"),
    [
        ("--json", "--compiler-json"),
        ("--json", "--compiler-jsonl"),
        ("--jsonl", "--compiler-json"),
        ("--jsonl", "--compiler-jsonl"),
    ],
)
def test_benchmark_and_compiler_outputs_must_not_collide(
    tmp_path, benchmark_option: str, compiler_option: str
) -> None:
    path = tmp_path / "records.json"
    arguments = _parse_args(
        ["s8-s8", benchmark_option, str(path), compiler_option, str(path)]
    )

    with pytest.raises(SystemExit, match="must be different"):
        _benchmark_output(arguments)
