"""The portable CI selection must collect without the optional linear extras."""

import subprocess
import sys
from pathlib import Path


def test_portable_suite_collects_without_torchao():
    script = """
import importlib.abc
import sys

class WithoutTorchAO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torchao" or fullname.startswith("torchao."):
            raise ModuleNotFoundError("TorchAO intentionally unavailable", name="torchao")

sys.meta_path.insert(0, WithoutTorchAO())

import pytest

raise SystemExit(pytest.main([
    "tests/attention", "tests/triton", "-m", "not gpu", "--collect-only", "-q",
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_probe_inspects_requested_gpu_not_current_gpu" in result.stdout
    assert "test_every_native_launch_and_descriptor_has_an_explicit_device_context" in result.stdout
