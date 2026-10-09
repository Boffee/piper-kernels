"""Shared inert metadata for benchmark unit tests."""

import pytest
from lib.environment import EnvironmentInfo


@pytest.fixture
def environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        captured_at_utc="2026-08-08T00:00:00+00:00",
        python_version="3.14.0",
        platform="test",
        torch_version="2.12.0",
        triton_version="3.7.1",
        accelerator_backend="cuda",
        accelerator_runtime_version="13.0",
        accelerator_driver_version="580.0",
        gpu_name="test GPU",
        gpu_architecture="SM120",
        gpu_index=0,
        git_revision="a" * 40,
        git_dirty=False,
    )
