"""Repository-wide pytest configuration."""

import os

import pytest


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """Bound ``-n auto`` by what actually speeds the suite up.

    Past about 16 workers, worker start-up and memory outweigh parallelism: a
    48-worker CPU-only run was slower than 16 workers while using 2.6 times the
    RAM. GPU workers also share one device: on an RTX 5090, 8 workers peak near
    19 GiB and 16 near 27 GiB, so GPU runs default to 8. CI and the pre-commit
    hook select CPU-only runs by hiding CUDA. Set ``PYTEST_XDIST_AUTO_NUM_WORKERS``
    to override, for example for a larger or smaller device.
    """
    del config
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None  # xdist's own hook honors the explicit override.
    cores = os.process_cpu_count() or 1
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return min(cores, 16)
    return min(cores, 8)
