"""Repository-wide pytest configuration."""

import os
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from filelock import FileLock

_COMPILE_CACHE = pytest.StashKey[str]()


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Own compilation caches for this run, with cleanup after workers exit."""
    worker_input = getattr(config, "workerinput", None)
    if worker_input is None:
        ram = Path("/dev/shm")
        cache = TemporaryDirectory(
            prefix="piper-kernels-pytest-",
            dir=ram if ram.is_dir() and os.access(ram, os.W_OK) else None,
        )
        config.add_cleanup(cache.cleanup)
        root = cache.name
    else:
        root = worker_input["piper_compile_cache"]
    config.stash[_COMPILE_CACHE] = root
    environment = pytest.MonkeyPatch()
    config.add_cleanup(environment.undo)
    for variable, subdirectory in (
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
    ):
        environment.setenv(variable, str(Path(root) / subdirectory))


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:
    """Share the controller's cache without giving workers cleanup ownership."""
    node.workerinput["piper_compile_cache"] = node.config.stash[_COMPILE_CACHE]


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """Bound ``-n auto`` by what actually speeds the suite up.

    Past about 16 workers, worker start-up and memory outweigh parallelism: a
    48-worker CPU-only run was slower than 16 workers while using 2.6 times the
    RAM. Workers that run GPU tests also share one device: on an RTX 5090, 8
    workers peak near 19 GiB and 16 near 27 GiB, so runs with a visible
    accelerator default to 8. Set ``PYTEST_XDIST_AUTO_NUM_WORKERS`` to override,
    for example for a larger or smaller device.
    """
    if config.option.collectonly:
        return 0  # Collection gains nothing from workers that each import PyTorch.
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None  # xdist's own hook honors the explicit override.
    cores = os.process_cpu_count() or 1
    return min(cores, 8 if _accelerator_visible() else 16)


def _accelerator_visible() -> bool:
    """Match the tests' own GPU gate, which also covers ROCm and hidden devices."""
    # CI and the pre-commit hook hide CUDA this way; skip importing torch for them.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    import torch  # noqa: PLC0415 - deferred so the hidden-CUDA path stays light.

    return torch.cuda.is_available()


@pytest.fixture
def large_device_memory(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Run tests that need gigabytes of device memory one at a time.

    Such tests check free device memory and then allocate. On different xdist
    workers, two of them could both pass the check and then run out of memory
    together. Every worker's base temporary directory shares one parent, which
    scopes the lock to the test session.
    """
    with FileLock(tmp_path_factory.getbasetemp().parent / "large_device_memory.lock"):
        yield
