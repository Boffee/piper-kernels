"""Exercise cache ownership across real pytest controller/worker lifecycles."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryFile
from types import SimpleNamespace

import pytest
from _compile_cache import ram_cache_directory


@pytest.mark.parametrize(
    ("workers", "outcome", "exit_code", "fallback"),
    [
        (0, "pass", 0, None),
        (2, "pass", 0, None),
        (2, "fail", 1, None),
        (0, "interrupt", 2, None),
        (0, "pass", 0, "noexec"),
        (0, "pass", 0, "capacity"),
        (0, "pass", 0, "unknown_capacity"),
    ],
)
def test_compile_cache_cleanup(tmp_path, workers, outcome, exit_code, fallback):
    for name in ("conftest.py", "_compile_cache.py"):
        shutil.copyfile(Path(__file__).with_name(name), tmp_path / name)
    if fallback:
        with (tmp_path / "conftest.py").open("a") as conftest:
            conftest.write("\nimport _compile_cache\nfrom types import SimpleNamespace\n")
            if fallback == "noexec":
                conftest.write(
                    "def denied(*args, **kwargs):\n"
                    "    raise PermissionError('noexec shared memory')\n"
                    "_compile_cache.mmap.mmap = denied\n"
                )
            elif fallback == "capacity":
                conftest.write(
                    "_compile_cache.shutil.disk_usage = lambda path: "
                    "SimpleNamespace(free=64 * 1024**2)\n"
                )
            else:
                conftest.write(
                    "def unavailable(path):\n"
                    "    raise OSError('cannot query capacity')\n"
                    "_compile_cache.shutil.disk_usage = unavailable\n"
                )
    inherited = tmp_path / "existing-cache"
    inherited.mkdir()
    sentinel = inherited / "keep"
    sentinel.write_text("another application's cache")
    (tmp_path / "test_probe.py").write_text(
        "import os\n"
        "import ctypes\n"
        "import shutil\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import pytest\n"
        "libraries = []\n"
        "@pytest.mark.parametrize('index', range(2))\n"
        "def test_probe(index):\n"
        "    triton = Path(os.environ['TRITON_CACHE_DIR'])\n"
        "    inductor = Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])\n"
        "    assert triton.parent == inductor.parent\n"
        "    for cache in (triton, inductor):\n"
        "        cache.mkdir(exist_ok=True)\n"
        "        (cache / str(index)).write_text('compiled artifact')\n"
        "    if sys.platform == 'win32':\n"
        "        dll = inductor / f'version-{index}.dll'\n"
        "        shutil.copyfile(Path(os.environ['SystemRoot']) / 'System32/version.dll', dll)\n"
        "        libraries.append(ctypes.WinDLL(str(dll)))\n"
        "    Path(f'root-{index}').write_text(str(triton.parent))\n"
        "    if os.environ['PROBE_OUTCOME'] == 'fail':\n"
        "        pytest.fail('intentional failure')\n"
        "    if os.environ['PROBE_OUTCOME'] == 'interrupt':\n"
        "        raise KeyboardInterrupt\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-n", str(workers), "--confcutdir", str(tmp_path)],
        cwd=tmp_path,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "TRITON_CACHE_DIR": str(inherited),
            "TORCHINDUCTOR_CACHE_DIR": str(inherited),
            "PROBE_OUTCOME": outcome,
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    reports = list(tmp_path.glob("root-*"))
    assert len(reports) == (1 if outcome == "interrupt" else 2)
    roots = {Path(report.read_text()) for report in reports}
    assert len(roots) == 1
    root = roots.pop()
    assert root != inherited
    if fallback:
        assert root.parent != Path("/dev/shm")
    elif ram_cache_directory() is not None:
        assert root.parent == Path("/dev/shm")
    if sys.platform == "win32":
        deadline = time.monotonic() + 10
        while root.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    assert not root.exists()
    assert sentinel.read_text() == "another application's cache"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux shared-memory probe")
@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError, OSError])
def test_unusable_shared_memory_falls_back(monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error("shared memory unavailable")

    monkeypatch.setattr(
        "_compile_cache.shutil.disk_usage", lambda path: SimpleNamespace(free=64 * 1024**3)
    )
    monkeypatch.setattr("_compile_cache.TemporaryFile", unavailable)
    assert ram_cache_directory() is None


@pytest.mark.skipif(sys.platform != "linux", reason="Linux shared-memory probe")
@pytest.mark.parametrize(
    ("total", "free", "use_ram"),
    [
        (64 * 1024**2, 64 * 1024**2, False),
        (64 * 1024**3, 32 * 1024**3 - 1, False),
        (64 * 1024**3, 32 * 1024**3, True),
    ],
)
def test_ram_cache_requires_free_capacity(monkeypatch, tmp_path, total, free, use_ram):
    monkeypatch.setattr(
        "_compile_cache.shutil.disk_usage",
        lambda path: SimpleNamespace(total=total, used=total - free, free=free),
    )
    # Exercise the mapping probe on the test filesystem, independent of the host's shm mount.
    with TemporaryFile(dir=tmp_path) as probe:
        monkeypatch.setattr("_compile_cache.TemporaryFile", lambda **kwargs: probe)
        assert ram_cache_directory() == (Path("/dev/shm") if use_ram else None)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux shared-memory probe")
def test_ample_capacity_still_requires_executable_mappings(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "_compile_cache.shutil.disk_usage", lambda path: SimpleNamespace(free=64 * 1024**3)
    )

    def denied(*args, **kwargs):
        raise PermissionError("noexec shared memory")

    monkeypatch.setattr("_compile_cache.mmap.mmap", denied)
    with TemporaryFile(dir=tmp_path) as probe:
        monkeypatch.setattr("_compile_cache.TemporaryFile", lambda **kwargs: probe)
        assert ram_cache_directory() is None
