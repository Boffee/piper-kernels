"""Test persistent cache selection, reset, and native loads across pytest exits."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_tests.py"
_VARIABLES = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
_PROBE = """
import ctypes
import json
import mmap
import os
from pathlib import Path
import shutil
import time
import pytest

libraries = []

@pytest.mark.parametrize('index', range(2))
def test_probe(index):
    paths = [Path(os.environ[key]) for key in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')]
    for cache in paths:
        cache.mkdir(parents=True, exist_ok=True)
        if os.name == 'nt':
            library = cache / f'native-{os.getpid()}-{index}.dll'
            shutil.copyfile(Path(os.environ['SystemRoot']) / 'System32/version.dll', library)
            libraries.append(ctypes.WinDLL(str(library)))
        else:
            with (cache / f'native-{os.getpid()}-{index}').open('w+b') as library:
                library.write(bytes(mmap.PAGESIZE))
                library.flush()
                libraries.append(mmap.mmap(library.fileno(), mmap.PAGESIZE,
                    flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_EXEC))
    Path(f'report-{index}.json').write_text(json.dumps([str(path) for path in paths]))
    outcome = os.environ.get('PROBE_OUTCOME', 'pass')
    if outcome == 'fail':
        pytest.fail('intentional failure')
    if outcome == 'interrupt':
        raise KeyboardInterrupt
    if outcome == 'crash':
        os._exit(3)
    if outcome == 'wait':
        Path('ready').touch()
        time.sleep(60)
"""


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("test_launcher", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe_suite(tmp_path):
    (tmp_path / "test_probe.py").write_text(_PROBE)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    return tmp_path


def _environment(temp_root: Path, **overrides: str) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in (*_VARIABLES, "PYTEST_ADDOPTS"):
        environment.pop(variable, None)
    environment.update({variable: str(temp_root) for variable in ("TMPDIR", "TEMP", "TMP")})
    return {**environment, "CUDA_VISIBLE_DEVICES": "", **overrides}


@pytest.fixture(params=["default", "custom"])
def cache_location(request, runner, monkeypatch, tmp_path):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    if request.param == "custom":
        root = tmp_path / "custom" / "cache"
        return root, ["--cache-dir", str(root)]
    return runner._persistent_cache_dir(), []


def _reported_paths(directory: Path) -> list[list[Path]]:
    return [
        [Path(path) for path in json.loads(report.read_text())]
        for report in directory.glob("report-*.json")
    ]


@pytest.mark.parametrize(
    ("workers", "outcome", "exit_code"),
    [(0, "pass", 0), (2, "pass", 0), (2, "fail", 1), (0, "interrupt", 2), (0, "crash", 3)],
)
def test_caches_survive_native_loads_and_pytest_exit(
    probe_suite, cache_location, workers, outcome, exit_code
):
    root, options = cache_location
    result = subprocess.run(
        [
            sys.executable,
            str(_RUNNER),
            *options,
            "--",
            "-n",
            str(workers),
            f"--basetemp={probe_suite / 'pytest'}",
        ],
        cwd=probe_suite,
        env=_environment(probe_suite, PROBE_OUTCOME=outcome),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    reports = _reported_paths(probe_suite)
    assert len(reports) == (1 if outcome in ("interrupt", "crash") else 2)
    assert {path.parent for paths in reports for path in paths} == {root}
    assert all(path.is_dir() for paths in reports for path in paths)


@pytest.mark.parametrize("reset", [False, True])
def test_cache_selection_and_reset(runner, monkeypatch, tmp_path, cache_location, reset):
    root, options = cache_location
    root.mkdir(parents=True)
    sentinel = root / "keep"
    sentinel.write_text("unrelated data")
    caches = [root / name for name in ("triton", "inductor")]
    for cache in caches:
        cache.mkdir()
        (cache / "old-artifact").touch()
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    (supplied / "keep").touch()
    for variable in _VARIABLES:
        monkeypatch.setenv(variable, str(supplied))
    before = os.environ.copy()

    def run(arguments, environment):
        assert arguments == ["-n0", "-k", "example"]
        for variable, cache in zip(_VARIABLES, caches, strict=True):
            assert environment[variable] == str(cache)
            assert (cache / "old-artifact").exists() is not reset
            cache.mkdir(exist_ok=True)
            (cache / "new-artifact").touch()
        return 5

    monkeypatch.setattr(runner, "_run_pytest", run)
    if reset:
        options = [*options, "--reset-cache"]
    assert runner.main([*options, "--", "-n0", "-k", "example"]) == 5
    assert os.environ == before
    assert sentinel.read_text() == "unrelated data"
    assert (supplied / "keep").exists()
    assert all((cache / "new-artifact").exists() for cache in caches)


@pytest.mark.parametrize("failure", ["executable mappings denied", "No space left on device"])
def test_cache_must_load_libraries_before_reset_or_pytest(
    runner, monkeypatch, capsys, cache_location, failure
):
    root, options = cache_location
    cache = root / "triton"
    cache.mkdir(parents=True)
    sentinel = cache / "keep"
    sentinel.touch()
    # Exercise a failing native loader in a real subprocess.
    monkeypatch.setattr(runner, "_LOAD_PROBE", f"raise OSError({failure!r})")

    def unexpected(*args):
        pytest.fail("pytest must not start with an unsuitable cache")

    monkeypatch.setattr(runner, "_run_pytest", unexpected)
    assert runner.main([*options, "--reset-cache"]) == 2
    assert list(root.iterdir()) == [cache]
    assert sentinel.exists()
    error = capsys.readouterr().err
    assert failure in error
    assert "--cache-dir" in error


def test_invalid_explicit_root_fails_before_pytest(runner, monkeypatch, tmp_path, capsys):
    def unexpected(*args):
        pytest.fail("pytest must not start when its cache could not be created")

    monkeypatch.setattr(runner, "_run_pytest", unexpected)
    root = tmp_path / "file"
    root.touch()
    assert runner.main(["--cache-dir", str(root)]) == 2
    assert str(root) in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="POSIX console process-group interruption")
@pytest.mark.parametrize("workers", [0, 2])
def test_console_interrupt_preserves_cache_lifetime(probe_suite, cache_location, workers):
    root, options = cache_location
    with subprocess.Popen(
        [
            sys.executable,
            str(_RUNNER),
            *options,
            "--",
            "-n",
            str(workers),
            f"--basetemp={probe_suite / 'pytest'}",
        ],
        cwd=probe_suite,
        env=_environment(probe_suite, PROBE_OUTCOME="wait"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            deadline = time.monotonic() + 20
            while not (probe_suite / "ready").exists():
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.05)
            os.killpg(process.pid, signal.SIGINT)
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 2, stdout + stderr
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=10)
    reports = _reported_paths(probe_suite)
    assert reports
    assert all(path.is_dir() and path.parent == root for paths in reports for path in paths)


@pytest.fixture
def noexec_root():
    if sys.platform != "linux":
        pytest.skip("requires an existing writable noexec mount")
    location = Path("/run/lock")
    if (
        not location.is_dir()
        or not os.access(location, os.W_OK)
        or not os.statvfs(location).f_flag & os.ST_NOEXEC
    ):
        pytest.skip("no suitable noexec mount available")
    with TemporaryDirectory(prefix="piper-cache-test-", dir=location) as directory:
        yield Path(directory)


@pytest.mark.parametrize("explicit_root", [False, True])
def test_real_noexec_filesystem(probe_suite, noexec_root, explicit_root):
    environment = _environment(noexec_root)
    options = ["--cache-dir", str(noexec_root)] if explicit_root else []
    result = subprocess.run(
        [sys.executable, str(_RUNNER), *options, "--", "-n0"],
        cwd=probe_suite,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "cannot load native libraries" in result.stderr
    assert not _reported_paths(probe_suite)
    if not explicit_root:
        directories = list(noexec_root.iterdir())
        assert len(directories) == 1
        assert not list(directories[0].iterdir())
    else:
        assert not list(noexec_root.iterdir())


def test_concurrent_launchers_share_the_cache(tmp_path, cache_location):
    root, options = cache_location
    directories = [tmp_path / "first", tmp_path / "second"]
    for directory in directories:
        directory.mkdir()
        (directory / "test_probe.py").write_text(_PROBE)
        (directory / "pytest.ini").write_text("[pytest]\n")
    processes = []
    try:
        for directory in directories:
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(_RUNNER), *options, "--", "-n0"],
                    cwd=directory,
                    env=_environment(tmp_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    roots = {paths[0].parent for directory in directories for paths in _reported_paths(directory)}
    assert roots == {root}
    assert all((root / name).is_dir() for name in ("triton", "inductor"))


def test_reuse_and_reset_rebuild_the_same_cache(probe_suite, cache_location):
    root, options = cache_location
    paths = {root / name for name in ("triton", "inductor")}
    # Reset must work both on a missing cache and on a populated one.
    for reset in (True, False, True):
        reset_options = ["--reset-cache"] if reset else []
        result = subprocess.run(
            [sys.executable, str(_RUNNER), *options, *reset_options, "--", "-n0"],
            cwd=probe_suite,
            env=_environment(probe_suite),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert {path for report in _reported_paths(probe_suite) for path in report} == paths
        for path in paths:
            assert (path / "retained-artifact").exists() is not reset
            assert any(path.glob("native-*"))
            (path / "retained-artifact").write_text("reuse this")
        for report in probe_suite.glob("report-*.json"):
            report.unlink()


def test_reset_failure_prevents_pytest(runner, monkeypatch, tmp_path, capsys):
    cache = tmp_path / "triton"
    cache.mkdir()

    def fail_reset(path):
        raise PermissionError("cache is in use")

    def unexpected(*args):
        pytest.fail("pytest must not start after an incomplete reset")

    monkeypatch.setattr(runner.shutil, "rmtree", fail_reset)
    monkeypatch.setattr(runner, "_probe_cache", lambda root: None)
    monkeypatch.setattr(runner, "_run_pytest", unexpected)
    assert runner.main(["--cache-dir", str(tmp_path), "--reset-cache"]) == 2
    assert "cache is in use" in capsys.readouterr().err
    assert cache.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="symlink creation requires no special privileges")
def test_reset_does_not_follow_cache_symlinks(runner, monkeypatch, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "keep"
    sentinel.touch()
    (tmp_path / "triton").symlink_to(target, target_is_directory=True)

    def unexpected(*args):
        pytest.fail("pytest must not start after an incomplete reset")

    monkeypatch.setattr(runner, "_run_pytest", unexpected)
    assert runner.main(["--cache-dir", str(tmp_path), "--reset-cache"]) == 2
    assert sentinel.exists()
