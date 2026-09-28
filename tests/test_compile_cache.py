"""Test cache ownership at the launcher/process boundary, including native loads."""

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
            library = cache / f'native-{index}.dll'
            shutil.copyfile(Path(os.environ['SystemRoot']) / 'System32/version.dll', library)
            libraries.append(ctypes.WinDLL(str(library)))
        else:
            with (cache / f'native-{index}').open('w+b') as library:
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


def _environment(**overrides):
    environment = os.environ.copy()
    for variable in (*_VARIABLES, "PYTEST_ADDOPTS"):
        environment.pop(variable, None)
    return {**environment, "CUDA_VISIBLE_DEVICES": "", **overrides}


def _reported_paths(directory):
    return [
        [Path(path) for path in json.loads(report.read_text())]
        for report in directory.glob("report-*.json")
    ]


@pytest.mark.parametrize(
    ("workers", "outcome", "exit_code"),
    [(0, "pass", 0), (2, "pass", 0), (2, "fail", 1), (0, "interrupt", 2), (0, "crash", 3)],
)
def test_owned_caches_outlive_native_loads_and_are_removed(
    probe_suite, workers, outcome, exit_code
):
    storage = probe_suite / "storage"
    storage.mkdir()
    sentinel = storage / "keep"
    sentinel.write_text("unrelated data")
    result = subprocess.run(
        [sys.executable, str(_RUNNER), "--cache-root", str(storage), "--", "-n", str(workers)],
        cwd=probe_suite,
        env=_environment(PROBE_OUTCOME=outcome),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    reports = _reported_paths(probe_suite)
    assert len(reports) == (1 if outcome in ("interrupt", "crash") else 2)
    roots = {path.parent for paths in reports for path in paths}
    assert len(roots) == 1
    assert roots.pop().parent == storage
    assert list(storage.iterdir()) == [sentinel]
    assert sentinel.read_text() == "unrelated data"


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("inherited", [0, 1, 2])
def test_cache_ownership_and_parent_environment(runner, monkeypatch, tmp_path, explicit, inherited):
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    sentinel = supplied / "keep"
    sentinel.touch()
    for index, variable in enumerate(_VARIABLES):
        if index < inherited:
            monkeypatch.setenv(variable, str(supplied))
        else:
            monkeypatch.delenv(variable, raising=False)
    before = os.environ.copy()
    environments = []

    def run(arguments, environment):
        assert arguments == ["-n0", "-k", "example"]
        environments.append(environment)
        for variable in _VARIABLES:
            if not explicit and variable in before:
                assert environment[variable] == before[variable]
            else:
                path = Path(environment[variable])
                assert path.parent.is_dir()
                assert path.parent != supplied
        return 5

    monkeypatch.setattr(runner, "_run_pytest", run)
    options = ["--cache-root", str(tmp_path)] if explicit else []
    assert runner.main([*options, "--", "-n0", "-k", "example"]) == 5
    assert os.environ == before
    for variable, path in environments[0].items():
        if variable in _VARIABLES and (explicit or variable not in before):
            assert not Path(path).parent.exists()
    assert sentinel.exists()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("failure", ["executable mappings denied", "No space left on device"])
def test_every_managed_location_must_load_libraries(
    runner, monkeypatch, tmp_path, capsys, explicit, failure
):
    for variable in _VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    # Exercise a failing native loader in a real subprocess for both kinds of root.
    monkeypatch.setattr(runner, "_LOAD_PROBE", f"raise OSError({failure!r})")
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    def unexpected(*args):
        pytest.fail("pytest must not start with an unsuitable cache")

    monkeypatch.setattr(runner, "_run_pytest", unexpected)
    options = ["--cache-root", str(tmp_path)] if explicit else []
    assert runner.main(options) == 2
    assert not list(tmp_path.iterdir())
    error = capsys.readouterr().err
    assert failure in error
    assert "--cache-root" in error


def test_supplied_caches_bypass_unsuitable_default_location(runner, monkeypatch, tmp_path):
    for variable in _VARIABLES:
        monkeypatch.setenv(variable, str(tmp_path))

    def unexpected(*args, **kwargs):
        pytest.fail("must not create or probe a managed root for supplied caches")

    monkeypatch.setattr(runner, "TemporaryDirectory", unexpected)
    monkeypatch.setattr(runner, "_run_pytest", lambda *args: 0)
    assert runner.main([]) == 0


def test_invalid_explicit_root_fails_before_pytest(runner, monkeypatch, tmp_path, capsys):
    def unexpected(*args):
        pytest.fail("pytest must not start when its cache could not be created")

    monkeypatch.setattr(runner, "_run_pytest", unexpected)
    assert runner.main(["--cache-root", str(tmp_path / "missing")]) == 2
    assert "missing" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="POSIX console process-group interruption")
@pytest.mark.parametrize("workers", [0, 2])
def test_console_interrupt_waits_for_pytest_before_cleanup(probe_suite, workers):
    storage = probe_suite / "storage"
    storage.mkdir()
    with subprocess.Popen(
        [sys.executable, str(_RUNNER), "--cache-root", str(storage), "--", "-n", str(workers)],
        cwd=probe_suite,
        env=_environment(PROBE_OUTCOME="wait"),
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
    assert _reported_paths(probe_suite)
    assert not list(storage.iterdir())


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


@pytest.mark.parametrize("selection", ["default", "explicit", "supplied"])
def test_real_noexec_filesystem(probe_suite, noexec_root, selection):
    environment = _environment(TMPDIR=str(noexec_root))
    if selection == "supplied":
        environment.update({variable: str(probe_suite / variable) for variable in _VARIABLES})
    options = ["--cache-root", str(noexec_root)] if selection == "explicit" else []
    result = subprocess.run(
        [sys.executable, str(_RUNNER), *options, "--", "-n0"],
        cwd=probe_suite,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == (0 if selection == "supplied" else 2), result.stdout + result.stderr
    if selection == "supplied":
        assert _reported_paths(probe_suite)
        assert all(Path(environment[variable]).is_dir() for variable in _VARIABLES)
    else:
        assert "cannot load native libraries" in result.stderr
        assert not _reported_paths(probe_suite)
    assert not list(noexec_root.iterdir())


def test_concurrent_launchers_have_independent_cache_roots(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
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
                    [sys.executable, str(_RUNNER), "--cache-root", str(storage), "--", "-n0"],
                    cwd=directory,
                    env=_environment(),
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
    assert len(roots) == 2
    assert not list(storage.iterdir())
