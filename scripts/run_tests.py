"""Run pytest with persistent compiler caches, optionally resetting them first."""

import argparse
import getpass
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

_CACHE_VARIABLES = {"TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor"}
_LOAD_PROBE = """
import mmap
import os
from pathlib import Path
import shutil
import sys

directory = Path(sys.argv[1])
if os.name == 'nt':
    import ctypes
    library = directory / 'probe.dll'
    shutil.copyfile(Path(os.environ['SystemRoot']) / 'System32/version.dll', library)
    ctypes.WinDLL(str(library))
else:
    with (directory / 'probe').open('w+b') as library:
        library.write(bytes(mmap.PAGESIZE))
        library.flush()
        with mmap.mmap(library.fileno(), mmap.PAGESIZE, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ | mmap.PROT_EXEC):
            pass
"""


def _probe_cache(directory: Path) -> None:
    """Check native-library loading, then unload before deleting probe files."""
    with TemporaryDirectory(prefix="probe-", dir=directory) as probe:
        result = subprocess.run(
            [sys.executable, "-I", "-c", _LOAD_PROBE, probe],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise OSError(
                f"cannot load native libraries from {directory}:\n{result.stderr.strip()}"
            )


@contextmanager
def _supervise_console() -> Iterator[None]:
    # Ctrl-C reaches the entire console/process group. A Python handler keeps the
    # launcher alive through child exit without inheriting SIG_IGN into
    # the child (which must retain pytest's normal interrupt behavior).
    previous = signal.signal(signal.SIGINT, lambda _signum, _frame: None)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _run_pytest(arguments: Sequence[str], environment: dict[str, str]) -> int:
    with subprocess.Popen([sys.executable, "-m", "pytest", *arguments], env=environment) as child:
        returncode = child.wait()
    return returncode if returncode >= 0 else 128 - returncode


def _persistent_cache_dir() -> Path:
    """Keep both compiler caches in one stable per-user temporary directory."""
    username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
    return Path(gettempdir()) / f"piper-kernels-tests-{username}"


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="clear both compiler caches before running tests; requires no other cache users",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="persistent cache directory "
        "(default: system temporary directory/piper-kernels-tests-<user>)",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="pytest arguments after --")
    options = parser.parse_args(arguments)
    pytest_args = options.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    with _supervise_console():
        return _run_with_cache(
            pytest_args,
            cache_dir=options.cache_dir
            if options.cache_dir is not None
            else _persistent_cache_dir(),
            reset=options.reset_cache,
        )


def _run_with_cache(
    arguments: Sequence[str],
    *,
    cache_dir: Path,
    reset: bool,
) -> int:
    """Own both child cache paths; reset only their subdirectories before pytest."""
    environment = os.environ.copy()
    try:
        cache_dir = cache_dir.expanduser().resolve()
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _probe_cache(cache_dir)
        for variable, name in _CACHE_VARIABLES.items():
            cache = cache_dir / name
            if reset and cache.exists():
                shutil.rmtree(cache)
            environment[variable] = str(cache)
        return _run_pytest(arguments, environment)
    except OSError as error:
        print(  # noqa: T201
            f"Test launcher: {error}\nUse a writable, executable filesystem for --cache-dir.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
