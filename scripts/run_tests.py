"""Run pytest with compilation caches owned by this parent process."""

import argparse
import os
import signal
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

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


def _probe_cache(directory: str) -> None:
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
    # owner alive through child exit and cleanup without inheriting SIG_IGN into
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


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="existing directory for fresh owned caches; use /dev/shm to explicitly request RAM",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="pytest arguments after --")
    options = parser.parse_args(arguments)
    pytest_args = options.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    environment = os.environ.copy()
    # Explicit --cache-root opts into managing both caches. Otherwise externally
    # supplied cache paths keep their normal ownership, contents, and lifetime.
    owned = [
        variable
        for variable in _CACHE_VARIABLES
        if options.cache_root is not None or variable not in environment
    ]
    with _supervise_console():
        return _run_with_cache(pytest_args, environment, owned, options.cache_root)


def _run_with_cache(
    arguments: Sequence[str], environment: dict[str, str], owned: list[str], cache_root: Path | None
) -> int:
    try:
        if not owned:
            return _run_pytest(arguments, environment)
        with TemporaryDirectory(prefix="piper-kernels-tests-", dir=cache_root) as root:
            _probe_cache(root)
            for variable in owned:
                environment[variable] = str(Path(root, _CACHE_VARIABLES[variable]).resolve())
            return _run_pytest(arguments, environment)
    except OSError as error:
        print(  # noqa: T201
            f"Test launcher: {error}\n"
            "Choose an executable filesystem with --cache-root, or supply compiler cache paths.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
