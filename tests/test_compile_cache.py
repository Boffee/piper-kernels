"""Exercise cache ownership across real pytest controller/worker lifecycles."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("workers", "outcome", "exit_code"),
    [(0, "pass", 0), (2, "pass", 0), (2, "fail", 1), (0, "interrupt", 2)],
)
def test_compile_cache_cleanup(tmp_path, workers, outcome, exit_code):
    shutil.copyfile(Path(__file__).with_name("conftest.py"), tmp_path / "conftest.py")
    inherited = tmp_path / "existing-cache"
    inherited.mkdir()
    sentinel = inherited / "keep"
    sentinel.write_text("another application's cache")
    (tmp_path / "test_probe.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import pytest\n"
        "@pytest.mark.parametrize('index', range(2))\n"
        "def test_probe(index):\n"
        "    triton = Path(os.environ['TRITON_CACHE_DIR'])\n"
        "    inductor = Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])\n"
        "    assert triton.parent == inductor.parent\n"
        "    for cache in (triton, inductor):\n"
        "        cache.mkdir(exist_ok=True)\n"
        "        (cache / str(index)).write_text('compiled artifact')\n"
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
    if Path("/dev/shm").is_dir() and os.access("/dev/shm", os.W_OK):
        assert root.parent == Path("/dev/shm")
    assert not root.exists()
    assert sentinel.read_text() == "another application's cache"
