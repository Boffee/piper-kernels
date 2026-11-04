"""Resource usage of offline-compiled NVIDIA kernels."""

import re
import subprocess
import tempfile
from pathlib import Path

import triton

# Production attention schedules sit at the 255-register limit, so a compiler
# change that keeps one more tile live spills heavily. Triton 3.9 did this to
# SageAttention2++ (160-216 bytes, up to 36% slower); production frames are at
# most 104 bytes.
ATTENTION_STACK_BUDGET_BYTES = 128


def stack_frame_bytes(cubin: bytes) -> int:
    """Return the per-thread stack frame of a single-kernel cubin, which holds spills."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "kernel.cubin"
        path.write_bytes(cubin)
        report = subprocess.run(
            [triton.knobs.nvidia.cuobjdump.path, "--dump-resource-usage", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    frames = [int(value) for value in re.findall(r"\bSTACK:(\d+)", report)]
    assert len(frames) == 1, report
    return frames[0]
