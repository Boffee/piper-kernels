"""Resource usage of offline-compiled NVIDIA kernels."""

import io
import re
import subprocess
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, CompiledKernel

# Production attention schedules sit at the 255-register limit, so a compiler
# change that keeps one more tile live spills heavily. Triton 3.9 did this to
# SageAttention2++ (160-216 bytes, up to 36% slower); production frames are at
# most 104 bytes.
ATTENTION_STACK_BUDGET_BYTES = 128

# Gluon projection programs also run at the 255-register limit. Their epilogues may
# spill a few words outside the K loop; spills inside it, or of whole epilogue vectors,
# made masked SM89 projections 6-41% slower at 32K H3 rows.
PROJECTION_SPILL_STORE_BUDGET_BYTES = 32


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


def compile_with_spill_stores(
    source: ASTSource, target: GPUTarget, options: dict[str, object]
) -> tuple[CompiledKernel, int]:
    """Compile one kernel and return it with the bytes of spill stores ptxas reports.

    The compilation bypasses Triton's cache so that ptxas runs and logs. Unlike
    ``stack_frame_bytes``, this needs no cuobjdump, which triton-windows omits, and it
    ignores stack frames that ptxas reserves without spilling into them.
    """
    log = io.StringIO()
    with triton.knobs.compilation.scope(), triton.knobs.nvidia.scope(), redirect_stdout(log):
        triton.knobs.compilation.always_compile = True
        triton.knobs.nvidia.dump_ptxas_log = True
        compiled = triton.compile(source, target=target, options=options)
    stores = [int(value) for value in re.findall(r"(\d+) bytes spill stores", log.getvalue())]
    assert len(stores) == 1, log.getvalue()
    return compiled, stores[0]
