"""Resource usage of offline-compiled NVIDIA kernels."""

import io
import json
import os
import re
import subprocess
import tempfile
import uuid
from contextlib import redirect_stdout, suppress
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
# made SM89 projections 4-41% slower at 32K H3 rows.
PROJECTION_SPILL_STORE_BUDGET_BYTES = 32

_SPILL_STORES_SIDECAR = "ptxas_spill_stores.json"
_SPILL_STORES_PATTERN = re.compile(r"(\d+) bytes spill stores")


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


def _parse_spill_stores(ptxas_log: str) -> int:
    stores = [int(value) for value in _SPILL_STORES_PATTERN.findall(ptxas_log)]
    assert len(stores) == 1, ptxas_log
    return stores[0]


def _sidecar_path(kernel_hash: str) -> Path | None:
    cache_dir = triton.knobs.cache.dir
    if not cache_dir:
        return None
    return Path(cache_dir) / kernel_hash / _SPILL_STORES_SIDECAR


def _read_cached_spill_stores(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("triton_version") != triton.__version__:
        return None
    value = payload.get("spill_store_bytes")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _write_cached_spill_stores(path: Path | None, spill_stores: int) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        tmp_path.write_text(
            json.dumps(
                {
                    "triton_version": triton.__version__,
                    "spill_store_bytes": spill_stores,
                }
            ),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        # Concurrent workers can race to publish the same sidecar (notably on Windows).
        if _read_cached_spill_stores(path) != spill_stores:
            raise


def _compile_capturing_ptxas_log(
    source: ASTSource,
    target: GPUTarget,
    options: dict[str, object],
    *,
    force_compile: bool,
) -> tuple[CompiledKernel, str]:
    log = io.StringIO()
    with triton.knobs.compilation.scope(), triton.knobs.nvidia.scope(), redirect_stdout(log):
        if force_compile:
            triton.knobs.compilation.always_compile = True
        triton.knobs.nvidia.dump_ptxas_log = True
        compiled = triton.compile(source, target=target, options=options)
    return compiled, log.getvalue()


def compile_with_spill_stores(
    source: ASTSource, target: GPUTarget, options: dict[str, object]
) -> tuple[CompiledKernel, int]:
    """Compile one kernel and return it with the bytes of spill stores ptxas reports.

    Reuses Triton's persistent cache and stores the parsed ``ptxas -v`` spill-store
    count in an atomic JSON sidecar under ``TRITON_CACHE_DIR/<compiled.hash>/``. Only
    a Triton cache miss or missing sidecar invokes ``ptxas``. Unlike
    ``stack_frame_bytes``, this needs no cuobjdump, which triton-windows omits, and it
    ignores stack frames that ptxas reserves without spilling into them.
    """
    compiled, ptxas_log = _compile_capturing_ptxas_log(source, target, options, force_compile=False)
    sidecar = _sidecar_path(compiled.hash)
    if ptxas_log:
        spill_stores = _parse_spill_stores(ptxas_log)
        _write_cached_spill_stores(sidecar, spill_stores)
        return compiled, spill_stores

    cached = _read_cached_spill_stores(sidecar)
    if cached is not None:
        return compiled, cached

    # Triton hit a cached binary whose spill sidecar is missing or stale (for example,
    # populated by a plain triton.compile call); compile once to capture ptxas output.
    compiled, ptxas_log = _compile_capturing_ptxas_log(source, target, options, force_compile=True)
    spill_stores = _parse_spill_stores(ptxas_log)
    _write_cached_spill_stores(_sidecar_path(compiled.hash), spill_stores)
    return compiled, spill_stores
