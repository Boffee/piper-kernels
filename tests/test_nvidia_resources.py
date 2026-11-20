"""Caching and invalidation tests for tests/_nvidia_resources.py."""

import json
import subprocess

import pytest

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402
from _nvidia_resources import _SPILL_STORES_SIDECAR, compile_with_spill_stores  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402


@triton.jit
def _tiny_add_kernel(x_ptr, y_ptr, block_size: tl.constexpr, extra_live_vars: tl.constexpr):
    offsets = tl.arange(0, block_size)
    acc = tl.load(x_ptr + offsets)
    if extra_live_vars:
        # Hold many distinct loaded vectors live simultaneously so a tight maxnreg cap spills.
        a = tl.load(x_ptr + offsets + block_size)
        b = tl.load(x_ptr + offsets + 2 * block_size)
        c = tl.load(x_ptr + offsets + 3 * block_size)
        d = tl.load(x_ptr + offsets + 4 * block_size)
        e = tl.load(x_ptr + offsets + 5 * block_size)
        f = tl.load(x_ptr + offsets + 6 * block_size)
        g = tl.load(x_ptr + offsets + 7 * block_size)
        h = tl.load(x_ptr + offsets + 8 * block_size)
        acc = acc + a * b + c * d + e * f + g * h + a * c + b * d + e * g + f * h
    tl.store(y_ptr + offsets, acc)


def _source(*, block_size: int = 32, extra_live_vars: bool = False) -> ASTSource:
    return ASTSource(
        _tiny_add_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={"block_size": block_size, "extra_live_vars": extra_live_vars},
    )


def test_compile_with_spill_stores_caches_sidecar_and_invalidates_on_change(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    target = GPUTarget("cuda", 120, 32)
    source = _source()

    events: list[bool] = []
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.listener = lambda **kwargs: events.append(
            bool(kwargs["cache_hit"])
        )

        # Cold compile: misses Triton cache, runs ptxas once, and writes the sidecar.
        compiled, spill_stores = compile_with_spill_stores(source, target, {"num_warps": 4})
        assert events == [False]
        assert spill_stores == 0

        sidecar = tmp_path / compiled.hash / _SPILL_STORES_SIDECAR
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text(encoding="utf-8")) == {
            "triton_version": triton.__version__,
            "spill_store_bytes": 0,
        }

        # Warm repeat: hits Triton cache + sidecar with zero ptxas subprocesses.
        def _forbid_subprocess(*args, **kwargs):
            raise AssertionError("warm compile_with_spill_stores must not invoke ptxas")

        with monkeypatch.context() as patched:
            patched.setattr(subprocess, "run", _forbid_subprocess)
            cached_compiled, cached_spills = compile_with_spill_stores(
                source, target, {"num_warps": 4}
            )
        assert events == [False, True]
        assert cached_compiled.hash == compiled.hash
        assert cached_spills == spill_stores

        # Missing sidecar (e.g., cache entry populated by plain triton.compile): recovers once.
        sidecar.unlink()
        recovered_compiled, recovered_spills = compile_with_spill_stores(
            source, target, {"num_warps": 4}
        )
        assert events == [False, True, True, False]
        assert recovered_compiled.hash == compiled.hash
        assert recovered_spills == spill_stores
        assert sidecar.is_file()

        # Corrupt or version-mismatched sidecar also triggers a clean single recovery.
        sidecar.write_text(
            json.dumps({"triton_version": "0.0.0-stale", "spill_store_bytes": 999}),
            encoding="utf-8",
        )
        _, refreshed_spills = compile_with_spill_stores(source, target, {"num_warps": 4})
        assert refreshed_spills == spill_stores
        assert json.loads(sidecar.read_text(encoding="utf-8"))["triton_version"] == (
            triton.__version__
        )

        # Modified kernel / register cap gets a distinct hash and reports exact spill bytes.
        spilling_source = _source(block_size=2048, extra_live_vars=True)
        spilling_compiled, spilling_bytes = compile_with_spill_stores(
            spilling_source, target, {"num_warps": 4, "maxnreg": 24}
        )
        assert spilling_compiled.hash != compiled.hash
        assert spilling_bytes > 0
        spilling_sidecar = tmp_path / spilling_compiled.hash / _SPILL_STORES_SIDECAR
        assert json.loads(spilling_sidecar.read_text(encoding="utf-8"))["spill_store_bytes"] == (
            spilling_bytes
        )

        # Warm hit on the spilling kernel also skips ptxas and returns the exact non-zero count.
        with monkeypatch.context() as patched:
            patched.setattr(subprocess, "run", _forbid_subprocess)
            warm_spilling, warm_spilling_bytes = compile_with_spill_stores(
                spilling_source, target, {"num_warps": 4, "maxnreg": 24}
            )
        assert warm_spilling.hash == spilling_compiled.hash
        assert warm_spilling_bytes == spilling_bytes
