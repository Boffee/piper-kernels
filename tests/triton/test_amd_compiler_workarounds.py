"""Tests for Piper's Triton 3.9 RDNA4 compiler workarounds."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from piper_kernels._triton import amd_compiler_workarounds

_PINNED_LOAD_BARRIER = "  call void @llvm.amdgcn.sched.barrier(i32 1), !dbg !7\n"
_STAGE_BARRIER = "  call void @llvm.amdgcn.sched.barrier(i32 0)\n"
_KERNEL_IR = (
    "define amdgpu_kernel void @kernel(ptr addrspace(1) %0) {\n"
    "  %1 = load <1 x i64>, ptr addrspace(1) %0, align 8\n"
    f"{_PINNED_LOAD_BARRIER}"
    "  %2 = load <1 x i64>, ptr addrspace(1) %0, align 8\n"
    f"  tail {_PINNED_LOAD_BARRIER.lstrip()}"
    f"{_STAGE_BARRIER}"
    "  ret void\n"
    "}\n"
)
_OPT_IN = SimpleNamespace(
    llvm_fn_attrs=(
        ("target-features", "+cumode"),
        (amd_compiler_workarounds._UNPINNED_WARP_PIPELINE_MARKER, ""),
    )
)
_NO_OPT_IN = SimpleNamespace(llvm_fn_attrs=(("target-features", "+cumode"),))


@pytest.mark.parametrize(
    ("version", "fixed", "applies"),
    [
        ((3, 8), None, False),
        ((3, 9), None, True),
        ((3, 12), None, True),
        ((3, 9), (3, 10), True),
        ((3, 10), (3, 10), False),
    ],
)
def test_workarounds_apply_from_first_release_until_fixed(
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int, int],
    fixed: tuple[int, int] | None,
    applies: bool,
) -> None:
    monkeypatch.setattr(amd_compiler_workarounds, "_TRITON_VERSION", version)

    assert amd_compiler_workarounds._applies_to_installed_triton((3, 9), fixed) is applies


def test_strip_removes_pinned_load_barriers_and_keeps_stage_boundaries() -> None:
    stripped = amd_compiler_workarounds.strip_memory_order_barriers(_KERNEL_IR)

    assert "sched.barrier(i32 1)" not in stripped
    assert stripped.count(_STAGE_BARRIER) == 1
    assert stripped.count("load <1 x i64>") == 2


@pytest.mark.parametrize(("options", "unpinned"), [(_OPT_IN, True), (_NO_OPT_IN, False)])
def test_compiler_hook_unpins_only_opted_in_kernels(options: object, unpinned: bool) -> None:
    calls: list[tuple[object, ...]] = []

    def previous(*args: object) -> tuple[str, str]:
        if args:
            calls.append(args)
        return "previous-key", "previous-hash"

    hook = amd_compiler_workarounds._StageHook(previous)
    key, digest = hook()
    stages: dict[str, object] = {"llir": lambda _src, _metadata: _KERNEL_IR}

    hook("backend", stages, options, "language", None)

    assert key == f"previous-key\0{amd_compiler_workarounds._CACHE_KEY}"
    assert digest != "previous-hash"
    assert len(digest) == 64
    assert len(calls) == 1
    llvm_ir = stages["llir"](None, {})  # type: ignore[operator]
    assert ("sched.barrier(i32 1)" not in llvm_ir) is unpinned


@pytest.mark.parametrize(
    ("hip_version", "triton_version", "triton_3_9", "registered"),
    [
        ("7.15", "3.9.0", True, True),
        ("7.15", "3.10.0+gitabc", True, True),
        ("7.15", "3.8.0", False, False),
        (None, "3.9.0", True, False),
    ],
)
def test_import_applies_workarounds_only_for_triton_3_9(
    hip_version: str | None, triton_version: str, triton_3_9: bool, registered: bool
) -> None:
    script = f"""
import torch
import triton
from triton import knobs

torch.version.hip = {hip_version!r}
torch.version.cuda = None  # ROCm and CPU builds; CUDA builds also register mixed_int8
triton.__version__ = {triton_version!r}

def previous(*args):
    return 'previous', 'previous-hash'

knobs.runtime.add_stages_inspection_hook = previous
from piper_kernels._triton import amd_compiler_workarounds as workarounds

assert workarounds.USE_SCALAR_FMA is {triton_3_9!r}
assert bool(workarounds.UNPINNED_WARP_PIPELINE_ATTRS) is {triton_3_9!r}
hook = knobs.runtime.add_stages_inspection_hook
if {registered!r}:
    assert isinstance(hook, workarounds._StageHook)
    assert hook.previous is previous
    workarounds._register()
    assert knobs.runtime.add_stages_inspection_hook is hook
else:
    assert hook is previous
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
