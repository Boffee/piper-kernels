"""Tests for Piper's RDNA4 Triton compiler workarounds."""

import subprocess
import sys
from collections.abc import Callable
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
_FENCE_IR = (
    "define amdgpu_kernel void @kernel() {\n"
    '  fence syncscope("workgroup") release, !dbg !7\n'
    "  call void @llvm.amdgcn.s.barrier()\n"
    '  fence syncscope("workgroup") acquire\n'
    '  fence syncscope("workgroup") acquire, !mmra !3\n'
    '  fence syncscope("agent") release\n'
    "  ret void\n"
    "}\n"
    '!3 = !{!"amdgpu-synchronize-as", !"global"}\n'
    "!7 = !DILocation(line: 1, scope: !3)\n"
)
_OPT_IN = SimpleNamespace(
    llvm_fn_attrs=(
        ("target-features", "+cumode"),
        (amd_compiler_workarounds._UNPINNED_WARP_PIPELINE_MARKER, ""),
    )
)
_LDS_FENCE_OPT_IN = SimpleNamespace(
    llvm_fn_attrs=((amd_compiler_workarounds._LDS_ONLY_FENCE_MARKER, ""),)
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


def test_lds_only_fences_tag_untagged_workgroup_fences() -> None:
    tagged = amd_compiler_workarounds.restrict_workgroup_fences_to_lds(_FENCE_IR)

    assert 'fence syncscope("workgroup") release, !dbg !7, !mmra !8\n' in tagged
    assert 'fence syncscope("workgroup") acquire, !mmra !8\n' in tagged
    assert 'fence syncscope("workgroup") acquire, !mmra !3\n' in tagged
    assert 'fence syncscope("agent") release\n' in tagged
    assert tagged.endswith('!8 = !{!"amdgpu-synchronize-as", !"local"}\n')


def test_lds_only_fences_leave_kernels_without_untagged_fences_unchanged() -> None:
    assert amd_compiler_workarounds.restrict_workgroup_fences_to_lds(_KERNEL_IR) == _KERNEL_IR
    already_tagged = _FENCE_IR.replace(
        'fence syncscope("workgroup") release, !dbg !7',
        'fence syncscope("workgroup") release, !dbg !7, !mmra !3',
    ).replace('fence syncscope("workgroup") acquire\n', "")

    assert amd_compiler_workarounds.restrict_workgroup_fences_to_lds(already_tagged) == (
        already_tagged
    )


@pytest.mark.parametrize("installed_triton_needs_it", [False, True])
@pytest.mark.parametrize("requested", [False, True])
def test_lds_only_fence_options_follow_plan_and_installed_triton(
    monkeypatch: pytest.MonkeyPatch, installed_triton_needs_it: bool, requested: bool
) -> None:
    attributes = ((amd_compiler_workarounds._LDS_ONLY_FENCE_MARKER, ""),)
    monkeypatch.setattr(
        amd_compiler_workarounds,
        "LDS_ONLY_FENCE_ATTRS",
        attributes if installed_triton_needs_it else (),
    )

    options = amd_compiler_workarounds.lds_only_fence_options(requested)

    # Plans that do not ask, and Triton releases without the issue, pass no option at all.
    expected = {"llvm_fn_attrs": attributes} if requested and installed_triton_needs_it else {}
    assert options == expected


@pytest.mark.parametrize(
    ("options", "llvm_ir", "rewritten"),
    [
        (_OPT_IN, _KERNEL_IR, lambda ir: "sched.barrier(i32 1)" not in ir),
        (_NO_OPT_IN, _KERNEL_IR, lambda ir: "sched.barrier(i32 1)" in ir),
        (_LDS_FENCE_OPT_IN, _FENCE_IR, lambda ir: 'amdgpu-synchronize-as", !"local' in ir),
        (_NO_OPT_IN, _FENCE_IR, lambda ir: ir == _FENCE_IR),
    ],
)
def test_compiler_hook_rewrites_only_opted_in_kernels(
    options: object, llvm_ir: str, rewritten: Callable[[str], bool]
) -> None:
    calls: list[tuple[object, ...]] = []

    def previous(*args: object) -> tuple[str, str]:
        if args:
            calls.append(args)
        return "previous-key", "previous-hash"

    hook = amd_compiler_workarounds._StageHook(previous)
    key, digest = hook()
    stages: dict[str, object] = {"llir": lambda _src, _metadata: llvm_ir}

    hook("backend", stages, options, "language", None)

    assert key == f"previous-key\0{amd_compiler_workarounds._CACHE_KEY}"
    assert digest != "previous-hash"
    assert len(digest) == 64
    assert len(calls) == 1
    assert rewritten(stages["llir"](None, {}))  # type: ignore[operator]


@pytest.mark.parametrize(
    ("hip_version", "triton_version", "triton_3_9", "lds_only_fences", "registered"),
    [
        ("7.15", "3.9.0", True, True, True),
        ("7.15", "3.10.0+gitabc", True, True, True),
        ("7.15", "3.8.0", False, True, True),
        ("7.15", "3.7.0", False, False, False),
        (None, "3.9.0", True, True, False),
    ],
)
def test_import_applies_workarounds_by_triton_release(
    hip_version: str | None,
    triton_version: str,
    triton_3_9: bool,
    lds_only_fences: bool,
    registered: bool,
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
assert bool(workarounds.LDS_ONLY_FENCE_ATTRS) is {lds_only_fences!r}
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
