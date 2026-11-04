"""Tests for removing Triton 3.9's per-memory-op warp-pipeline barriers."""

import subprocess
import sys

import pytest

from piper_kernels._triton import amd_warp_pipeline

_MEMORY_BARRIER = "  call void @llvm.amdgcn.sched.barrier(i32 1), !dbg !7\n"
_STAGE_BARRIER = "  call void @llvm.amdgcn.sched.barrier(i32 0)\n"


def _kernel_ir(name: str) -> str:
    return (
        f"define amdgpu_kernel void @{name}(ptr addrspace(1) %0) {{\n"
        "  %1 = load <1 x i64>, ptr addrspace(1) %0, align 8\n"
        f"{_MEMORY_BARRIER}"
        "  %2 = load <1 x i64>, ptr addrspace(1) %0, align 8\n"
        f"  tail {_MEMORY_BARRIER.lstrip()}"
        f"{_STAGE_BARRIER}"
        "  ret void\n"
        "}\n"
    )


@pytest.mark.parametrize("name", ["_dense_piper_kernel", "_sparse_piper_attention_kernel"])
def test_strip_removes_only_memory_order_barriers_from_piper_kernels(name: str) -> None:
    stripped = amd_warp_pipeline.strip_memory_order_barriers(_kernel_ir(name))

    assert "sched.barrier(i32 1)" not in stripped
    assert stripped.count(_STAGE_BARRIER) == 1
    assert stripped.count("load <1 x i64>") == 2


def test_strip_leaves_other_kernels_unchanged() -> None:
    llvm_ir = _kernel_ir("user_gemm_kernel")

    assert amd_warp_pipeline.strip_memory_order_barriers(llvm_ir) == llvm_ir


def test_compiler_hook_composes_cache_identity_and_stage_order() -> None:
    calls: list[tuple[object, ...]] = []

    def previous(*args: object) -> tuple[str, str]:
        if args:
            calls.append(args)
        return "previous-key", "previous-hash"

    hook = amd_warp_pipeline._StageHook(previous)
    key, digest = hook()
    stages: dict[str, object] = {"llir": lambda _src, _metadata: _kernel_ir("_dense_piper_kernel")}

    hook("backend", stages, "options", "language", None)

    assert key == f"previous-key\0{amd_warp_pipeline._CACHE_KEY}"
    assert digest != "previous-hash"
    assert len(digest) == 64
    assert len(calls) == 1
    assert "sched.barrier(i32 1)" not in stages["llir"](None, {})  # type: ignore[operator]


@pytest.mark.parametrize(
    ("hip_version", "triton_version", "registered"),
    [
        ("7.15", "3.9.0", True),
        ("7.15", "3.10.0+gitabc", True),
        ("7.15", "3.8.0", False),
        (None, "3.9.0", False),
    ],
)
def test_import_registers_hook_only_for_rocm_with_triton_3_9(
    hip_version: str | None, triton_version: str, registered: bool
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
from piper_kernels._triton import amd_warp_pipeline

hook = knobs.runtime.add_stages_inspection_hook
if {registered!r}:
    assert isinstance(hook, amd_warp_pipeline._StageHook)
    assert hook.previous is previous
    amd_warp_pipeline._register()
    assert knobs.runtime.add_stages_inspection_hook is hook
else:
    assert hook is previous
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
