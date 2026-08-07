"""Tests for stock-Triton mixed-sign integer-dot lowering."""

import inspect
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from lib.triton_inspection import compiled_artifact

from piper_kernels._triton import mixed_int8
from piper_kernels._triton.mixed_int8 import (
    MixedInt8DotCompatibilityError,
    install_uint8_int8_dot_hook,
    rewrite_uint8_int8_dot_llvm,
    uint8_int8_dot,
)

_NVIDIA_GPU_AVAILABLE = torch.cuda.is_available() and torch.version.cuda is not None

_MMA_PREFIX = (
    '  {name} = tail call {{ i32, i32, i32, i32 }} asm sideeffect '
    '"mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 '
    '{{ $0, $1, $2, $3 }}, {{ $8, $9, $10, $11 }}, {{ $12, $13 }}, '
    '{{ $4, $5, $6, $7 }};", "=r,=r,=r,=r,0,1,2,3,r,r,r,r,r,r"'
)


@triton.jit
def _uint8_int8_dot_kernel(lhs_ptr, rhs_ptr, output_ptr):
    offsets_m = tl.arange(0, 64)
    offsets_n = tl.arange(0, 64)
    offsets_k = tl.arange(0, 64)
    lhs = tl.load(lhs_ptr + offsets_m[:, None] * 64 + offsets_k[None, :])
    rhs = tl.load(rhs_ptr + offsets_k[:, None] * 64 + offsets_n[None, :])
    output = uint8_int8_dot(lhs, rhs)
    tl.store(output_ptr + offsets_m[:, None] * 64 + offsets_n[None, :], output)


@triton.jit
def _mixed_and_signed_dot_kernel(
    unsigned_lhs_ptr,
    signed_lhs_ptr,
    rhs_ptr,
    mixed_output_ptr,
    signed_output_ptr,
):
    offsets_m = tl.arange(0, 64)
    offsets_n = tl.arange(0, 64)
    offsets_k = tl.arange(0, 64)
    unsigned_lhs = tl.load(
        unsigned_lhs_ptr + offsets_m[:, None] * 64 + offsets_k[None, :]
    )
    signed_lhs = tl.load(
        signed_lhs_ptr + offsets_m[:, None] * 64 + offsets_k[None, :]
    )
    rhs = tl.load(rhs_ptr + offsets_k[:, None] * 64 + offsets_n[None, :])
    mixed_output = uint8_int8_dot(unsigned_lhs, rhs)
    signed_output = tl.dot(signed_lhs, rhs, out_dtype=tl.int32)
    output_offsets = offsets_m[:, None] * 64 + offsets_n[None, :]
    tl.store(mixed_output_ptr + output_offsets, mixed_output)
    tl.store(signed_output_ptr + output_offsets, signed_output)


def _mma(name: str, accumulators: tuple[str, str, str, str], a: str, b: str) -> str:
    arguments = ", ".join(
        [*(f"i32 {value}" for value in accumulators), f"i32 {a}", f"i32 {b}"]
    )
    return f"{_MMA_PREFIX.format(name=name)}({arguments})"


def _marker(result: str, value: str) -> str:
    return (
        f'  {result} = tail call i32 asm sideeffect "piper_u8s8_dot_marker $0, $1;", '
        f'"=r,r"(i32 {value})'
    )


def test_rewrite_leaves_unmarked_llvm_unchanged() -> None:
    llvm_ir = "define void @unmarked() {\n  ret void\n}\n"

    assert rewrite_uint8_int8_dot_llvm(llvm_ir) is llvm_ir


def test_mixed_dot_api_excludes_external_accumulators() -> None:
    parameters = inspect.signature(uint8_int8_dot.fn).parameters

    assert tuple(parameters) == ("lhs", "rhs")


def test_rewrite_marks_only_the_dot_accumulator_chain() -> None:
    llvm_ir = "\n".join(
        [
            _mma("%qk", ("0", "0", "0", "0"), "%query", "%key"),
            "  %qk0 = extractvalue { i32, i32, i32, i32 } %qk, 0",
            "  %external_accumulator = add i32 %qk0, 1",
            # Both an A operand and an external accumulator depend on QK. Dot-chain
            # traversal must not cross the arithmetic that prepared either operand.
            _mma(
                "%pv0",
                ("%external_accumulator", "0", "0", "0"),
                "%qk0",
                "%value0",
            ),
            "  %pv00 = extractvalue { i32, i32, i32, i32 } %pv0, 0",
            "  %pv01 = extractvalue { i32, i32, i32, i32 } %pv0, 1",
            "  %pv02 = extractvalue { i32, i32, i32, i32 } %pv0, 2",
            "  %pv03 = extractvalue { i32, i32, i32, i32 } %pv0, 3",
            _mma("%pv1", ("%pv00", "%pv01", "%pv02", "%pv03"), "%p", "%value1"),
            "  %result = extractvalue { i32, i32, i32, i32 } %pv1, 0",
            _marker("%marked", "%result"),
            "  %used = add i32 %marked, 1",
        ]
    )

    rewritten = rewrite_uint8_int8_dot_llvm(llvm_ir)

    qk_line = next(line for line in rewritten.splitlines() if "%qk =" in line)
    assert ".s32.s8.s8.s32" in qk_line
    assert rewritten.count(".s32.u8.s8.s32") == 2
    assert "piper_u8s8_dot_marker" not in rewritten
    assert "%used = add i32 %result, 1" in rewritten


def test_rewrite_rejects_each_marker_without_integer_mma() -> None:
    llvm_ir = "\n".join(
        [
            _mma("%pv", ("0", "0", "0", "0"), "%p", "%value"),
            "  %result = extractvalue { i32, i32, i32, i32 } %pv, 0",
            _marker("%supported", "%result"),
            "  %not_a_dot = add i32 %left, %right",
            _marker("%unsupported", "%not_a_dot"),
        ]
    )

    with pytest.raises(MixedInt8DotCompatibilityError, match="did not reach"):
        rewrite_uint8_int8_dot_llvm(llvm_ir)


def test_rewrite_rejects_an_unsupported_integer_mma_shape() -> None:
    llvm_ir = "\n".join(
        [
            _mma("%pv", ("0", "0", "0", "0"), "%p", "%value").replace(
                "m16n8k32", "m8n8k16"
            ),
            "  %result = extractvalue { i32, i32, i32, i32 } %pv, 0",
            _marker("%marked", "%result"),
        ]
    )

    with pytest.raises(MixedInt8DotCompatibilityError, match="unsupported Triton MMA"):
        rewrite_uint8_int8_dot_llvm(llvm_ir)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (None, "active NVIDIA target"),
        (SimpleNamespace(backend="hip", arch="gfx1201"), "got backend 'hip'"),
        (SimpleNamespace(backend="cuda", arch=75), "capability 8.0 or newer"),
        (SimpleNamespace(backend="cuda", arch="sm80"), "capability 8.0 or newer"),
    ],
)
def test_target_validation_fails_clearly(target: object, message: str) -> None:
    with pytest.raises(MixedInt8DotCompatibilityError, match=message):
        mixed_int8._validate_target(target)


def test_compiler_hook_composes_cache_identity_and_stage_order() -> None:
    calls: list[tuple[object, ...]] = []

    def previous(*args: object) -> tuple[str, str]:
        if not args:
            return "previous-key", "previous-hash"
        calls.append(args)
        return "previous-key", "previous-hash"

    hook = mixed_int8._MixedInt8StageHook(previous)
    key, digest = hook()
    stages: dict[str, object] = {"llir": lambda _src, _metadata: "unmarked"}

    hook("backend", stages, "options", "language", 120)

    assert key == f"previous-key\0{mixed_int8._CACHE_KEY}"
    assert digest != "previous-hash"
    assert len(digest) == 64
    assert len(calls) == 1
    assert callable(stages["llir"])
    assert stages["llir"](None, {}) == "unmarked"


@pytest.mark.gpu
@pytest.mark.skipif(not _NVIDIA_GPU_AVAILABLE, reason="requires an NVIDIA GPU")
def test_uint8_int8_dot_is_exact_above_signed_range() -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires MMAv2 integer tensor cores")
    lhs = torch.arange(64 * 64, device="cuda", dtype=torch.int64)
    lhs = lhs.remainder(256).to(torch.uint8).reshape(64, 64)
    generator = torch.Generator(device="cuda").manual_seed(912)
    rhs = torch.randint(
        -128,
        128,
        (64, 64),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    output = torch.empty((64, 64), device="cuda", dtype=torch.int32)

    assert torch.any(lhs > 127)
    install_uint8_int8_dot_hook()
    _uint8_int8_dot_kernel[(1,)](lhs, rhs, output, num_warps=4)

    expected = lhs.cpu().to(torch.int32) @ rhs.cpu().to(torch.int32)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _NVIDIA_GPU_AVAILABLE, reason="requires an NVIDIA GPU")
def test_generated_ptx_changes_only_the_marked_dot() -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires MMAv2 integer tensor cores")
    unsigned_lhs = torch.zeros((64, 64), device="cuda", dtype=torch.uint8)
    signed_lhs = torch.zeros((64, 64), device="cuda", dtype=torch.int8)
    rhs = torch.zeros((64, 64), device="cuda", dtype=torch.int8)
    mixed_output = torch.empty((64, 64), device="cuda", dtype=torch.int32)
    signed_output = torch.empty_like(mixed_output)

    install_uint8_int8_dot_hook()
    _mixed_and_signed_dot_kernel[(1,)](
        unsigned_lhs,
        signed_lhs,
        rhs,
        mixed_output,
        signed_output,
        num_warps=4,
    )
    ptx = compiled_artifact(_mixed_and_signed_dot_kernel, "ptx")

    assert "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32" in ptx
    assert "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32" in ptx
    assert "piper_u8s8_dot_marker" not in ptx
