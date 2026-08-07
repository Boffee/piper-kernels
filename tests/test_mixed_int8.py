"""Tests for Piper's stock-Triton mixed integer-dot lowering."""

import pytest
import torch
import triton
import triton.language as tl

from piper_kernels._triton.mixed_int8 import (
    enable_uint8_int8_dot,
    rewrite_uint8_int8_dot_llvm,
    uint8_int8_dot,
)

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


def _mma(name: str, accumulators: tuple[str, str, str, str], a: str, b: str) -> str:
    arguments = ", ".join(
        [*(f"i32 {value}" for value in accumulators), f"i32 {a}", f"i32 {b}"]
    )
    return f"{_MMA_PREFIX.format(name=name)}({arguments})"


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
            _mma("%pv1", ("%pv00", "%pv01", "%pv02", "%pv03"), "%probability", "%value1"),
            "  %result = extractvalue { i32, i32, i32, i32 } %pv1, 0",
            '  %marked = tail call i32 asm sideeffect "piper_u8s8_dot_marker $0, $1;", '
            '"=r,r"(i32 %result)',
            "  %used = add i32 %marked, 1",
        ]
    )

    rewritten = rewrite_uint8_int8_dot_llvm(llvm_ir)

    qk_line = next(line for line in rewritten.splitlines() if "%qk =" in line)
    assert ".s32.s8.s8.s32" in qk_line
    assert rewritten.count(".s32.u8.s8.s32") == 2
    assert "piper_u8s8_dot_marker" not in rewritten
    assert "%used = add i32 %result, 1" in rewritten


def test_rewrite_rejects_a_marker_without_integer_mma() -> None:
    llvm_ir = "\n".join(
        [
            "  %result = add i32 %left, %right",
            '  %marked = tail call i32 asm sideeffect "piper_u8s8_dot_marker $0, $1;", '
            '"=r,r"(i32 %result)',
        ]
    )

    with pytest.raises(RuntimeError, match="did not reach an integer MMA"):
        rewrite_uint8_int8_dot_llvm(llvm_ir)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA GPU")
def test_uint8_int8_dot_is_exact_on_stock_triton() -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires MMAv2 integer tensor cores")
    torch.manual_seed(912)
    lhs = torch.randint(0, 256, (64, 64), device="cuda", dtype=torch.uint8)
    rhs = torch.randint(-128, 128, (64, 64), device="cuda", dtype=torch.int8)
    output = torch.empty((64, 64), device="cuda", dtype=torch.int32)

    enable_uint8_int8_dot()
    _uint8_int8_dot_kernel[(1,)](lhs, rhs, output, num_warps=4)

    expected = lhs.cpu().to(torch.int32) @ rhs.cpu().to(torch.int32)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
