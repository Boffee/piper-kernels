"""Native UINT8-by-INT8 dot support for stock Triton.

Triton lowers integer dot operands through a signless i8 IR type. This module lets
Triton perform its normal dot layout, scheduling, and LLVM lowering, then rewrites
only explicitly marked signed MMAv2 accumulator chains to their unsigned-left-hand-
side equivalent. The marker is removed before PTX generation, so it adds no device
instruction.

The extension is deliberately fail-closed and limited to NVIDIA's
``m16n8k32`` MMAv2 integer MMA on SM8x and consumer Blackwell SM12x. Other
Triton backends remain usable, but they cannot request this operation.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable
from typing import Any, cast

import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver

from piper_kernels._triton.targets import AcceleratorTarget

_MARKER = "piper_u8s8_dot_marker"
_SIGNED_MMA = "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"
_MIXED_MMA = "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32"
_SSA_VALUE = re.compile(r"%[-a-zA-Z$._0-9]+")
_SSA_DEFINITION = re.compile(r"\s*(%[-a-zA-Z$._0-9]+)\s*=")
_CACHE_KEY = "piper-u8s8-dot-llvm-v3"
_CACHE_HASH = hashlib.sha256(_CACHE_KEY.encode()).hexdigest()
_INSTALL_LOCK = threading.Lock()
_DOT_RESULT_OPERATIONS = ("extractvalue", "insertvalue", "bitcast")


class MixedInt8DotError(RuntimeError):
    """Base error for Piper's mixed-sign integer dot extension."""


class MixedInt8DotCompatibilityError(MixedInt8DotError):
    """The active Triton compiler or accelerator cannot provide mixed-sign dot."""


@triton.jit
def _mark_uint8_int8_dot(value):
    # This deliberately invalid PTX mnemonic makes a missing compiler hook fail during
    # assembly instead of silently executing a signed dot. The LLVM-stage rewrite removes it.
    return tl.inline_asm_elementwise(
        asm="piper_u8s8_dot_marker $0, $1;",
        constraints="=r,r",
        args=[value],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def uint8_int8_dot(lhs, rhs):
    """Emit a marked UINT8-by-INT8 dot with INT32 accumulation.

    External accumulators are intentionally unsupported: the LLVM marker traces
    Triton's tied MMA accumulator chain backward, so accepting an independent dot
    result could incorrectly rewrite that producer as mixed-sign arithmetic.
    """
    tl.static_assert(lhs.dtype == tl.uint8, "uint8_int8_dot requires a UINT8 lhs")
    tl.static_assert(rhs.dtype == tl.int8, "uint8_int8_dot requires an INT8 rhs")
    lhs_bits = lhs.to(tl.int8)
    result = tl.dot(lhs_bits, rhs, out_dtype=tl.int32)
    return _mark_uint8_int8_dot(result)


def _mma_accumulator_dependencies(line: str) -> list[str]:
    """Return only the four tied accumulator inputs of an MMAv2 inline-asm call."""
    try:
        argument_text = line.rsplit("(", maxsplit=1)[1].rsplit(")", maxsplit=1)[0]
    except IndexError as error:
        raise MixedInt8DotCompatibilityError(
            f"cannot parse integer MMA operands in Triton LLVM IR: {line}"
        ) from error
    accumulator_arguments = argument_text.split(",", maxsplit=4)[:4]
    if len(accumulator_arguments) != 4:
        raise MixedInt8DotCompatibilityError(
            f"expected four integer MMA accumulator operands in Triton LLVM IR: {line}"
        )
    return [
        value
        for argument in accumulator_arguments
        for value in _SSA_VALUE.findall(argument)
    ]


def _trace_marked_mma_chain(
    marker_input: str,
    *,
    lines: list[str],
    definitions: dict[str, int],
) -> set[int]:
    """Trace one marker through dot-result plumbing and tied MMA accumulators."""
    mma_lines: set[int] = set()
    pending = [marker_input]
    visited: set[str] = set()
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        line_index = definitions.get(value)
        if line_index is None:
            continue
        line = lines[line_index]
        if "mma.sync" in line:
            if _SIGNED_MMA not in line:
                raise MixedInt8DotCompatibilityError(
                    f"UINT8-by-INT8 marker reached an unsupported Triton MMA: {line}"
                )
            mma_lines.add(line_index)
            pending.extend(_mma_accumulator_dependencies(line))
        elif any(operation in line for operation in _DOT_RESULT_OPERATIONS):
            pending.extend(_SSA_VALUE.findall(line.partition("=")[2]))

    if not mma_lines:
        raise MixedInt8DotCompatibilityError(
            "a UINT8-by-INT8 marker did not reach Triton's supported m16n8k32 integer MMA"
        )
    return mma_lines


def rewrite_uint8_int8_dot_llvm(llvm_ir: str) -> str:
    """Rewrite marked signed MMAv2 chains to unsigned-left-hand-side MMAs."""
    lines = llvm_ir.splitlines()
    definitions: dict[str, int] = {}
    marker_inputs: dict[str, str] = {}
    for index, line in enumerate(lines):
        definition = _SSA_DEFINITION.match(line)
        if definition is not None:
            definitions[definition.group(1)] = index
        if _MARKER not in line:
            continue
        if definition is None:
            raise MixedInt8DotCompatibilityError(
                f"unexpected UINT8-by-INT8 marker in Triton LLVM IR: {line}"
            )
        values = _SSA_VALUE.findall(line)
        if len(values) != 2:
            raise MixedInt8DotCompatibilityError(
                f"unexpected UINT8-by-INT8 marker in Triton LLVM IR: {line}"
            )
        marker_inputs[values[0]] = values[1]

    if not marker_inputs:
        return llvm_ir

    mixed_mma_lines: set[int] = set()
    for marker_input in marker_inputs.values():
        mixed_mma_lines.update(
            _trace_marked_mma_chain(
                marker_input,
                lines=lines,
                definitions=definitions,
            )
        )

    for index in mixed_mma_lines:
        lines[index] = lines[index].replace(_SIGNED_MMA, _MIXED_MMA)
    for marker_result, marker_input in marker_inputs.items():
        pattern = rf"(?<![-a-zA-Z$._0-9]){re.escape(marker_result)}(?![-a-zA-Z$._0-9])"
        lines = [re.sub(pattern, marker_input, line) for line in lines]
    lines = [line for line in lines if _MARKER not in line]
    return "\n".join(lines) + "\n"


class _MixedInt8StageHook:
    def __init__(self, previous: Callable[..., tuple[str, str]] | None) -> None:
        self.previous = previous

    def _cache_identity(self) -> tuple[str, str]:
        if self.previous is None:
            return _CACHE_KEY, _CACHE_HASH
        previous_identity = self.previous()
        if (
            not isinstance(previous_identity, tuple)
            or len(previous_identity) != 2
            or not all(isinstance(item, str) for item in previous_identity)
        ):
            raise MixedInt8DotCompatibilityError(
                "the existing Triton compiler-stage hook returned an invalid cache identity"
            )
        previous_key, previous_hash = previous_identity
        combined_key = f"{previous_key}\0{_CACHE_KEY}"
        combined_hash = hashlib.sha256(
            f"{previous_hash}\0{_CACHE_HASH}".encode()
        ).hexdigest()
        return combined_key, combined_hash

    def __call__(
        self,
        backend: object | None = None,
        stages: dict[str, Callable[..., object]] | None = None,
        options: object | None = None,
        language: object | None = None,
        capability: object | None = None,
    ) -> tuple[str, str]:
        if all(item is None for item in (backend, stages, options, language, capability)):
            return self._cache_identity()
        if stages is None:
            raise MixedInt8DotCompatibilityError(
                "Triton did not provide compiler stages to the mixed-sign integer-dot hook"
            )
        if self.previous is not None:
            self.previous(backend, stages, options, language, capability)
        make_llir = stages.get("llir")
        if make_llir is None:
            raise MixedInt8DotCompatibilityError(
                "Triton's compiler pipeline does not expose an LLVM IR stage"
            )
        stages["llir"] = lambda src, metadata: rewrite_uint8_int8_dot_llvm(
            cast(str, make_llir(src, metadata))
        )
        return self._cache_identity()


def _validate_target(target: object | None) -> None:
    if target is None:
        raise MixedInt8DotCompatibilityError(
            "native UINT8-by-INT8 dot requires an active NVIDIA target"
        )
    backend = getattr(target, "backend", None)
    if backend != "cuda":
        raise MixedInt8DotCompatibilityError(
            f"native UINT8-by-INT8 dot requires NVIDIA MMAv2, got backend {backend!r}"
        )
    architecture = getattr(target, "arch", None)
    accelerator_target = AcceleratorTarget.from_compiler_target(target)
    if not accelerator_target.supports_uint8_int8_mma:
        raise MixedInt8DotCompatibilityError(
            "native UINT8-by-INT8 dot requires NVIDIA SM8x or consumer Blackwell SM12x, "
            f"got {architecture!r}"
        )


def install_uint8_int8_dot_hook() -> None:
    """Install the idempotent compiler-stage hook used by :func:`uint8_int8_dot`.

    Call this before launching a kernel that uses :func:`uint8_int8_dot`. Installing
    the hook changes Triton's explicit cache identity and does not alter kernels with
    no marked mixed-sign dot.
    """
    _validate_target(driver.active.get_current_target())
    runtime_knobs = cast(Any, knobs.runtime)
    if not hasattr(runtime_knobs, "add_stages_inspection_hook"):
        raise MixedInt8DotCompatibilityError(
            f"native UINT8-by-INT8 dot requires Triton compiler-stage hooks; "
            f"installed Triton is {triton.__version__}"
        )
    with _INSTALL_LOCK:
        current = runtime_knobs.add_stages_inspection_hook
        if isinstance(current, _MixedInt8StageHook):
            return
        if current is not None and not callable(current):
            raise MixedInt8DotCompatibilityError(
                "Triton's existing compiler-stage hook is not callable"
            )
        previous = cast(Callable[..., tuple[str, str]] | None, current)
        runtime_knobs.add_stages_inspection_hook = _MixedInt8StageHook(previous)
