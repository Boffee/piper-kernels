"""Native UINT8-by-INT8 dot support for stock Triton.

Triton currently lowers integer dot operands through a signless i8 IR type. This module lets
Triton perform its normal dot layout, scheduling, and LLVM lowering, then rewrites only explicitly
marked signed MMA chains to their unsigned-left-hand-side equivalent. The marker is removed before
PTX generation, so it adds no device instructions. This extension targets NVIDIA MMAv2 kernels;
other Triton backends remain usable but cannot request this operation.
"""

# Triton's JIT values intentionally omit Python annotations.
# ruff: noqa: ANN001, ANN202, PLR0912

import hashlib
import re
import threading
from collections.abc import Callable
from typing import Any, cast

import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver

_MARKER = "piper_u8s8_dot_marker"
_SIGNED_MMA_SUFFIX = ".s32.s8.s8.s32"
_MIXED_MMA_SUFFIX = ".s32.u8.s8.s32"
_SSA_VALUE = re.compile(r"%[-a-zA-Z$._0-9]+")
_SSA_DEFINITION = re.compile(r"\s*(%[-a-zA-Z$._0-9]+)\s*=")
_CACHE_KEY = "piper-u8s8-dot-llvm-v2"
_CACHE_HASH = hashlib.sha256(_CACHE_KEY.encode()).hexdigest()
_INSTALL_LOCK = threading.Lock()
_DOT_RESULT_OPERATIONS = ("extractvalue", "insertvalue", "bitcast")


@triton.jit
def _mark_uint8_int8_dot(value):
    # This deliberately invalid PTX mnemonic makes a missing compiler hook fail during assembly
    # instead of silently executing a signed dot. The LLVM-stage rewrite removes the marker.
    return tl.inline_asm_elementwise(
        asm="piper_u8s8_dot_marker $0, $1;",
        constraints="=r,r",
        args=[value],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def uint8_int8_dot(lhs, rhs, accumulator=None):
    """Emit a marked UINT8-by-INT8 dot with an INT32 accumulator."""
    lhs_bits = lhs.to(tl.int8)
    if accumulator is None:
        result = tl.dot(lhs_bits, rhs, out_dtype=tl.int32)
    else:
        result = tl.dot(lhs_bits, rhs, accumulator, out_dtype=tl.int32)
    return _mark_uint8_int8_dot(result)


def _mma_accumulator_dependencies(line: str) -> list[str]:
    """Return only the four tied accumulator inputs of an MMAv2 inline-asm call."""
    try:
        argument_text = line.rsplit('"(', maxsplit=1)[1].rsplit(")", maxsplit=1)[0]
    except IndexError as error:
        raise RuntimeError(f"cannot parse integer MMA operands: {line}") from error
    accumulator_arguments = argument_text.split(",", maxsplit=4)[:4]
    return [value for argument in accumulator_arguments for value in _SSA_VALUE.findall(argument)]


def rewrite_uint8_int8_dot_llvm(llvm_ir: str) -> str:
    """Rewrite marked signed MMAv2 chains to unsigned-left-hand-side MMAs."""
    lines = llvm_ir.splitlines()
    definitions: dict[str, int] = {}
    marker_inputs: dict[str, str] = {}
    for index, line in enumerate(lines):
        definition = _SSA_DEFINITION.match(line)
        if definition is not None:
            definitions[definition.group(1)] = index
        if _MARKER in line and definition is not None:
            values = _SSA_VALUE.findall(line)
            if len(values) != 2:
                raise RuntimeError(f"unexpected UINT8-by-INT8 marker: {line}")
            marker_inputs[values[0]] = values[1]

    mixed_mma_lines: set[int] = set()
    pending = list(marker_inputs.values())
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
            if _SIGNED_MMA_SUFFIX not in line:
                raise RuntimeError(f"UINT8-by-INT8 marker reached unsupported MMA: {line}")
            mixed_mma_lines.add(line_index)
            pending.extend(_mma_accumulator_dependencies(line))
        elif any(operation in line for operation in _DOT_RESULT_OPERATIONS):
            pending.extend(_SSA_VALUE.findall(line.partition("=")[2]))

    if marker_inputs and not mixed_mma_lines:
        raise RuntimeError("UINT8-by-INT8 markers did not reach an integer MMA")
    for index in mixed_mma_lines:
        lines[index] = lines[index].replace(_SIGNED_MMA_SUFFIX, _MIXED_MMA_SUFFIX)
    for marker_result, marker_input in marker_inputs.items():
        pattern = rf"(?<![-a-zA-Z$._0-9]){re.escape(marker_result)}(?![-a-zA-Z$._0-9])"
        lines = [re.sub(pattern, marker_input, line) for line in lines]
    lines = [line for line in lines if _MARKER not in line]
    return "\n".join(lines) + "\n"


class _MixedInt8StageHook:
    def __init__(self, previous: Callable[..., tuple[str, str]] | None) -> None:
        self.previous = previous

    def __call__(
        self,
        backend: object | None = None,
        stages: dict[str, Callable[..., object]] | None = None,
        options: object | None = None,
        language: object | None = None,
        capability: object | None = None,
    ) -> tuple[str, str]:
        if all(item is None for item in (backend, stages, options, language, capability)):
            previous_key = self.previous() if self.previous is not None else ("", "")
            combined = f"{previous_key!r}:{_CACHE_HASH}"
            return _CACHE_KEY, hashlib.sha256(combined.encode()).hexdigest()
        if stages is None:
            raise RuntimeError("Triton did not provide compiler stages to the mixed-INT8 hook")
        if self.previous is not None:
            self.previous(backend, stages, options, language, capability)
        make_llir = stages["llir"]
        stages["llir"] = lambda src, metadata: rewrite_uint8_int8_dot_llvm(
            cast(str, make_llir(src, metadata))
        )
        return self()


def enable_uint8_int8_dot() -> None:
    """Install the idempotent compiler-stage hook used by :func:`uint8_int8_dot`."""
    target = driver.active.get_current_target()
    if target is None:
        raise RuntimeError("native UINT8-by-INT8 dot requires an active NVIDIA target")
    if target.backend != "cuda":
        raise RuntimeError(
            f"native UINT8-by-INT8 dot requires NVIDIA MMAv2, got {target.backend!r}"
        )
    if not isinstance(target.arch, int) or target.arch < 80:
        raise RuntimeError(
            "native UINT8-by-INT8 dot requires NVIDIA compute capability 8.0 or newer, "
            f"got {target.arch!r}"
        )
    runtime_knobs = cast(Any, knobs.runtime)
    if not hasattr(runtime_knobs, "add_stages_inspection_hook"):
        raise RuntimeError("native UINT8-by-INT8 dot requires Triton compiler-stage hooks")
    with _INSTALL_LOCK:
        current = runtime_knobs.add_stages_inspection_hook
        if isinstance(current, _MixedInt8StageHook):
            return
        runtime_knobs.add_stages_inspection_hook = _MixedInt8StageHook(current)
