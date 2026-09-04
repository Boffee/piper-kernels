"""GGUF quantization metadata used by device-side conversion kernels.

The integer values and block layouts are part of the GGUF/GGML file format.
Keeping this small table locally lets Piper Kernels consume tensors produced by
``gguf``, Diffusers, or another reader without taking a runtime dependency on
any one parser.
"""

from __future__ import annotations

from enum import IntEnum


class GGUFQuantizationType(IntEnum):
    """GGUF quantization types supported by the conversion kernels."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    IQ4_NL = 20
    IQ4_XS = 23
    BF16 = 30


GGUF_QUANT_SIZES: dict[GGUFQuantizationType, tuple[int, int]] = {
    GGUFQuantizationType.F32: (1, 4),
    GGUFQuantizationType.F16: (1, 2),
    GGUFQuantizationType.Q4_0: (32, 18),
    GGUFQuantizationType.Q4_1: (32, 20),
    GGUFQuantizationType.Q5_0: (32, 22),
    GGUFQuantizationType.Q5_1: (32, 24),
    GGUFQuantizationType.Q8_0: (32, 34),
    GGUFQuantizationType.Q2_K: (256, 84),
    GGUFQuantizationType.Q3_K: (256, 110),
    GGUFQuantizationType.Q4_K: (256, 144),
    GGUFQuantizationType.Q5_K: (256, 176),
    GGUFQuantizationType.Q6_K: (256, 210),
    GGUFQuantizationType.IQ4_NL: (32, 18),
    GGUFQuantizationType.IQ4_XS: (256, 136),
    GGUFQuantizationType.BF16: (1, 2),
}

SUPPORTED_GGUF_QUANT_TYPES = frozenset(GGUF_QUANT_SIZES)


def normalize_quant_type(quant_type: int) -> GGUFQuantizationType:
    """Return a supported quantization type from an integer-like enum value."""
    if isinstance(quant_type, bool):
        raise TypeError("GGUF quantization type must be an integer, not bool")
    try:
        normalized = GGUFQuantizationType(int(quant_type))
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported GGUF quantization type: {quant_type!r}") from error
    if normalized not in SUPPORTED_GGUF_QUANT_TYPES:
        raise ValueError(f"unsupported GGUF quantization type: {normalized!r}")
    return normalized


def logical_shape(
    packed_shape: tuple[int, ...],
    quant_type: int,
) -> tuple[int, ...]:
    """Recover a GGUF tensor's logical shape from its byte-level shape."""
    normalized = normalize_quant_type(quant_type)
    if not packed_shape:
        raise ValueError("GGUF packed tensors must be non-scalar")
    block_size, type_size = GGUF_QUANT_SIZES[normalized]
    row_bytes = packed_shape[-1]
    if row_bytes % type_size:
        raise ValueError(
            f"GGUF {normalized.name} row uses {row_bytes} bytes, which is not divisible "
            f"by its {type_size}-byte block storage"
        )
    return (*packed_shape[:-1], row_bytes // type_size * block_size)


__all__ = [
    "GGUF_QUANT_SIZES",
    "SUPPORTED_GGUF_QUANT_TYPES",
    "GGUFQuantizationType",
    "logical_shape",
]
