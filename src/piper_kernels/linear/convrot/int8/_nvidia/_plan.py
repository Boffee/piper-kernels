"""NVIDIA ConvRot INT8 plan values, validation, and benchmark reporting."""

from dataclasses import dataclass
from typing import Literal, NamedTuple

from .._plan import LinearExecutionPlan

_FUSED_NUM_WARPS_VALUES = (1, 2, 4, 8, 16)
_ROTATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_QUANTIZATION_NUM_WARPS_VALUES = (1, 2, 4, 8)
_MATMUL_BLOCK_M_VALUES = (16, 32, 64, 128)
_MATMUL_BLOCK_N_VALUES = (16, 32, 64, 128, 256)
_MATMUL_BLOCK_K_VALUES = (32, 64, 128)
_MATMUL_NUM_WARPS_VALUES = (2, 4, 8)
_MATMUL_NUM_STAGES_VALUES = (1, 2, 3, 4)
_MATMUL_GROUP_M_VALUES = (0, 8, 16)
_MATMUL_KERNEL_VALUES = ("triton", "gluon_async_copy")


def _choices(values: tuple[int, ...]) -> str:
    *leading, last = values
    return f"{', '.join(map(str, leading))}, or {last}"


@dataclass(frozen=True, slots=True)
class NvidiaExecutionPlan(LinearExecutionPlan):
    """Explicit implementation and schedule, independent of the selecting architecture.

    The ``triton_*`` options apply only to Triton. The read-only ``matmul_*``
    properties report the selected implementation's effective behavior. Gluon's
    async-copy implementation always uses dynamic M and explicit bias FMAs,
    including in its unaligned-operand fallback.
    """

    matmul_kernel: Literal["triton", "gluon_async_copy"] = "triton"
    matmul_group_m: int = 0
    triton_specialize_m: bool = True
    triton_explicit_bias_fma: bool = False

    @property
    def matmul_specialize_m(self) -> bool:
        """Return whether the selected implementation specializes on the row count."""
        return self.matmul_kernel == "triton" and self.triton_specialize_m

    @property
    def matmul_explicit_bias_fma(self) -> bool:
        """Return whether the selected implementation uses explicit bias FMAs."""
        return self.matmul_kernel == "gluon_async_copy" or self.triton_explicit_bias_fma

    def as_dict(self) -> dict[str, int | bool | str]:
        """Report effective execution choices, omitting inactive Triton options."""
        choices = LinearExecutionPlan.as_dict(self)
        del choices["triton_specialize_m"], choices["triton_explicit_bias_fma"]
        return choices | {
            "matmul_specialize_m": self.matmul_specialize_m,
            "matmul_explicit_bias_fma": self.matmul_explicit_bias_fma,
        }

    def __post_init__(self) -> None:
        LinearExecutionPlan.__post_init__(self)
        if self.fused_num_warps not in _FUSED_NUM_WARPS_VALUES:
            raise ValueError(
                f"ConvRot fused preparation num_warps must be {_choices(_FUSED_NUM_WARPS_VALUES)}"
            )
        if self.rotation_num_warps not in _ROTATION_NUM_WARPS_VALUES:
            raise ValueError(
                f"ConvRot split rotation num_warps must be {_choices(_ROTATION_NUM_WARPS_VALUES)}"
            )
        if self.quantization_num_warps not in _QUANTIZATION_NUM_WARPS_VALUES:
            raise ValueError(
                "ConvRot split quantization num_warps must be "
                f"{_choices(_QUANTIZATION_NUM_WARPS_VALUES)}"
            )
        self._validate_matmul()

    def _validate_matmul(self) -> None:
        """Validate implementation capabilities independently of measured target defaults."""
        if self.matmul_kernel not in _MATMUL_KERNEL_VALUES:
            raise ValueError("ConvRot matmul kernel must be triton or gluon_async_copy")
        for name in ("triton_specialize_m", "triton_explicit_bias_fma"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"ConvRot {name} must be boolean")
        group_m = self.matmul_group_m
        if type(group_m) is not int or group_m not in _MATMUL_GROUP_M_VALUES:
            raise ValueError(f"ConvRot matmul group_m must be {_choices(_MATMUL_GROUP_M_VALUES)}")
        if self.matmul_kernel == "gluon_async_copy":
            # Two 64-column warp tiles and two/four 64-row warp tiles per CTA.
            if (
                self.matmul_block_m not in (128, 256)
                or self.matmul_block_n != 128
                or self.matmul_block_k != 64
                or self.matmul_num_warps != self.matmul_block_m // 32
                or self.matmul_num_stages not in (3, 4)
            ):
                raise ValueError(
                    "ConvRot async-copy GEMM requires 128x128x64/4-warps or "
                    "256x128x64/8-warps, with 3 or 4 stages"
                )
            return

        if self.matmul_block_m not in _MATMUL_BLOCK_M_VALUES:
            raise ValueError(
                f"ConvRot Triton matmul block_m must be {_choices(_MATMUL_BLOCK_M_VALUES)}"
            )
        if self.matmul_block_n not in _MATMUL_BLOCK_N_VALUES:
            raise ValueError(f"ConvRot matmul block_n must be {_choices(_MATMUL_BLOCK_N_VALUES)}")
        if self.matmul_block_k not in _MATMUL_BLOCK_K_VALUES:
            raise ValueError(f"ConvRot matmul block_k must be {_choices(_MATMUL_BLOCK_K_VALUES)}")
        if self.matmul_num_warps not in _MATMUL_NUM_WARPS_VALUES:
            raise ValueError(
                f"ConvRot matmul num_warps must be {_choices(_MATMUL_NUM_WARPS_VALUES)}"
            )
        if self.matmul_num_stages not in _MATMUL_NUM_STAGES_VALUES:
            raise ValueError(
                f"ConvRot matmul num_stages must be {_choices(_MATMUL_NUM_STAGES_VALUES)}"
            )


class MatmulSchedule(NamedTuple):
    """Complete immutable GEMM choices, independent of input preparation."""

    matmul_block_m: int
    matmul_block_n: int
    matmul_block_k: int
    matmul_num_warps: int
    matmul_num_stages: int
    matmul_group_m: int
    matmul_kernel: Literal["triton", "gluon_async_copy"] = "triton"
    triton_specialize_m: bool = True
    triton_explicit_bias_fma: bool = False
