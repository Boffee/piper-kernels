"""Shared host orchestration and target selection for ConvRot INT8 linear execution.

This module imports without Triton. Vendor modules supply Triton-free policy
selection and GPU kernel launchers; shared buffer allocation, operand reshaping,
validation, and residual-epilogue fallback live once in :class:`LinearBackend`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol

import torch

from piper_kernels._input_activations import apply_input_activation, input_activation_width
from piper_kernels._residual import ResidualAdd
from piper_kernels._triton import runtime
from piper_kernels._triton.targets import AcceleratorTarget

from . import _generic
from ._amd import policy as amd_policy
from ._nvidia import policy as nvidia_policy
from ._plan import LinearExecutionPlan

type PreparedInput = tuple[torch.Tensor, torch.Tensor]
type SecondProjection = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
type DequantizedMean = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


class PreparationBackend(Protocol):
    """Input preparation can run without an optimized matrix implementation."""

    def prepare_input(
        self,
        input: torch.Tensor,  # noqa: A002
        group_size: int,
        activation_fn: str | None = None,
        input_scale: torch.Tensor | None = None,
        *,
        out: PreparedInput | None = None,
    ) -> PreparedInput: ...


class _LinearPolicy(Protocol):
    def supports_target(self, target: AcceleratorTarget) -> bool: ...

    def select_execution_plan(
        self,
        target: AcceleratorTarget,
        *,
        in_features: int,
        rows: int | None = None,
        out_features: int | None = None,
    ) -> LinearExecutionPlan: ...


class _LinearKernels(Protocol):
    def fused_rotate_quantize_input(
        self,
        input: torch.Tensor,  # noqa: A002
        input_qdata: torch.Tensor,
        row_scales: torch.Tensor,
        group_size: int,
        *,
        activation_fn: str | None = None,
        input_scale: torch.Tensor | None = None,
        num_warps: int,
        target: AcceleratorTarget | None = None,
    ) -> None: ...

    def quantize_input(
        self,
        rotated: torch.Tensor,
        input_qdata: torch.Tensor,
        row_scales: torch.Tensor,
        *,
        input_scale: torch.Tensor | None = None,
        num_warps: int,
    ) -> None: ...

    def launch_prepared_linear(  # noqa: PLR0913
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        output: torch.Tensor,
        second_weight: torch.Tensor,
        second_scale: torch.Tensor,
        second_bias: torch.Tensor | None,
        *,
        paired: bool,
        logical_dtype: torch.dtype,
        execution_plan: LinearExecutionPlan,
        target: AcceleratorTarget | None = None,
    ) -> None: ...

    def supports_prepared_swiglu(
        self,
        weight_qdata: torch.Tensor,
        second_weight: torch.Tensor,
        logical_dtype: torch.dtype,
        *,
        rows: int | None = None,
        execution_plan: LinearExecutionPlan,
        target: AcceleratorTarget,
    ) -> bool: ...

    def try_launch_prepared_swiglu(  # noqa: PLR0913
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        output: torch.Tensor,
        second_weight: torch.Tensor,
        second_scale: torch.Tensor,
        second_bias: torch.Tensor | None,
        *,
        execution_plan: LinearExecutionPlan,
        target: AcceleratorTarget,
    ) -> bool: ...

    def try_launch_prepared_residual(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        output: torch.Tensor,
        residual_add: ResidualAdd,
        *,
        row_offset: int,
        execution_plan: LinearExecutionPlan,
        target: AcceleratorTarget,
    ) -> bool: ...


try:
    from piper_kernels._residual.triton import apply_residual_add
    from piper_kernels.weights.convrot import triton as _convrot_triton

    from ._generic import mean as _mean_backend
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    apply_residual_add = None
    _convrot_triton = None
    _mean_backend = None


class LinearBackend:
    """Shared host orchestration bound to one accelerator policy and kernel launcher.

    Prepared inputs contain contiguous INT8 data of shape ``[..., K]`` and FP32
    row scales of shape ``[...]``. Preparation depends on input width, dtype,
    group size, activation, and optional static input scale, never on output width.
    Projections apply row scales after INT32 accumulation and preserve the
    existing bias and logical-dtype rounding contract.

    Caller-owned preparation buffers are contiguous and returned unchanged.
    Projection outputs may have a row stride but must be column-contiguous;
    paired projections write adjacent columns for two equal-shaped weights.
    Neither operation may write beyond the supplied output views.
    """

    def __init__(self, policy: _LinearPolicy, kernels: _LinearKernels) -> None:
        self.policy = policy
        self.kernels = kernels

    def default_execution_plan(
        self,
        weight_qdata: torch.Tensor,
        *,
        target: AcceleratorTarget | None = None,
        rows: int | None = None,
    ) -> LinearExecutionPlan:
        """Resolve production policy, accepting an explicit target for offline tuning."""
        target = AcceleratorTarget.from_device(weight_qdata.device) if target is None else target
        return self.policy.select_execution_plan(
            target,
            in_features=weight_qdata.shape[1],
            rows=rows,
            out_features=weight_qdata.shape[0],
        )

    def prepare_input_with_plan(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        in_features: int,
        group_size: int,
        *,
        activation_fn: str | None,
        input_scale: torch.Tensor | None = None,
        execution_plan: LinearExecutionPlan,
        target: AcceleratorTarget,
        out: PreparedInput | None = None,
    ) -> PreparedInput:
        """Rotate and quantize an input, optionally into caller-owned storage."""
        input_2d = input.reshape(-1, input.shape[-1]).contiguous()
        m = input_2d.shape[0]
        if out is None:
            input_qdata = torch.empty(
                (m, in_features),
                device=input.device,
                dtype=torch.int8,
            )
            row_scales = torch.empty(m, device=input.device, dtype=torch.float32)
            result = (
                input_qdata.reshape(*input.shape[:-1], in_features),
                row_scales.reshape(input.shape[:-1]),
            )
        else:
            result = out
        input_qdata = result[0].reshape(m, in_features)
        row_scales = result[1].reshape(m)
        if execution_plan.fuse_rotation_quantization:
            self.kernels.fused_rotate_quantize_input(
                input_2d,
                input_qdata,
                row_scales,
                group_size,
                activation_fn=activation_fn,
                input_scale=input_scale,
                num_warps=execution_plan.fused_num_warps,
                target=target,
            )
        else:
            assert _convrot_triton is not None
            transformed_input = apply_input_activation(input_2d, activation_fn)
            rotated = torch.empty_like(transformed_input)
            _convrot_triton.rotate_input(
                transformed_input,
                rotated,
                group_size,
                num_warps=execution_plan.rotation_num_warps,
            )
            self.kernels.quantize_input(
                rotated,
                input_qdata,
                row_scales,
                input_scale=input_scale,
                num_warps=execution_plan.quantization_num_warps,
            )
        return result

    def execute_prepared_linear(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        logical_dtype: torch.dtype,
        execution_plan: LinearExecutionPlan,
        *,
        out: torch.Tensor | None = None,
        second_projection: SecondProjection | None = None,
        target: AcceleratorTarget | None = None,
    ) -> torch.Tensor:
        """Project prepared INT8 input, optionally to two equal-width independent weights.

        Paired projections share a launch and write adjacent output columns without
        packing weights. ``out`` may provide caller-owned, row-strided storage.
        """
        leading_shape = input_qdata.shape[:-1]
        m = math.prod(leading_shape)
        k = input_qdata.shape[-1]
        n = weight_qdata.shape[0]
        paired = second_projection is not None
        if second_projection is None:
            second_weight, second_scale, second_bias = weight_qdata, weight_scale, None
        else:
            second_weight, second_scale, second_bias = second_projection
            if second_weight.shape != weight_qdata.shape:
                raise ValueError("paired INT8 projections must have matching weight shapes")
        output_features = n * (2 if paired else 1)
        if out is None:
            output = torch.empty(
                (m, output_features),
                device=input_qdata.device,
                dtype=logical_dtype,
            )
            result = output.reshape(*leading_shape, output_features)
        else:
            result = out
        input_qdata_2d = input_qdata.reshape(m, k)
        input_scale_1d = input_scale.reshape(m)
        output = result.reshape(m, output_features)
        if output.stride(1) != 1:
            raise ValueError("prepared INT8 GEMM output must be column-contiguous")
        if not m or not n:
            return result
        self.kernels.launch_prepared_linear(
            input_qdata_2d,
            input_scale_1d,
            weight_qdata,
            weight_scale,
            bias,
            output,
            second_weight,
            second_scale,
            second_bias,
            paired=paired,
            logical_dtype=logical_dtype,
            execution_plan=execution_plan,
            target=target,
        )
        return result

    def run_linear(
        self,
        input: torch.Tensor,  # noqa: A002
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
        *,
        activation_fn: str | None = None,
        input_scale: torch.Tensor | None = None,
        execution_plan: LinearExecutionPlan | None = None,
        target: AcceleratorTarget | None = None,
    ) -> torch.Tensor:
        """Run ConvRot input preparation and INT8 GEMM under one target and plan."""
        original_shape = input.shape
        k = weight_qdata.shape[1]
        expected_width = k * input_activation_width(activation_fn)
        if original_shape[-1] != expected_width:
            operation = "activated linear input" if activation_fn is not None else "linear input"
            raise ValueError(
                f"{operation} has {original_shape[-1]} features, expected {expected_width}"
            )
        target = AcceleratorTarget.from_device(input.device) if target is None else target
        plan = (
            execution_plan
            if execution_plan is not None
            else self.default_execution_plan(
                weight_qdata,
                target=target,
                rows=math.prod(original_shape[:-1]),
            )
        )
        input_qdata, row_scales = self.prepare_input_with_plan(
            input,
            k,
            group_size,
            activation_fn=activation_fn,
            input_scale=input_scale,
            execution_plan=plan,
            target=target,
        )
        return self.execute_prepared_linear(
            input_qdata,
            row_scales,
            weight_qdata,
            weight_scale,
            bias,
            input.dtype,
            plan,
            target=target,
        )

    def linear(
        self,
        input: torch.Tensor,  # noqa: A002
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
        activation_fn: str | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run ConvRot input rotation, quantization, and INT8 GEMM."""
        return self.run_linear(
            input,
            weight_qdata,
            weight_scale,
            bias,
            group_size,
            activation_fn=activation_fn,
            input_scale=input_scale,
        )

    def prepare_input(
        self,
        input: torch.Tensor,  # noqa: A002 - match linear terminology
        group_size: int,
        activation_fn: str | None = None,
        input_scale: torch.Tensor | None = None,
        *,
        out: PreparedInput | None = None,
        target: AcceleratorTarget | None = None,
    ) -> PreparedInput:
        """Apply an optional activation, then rotate and quantize a linear input."""
        in_features = input.shape[-1] // input_activation_width(activation_fn)
        target = AcceleratorTarget.from_device(input.device) if target is None else target
        plan = self.policy.select_execution_plan(target, in_features=in_features)
        return self.prepare_input_with_plan(
            input,
            in_features,
            group_size,
            activation_fn=activation_fn,
            input_scale=input_scale,
            execution_plan=plan,
            target=target,
            out=out,
        )

    def linear_prepared(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        logical_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
        second_projection: SecondProjection | None = None,
        target: AcceleratorTarget | None = None,
    ) -> torch.Tensor:
        """Apply one weight to an input prepared by the matching operator."""
        target = AcceleratorTarget.from_device(input_qdata.device) if target is None else target
        plan = self.default_execution_plan(
            weight_qdata, target=target, rows=math.prod(input_qdata.shape[:-1])
        )
        return self.execute_prepared_linear(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            logical_dtype,
            plan,
            out=out,
            second_projection=second_projection,
            target=target,
        )

    def supports_prepared_swiglu(
        self,
        weight_qdata: torch.Tensor,
        second_weight: torch.Tensor,
        logical_dtype: torch.dtype,
        *,
        rows: int | None = None,
        execution_plan: LinearExecutionPlan | None = None,
        target: AcceleratorTarget | None = None,
    ) -> bool:
        """Return whether paired INT8 inputs can fuse SwiGLU into the projection."""
        target = AcceleratorTarget.from_device(weight_qdata.device) if target is None else target
        plan = (
            execution_plan
            if execution_plan is not None
            else self.default_execution_plan(weight_qdata, target=target, rows=rows)
        )
        return self.kernels.supports_prepared_swiglu(
            weight_qdata,
            second_weight,
            logical_dtype,
            rows=rows,
            execution_plan=plan,
            target=target,
        )

    def linear_prepared_swiglu(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        second_projection: SecondProjection,
        logical_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
        execution_plan: LinearExecutionPlan | None = None,
        target: AcceleratorTarget | None = None,
    ) -> torch.Tensor:
        """Project prepared INT8 input through paired up/gate weights and apply SwiGLU."""
        leading_shape = input_qdata.shape[:-1]
        m = math.prod(leading_shape)
        k = input_qdata.shape[-1]
        n = weight_qdata.shape[0]
        second_weight, second_scale, second_bias = second_projection
        if second_weight.shape != weight_qdata.shape:
            raise ValueError("paired INT8 projections must have matching weight shapes")
        expected_shape = (*leading_shape, n)
        if out is None:
            output = torch.empty((m, n), device=input_qdata.device, dtype=logical_dtype)
            result = output.reshape(expected_shape)
        else:
            if (
                out.shape != expected_shape
                or out.dtype is not logical_dtype
                or out.device != input_qdata.device
            ):
                raise ValueError(
                    "prepared INT8 SwiGLU output must match expected shape "
                    f"{expected_shape}, dtype {logical_dtype}, and device {input_qdata.device}"
                )
            if out.ndim == 0 or out.stride(-1) != 1:
                raise ValueError("prepared INT8 GEMM output must be column-contiguous")
            result = out
            try:
                output = result.view(m, n)
            except RuntimeError as error:
                raise ValueError(
                    "prepared INT8 SwiGLU output must be viewable as a 2-D row-strided matrix"
                ) from error
        if not m or not n:
            return result
        target = AcceleratorTarget.from_device(input_qdata.device) if target is None else target
        plan = (
            execution_plan
            if execution_plan is not None
            else self.default_execution_plan(weight_qdata, target=target, rows=m)
        )
        if self.kernels.try_launch_prepared_swiglu(
            input_qdata.reshape(m, k),
            input_scale.reshape(m),
            weight_qdata,
            weight_scale,
            bias,
            output,
            second_weight,
            second_scale,
            second_bias,
            execution_plan=plan,
            target=target,
        ):
            return result
        paired = self.execute_prepared_linear(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            logical_dtype,
            plan,
            second_projection=second_projection,
            target=target,
        )
        activated = apply_input_activation(paired, "swiglu").to(logical_dtype)
        if out is None:
            return activated
        result.copy_(activated)
        return result

    def linear_prepared_residual(
        self,
        input_qdata: torch.Tensor,
        input_scale: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        residual_add: ResidualAdd,
        *,
        row_offset: int = 0,
        projection_scratch: torch.Tensor | None = None,
        target: AcceleratorTarget | None = None,
    ) -> torch.Tensor:
        """Project one chunk and update the corresponding residual output rows.

        The owning fusion validates ``residual_add`` once. ``row_offset`` locates
        this chunk in its flattened rows; the result has the input's leading shape
        and aliases ``residual_add.output``. The residual owns the output dtype.
        Both native and standalone epilogues preserve projection rounding before
        FP32 residual arithmetic and the final cast.

        Optional ``projection_scratch`` is a column-contiguous chunk buffer with
        the projection's shape, dtype, and device, disjoint from live operands.
        Only the standalone fallback uses it; the backend owns that decision.
        """
        leading_shape = input_qdata.shape[:-1]
        m = math.prod(leading_shape)
        n, k = weight_qdata.shape
        target = AcceleratorTarget.from_device(input_qdata.device) if target is None else target
        plan = self.default_execution_plan(weight_qdata, target=target, rows=m)
        output = residual_add.output.view(-1, n)[row_offset : row_offset + m]
        if self.kernels.try_launch_prepared_residual(
            input_qdata.reshape(m, k),
            input_scale.reshape(m),
            weight_qdata,
            weight_scale,
            bias,
            output,
            residual_add,
            row_offset=row_offset,
            execution_plan=plan,
            target=target,
        ):
            return output.view(*leading_shape, n)
        projection = self.execute_prepared_linear(
            input_qdata,
            input_scale,
            weight_qdata,
            weight_scale,
            bias,
            residual_add.base.dtype,
            plan,
            out=projection_scratch,
            target=target,
        )
        assert apply_residual_add is not None
        return apply_residual_add(projection, residual_add, row_offset)


# Dynamic backend slots accept test doubles in unit tests.
_nvidia_backend: Any = None
_amd_backend: Any = None

try:
    from ._nvidia import triton as _nvidia_kernels
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
else:
    _nvidia_backend = LinearBackend(nvidia_policy, _nvidia_kernels)

try:
    from ._amd import triton as _amd_kernels
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
else:
    _amd_backend = LinearBackend(amd_policy, _amd_kernels)


def select_linear_backend_for_target(target: AcceleratorTarget) -> LinearBackend | None:
    """Select the optimized backend for an explicit accelerator target, or None."""
    if _nvidia_backend is None and _amd_backend is None:
        return None
    if nvidia_policy.supports_target(target):
        return _nvidia_backend
    return _amd_backend if amd_policy.supports_target(target) else None


def select_linear_backend(
    input: torch.Tensor,  # noqa: A002
    *,
    target: AcceleratorTarget | None = None,
) -> LinearBackend | None:
    """Select compatible linear/preparation/projection operations, or a portable fallback."""
    if _nvidia_backend is None and _amd_backend is None:
        return None
    resolved_target = AcceleratorTarget.from_device(input.device) if target is None else target
    return select_linear_backend_for_target(resolved_target)


def require_linear_backend(input: torch.Tensor) -> LinearBackend:  # noqa: A002
    """Resolve an optimized op emitted by the compiler or called explicitly."""
    backend = select_linear_backend(input)
    if backend is None:
        raise ValueError(f"ConvRot INT8 optimized linear is unavailable on {input.device}")
    return backend


def select_preparation_backend(input: torch.Tensor) -> PreparationBackend:  # noqa: A002
    """Prefer tuned preparation, with generic execution on other devices."""
    return select_linear_backend(input) or _generic


def select_dequantized_mean(input: torch.Tensor) -> DequantizedMean | None:  # noqa: A002
    """Use the shared reduction independently of optimized INT8 matrix support."""
    if _mean_backend is not None and runtime.supports_device(input.device):
        return _mean_backend.dequantized_input_mean
    return None
