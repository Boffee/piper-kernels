"""Shared workload and provider adapters for ConvRot benchmarking and tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot import ConvRotInt8Tensor, convrot_linear
from piper_kernels.linear.convrot.int8 import _policy as convrot_policy
from piper_kernels.linear.convrot.int8 import triton as convrot_backend
from piper_kernels.linear.convrot.int8.reference import (
    convrot_int8_linear,
    convrot_int8_swiglu_linear,
)

from .convrot import ConvRotConfig, ConvRotInputs, ConvRotShape, make_convrot_inputs
from .providers import BenchmarkProvider


@dataclass(frozen=True, slots=True)
class ConvRotWorkload:
    """Tensors and production policy for one ConvRot case."""

    shape: ConvRotShape
    config: ConvRotConfig
    inputs: ConvRotInputs
    production_plan: convrot_policy.ConvRotInt8LinearExecutionPlan

    @property
    def input_preparation(self) -> str | None:
        """Return the selected public SwiGLU preparation description."""
        if self.shape.input_activation is None:
            return None
        return "fused" if self.production_plan.fuse_rotation_quantization else "materialized"

    def common_configuration(self) -> dict[str, object]:
        """Return provider-neutral workload metadata."""
        logical_input_layout = "up_gate" if self.shape.input_activation == "swiglu" else "plain"
        return {
            **self.config.as_dict(),
            "input_activation": self.shape.input_activation or "none",
            "logical_input_layout": logical_input_layout,
            "provider_input_layout": logical_input_layout,
            "has_bias": self.shape.has_bias,
            "prepared_execution_scope": "complete_operator_on_fixed_source_tensors",
        }

    def reference(self) -> torch.Tensor:
        """Evaluate the complete workload with the shared portable reference."""
        return _run_convrot_reference(self, self.inputs)


def make_convrot_workload(
    shape: ConvRotShape,
    config: ConvRotConfig,
    *,
    device: torch.device,
    target: AcceleratorTarget | None = None,
) -> ConvRotWorkload:
    """Create shared tensors and resolve the production execution plan."""
    inputs = make_convrot_inputs(shape, config, device=device)
    activation, qdata, _scale, _bias = inputs
    production_plan = convrot_backend.default_convrot_int8_execution_plan(
        activation,
        qdata,
        config.group_size,
        apply_swiglu=shape.input_activation == "swiglu",
        target=target,
    )
    return ConvRotWorkload(
        shape=shape,
        config=config,
        inputs=inputs,
        production_plan=production_plan,
    )


def _run_convrot_reference(
    workload: ConvRotWorkload,
    inputs: ConvRotInputs,
) -> torch.Tensor:
    """Run the matching portable reference on the supplied workload inputs."""
    activation, qdata, scale, bias = inputs
    if workload.shape.input_activation == "swiglu":
        return convrot_int8_swiglu_linear(
            activation,
            qdata,
            scale,
            workload.config.group_size,
            bias,
        )
    return convrot_int8_linear(
        activation,
        qdata,
        scale,
        workload.config.group_size,
        bias,
    )


def make_public_convrot_provider(
    workload: ConvRotWorkload,
) -> BenchmarkProvider[ConvRotInputs, torch.Tensor]:
    """Build a provider that exercises normal production dispatch."""
    shape = workload.shape
    config = workload.config
    _activation, qdata, scale, bias = workload.inputs
    weight = ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=config.group_size,
        logical_dtype=config.dtype,
    )

    def run(prepared: ConvRotInputs) -> torch.Tensor:
        prepared_activation = prepared[0]
        if shape.input_activation == "swiglu":
            return convrot_linear(
                prepared_activation,
                weight,
                bias,
                input_activation="swiglu",
            )
        return torch.nn.functional.linear(prepared_activation, weight, bias)

    return BenchmarkProvider(
        name="piper-convrot",
        prepare=lambda: workload.inputs,
        run=run,
        synchronize=torch.cuda.synchronize,
        configuration={
            **workload.common_configuration(),
            "operation_entrypoint": (
                "piper_kernels.linear.convrot.convrot_linear"
                if shape.input_activation == "swiglu"
                else "torch.nn.functional.linear"
            ),
            "input_preparation": workload.input_preparation or "none",
            **workload.production_plan.as_dict(),
        },
    )


def planned_convrot_configuration(
    workload: ConvRotWorkload,
    plan: convrot_policy.ConvRotInt8LinearExecutionPlan,
) -> dict[str, object]:
    """Return complete metadata for one explicitly injected execution plan."""
    return {
        **workload.common_configuration(),
        "algorithm": "convrot_int8_linear",
        **plan.as_dict(),
    }


def make_planned_convrot_provider(
    workload: ConvRotWorkload,
    plan: convrot_policy.ConvRotInt8LinearExecutionPlan,
    *,
    name: str,
) -> BenchmarkProvider[ConvRotInputs, torch.Tensor]:
    """Build a provider that injects one plan into the complete device pipeline."""

    def run(prepared: ConvRotInputs) -> torch.Tensor:
        activation, qdata, scale, bias = prepared
        return convrot_backend.run_convrot_int8_linear(
            activation,
            qdata,
            scale,
            bias,
            workload.config.group_size,
            apply_swiglu=workload.shape.input_activation == "swiglu",
            execution_plan=plan,
        )

    return BenchmarkProvider(
        name=name,
        prepare=lambda: workload.inputs,
        run=run,
        synchronize=torch.cuda.synchronize,
        configuration=planned_convrot_configuration(workload, plan),
    )
