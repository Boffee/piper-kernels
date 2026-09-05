"""Shared workload and provider adapters for ConvRot INT8 benchmarking and tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels.linear.convrot import ConvRotInt8Tensor, convrot_int8_linear
from piper_kernels.linear.convrot.int8 import _policy as convrot_int8_policy
from piper_kernels.linear.convrot.int8 import triton as convrot_int8_backend
from piper_kernels.linear.convrot.int8.reference import linear as reference_linear

from .convrot import ConvRotConfig, ConvRotInputs, ConvRotShape, make_convrot_inputs
from .providers import BenchmarkProvider


@dataclass(frozen=True, slots=True)
class ConvRotInt8Workload:
    """Tensors and production policy for one ConvRot INT8 case."""

    shape: ConvRotShape
    config: ConvRotConfig
    inputs: ConvRotInputs
    production_plan: convrot_int8_policy.LinearExecutionPlan

    @property
    def input_preparation(self) -> str | None:
        """Return the selected public input-preparation description."""
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
        return _run_convrot_int8_reference(self, self.inputs)


def make_convrot_int8_workload(
    shape: ConvRotShape,
    config: ConvRotConfig,
    *,
    device: torch.device,
) -> ConvRotInt8Workload:
    """Create shared tensors and resolve the production execution plan."""
    inputs = make_convrot_inputs(shape, config, device=device)
    qdata = inputs[1]
    production_plan = convrot_int8_backend.default_execution_plan(qdata)
    return ConvRotInt8Workload(
        shape=shape,
        config=config,
        inputs=inputs,
        production_plan=production_plan,
    )


def _run_convrot_int8_reference(
    workload: ConvRotInt8Workload,
    inputs: ConvRotInputs,
) -> torch.Tensor:
    """Run the matching portable reference on the supplied workload inputs."""
    activation, qdata, scale, bias = inputs
    return reference_linear(
        activation,
        qdata,
        scale,
        workload.config.group_size,
        bias,
        activation_fn=workload.shape.input_activation,
    )


def make_public_convrot_int8_provider(
    workload: ConvRotInt8Workload,
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
        if shape.input_activation is not None:
            return convrot_int8_linear(
                prepared_activation,
                weight,
                bias,
                activation_fn=shape.input_activation,
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
                "piper_kernels.linear.convrot.convrot_int8_linear"
                if shape.input_activation is not None
                else "torch.nn.functional.linear"
            ),
            "input_preparation": workload.input_preparation or "none",
            **workload.production_plan.as_dict(),
        },
    )


def planned_convrot_int8_configuration(
    workload: ConvRotInt8Workload,
    plan: convrot_int8_policy.LinearExecutionPlan,
) -> dict[str, object]:
    """Return complete metadata for one explicitly injected execution plan."""
    return {
        **workload.common_configuration(),
        "algorithm": "convrot_int8_linear",
        **plan.as_dict(),
    }


def make_planned_convrot_int8_provider(
    workload: ConvRotInt8Workload,
    plan: convrot_int8_policy.LinearExecutionPlan,
    *,
    name: str,
) -> BenchmarkProvider[ConvRotInputs, torch.Tensor]:
    """Build a provider that injects one plan into the complete device pipeline."""

    def run(prepared: ConvRotInputs) -> torch.Tensor:
        activation, qdata, scale, bias = prepared
        return convrot_int8_backend.run_linear(
            activation,
            qdata,
            scale,
            bias,
            workload.config.group_size,
            activation_fn=workload.shape.input_activation,
            execution_plan=plan,
        )

    return BenchmarkProvider(
        name=name,
        prepare=lambda: workload.inputs,
        run=run,
        synchronize=torch.cuda.synchronize,
        configuration=planned_convrot_int8_configuration(workload, plan),
    )
