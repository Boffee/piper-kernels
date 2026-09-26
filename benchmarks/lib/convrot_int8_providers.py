"""Shared workload and provider adapters for ConvRot INT8 benchmarking and tuning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot import convrot_int8_linear
from piper_kernels.linear.convrot.int8._amd import triton as amd
from piper_kernels.linear.convrot.int8._nvidia import dispatch as nvidia
from piper_kernels.linear.convrot.int8._plan import LinearExecutionPlan
from piper_kernels.linear.convrot.int8.reference import linear as reference_linear
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

from .convrot import ConvRotConfig, ConvRotInputs, ConvRotShape, make_convrot_inputs
from .providers import BenchmarkProvider


@dataclass(frozen=True, slots=True)
class ConvRotInt8Workload:
    """Tensors and production policy for one ConvRot INT8 case."""

    shape: ConvRotShape
    config: ConvRotConfig
    inputs: ConvRotInputs
    production_plan: LinearExecutionPlan
    backend: ModuleType

    @property
    def input_preparation(self) -> str:
        """Return the selected public input-preparation description."""
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


def select_convrot_int8_backend(target: AcceleratorTarget) -> ModuleType:
    """Resolve the production policy while allowing explicit offline targets."""
    for backend in (nvidia, amd):
        if backend.policy.supports_target(target):
            return backend
    raise ValueError(f"ConvRot INT8 benchmarking has no optimized backend for {target}")


def make_convrot_int8_workload(
    shape: ConvRotShape,
    config: ConvRotConfig,
    *,
    device: torch.device,
    target: AcceleratorTarget | None = None,
) -> ConvRotInt8Workload:
    """Create tensors and resolve policy; an explicit target permits offline inspection."""
    target = AcceleratorTarget.from_device(device) if target is None else target
    backend = select_convrot_int8_backend(target)
    inputs = make_convrot_inputs(shape, config, device=device)
    qdata = inputs[1]
    production_plan = (
        backend.default_execution_plan(qdata, target=target, rows=shape.rows)
        if target.is_nvidia_cuda
        else backend.default_execution_plan(qdata, target=target)
    )
    return ConvRotInt8Workload(
        shape=shape,
        config=config,
        inputs=inputs,
        production_plan=production_plan,
        backend=backend,
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
            "input_preparation": workload.input_preparation,
            **workload.production_plan.as_dict(),
        },
    )


def make_convrot_int8_phase_operations(
    workload: ConvRotInt8Workload,
) -> dict[str, Callable[[], object]]:
    """Check and bind production preparation/GEMM with reusable phase buffers."""
    backend = workload.backend
    activation, qdata, scale, bias = workload.inputs
    prepared = backend.prepare_input(
        activation, workload.config.group_size, activation_fn=workload.shape.input_activation
    )
    output = activation.new_empty((workload.shape.rows, workload.shape.out_features))

    def prepare() -> tuple[torch.Tensor, torch.Tensor]:
        return backend.prepare_input(
            activation,
            workload.config.group_size,
            activation_fn=workload.shape.input_activation,
            out=prepared,
        )

    def project() -> torch.Tensor:
        return backend.linear_prepared(*prepared, qdata, scale, bias, activation.dtype, out=output)

    m, n = output.shape
    padded_input = torch.nn.functional.pad(prepared[0], (0, 0, 0, (-m) % 32))
    padded_weight = torch.nn.functional.pad(qdata, (0, 0, 0, (-n) % 8))
    expected = torch._int_mm(padded_input, padded_weight.T)[:m, :n].float()
    expected.mul_(prepared[1].reshape(m, 1))
    if bias is not None:
        # Match the FP32 fused weight-scale/bias epilogue using independent FP64 math.
        expected = (expected.double() * scale.reshape(1, n).double() + bias.double()).float()
    else:
        expected.mul_(scale.reshape(1, n))
    expected = expected.to(activation.dtype)
    public = make_public_convrot_int8_provider(workload)
    torch.testing.assert_close(project(), expected, rtol=0, atol=0)
    torch.testing.assert_close(public.run_operator(), expected, rtol=0, atol=0)
    return {"prepare": prepare, "prepared_gemm": project, "linear": public.run_operator}


def planned_convrot_int8_configuration(
    workload: ConvRotInt8Workload,
    plan: LinearExecutionPlan,
) -> dict[str, object]:
    """Return complete metadata for one explicitly injected execution plan."""
    return {
        **workload.common_configuration(),
        "algorithm": "convrot_int8_linear",
        **plan.as_dict(),
    }


def make_planned_convrot_int8_provider(
    workload: ConvRotInt8Workload,
    plan: LinearExecutionPlan,
    *,
    name: str,
) -> BenchmarkProvider[ConvRotInputs, torch.Tensor]:
    """Build a provider that injects one plan into the complete device pipeline."""

    def run(prepared: ConvRotInputs) -> torch.Tensor:
        activation, qdata, scale, bias = prepared
        return workload.backend.run_linear(
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
