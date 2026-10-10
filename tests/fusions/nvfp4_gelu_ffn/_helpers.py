"""Shared operands, models, and references for standard NVFP4 GELU FFN tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

from piper_kernels.linear.nvfp4 import reference as nvfp4_reference

from .._nvfp4 import Linear, make_weight, precise_linear


@dataclass(frozen=True, slots=True)
class Operands:
    input: torch.Tensor
    up: Linear
    down: Linear

    def arguments(self, chunk_rows: int) -> tuple[object, ...]:
        return self.input, *self.up.arguments(), *self.down.arguments(), chunk_rows


def activated_down(input: torch.Tensor, linear: Linear) -> torch.Tensor:  # noqa: A002
    """Apply FP32 GELU and directly prepare the down projection input."""
    prepared = nvfp4_reference.prepare_input(
        input,
        linear.activation_scale,
        linear.dynamic,
        "gelu_tanh",
        linear.weight.high_first,
    )
    return nvfp4_reference.linear_prepared(
        *prepared,
        linear.weight.qdata,
        linear.weight.scale,
        linear.weight.per_tensor_scale,
        linear.bias,
        input.dtype,
    ).reshape(*input.shape[:-1], linear.weight.shape[0])


def materialized(operands: Operands) -> torch.Tensor:
    """Run the equivalent two-projection path with direct GELU preparation."""
    return activated_down(precise_linear(operands.input, operands.up), operands.down)


def make_operands(  # noqa: PLR0913
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    up_dynamic: bool,
    down_dynamic: bool,
    dtype: torch.dtype = torch.bfloat16,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    up_high_first: bool = False,
    down_high_first: bool = False,
    seed: int = 951,
) -> Operands:
    torch.manual_seed(seed)
    input = torch.randn(rows, input_features, device="cuda", dtype=dtype)  # noqa: A001
    up_dense = torch.randn(
        intermediate_features,
        input_features,
        device="cuda",
        dtype=dtype,
    )
    down_dense = torch.randn(
        output_features,
        intermediate_features,
        device="cuda",
        dtype=dtype,
    )

    def make_linear(
        dense: torch.Tensor,
        activation_scale: torch.Tensor | None,
        dynamic: bool,
        high_first: bool,
    ) -> Linear:
        bias = (
            torch.randn(dense.shape[0], device="cuda", dtype=bias_dtype)
            if bias_dtype is not None
            else None
        )
        return Linear(
            make_weight(dense, activation_scale, dynamic, high_first),
            activation_scale,
            bias,
            dynamic,
        )

    up_scale = None if up_dynamic else per_tensor_amax_to_scale(input.abs().amax())
    up = make_linear(up_dense, up_scale, up_dynamic, up_high_first)
    down_scale = None
    if not down_dynamic:
        projected = precise_linear(input, up)
        activated = torch.nn.functional.gelu(projected.float(), approximate="tanh")
        down_scale = per_tensor_amax_to_scale(activated.abs().amax())
    down = make_linear(down_dense, down_scale, down_dynamic, down_high_first)
    return Operands(input, up, down)


class GeluFfn(torch.nn.Module):
    """Two semantic linears around promoted tanh-GELU."""

    def __init__(
        self,
        operands: Operands,
        *,
        expose_up: bool = False,
        explicit_promotion: bool = False,
    ) -> None:
        super().__init__()
        self.expose_up = expose_up
        self.explicit_promotion = explicit_promotion
        self.up = self._linear(operands.up)
        self.down = self._linear(operands.down)

    @staticmethod
    def _linear(operands: Linear) -> torch.nn.Linear:
        out_features, in_features = operands.weight.shape
        linear = torch.nn.Linear(
            in_features,
            out_features,
            bias=operands.bias is not None,
            device="cuda",
            dtype=operands.weight.dtype,
        )
        linear.weight = torch.nn.Parameter(operands.weight, requires_grad=False)
        if operands.bias is not None:
            linear.bias = torch.nn.Parameter(operands.bias, requires_grad=False)
        return linear

    def forward(
        self,
        activation: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        up = self.up(activation)
        activated = F.gelu(
            up.float() if self.explicit_promotion else up,
            approximate="tanh",
        )
        if self.explicit_promotion:
            activated = activated.to(up.dtype)
        output = self.down(activated)
        return (output, up) if self.expose_up else output


class GatedUpdates(torch.nn.Module):
    """H3-style indexed updates containing the GELU FFN."""

    def __init__(self, operands: Operands, *, python_indexing: bool = False) -> None:
        super().__init__()
        self.ffn = GeluFfn(operands)
        self.python_indexing = python_indexing
        self.input_features = operands.up.weight.shape[1]
        output_features = operands.down.weight.shape[0]
        self.update = torch.nn.Linear(
            output_features,
            output_features,
            bias=False,
            device="cuda",
            dtype=operands.input.dtype,
        )
        self.update.weight.requires_grad_(False)

    def forward(
        self,
        base: torch.Tensor,
        update_source: torch.Tensor,
        update_gate: torch.Tensor,
        ffn_gate: torch.Tensor,
        gate_indices: torch.Tensor,
    ) -> torch.Tensor:
        reusable_update = self.update(update_source)
        if self.python_indexing:
            selected_update_gate = update_gate[gate_indices]
            selected_ffn_gate = ffn_gate[gate_indices]
        else:
            selected_update_gate = update_gate.index_select(0, gate_indices)
            selected_ffn_gate = ffn_gate.index_select(0, gate_indices)
        hidden = base + selected_update_gate * reusable_update
        ffn = self.ffn(hidden[..., : self.input_features].contiguous())
        assert isinstance(ffn, torch.Tensor)
        return hidden + selected_ffn_gate * ffn


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual.float() - expected.float()).norm() / expected.float().norm()


def make_gated_update_arguments(
    rows: int,
    features: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    base = torch.randn(rows, features, dtype=dtype, device="cuda")
    update_source = torch.randn_like(base)
    update_gate = torch.randn(7, features, dtype=dtype, device="cuda")
    ffn_gate = torch.randn(7, features, dtype=dtype, device="cuda")
    gate_indices = torch.randint(0, 7, (rows,), dtype=torch.int64, device="cuda")
    return base, update_source, update_gate, ffn_gate, gate_indices
