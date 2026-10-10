"""Shared operands and model builders for ConvRot INT8 SwiGLU FFN tests."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

from .._convrot_int8 import Linear, make_linear


@dataclass(frozen=True, slots=True)
class Operands:
    input: torch.Tensor
    gate: Linear
    value: Linear
    down: Linear

    def arguments(self, chunk_rows: int) -> tuple[object, ...]:
        return (
            self.input,
            *self.gate.arguments(),
            *self.value.arguments(),
            *self.down.arguments(),
            chunk_rows,
        )


def make_operands(
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 256,
    down_group_size: int = 256,
) -> Operands:
    input = torch.randn(rows, input_features, dtype=dtype, device="cuda")  # noqa: A001
    return Operands(
        input,
        make_linear(intermediate_features, input_features, bias_dtype, group_size),
        make_linear(intermediate_features, input_features, bias_dtype, group_size),
        make_linear(output_features, intermediate_features, bias_dtype, down_group_size),
    )


class SwiGluFfn(torch.nn.Module):
    input_features = 256
    intermediate_features = 512
    output_features = 384

    def __init__(
        self,
        *,
        promote_gate: bool = False,
        reverse_multiply: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        bias_dtype: torch.dtype | None = torch.bfloat16,
        expose_gate: bool = False,
    ) -> None:
        super().__init__()
        self.promote_gate = promote_gate
        self.reverse_multiply = reverse_multiply
        self.expose_gate = expose_gate
        self.gate = self._linear(self.intermediate_features, self.input_features, bias_dtype, dtype)
        self.value = self._linear(
            self.intermediate_features, self.input_features, bias_dtype, dtype
        )
        self.down = self._linear(
            self.output_features, self.intermediate_features, bias_dtype, dtype
        )

    @staticmethod
    def _linear(
        out_features: int,
        in_features: int,
        bias_dtype: torch.dtype | None,
        dtype: torch.dtype,
    ) -> torch.nn.Linear:
        qdata = torch.randint(
            -127,
            128,
            (out_features, in_features),
            dtype=torch.int8,
            device="cuda",
        )
        scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
        weight = ConvRotInt8Tensor.from_quantized(qdata, scale, group_size=256, logical_dtype=dtype)
        linear = torch.nn.Linear(
            in_features,
            out_features,
            bias=bias_dtype is not None,
            dtype=dtype,
            device="cuda",
        )
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias_dtype is not None:
            assert linear.bias is not None
            linear.bias = torch.nn.Parameter(linear.bias.to(bias_dtype), requires_grad=False)
        return linear

    def forward(
        self,
        activation: torch.Tensor,
        value_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(activation)
        value = self.value(activation if value_input is None else value_input)
        activated_gate = F.silu(gate.float()).to(gate.dtype) if self.promote_gate else F.silu(gate)
        activated = activated_gate * value if self.reverse_multiply else value * activated_gate
        output = self.down(activated)
        return (output, gate) if self.expose_gate else output


class GatedUpdates(torch.nn.Module):
    def __init__(
        self,
        *,
        expose: Literal["none", "ffn", "hidden"] = "none",
        update_mode: Literal["materialized", "direct", "alias"] = "materialized",
        python_indexing: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.ffn = SwiGluFfn(promote_gate=True, reverse_multiply=True, dtype=dtype)
        self.dtype = dtype
        self.expose = expose
        self.update_mode = update_mode
        self.python_indexing = python_indexing
        self.update = torch.nn.Linear(
            self.ffn.output_features,
            self.ffn.output_features,
            bias=False,
            dtype=dtype,
            device="cuda",
        )
        self.update.weight.requires_grad_(False)

    def forward(
        self,
        base: torch.Tensor,
        update_source: torch.Tensor,
        update_gate: torch.Tensor,
        ffn_gate: torch.Tensor,
        gate_indices: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.update_mode == "materialized":
            reusable_update = self.update(update_source)
        elif self.update_mode == "alias":
            reusable_update = update_source[1:]
        else:
            reusable_update = update_source
        if self.python_indexing:
            selected_update_gate = update_gate[gate_indices]
            selected_ffn_gate = ffn_gate[gate_indices]
        else:
            selected_update_gate = update_gate.index_select(0, gate_indices)
            selected_ffn_gate = ffn_gate.index_select(0, gate_indices)
        hidden = base + selected_update_gate * reusable_update
        ffn = self.ffn(hidden[..., : self.ffn.input_features].contiguous())
        assert isinstance(ffn, torch.Tensor)
        output = hidden + selected_ffn_gate * ffn
        if self.expose == "ffn":
            return output, ffn
        if self.expose == "hidden":
            return output, hidden
        return output


def make_gated_update_arguments(
    model: GatedUpdates,
    rows: int,
) -> tuple[torch.Tensor, ...]:
    features = model.ffn.output_features
    base = torch.randn(rows, features, dtype=model.dtype, device="cuda")
    update_source = torch.randn(
        rows + int(model.update_mode == "alias"),
        features,
        dtype=model.dtype,
        device="cuda",
    )
    gate_storage = torch.randn(7, 6 * features, dtype=model.dtype, device="cuda")
    update_gate = gate_storage[:, 2 * features : 3 * features]
    ffn_gate = gate_storage[:, 5 * features :]
    gate_indices = torch.randint(0, 7, (rows,), dtype=torch.int64, device="cuda")
    return base, update_source, update_gate, ffn_gate, gate_indices


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual.float() - expected.float()).norm() / expected.float().norm()
