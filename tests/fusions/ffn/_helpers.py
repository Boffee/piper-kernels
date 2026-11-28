"""Shared operands, modules, references, and assertions for FFN fusion tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from _accelerators import ensure_convrot_int8_linear_ops
from _compile_capture import FunctionalTargetCapturePass
from torch._dynamo.utils import ReinplaceCounters
from torch.nn import functional as F  # noqa: N812

from piper_kernels import compile_options
from piper_kernels.linear.nvfp4 import reference as nvfp4_reference
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

from .._convrot_int8 import Linear as Int8Linear
from .._convrot_int8 import make_linear as make_int8_linear
from .._convrot_nvfp4 import make_weight as make_convrot_nvfp4_weight
from .._nvfp4 import make_weight as make_standard_nvfp4_weight

type Activation = Literal["gelu_tanh", "swiglu"]
type Format = Literal["convrot_int8", "nvfp4", "convrot_nvfp4"]


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual.float() - expected.float()).norm() / expected.float().norm()


@dataclass(frozen=True, slots=True)
class Int8Operands:
    input: torch.Tensor
    activation: Activation
    up: Int8Linear
    down: Int8Linear
    gate: Int8Linear | None = None

    def arguments(
        self,
        chunk_rows: int,
        *,
        up_input_scale: torch.Tensor | None = None,
        gate_input_scale: torch.Tensor | None = None,
        down_input_scale: torch.Tensor | None = None,
    ) -> dict[str, object]:
        return {
            "input": self.input,
            "up_weight_qdata": self.up.qdata,
            "up_weight_scale": self.up.scale,
            "up_bias": self.up.bias,
            "up_group_size": self.up.group_size,
            "down_weight_qdata": self.down.qdata,
            "down_weight_scale": self.down.scale,
            "down_bias": self.down.bias,
            "down_group_size": self.down.group_size,
            "activation_fn": self.activation,
            "chunk_rows": chunk_rows,
            "gate_weight_qdata": None if self.gate is None else self.gate.qdata,
            "gate_weight_scale": None if self.gate is None else self.gate.scale,
            "gate_bias": None if self.gate is None else self.gate.bias,
            "gate_group_size": None if self.gate is None else self.gate.group_size,
            "up_input_scale": up_input_scale,
            "gate_input_scale": gate_input_scale,
            "down_input_scale": down_input_scale,
        }


def make_int8_operands(
    activation: Activation,
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 256,
    down_group_size: int = 256,
) -> Int8Operands:
    source = torch.randn(rows, input_features, dtype=dtype, device="cuda")
    up = make_int8_linear(intermediate_features, input_features, bias_dtype, group_size)
    gate = (
        make_int8_linear(intermediate_features, input_features, bias_dtype, group_size)
        if activation == "swiglu"
        else None
    )
    down = make_int8_linear(output_features, intermediate_features, bias_dtype, down_group_size)
    return Int8Operands(source, activation, up, down, gate)


def materialized_int8(
    operands: Int8Operands,
    *,
    up_input_scale: torch.Tensor | None = None,
    gate_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    ensure_convrot_int8_linear_ops()
    linear = torch.ops.piper_kernels.convrot_int8_linear
    up = linear(operands.input, *operands.up.arguments(), None, up_input_scale)
    if operands.activation == "gelu_tanh":
        return linear(up, *operands.down.arguments(), "gelu_tanh", down_input_scale)
    assert operands.gate is not None
    gate = linear(operands.input, *operands.gate.arguments(), None, gate_input_scale)
    return linear(
        torch.cat((up, gate), dim=-1),
        *operands.down.arguments(),
        "swiglu",
        down_input_scale,
    )


def _make_int8_module_linear(
    out_features: int,
    in_features: int,
    bias_dtype: torch.dtype | None,
    dtype: torch.dtype,
) -> torch.nn.Linear:
    weight = ConvRotInt8Tensor.from_quantized(
        torch.randint(-127, 128, (out_features, in_features), dtype=torch.int8, device="cuda"),
        torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01,
        group_size=256,
        logical_dtype=dtype,
    )
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


class Int8SwiGluFfn(torch.nn.Module):
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
        self.gate = _make_int8_module_linear(
            self.intermediate_features, self.input_features, bias_dtype, dtype
        )
        self.value = _make_int8_module_linear(
            self.intermediate_features, self.input_features, bias_dtype, dtype
        )
        self.down = _make_int8_module_linear(
            self.output_features, self.intermediate_features, bias_dtype, dtype
        )

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


class Int8GeluFfn(torch.nn.Module):
    input_features = 256
    intermediate_features = 512
    output_features = 384

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.bfloat16,
        bias_dtype: torch.dtype | None = torch.bfloat16,
        expose_up: bool = False,
    ) -> None:
        super().__init__()
        self.expose_up = expose_up
        self.up = _make_int8_module_linear(
            self.intermediate_features, self.input_features, bias_dtype, dtype
        )
        self.down = _make_int8_module_linear(
            self.output_features, self.intermediate_features, bias_dtype, dtype
        )

    def forward(self, activation: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        up = self.up(activation)
        output = self.down(F.gelu(up, approximate="tanh"))
        return (output, up) if self.expose_up else output


class Int8GatedUpdates(torch.nn.Module):
    def __init__(
        self,
        *,
        activation: Activation = "swiglu",
        expose: Literal["none", "ffn", "hidden"] = "none",
        update_mode: Literal["materialized", "direct", "alias"] = "materialized",
        python_indexing: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.ffn: Int8SwiGluFfn | Int8GeluFfn = (
            Int8GeluFfn(dtype=dtype)
            if activation == "gelu_tanh"
            else Int8SwiGluFfn(promote_gate=True, reverse_multiply=True, dtype=dtype)
        )
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


def make_int8_gated_update_arguments(
    model: Int8GatedUpdates,
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


@dataclass(frozen=True, slots=True)
class Nvfp4Linear:
    weight: PiperNVFP4Tensor | ConvRotNVFP4Tensor
    activation_scale: torch.Tensor | None
    bias: torch.Tensor | None
    dynamic: bool
    group_size: int | None

    def kwargs(self, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_weight_qdata": self.weight.qdata,
            f"{prefix}_weight_scale": self.weight.scale,
            f"{prefix}_weight_per_tensor_scale": self.weight.per_tensor_scale,
            f"{prefix}_activation_per_tensor_scale": self.activation_scale,
            f"{prefix}_bias": self.bias,
            f"{prefix}_dynamic_activation_scale": self.dynamic,
            f"{prefix}_group_size": self.group_size,
            f"{prefix}_high_first": self.weight.high_first,
        }


@dataclass(frozen=True, slots=True)
class Nvfp4Operands:
    input: torch.Tensor
    activation: Activation
    up: Nvfp4Linear
    down: Nvfp4Linear
    gate: Nvfp4Linear | None = None

    def arguments(self, chunk_rows: int) -> dict[str, object]:
        args: dict[str, object] = {
            "input": self.input,
            **self.up.kwargs("up"),
            **self.down.kwargs("down"),
            "activation_fn": self.activation,
            "chunk_rows": chunk_rows,
        }
        if self.gate is not None:
            args.update(self.gate.kwargs("gate"))
        return args


def _nvfp4_activation_scale(value: torch.Tensor, group_size: int | None) -> torch.Tensor:
    prepared = value if group_size is None else rotate_groups(value, group_size)
    return prepared.abs().amax().float() / (448.0 * 6.0)


def precise_nvfp4_linear(source: torch.Tensor, linear: Nvfp4Linear) -> torch.Tensor:
    return nvfp4_reference.linear(
        source,
        linear.weight.qdata,
        linear.weight.scale,
        linear.weight.per_tensor_scale,
        linear.activation_scale,
        linear.bias,
        linear.dynamic,
        linear.weight.high_first,
        group_size=linear.group_size,
    )


def materialized_nvfp4(operands: Nvfp4Operands) -> torch.Tensor:
    up = precise_nvfp4_linear(operands.input, operands.up)
    if operands.activation == "gelu_tanh":
        prepared = nvfp4_reference.prepare_input(
            up,
            operands.down.activation_scale,
            operands.down.dynamic,
            "gelu_tanh",
            operands.down.weight.high_first,
            group_size=operands.down.group_size,
        )
        return nvfp4_reference.linear_prepared(
            *prepared,
            operands.down.weight.qdata,
            operands.down.weight.scale,
            operands.down.weight.per_tensor_scale,
            operands.down.bias,
            up.dtype,
        ).reshape(*up.shape[:-1], operands.down.weight.shape[0])
    assert operands.gate is not None
    gate = precise_nvfp4_linear(operands.input, operands.gate)
    return precise_nvfp4_linear(up * F.silu(gate), operands.down)


def make_nvfp4_operands(  # noqa: PLR0913
    activation: Activation,
    *,
    rows: int = 385,
    input_features: int = 256,
    intermediate_features: int = 512,
    output_features: int = 384,
    up_dynamic: bool,
    down_dynamic: bool,
    up_group_size: int | None = None,
    down_group_size: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    bias_dtype: torch.dtype | None = torch.bfloat16,
    up_high_first: bool = False,
    down_high_first: bool = False,
    distinct_input_scales: bool = False,
    seed: int = 951,
) -> Nvfp4Operands:
    torch.manual_seed(seed)
    source = torch.randn(rows, input_features, device="cuda", dtype=dtype)
    up_dense = torch.randn(intermediate_features, input_features, device="cuda", dtype=dtype)
    gate_dense = (
        torch.randn(intermediate_features, input_features, device="cuda", dtype=dtype)
        if activation == "swiglu"
        else None
    )
    down_dense = torch.randn(output_features, intermediate_features, device="cuda", dtype=dtype)

    def _make_linear(
        dense: torch.Tensor,
        scale: torch.Tensor | None,
        dynamic: bool,
        group_size: int | None,
        high_first: bool,
    ) -> Nvfp4Linear:
        bias = (
            torch.randn(dense.shape[0], device="cuda", dtype=bias_dtype)
            if bias_dtype is not None
            else None
        )
        weight: PiperNVFP4Tensor | ConvRotNVFP4Tensor = (
            make_standard_nvfp4_weight(dense, scale, dynamic, high_first)
            if group_size is None
            else make_convrot_nvfp4_weight(dense, scale, dynamic, group_size, high_first)
        )
        return Nvfp4Linear(weight, scale, bias, dynamic, group_size)

    up_scale = None if up_dynamic else _nvfp4_activation_scale(source, up_group_size)
    gate_scale = up_scale * 0.875 if distinct_input_scales and up_scale is not None else up_scale
    up = _make_linear(up_dense, up_scale, up_dynamic, up_group_size, up_high_first)
    gate = (
        None
        if gate_dense is None
        else _make_linear(gate_dense, gate_scale, up_dynamic, up_group_size, up_high_first)
    )
    down_scale = None
    if not down_dynamic:
        up_proj = precise_nvfp4_linear(source, up)
        if activation == "gelu_tanh":
            activated = F.gelu(up_proj.float(), approximate="tanh")
        else:
            assert gate is not None
            activated = up_proj * F.silu(precise_nvfp4_linear(source, gate))
        down_scale = _nvfp4_activation_scale(activated, down_group_size)
    down = _make_linear(down_dense, down_scale, down_dynamic, down_group_size, down_high_first)
    return Nvfp4Operands(source, activation, up, down, gate)


def _make_nvfp4_module_linear(operands: Nvfp4Linear) -> torch.nn.Linear:
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


class Nvfp4FfnModule(torch.nn.Module):
    def __init__(
        self,
        operands: Nvfp4Operands,
        *,
        promote_gate: bool = False,
        reverse_multiply: bool = False,
        explicit_promotion: bool = False,
        expose_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.activation = operands.activation
        self.promote_gate = promote_gate
        self.reverse_multiply = reverse_multiply
        self.explicit_promotion = explicit_promotion
        self.expose_intermediate = expose_intermediate
        self.up = _make_nvfp4_module_linear(operands.up)
        self.gate = None if operands.gate is None else _make_nvfp4_module_linear(operands.gate)
        self.down = _make_nvfp4_module_linear(operands.down)

    def forward(
        self,
        source: torch.Tensor,
        value_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.activation == "gelu_tanh":
            up = self.up(source)
            activated = F.gelu(
                up.float() if self.explicit_promotion else up,
                approximate="tanh",
            )
            if self.explicit_promotion:
                activated = activated.to(up.dtype)
            output = self.down(activated)
            return (output, up) if self.expose_intermediate else output
        assert self.gate is not None
        gate = self.gate(source)
        value = self.up(source if value_input is None else value_input)
        activated_gate = F.silu(gate.float()).to(gate.dtype) if self.promote_gate else F.silu(gate)
        activated = activated_gate * value if self.reverse_multiply else value * activated_gate
        output = self.down(activated)
        return (output, gate) if self.expose_intermediate else output


class Nvfp4GatedUpdates(torch.nn.Module):
    def __init__(self, operands: Nvfp4Operands, *, python_indexing: bool = False) -> None:
        super().__init__()
        self.ffn = Nvfp4FfnModule(operands, promote_gate=True, reverse_multiply=True)
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


def make_nvfp4_gated_update_arguments(
    rows: int,
    features: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    base = torch.randn(rows, features, dtype=dtype, device="cuda")
    update_source = torch.randn_like(base)
    gate_storage = torch.randn(7, 6 * features, dtype=dtype, device="cuda")
    update_gate = gate_storage[:, 2 * features : 3 * features]
    ffn_gate = gate_storage[:, 5 * features :]
    gate_indices = torch.randint(0, 7, (rows,), dtype=torch.int64, device="cuda")
    return base, update_source, update_gate, ffn_gate, gate_indices


def assert_residual_add_matches_gated_updates(
    update_op: Callable[..., None],
    ffn_kwargs: Mapping[str, object],
    output_features: int,
    *,
    python_indexing: bool,
) -> None:
    """Assert residual-add mode (base=None) matches gated-updates mode with a zero buffer."""
    source = ffn_kwargs["input"]
    assert isinstance(source, torch.Tensor)
    base = torch.randn(
        (*source.shape[:-1], output_features), device=source.device, dtype=source.dtype
    )
    ffn_gate = torch.randn(7, output_features, device=source.device, dtype=source.dtype)
    gate_indices = torch.randint(
        -7 if python_indexing else 0,
        7,
        (source.shape[:-1].numel(),),
        device=source.device,
    )
    expected = torch.zeros_like(base)
    update_op(
        **ffn_kwargs,
        output=expected,
        base=base,
        update_gate=ffn_gate,
        ffn_gate=ffn_gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    actual = base.clone()
    result = update_op(
        **ffn_kwargs,
        output=actual,
        ffn_gate=ffn_gate,
        gate_indices=gate_indices,
        python_indexing=python_indexing,
    )
    assert result is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def assert_copy_back_computes_ffn_inside_its_add(
    model: Callable[..., torch.Tensor],
    arguments: Sequence[torch.Tensor],
    update_op_name: str,
    ordinary_options: Mapping[str, object],
    fusion_options: Mapping[str, object],
    tolerance: float,
) -> None:
    """Assert a copy-back gated-update block folds its FFN inside an in-place residual add."""
    base, *operands = arguments
    capture = FunctionalTargetCapturePass()

    def copy_back(base: torch.Tensor, *operands: torch.Tensor) -> torch.Tensor:
        return base.copy_(model(base, *operands))

    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=dict(ordinary_options))(
            base.clone(), *operands
        )
        torch._dynamo.reset()
        ReinplaceCounters.clear()
        options = capture.wrap_options(compile_options(fusion_options, residual_updates=True))
        result = torch.compile(copy_back, fullgraph=True, options=options)(base, *operands)

    assert result.data_ptr() == base.data_ptr()
    assert relative_l2(base, expected) < tolerance
    assert ReinplaceCounters.get_total_missed() == 0
    kernels = torch.ops.piper_kernels
    assert capture.targets.count(getattr(kernels, update_op_name).default) == 1
    assert capture.targets.count(kernels.residual_add_.default) == 1
