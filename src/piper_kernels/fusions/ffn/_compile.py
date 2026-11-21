"""Unified imperative anchor-based FX pass for quantized FFN fusions.

Matches semantic ConvRot INT8, NVFP4, and ConvRot NVFP4 GELU-tanh and SwiGLU
FFNs directly from each ``down`` projection anchor using schema argument binding,
folding optional indexed gated updates and copy-back residual adds while keeping
format extras and Triton lazily imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch.fx.node import Argument

from piper_kernels import _input_activations as input_activations
from piper_kernels.fusions.residual_updates import _gates as residual_gates
from piper_kernels.linear import _bias
from piper_kernels.linear import _compile_fx as linear_fx
from piper_kernels.linear import _input_activation_compile as input_activation_compile
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear import _projection_views as projection_views

_aten = torch.ops.aten

_CONVROT_INT8_LINEAR = "piper_kernels::convrot_int8_linear"
_CONVROT_NVFP4_LINEAR = "piper_kernels::convrot_nvfp4_linear"
_CONVROT_INT8_FFN = "piper_kernels::convrot_int8_ffn"
_NVFP4_FFN = "piper_kernels::nvfp4_ffn"
_NVFP4_LINEAR = "piper_kernels::nvfp4_linear"

type _Format = Literal["convrot_int8", "nvfp4"]


@dataclass(frozen=True, slots=True)
class _BoundProjection:
    """One semantic linear node parsed via its PyTorch operator schema."""

    node: torch.fx.Node
    schema_name: str
    args: dict[str, Argument]

    @property
    def format(self) -> _Format:
        return "convrot_int8" if self.schema_name == _CONVROT_INT8_LINEAR else "nvfp4"

    @property
    def input_node(self) -> torch.fx.Node:
        value = self.args["input"]
        assert isinstance(value, torch.fx.Node)
        return value


@dataclass(frozen=True, slots=True)
class _GatedUpdatesMatch:
    """Validated two-update indexed residual wrapper around an FFN."""

    output_node: torch.fx.Node
    base: torch.fx.Node
    reusable_update: torch.fx.Node
    update_gate: torch.fx.Node
    ffn_gate: torch.fx.Node
    gate_indices: torch.fx.Node
    python_indexing: bool


def _parse_projection(node: object) -> _BoundProjection | None:
    if not isinstance(node, torch.fx.Node):
        return None
    name = linear_fx.schema_name(node)
    if name not in linear_fx.SEMANTIC_LINEAR_NAMES:
        return None
    bound = linear_fx.bind_arguments(node)
    if bound is None or not isinstance(bound.get("input"), torch.fx.Node):
        return None
    if name == _CONVROT_INT8_LINEAR and bound.get("activation_fn") is not None:
        return None
    return _BoundProjection(node=node, schema_name=name, args=bound)


def _node_metadata(value: Argument) -> torch.Tensor | None:
    return preparation_sharing.tensor_metadata(value) if isinstance(value, torch.fx.Node) else None


def _dimension_matches(left: int | torch.SymInt, right: int | torch.SymInt) -> bool:
    return preparation_sharing.dimension_key(left) == preparation_sharing.dimension_key(right)


def _shape_matches(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.ndim == right.ndim and all(
        _dimension_matches(a, b) for a, b in zip(left.shape, right.shape, strict=True)
    )


def _match_gelu_tanh(
    activated: torch.fx.Node,
) -> tuple[_BoundProjection, torch.dtype] | None:
    """Match PyTorch's promoted FP32 tanh-GELU decomposition feeding a down linear."""
    matched = input_activation_compile.match_gelu_tanh_input(
        activated,
        allow_unpromoted_fp32=False,
    )
    if matched is None:
        return None
    source, logical_dtype = matched
    if logical_dtype is None or len(source.users) != 1:
        return None
    up_projection = _parse_projection(source)
    return None if up_projection is None else (up_projection, logical_dtype)


def _match_decomposed_silu(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    """Match ``x / (exp(-x) + 1)`` and return ``(x, neg_node)``."""
    div_args = linear_fx.binary_args(node, _aten.div.Tensor)
    if div_args is None:
        return None
    numerator, denominator = div_args
    if (
        not isinstance(numerator, torch.fx.Node)
        or not isinstance(denominator, torch.fx.Node)
        or len(denominator.users) != 1
    ):
        return None
    exp_node = linear_fx.match_node_and_scalar(denominator, _aten.add.Tensor, 1.0)
    neg_node = (
        linear_fx.unary_arg(exp_node, _aten.exp.default)
        if exp_node is not None and len(exp_node.users) == 1
        else None
    )
    if (
        neg_node is None
        or len(neg_node.users) != 1
        or linear_fx.unary_arg(neg_node, _aten.neg.default) is not numerator
    ):
        return None
    return numerator, neg_node


def _match_promoted_silu(
    div_node: torch.fx.Node,
    logical_dtype: torch.dtype,
) -> tuple[_BoundProjection, torch.dtype] | None:
    decomposed = _match_decomposed_silu(div_node) if len(div_node.users) == 1 else None
    if decomposed is None:
        return None
    gate_fp32, neg_node = decomposed
    if set(gate_fp32.users) != {div_node, neg_node}:
        return None
    upcast = linear_fx.convert_dtype(gate_fp32)
    if upcast is None or upcast[1] is not torch.float32 or len(upcast[0].users) != 1:
        return None
    projection = _parse_projection(upcast[0])
    return None if projection is None else (projection, logical_dtype)


def _match_silu(
    silu_node: torch.fx.Node,
) -> tuple[_BoundProjection, torch.dtype | None] | None:
    """Match ``aten.silu`` or its decomposed/FP32-promoted form feeding SwiGLU."""
    if len(silu_node.users) != 1:
        return None
    direct_gate = linear_fx.unary_arg(silu_node, _aten.silu.default)
    if direct_gate is not None:
        projection = _parse_projection(direct_gate) if len(direct_gate.users) == 1 else None
        return None if projection is None else (projection, None)

    downcast = linear_fx.convert_dtype(silu_node)
    if downcast is not None:
        return _match_promoted_silu(*downcast)

    decomposed = _match_decomposed_silu(silu_node)
    if decomposed is None or set(decomposed[0].users) != {silu_node, decomposed[1]}:
        return None
    projection = _parse_projection(decomposed[0])
    return None if projection is None else (projection, None)


def _match_swiglu(
    activated: torch.fx.Node,
) -> tuple[_BoundProjection, _BoundProjection, torch.dtype | None] | None:
    """Match ``value * silu(gate)`` in either operand order."""
    if len(activated.users) != 1:
        return None
    mul_args = linear_fx.binary_args(activated, _aten.mul.Tensor)
    if mul_args is None:
        return None
    for value_arg, silu_arg in (mul_args, mul_args[::-1]):
        if not isinstance(value_arg, torch.fx.Node) or not isinstance(silu_arg, torch.fx.Node):
            continue
        if len(value_arg.users) != 1:
            continue
        value_projection = _parse_projection(value_arg)
        if value_projection is None:
            continue
        matched_silu = _match_silu(silu_arg)
        if matched_silu is None:
            continue
        gate_projection, logical_dtype = matched_silu
        return value_projection, gate_projection, logical_dtype
    return None


def _valid_convrot_int8_bias(
    bias: Argument,
    *,
    features: int | torch.SymInt,
    input_value: torch.Tensor,
) -> bool:
    if bias is None:
        return True
    value = _node_metadata(bias)
    return bool(
        value is not None
        and value.ndim == 1
        and _dimension_matches(value.shape[0], features)
        and _bias.is_supported_dtype(value.dtype)
        and value.device == input_value.device
        and value.layout is torch.strided
        and value.is_contiguous()
    )


def _valid_convrot_int8_ffn(  # noqa: PLR0911
    sources: tuple[_BoundProjection, ...],
    down: _BoundProjection,
    promoted_dtype: torch.dtype | None,
) -> bool:
    from piper_kernels.linear.convrot.int8 import _backend, _compile_fx  # noqa: PLC0415

    if any(p.format != "convrot_int8" for p in (*sources, down)):
        return False
    if any(source.input_node is not sources[0].input_node for source in sources[1:]):
        return False
    input_value = _node_metadata(sources[0].input_node)
    output_value = _node_metadata(down.node)
    projections = (*sources, down)
    weights = tuple(_node_metadata(p.args["weight_qdata"]) for p in projections)
    scales = tuple(_node_metadata(p.args["weight_scale"]) for p in projections)
    if (
        input_value is None
        or output_value is None
        or any(value is None for value in (*weights, *scales))
    ):
        return False
    if promoted_dtype is not None and promoted_dtype is not input_value.dtype:
        return False
    concrete_weights = cast(tuple[torch.Tensor, ...], weights)
    concrete_scales = cast(tuple[torch.Tensor, ...], scales)
    rows = math.prod(input_value.shape[:-1])
    if any(
        not _compile_fx.valid_input_scale(p.args["input_scale"], input_value.device)
        for p in projections
    ):
        return False
    if (
        input_value.ndim == 0
        or (isinstance(rows, int) and rows < 1)
        or input_value.dtype not in (torch.float16, torch.bfloat16)
        or input_value.layout is not torch.strided
        or not input_value.is_contiguous()
        or any(
            weight.ndim != 2
            or weight.dtype is not torch.int8
            or weight.device != input_value.device
            or weight.layout is not torch.strided
            or not weight.is_contiguous()
            for weight in concrete_weights
        )
        or any(
            scale.dtype is not torch.float32
            or scale.device != input_value.device
            or scale.layout is not torch.strided
            or not scale.is_contiguous()
            for scale in concrete_scales
        )
        or _backend.select_linear_backend(input_value) is None
    ):
        return False

    source_weights = concrete_weights[:-1]
    source_scales = concrete_scales[:-1]
    down_weight = concrete_weights[-1]
    down_scale = concrete_scales[-1]
    input_features = source_weights[0].shape[1]
    intermediate_features = source_weights[0].shape[0]
    output_features = down_weight.shape[0]
    if (
        not _dimension_matches(input_value.shape[-1], input_features)
        or any(not _shape_matches(source_weights[0], source) for source in source_weights[1:])
        or not _dimension_matches(down_weight.shape[1], intermediate_features)
        or any(scale.shape != (intermediate_features, 1) for scale in source_scales)
        or down_scale.shape != (output_features, 1)
        or output_value.dtype is not input_value.dtype
        or output_value.device != input_value.device
        or output_value.ndim != input_value.ndim
        or any(
            not _dimension_matches(out_dim, in_dim)
            for out_dim, in_dim in zip(output_value.shape[:-1], input_value.shape[:-1], strict=True)
        )
        or not _dimension_matches(output_value.shape[-1], output_features)
    ):
        return False

    group_sizes = tuple(p.args["group_size"] for p in projections)
    if any(
        isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1
        for group_size in group_sizes
    ):
        return False
    typed_groups = cast(tuple[int, ...], group_sizes)
    source_groups = typed_groups[:-1]
    down_group = typed_groups[-1]
    if (
        any(group != source_groups[0] for group in source_groups[1:])
        or (isinstance(input_features, int) and input_features % source_groups[0])
        or (isinstance(intermediate_features, int) and intermediate_features % down_group)
    ):
        return False
    return all(
        _valid_convrot_int8_bias(p.args["bias"], features=features, input_value=input_value)
        for p, features in (
            *((source, intermediate_features) for source in sources),
            (down, output_features),
        )
    )


def _parse_nvfp4_projections(
    projections: tuple[_BoundProjection, ...],
) -> tuple[tuple[object, int | None], ...] | None:
    from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx  # noqa: PLC0415

    parsed: list[tuple[object, int | None]] = []
    for projection in projections:
        linear = nvfp4_compile_fx.SemanticLinearNodes.from_values(
            tuple(
                projection.args[name]
                for name in (
                    "input",
                    "weight_qdata",
                    "weight_scale",
                    "weight_per_tensor_scale",
                    "activation_per_tensor_scale",
                    "bias",
                    "dynamic_activation_scale",
                    "high_first",
                )
            )
        )
        if linear is None:
            return None
        group_size: int | None = None
        if projection.schema_name == _CONVROT_NVFP4_LINEAR:
            raw_group = projection.args["group_size"]
            if isinstance(raw_group, bool) or not isinstance(raw_group, int):
                return None
            group_size = raw_group
        parsed.append((linear, group_size))
    return tuple(parsed)


def _valid_nvfp4_convrot_groups(
    parsed: tuple[tuple[object, int | None], ...],
) -> bool:
    if not any(group is not None for _linear, group in parsed):
        return True
    from piper_kernels.linear.convrot.nvfp4 import (  # noqa: PLC0415
        _compile_fx as convrot_nvfp4_compile_fx,
    )
    from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx  # noqa: PLC0415

    return all(
        convrot_nvfp4_compile_fx.validated_semantic_linear(
            convrot_nvfp4_compile_fx.SemanticLinearNodes(
                cast(nvfp4_compile_fx.SemanticLinearNodes, linear), group
            ),
            f"ConvRot NVFP4 FFN compiler projection {index}",
        )
        is not None
        for index, (linear, group) in enumerate(parsed)
        if group is not None
    )


def _valid_nvfp4_ffn(
    sources: tuple[_BoundProjection, ...],
    down: _BoundProjection,
    promoted_dtype: torch.dtype | None,
) -> bool:
    from piper_kernels.linear.nvfp4 import _compile_fx as nvfp4_compile_fx  # noqa: PLC0415

    if any(p.format != "nvfp4" for p in (*sources, down)):
        return False
    if any(source.input_node is not sources[0].input_node for source in sources[1:]):
        return False
    parsed = _parse_nvfp4_projections((*sources, down))
    if parsed is None:
        return False
    source_groups = tuple(group for _linear, group in parsed[:-1])
    if any(group != source_groups[0] for group in source_groups[1:]) or not (
        _valid_nvfp4_convrot_groups(parsed)
    ):
        return False

    linears = tuple(cast(nvfp4_compile_fx.SemanticLinearNodes, linear) for linear, _ in parsed)
    validated_sources = tuple(
        nvfp4_compile_fx.validated_semantic_linear(
            linear,
            f"NVFP4 FFN compiler source projection {index}",
        )
        for index, linear in enumerate(linears[:-1])
    )
    validated_down = nvfp4_compile_fx.validated_semantic_linear(
        linears[-1],
        "NVFP4 FFN compiler down projection",
    )
    output_value = _node_metadata(down.node)
    if (
        validated_down is None
        or output_value is None
        or any(validated is None for validated in validated_sources)
    ):
        return False
    source_values = tuple(v for v in validated_sources if v is not None)
    input_value, source_shape = source_values[0]
    down_input, down_shape = validated_down
    source_linears = linears[:-1]
    return bool(
        (promoted_dtype is None or promoted_dtype is input_value.dtype)
        and input_value.dtype in (torch.float16, torch.bfloat16)
        and input_value.layout is torch.strided
        and input_value.is_contiguous()
        and all(value.dtype is input_value.dtype for value, _shape in source_values[1:])
        and down_input.dtype is input_value.dtype
        and all(
            _dimension_matches(source_shape.rows, shape.rows) for _value, shape in source_values
        )
        and _dimension_matches(source_shape.rows, down_shape.rows)
        and all(
            _dimension_matches(source_shape.input_features, shape.input_features)
            for _value, shape in source_values
        )
        and all(
            _dimension_matches(source_shape.output_features, shape.output_features)
            for _value, shape in source_values
        )
        and _dimension_matches(down_shape.input_features, source_shape.output_features)
        and all(linear.high_first == source_linears[0].high_first for linear in source_linears[1:])
        and output_value.dtype is input_value.dtype
        and output_value.device == input_value.device
        and output_value.ndim == input_value.ndim
        and all(
            _dimension_matches(out_dim, in_dim)
            for out_dim, in_dim in zip(output_value.shape[:-1], input_value.shape[:-1], strict=True)
        )
        and _dimension_matches(output_value.shape[-1], down_shape.output_features)
    )


def _match_gated_update_term(
    gated_update_arg: Argument,
    gate_indices: torch.fx.Node,
    python_indexing: bool,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    if not isinstance(gated_update_arg, torch.fx.Node) or len(gated_update_arg.users) != 1:
        return None
    update_mul = linear_fx.binary_args(gated_update_arg, _aten.mul.Tensor)
    if update_mul is None:
        return None
    for update_gate_sel, reusable_arg in (update_mul, update_mul[::-1]):
        if (
            not isinstance(update_gate_sel, torch.fx.Node)
            or not isinstance(reusable_arg, torch.fx.Node)
            or len(update_gate_sel.users) != 1
            or len(reusable_arg.users) != 1
            or not preparation_sharing.operator_returns_fresh_tensor(reusable_arg)
        ):
            continue
        selected_update = residual_gates.selected_gate(update_gate_sel)
        if selected_update is None:
            continue
        update_gate, update_indices, update_py_idx = selected_update
        if update_indices is gate_indices and update_py_idx == python_indexing:
            return update_gate, reusable_arg
    return None


def _valid_gated_updates_metadata(
    ffn_input: torch.fx.Node,
    base_arg: torch.fx.Node,
    reusable_arg: torch.fx.Node,
    update_gate: torch.fx.Node,
    gate_indices: torch.fx.Node,
    ffn_gate: torch.fx.Node,
    output_node: torch.fx.Node,
) -> bool:
    input_val = _node_metadata(ffn_input)
    base_val = _node_metadata(base_arg)
    reusable_val = _node_metadata(reusable_arg)
    update_gate_val = _node_metadata(update_gate)
    indices_val = _node_metadata(gate_indices)
    ffn_gate_val = _node_metadata(ffn_gate)
    output_val = _node_metadata(output_node)
    if (
        input_val is None
        or base_val is None
        or reusable_val is None
        or update_gate_val is None
        or indices_val is None
        or ffn_gate_val is None
        or output_val is None
    ):
        return False
    from piper_kernels.fusions.residual_updates import _ops as residual_ops  # noqa: PLC0415

    return bool(
        _shape_matches(base_val, output_val)
        and base_val.dtype is input_val.dtype
        and base_val.device == input_val.device
        and residual_ops.supports_residual_add(base_val, reusable_val, update_gate_val, indices_val)
        and residual_ops.supports_residual_add(base_val, None, ffn_gate_val, indices_val)
    )


def _match_gated_updates(
    down_node: torch.fx.Node,
    ffn_input: torch.fx.Node,
) -> _GatedUpdatesMatch | None:
    """Match ``base + update_gate[i] * reusable_update + ffn_gate[i] * down``."""
    if len(down_node.users) != 1:
        return None
    gated_ffn = next(iter(down_node.users))
    mul_args = linear_fx.binary_args(gated_ffn, _aten.mul.Tensor)
    if mul_args is None or len(gated_ffn.users) != 1:
        return None
    ffn_gate_sel = mul_args[1] if mul_args[0] is down_node else mul_args[0]
    selected_ffn = (
        residual_gates.selected_gate(ffn_gate_sel)
        if isinstance(ffn_gate_sel, torch.fx.Node) and len(ffn_gate_sel.users) == 1
        else None
    )
    output_node = next(iter(gated_ffn.users))
    out_add = linear_fx.binary_args(output_node, _aten.add.Tensor)
    if selected_ffn is None or out_add is None:
        return None
    ffn_gate, gate_indices, python_indexing = selected_ffn

    hidden = out_add[1] if out_add[0] is gated_ffn else out_add[0]
    hidden_add = (
        linear_fx.binary_args(hidden, _aten.add.Tensor)
        if isinstance(hidden, torch.fx.Node) and len(hidden.users) == 2
        else None
    )
    if hidden_add is None:
        return None

    for base_arg, gated_update_arg in (hidden_add, hidden_add[::-1]):
        if not isinstance(base_arg, torch.fx.Node):
            continue
        matched_term = _match_gated_update_term(gated_update_arg, gate_indices, python_indexing)
        if matched_term is None:
            continue
        update_gate, reusable_arg = matched_term
        if _valid_gated_updates_metadata(
            ffn_input, base_arg, reusable_arg, update_gate, gate_indices, ffn_gate, output_node
        ):
            return _GatedUpdatesMatch(
                output_node=output_node,
                base=base_arg,
                reusable_update=reusable_arg,
                update_gate=update_gate,
                ffn_gate=ffn_gate,
                gate_indices=gate_indices,
                python_indexing=python_indexing,
            )
    return None


def _emit_convrot_int8_ffn(
    graph: torch.fx.Graph,
    sources: tuple[_BoundProjection, ...],
    down: _BoundProjection,
    activation_fn: input_activations.InputActivation,
    gated: _GatedUpdatesMatch | None,
) -> None:
    from . import _convrot_int8  # noqa: PLC0415

    up = sources[0]
    gate = sources[1] if len(sources) == 2 else None
    input_val = _node_metadata(up.input_node)
    up_weight_val = _node_metadata(up.args["weight_qdata"])
    down_weight_val = _node_metadata(down.args["weight_qdata"])
    assert input_val is not None
    assert up_weight_val is not None
    assert down_weight_val is not None
    chunk_rows = _convrot_int8.default_chunk_rows(
        input_val,
        up_weight_val,
        down_weight_val,
        activation_fn,
        adds_residual=gated is not None,
    )
    common = (
        up.input_node,
        up.args["weight_qdata"],
        up.args["weight_scale"],
        up.args["bias"],
        up.args["group_size"],
        down.args["weight_qdata"],
        down.args["weight_scale"],
        down.args["bias"],
        down.args["group_size"],
        activation_fn,
    )
    trailing = (
        None if gate is None else gate.args["weight_qdata"],
        None if gate is None else gate.args["weight_scale"],
        None if gate is None else gate.args["bias"],
        None if gate is None else gate.args["group_size"],
        up.args["input_scale"],
        None if gate is None else gate.args["input_scale"],
        down.args["input_scale"],
    )
    if gated is not None:
        with graph.inserting_before(gated.output_node):
            mutation = graph.call_function(
                torch.ops.piper_kernels.convrot_int8_ffn_update_.default,
                args=(
                    *common,
                    gated.reusable_update,
                    gated.ffn_gate,
                    gated.gate_indices,
                    gated.python_indexing,
                    chunk_rows,
                    gated.base,
                    gated.update_gate,
                    *trailing,
                ),
            )
        mutation.meta["val"] = None
        gated.output_node.replace_all_uses_with(gated.reusable_update)
        graph.erase_node(gated.output_node)
        return

    with graph.inserting_before(down.node):
        replacement = graph.call_function(
            torch.ops.piper_kernels.convrot_int8_ffn.default,
            args=(*common, chunk_rows, *trailing),
        )
    replacement.meta = down.node.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    down.node.replace_all_uses_with(replacement)
    graph.erase_node(down.node)


def _nvfp4_projection_args(projection: _BoundProjection) -> tuple[Argument, ...]:
    group_size = (
        projection.args["group_size"] if projection.schema_name == _CONVROT_NVFP4_LINEAR else None
    )
    return (
        projection.args["weight_qdata"],
        projection.args["weight_scale"],
        projection.args["weight_per_tensor_scale"],
        projection.args["activation_per_tensor_scale"],
        projection.args["bias"],
        projection.args["dynamic_activation_scale"],
        group_size,
        projection.args["high_first"],
    )


def _emit_nvfp4_ffn(
    graph: torch.fx.Graph,
    sources: tuple[_BoundProjection, ...],
    down: _BoundProjection,
    activation_fn: input_activations.InputActivation,
    gated: _GatedUpdatesMatch | None,
) -> None:
    from . import _nvfp4  # noqa: PLC0415

    up = sources[0]
    gate = sources[1] if len(sources) == 2 else None
    input_val = _node_metadata(up.input_node)
    up_weight_val = _node_metadata(up.args["weight_qdata"])
    down_weight_val = _node_metadata(down.args["weight_qdata"])
    assert input_val is not None
    assert up_weight_val is not None
    assert down_weight_val is not None
    chunk_rows = _nvfp4.default_chunk_rows(
        input_val,
        up_weight_val,
        down_weight_val,
        activation_fn,
        adds_residual=gated is not None,
    )
    common = (
        up.input_node,
        *_nvfp4_projection_args(up),
        *_nvfp4_projection_args(down),
        activation_fn,
    )
    gate_args: tuple[Argument, ...] = (
        (None, None, None, None, None, False, None, False)
        if gate is None
        else _nvfp4_projection_args(gate)
    )
    if gated is not None:
        with graph.inserting_before(gated.output_node):
            mutation = graph.call_function(
                torch.ops.piper_kernels.nvfp4_ffn_update_.default,
                args=(
                    *common,
                    gated.reusable_update,
                    gated.ffn_gate,
                    gated.gate_indices,
                    gated.python_indexing,
                    chunk_rows,
                    gated.base,
                    gated.update_gate,
                    *gate_args,
                ),
            )
        mutation.meta["val"] = None
        gated.output_node.replace_all_uses_with(gated.reusable_update)
        graph.erase_node(gated.output_node)
        return

    with graph.inserting_before(down.node):
        replacement = graph.call_function(
            torch.ops.piper_kernels.nvfp4_ffn.default,
            args=(*common, chunk_rows, *gate_args),
        )
    replacement.meta = down.node.meta.copy()
    replacement.meta.pop("eager_input_vals", None)
    down.node.replace_all_uses_with(replacement)
    graph.erase_node(down.node)


def _fold_residual_adds(
    graph: torch.fx.Graph,
    *,
    has_convrot_int8: bool,
    has_nvfp4: bool,
) -> int:
    has_auto_func = any(
        node.op == "call_function" and node.target is torch.ops.higher_order.auto_functionalized
        for node in graph.nodes
    )
    if not has_auto_func:
        return 0
    from piper_kernels.fusions.residual_updates import _graph as residual_graph  # noqa: PLC0415

    replaced = 0
    if has_convrot_int8:
        from . import _convrot_int8  # noqa: PLC0415

        def _compute_int8(
            graph: torch.fx.Graph,
            update: torch.fx.Node,
            add: dict[str, Argument],
        ) -> torch.fx.Node:
            values = linear_fx.bind_arguments(update)
            assert values is not None
            input_val = _node_metadata(values["input"])
            up_weight = _node_metadata(values["up_weight_qdata"])
            down_weight = _node_metadata(values["down_weight_qdata"])
            assert input_val is not None
            assert up_weight is not None
            assert down_weight is not None
            activation_fn = cast(input_activations.InputActivation, values["activation_fn"])
            values.update(
                output=add["base"],
                base=None,
                update_gate=None,
                ffn_gate=add["gate"],
                gate_indices=add["gate_indices"],
                python_indexing=add["python_indexing"],
                chunk_rows=_convrot_int8.default_chunk_rows(
                    input_val,
                    up_weight,
                    down_weight,
                    activation_fn,
                    adds_residual=True,
                ),
            )
            return residual_graph.call_functional(
                graph,
                torch.ops.piper_kernels.convrot_int8_ffn_update_.default,
                values,
            )

        replaced += residual_graph.compute_inside_adds(
            graph,
            torch.ops.piper_kernels.convrot_int8_ffn.default,
            _compute_int8,
        )

    if has_nvfp4:
        from . import _nvfp4  # noqa: PLC0415

        def _compute_nvfp4(
            graph: torch.fx.Graph,
            update: torch.fx.Node,
            add: dict[str, Argument],
        ) -> torch.fx.Node:
            values = linear_fx.bind_arguments(update)
            assert values is not None
            input_val = _node_metadata(values["input"])
            up_weight = _node_metadata(values["up_weight_qdata"])
            down_weight = _node_metadata(values["down_weight_qdata"])
            assert input_val is not None
            assert up_weight is not None
            assert down_weight is not None
            activation_fn = cast(input_activations.InputActivation, values["activation_fn"])
            values.update(
                output=add["base"],
                base=None,
                update_gate=None,
                ffn_gate=add["gate"],
                gate_indices=add["gate_indices"],
                python_indexing=add["python_indexing"],
                chunk_rows=_nvfp4.default_chunk_rows(
                    input_val,
                    up_weight,
                    down_weight,
                    activation_fn,
                    adds_residual=True,
                ),
            )
            return residual_graph.call_functional(
                graph,
                torch.ops.piper_kernels.nvfp4_ffn_update_.default,
                values,
            )

        replaced += residual_graph.compute_inside_adds(
            graph,
            torch.ops.piper_kernels.nvfp4_ffn.default,
            _compute_nvfp4,
        )

    return replaced


def fold_ffns(graph: torch.fx.Graph) -> bool:
    """Fold semantic GELU-tanh and SwiGLU FFNs before ordinary linear normalization."""
    schema_names = {
        name for node in graph.nodes if (name := linear_fx.schema_name(node)) is not None
    }
    has_convrot_int8 = _CONVROT_INT8_LINEAR in schema_names or _CONVROT_INT8_FFN in schema_names
    has_nvfp4 = (
        _NVFP4_LINEAR in schema_names
        or _CONVROT_NVFP4_LINEAR in schema_names
        or _NVFP4_FFN in schema_names
    )
    if not (has_convrot_int8 or has_nvfp4):
        return False

    changed = projection_views.normalize_projection_views(graph)
    changed = linear_fx.coalesce_input_casts(graph) or changed
    for node in list(graph.nodes):
        down = _parse_projection(node)
        if down is None:
            continue
        activated = down.input_node
        gelu = _match_gelu_tanh(activated)
        if gelu is not None:
            up_proj, promoted_dtype = gelu
            activation_fn: input_activations.InputActivation = "gelu_tanh"
            sources: tuple[_BoundProjection, ...] = (up_proj,)
        else:
            swiglu = _match_swiglu(activated)
            if swiglu is None:
                continue
            value_proj, gate_proj, promoted_dtype = swiglu
            activation_fn = "swiglu"
            sources = (value_proj, gate_proj)

        if down.format == "convrot_int8":
            if not _valid_convrot_int8_ffn(sources, down, promoted_dtype):
                continue
            gated = _match_gated_updates(down.node, sources[0].input_node)
            _emit_convrot_int8_ffn(graph, sources, down, activation_fn, gated)
        else:
            if not _valid_nvfp4_ffn(sources, down, promoted_dtype):
                continue
            gated = _match_gated_updates(down.node, sources[0].input_node)
            _emit_nvfp4_ffn(graph, sources, down, activation_fn, gated)
        changed = True

    residual_replaced = _fold_residual_adds(
        graph,
        has_convrot_int8=has_convrot_int8,
        has_nvfp4=has_nvfp4,
    )
    changed = changed or residual_replaced > 0
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed


__all__ = ["fold_ffns"]
