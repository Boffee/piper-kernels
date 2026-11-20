"""Shared fake graphs for ordinary and ConvRot NVFP4 sparse-attention fusion."""

from __future__ import annotations

import operator

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.sparse_piper_attention._profile import _SparsePiperRatioProfile
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MINMAX_ROUTING
from piper_kernels.fusions.nvfp4_sparse_piper._compile import (
    compile_pass as nvfp4_sparse_compile_pass,
)
from piper_kernels.linear.convrot.nvfp4._compile import (
    compile_pass as convrot_nvfp4_compile_pass,
)
from piper_kernels.linear.nvfp4._compile import compile_pass as nvfp4_compile_pass


def _placeholder(
    graph: torch.fx.Graph,
    name: str,
    value: torch.Tensor,
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def _profile_placeholder(graph: torch.fx.Graph) -> torch.fx.Node:
    node = graph.placeholder("ratio_profile")
    node.meta["val"] = _SparsePiperRatioProfile((5000, 10000))
    return node


def _semantic_linear(
    graph: torch.fx.Graph,
    input: torch.fx.Node,  # noqa: A002
    prefix: str,
    activation_scale: torch.fx.Node | None,
    *,
    dynamic: bool,
    output_features: int = 256,
    high_first: bool = False,
    group_size: int | None = None,
) -> torch.fx.Node:
    weight_qdata = _placeholder(
        graph,
        f"{prefix}_weight_qdata",
        torch.empty((output_features, 128), device="cuda", dtype=torch.uint8),
    )
    weight_scale = _placeholder(
        graph,
        f"{prefix}_weight_scale",
        torch.empty(
            (((output_features + 127) // 128) * 32, 64),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        ),
    )
    weight_global_scale = _placeholder(
        graph,
        f"{prefix}_weight_global_scale",
        torch.empty((), device="cuda", dtype=torch.float32),
    )
    if group_size is None:
        target = torch.ops.piper_kernels.nvfp4_linear.default
        args: tuple[object, ...] = (
            input,
            weight_qdata,
            weight_scale,
            weight_global_scale,
            activation_scale,
            None,
            dynamic,
            high_first,
        )
    else:
        target = torch.ops.piper_kernels.convrot_nvfp4_linear.default
        args = (
            input,
            weight_qdata,
            weight_scale,
            weight_global_scale,
            activation_scale,
            None,
            dynamic,
            group_size,
            high_first,
        )
    projected = graph.call_function(target, args=args)
    projected.meta["val"] = torch.empty(
        (1, 192, output_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    return projected


def _normalized_rope(
    graph: torch.fx.Graph,
    projected: torch.fx.Node,
    norm: torch.fx.Node | None,
    cos: torch.fx.Node,
    sin: torch.fx.Node,
) -> torch.fx.Node:
    reshaped = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(projected, (1, 192, 2, 128)),
    )
    promoted = graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(reshaped, torch.float32),
    )
    squared = graph.call_function(torch.ops.aten.pow.Tensor_Scalar, args=(promoted, 2))
    mean = graph.call_function(torch.ops.aten.mean.dim, args=(squared, [3], True))
    variance = graph.call_function(torch.ops.aten.add.Scalar, args=(mean, 1e-5))
    inverse_rms = graph.call_function(torch.ops.aten.rsqrt.default, args=(variance,))
    normalized = graph.call_function(torch.ops.aten.mul.Tensor, args=(promoted, inverse_rms))
    scaled = (
        graph.call_function(torch.ops.aten.mul.Tensor, args=(normalized, norm))
        if norm is not None
        else normalized
    )
    rounded = graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(scaled, torch.bfloat16),
    )
    rotary = graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(rounded, 3, 0, 96),
    )
    split = graph.call_function(torch.ops.aten.split.Tensor, args=(rotary, 48, -1))
    first = graph.call_function(operator.getitem, args=(split, 0))
    second = graph.call_function(operator.getitem, args=(split, 1))
    cos_bf16 = graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(cos, torch.bfloat16),
    )
    cos_bf16 = graph.call_function(torch.ops.aten.unsqueeze.default, args=(cos_bf16, 0))
    cos_bf16 = graph.call_function(torch.ops.aten.unsqueeze.default, args=(cos_bf16, 2))
    direct = graph.call_function(torch.ops.aten.mul.Tensor, args=(rotary, cos_bf16))
    negated = graph.call_function(torch.ops.aten.neg.default, args=(second,))
    rotated = graph.call_function(torch.ops.aten.cat.default, args=([negated, first], -1))
    sin_bf16 = graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(sin, torch.bfloat16),
    )
    sin_bf16 = graph.call_function(torch.ops.aten.unsqueeze.default, args=(sin_bf16, 0))
    sin_bf16 = graph.call_function(torch.ops.aten.unsqueeze.default, args=(sin_bf16, 2))
    rotated = graph.call_function(torch.ops.aten.mul.Tensor, args=(rotated, sin_bf16))
    rotary_output = graph.call_function(torch.ops.aten.add.Tensor, args=(direct, rotated))
    passthrough = graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(rounded, 3, 96, torch.iinfo(torch.int64).max),
    )
    return graph.call_function(
        torch.ops.aten.cat.default,
        args=([rotary_output, passthrough], -1),
    )


def build_attention_graph(  # noqa: PLR0913
    *,
    dynamic: bool,
    output_dynamic: bool | None = None,
    escape_attention: bool = False,
    routing_mode: int = _MINMAX_ROUTING,
    with_block_lengths: bool = False,
    with_coarse: bool = False,
    with_sparse_query_blocks: bool = False,
    high_first: bool = False,
    q_affine: bool = True,
    k_affine: bool = True,
    with_force_tiles: bool = False,
    group_size: int | None = None,
) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    with FakeTensorMode():
        input = _placeholder(  # noqa: A001
            graph,
            "input",
            torch.empty((1, 192, 256), device="cuda", dtype=torch.bfloat16),
        )
        activation_scales = tuple(
            None
            if dynamic
            else _placeholder(
                graph,
                f"{prefix}_activation_scale",
                torch.empty((), device="cuda", dtype=torch.float32),
            )
            for prefix in ("q", "k", "v")
        )
        projected = tuple(
            _semantic_linear(
                graph,
                input,
                prefix,
                activation_scale,
                dynamic=dynamic,
                high_first=high_first,
                group_size=group_size,
            )
            for prefix, activation_scale in zip(("q", "k", "v"), activation_scales, strict=True)
        )
        q_norm = _placeholder(
            graph,
            "q_norm",
            torch.empty(128, device="cuda", dtype=torch.bfloat16),
        )
        k_norm = _placeholder(
            graph,
            "k_norm",
            torch.empty(128, device="cuda", dtype=torch.bfloat16),
        )
        cos = _placeholder(
            graph,
            "cos",
            torch.empty((192, 96), device="cuda", dtype=torch.float32),
        )
        sin = _placeholder(
            graph,
            "sin",
            torch.empty((192, 96), device="cuda", dtype=torch.float32),
        )
        query = _normalized_rope(graph, projected[0], q_norm if q_affine else None, cos, sin)
        key = _normalized_rope(graph, projected[1], k_norm if k_affine else None, cos, sin)
        value = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(projected[2], (1, 192, 2, 128)),
        )
        block_lengths = (
            _placeholder(
                graph,
                "block_lengths",
                torch.empty(3, device="cuda", dtype=torch.int32),
            )
            if with_block_lengths
            else None
        )
        optional_attention_arguments = (
            (block_lengths, 2)
            if with_sparse_query_blocks
            else ((block_lengths,) if with_block_lengths else ())
        )
        force_arguments = (
            (
                _placeholder(
                    graph, "force_tile_ids", torch.empty((3, 2), device="cuda", dtype=torch.int32)
                ),
                _placeholder(
                    graph, "force_tile_counts", torch.empty(3, device="cuda", dtype=torch.int32)
                ),
            )
            if with_force_tiles
            else ()
        )
        if with_force_tiles:
            optional_attention_arguments = (block_lengths, 2 if with_sparse_query_blocks else None)
        output = graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention.default,
            args=(
                query,
                key,
                value,
                _profile_placeholder(graph),
                2,
                128**-0.5,
                routing_mode,
                *optional_attention_arguments,
                *force_arguments,
            ),
        )
        output.meta["val"] = torch.empty(
            (1, 192, 2, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        if with_coarse:
            coarse_gate = _placeholder(
                graph,
                "coarse_gate",
                torch.empty((1, 192, 2, 128), device="cuda", dtype=torch.bfloat16),
            )
            coarse_output = graph.call_function(
                torch.ops.piper_kernels.sparse_piper_coarse_residual.default,
                args=(
                    query,
                    key,
                    value,
                    coarse_gate,
                    3,
                    0.125,
                    routing_mode,
                    *((block_lengths,) if with_block_lengths else ()),
                ),
            )
            output = graph.call_function(
                torch.ops.aten.add.Tensor,
                args=(output, coarse_output),
            )
            output.meta["val"] = torch.empty(
                (1, 192, 2, 128),
                device="cuda",
                dtype=torch.bfloat16,
            )
        attention_output = output
        if output_dynamic is not None:
            reshaped_output = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(output, (1, 192, 256)),
            )
            reshaped_output.meta["val"] = torch.empty(
                (1, 192, 256),
                device="cuda",
                dtype=torch.bfloat16,
            )
            output_activation_scale = (
                None
                if output_dynamic
                else _placeholder(
                    graph,
                    "output_activation_scale",
                    torch.empty((), device="cuda", dtype=torch.float32),
                )
            )
            output = _semantic_linear(
                graph,
                reshaped_output,
                "output",
                output_activation_scale,
                dynamic=output_dynamic,
                output_features=320,
                high_first=high_first,
                group_size=group_size,
            )
        graph.output((output, attention_output) if escape_attention else output)
    torch.fx.GraphModule({}, graph)
    return graph


def _append_quantized_attention_output(
    graph: torch.fx.Graph,
    prefix: str,
    *,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
    high_first: bool = False,
    group_size: int | None = None,
) -> torch.fx.Node:
    tensors = {
        "query": torch.empty((1, 2, 192, 128), device="cuda", dtype=torch.int8),
        "query_scale": torch.empty((1, 2, 6), device="cuda", dtype=torch.float32),
        "query_summary": torch.empty((1, 2, 3, 128), device="cuda", dtype=torch.float32),
        "key": torch.empty((1, 2, 192, 128), device="cuda", dtype=torch.int8),
        "key_scale": torch.empty((1, 2, 3), device="cuda", dtype=torch.float32),
        "key_center": torch.empty((1, 2, 3, 2, 128), device="cuda", dtype=torch.int8),
        "key_center_scale": torch.empty((1, 2, 3, 2), device="cuda", dtype=torch.float32),
        "key_summary": torch.empty((1, 2, 3, 128), device="cuda", dtype=torch.float32),
        "key_aux": torch.empty((1, 2, 3, 128), device="cuda", dtype=torch.float32),
        "value": torch.empty((1, 2, 128, 192), device="cuda", dtype=torch.int8),
        "value_scale": torch.empty((1, 2, 3, 1), device="cuda", dtype=torch.float32),
        "value_mean": torch.empty((1, 2, 3, 128), device="cuda", dtype=torch.bfloat16),
    }
    operands = tuple(
        _placeholder(graph, f"{prefix}{name}", value) for name, value in tensors.items()
    )
    block_lengths = (
        _placeholder(
            graph,
            f"{prefix}block_lengths",
            torch.empty(3, device="cuda", dtype=torch.int32),
        )
        if with_block_lengths
        else None
    )
    if with_coarse:
        coarse_gate = _placeholder(
            graph,
            f"{prefix}coarse_gate",
            torch.empty((1, 192, 2, 128), device="cuda", dtype=torch.bfloat16),
        )
        attention = graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default,
            args=(
                *operands,
                coarse_gate,
                [5_000, 10_000],
                2,
                192,
                _MINMAX_ROUTING,
                0.125,
                3,
                block_lengths,
                *((2,) if with_sparse_query_blocks else ()),
            ),
        )
    else:
        arguments = (*operands, [5_000, 10_000], 2, 192, _MINMAX_ROUTING)
        optional_arguments = (
            (block_lengths, 2)
            if with_sparse_query_blocks
            else ((block_lengths,) if with_block_lengths else ())
        )
        attention = graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
            args=(*arguments, *optional_arguments),
        )
    attention.meta["val"] = torch.empty(
        (1, 192, 2, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    reshaped = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(attention, (1, 192, 256)),
    )
    reshaped.meta["val"] = torch.empty(
        (1, 192, 256),
        device="cuda",
        dtype=torch.bfloat16,
    )
    activation_scale = _placeholder(
        graph,
        f"{prefix}output_activation_scale",
        torch.empty((), device="cuda", dtype=torch.float32),
    )
    return _semantic_linear(
        graph,
        reshaped,
        f"{prefix}output",
        activation_scale,
        dynamic=False,
        output_features=320,
        high_first=high_first,
        group_size=group_size,
    )


def build_quantized_attention_output_graph(
    *,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
    high_first: bool = False,
    group_size: int | None = None,
) -> torch.fx.Graph:
    """Build an already-prepared attention boundary for output-only folding."""
    graph = torch.fx.Graph()
    with FakeTensorMode():
        output = _append_quantized_attention_output(
            graph,
            "",
            with_block_lengths=with_block_lengths,
            with_coarse=with_coarse,
            with_sparse_query_blocks=with_sparse_query_blocks,
            high_first=high_first,
            group_size=group_size,
        )
        graph.output(output)
    torch.fx.GraphModule({}, graph)
    return graph


def build_mixed_quantized_attention_output_graph(
    group_size: int = 64,
) -> torch.fx.Graph:
    """Build one graph containing both plain NVFP4 and ConvRot NVFP4 output blocks."""
    graph = torch.fx.Graph()
    with FakeTensorMode():
        plain = _append_quantized_attention_output(
            graph,
            "plain_",
            with_block_lengths=False,
            with_coarse=False,
            with_sparse_query_blocks=False,
            group_size=None,
        )
        convrot = _append_quantized_attention_output(
            graph,
            "convrot_",
            with_block_lengths=False,
            with_coarse=False,
            with_sparse_query_blocks=False,
            group_size=group_size,
        )
        graph.output((plain, convrot))
    torch.fx.GraphModule({}, graph)
    return graph


def run_nvfp4_passes(graph: torch.fx.Graph, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the unified NVFP4 + ConvRot NVFP4 sparse-Piper pass pipeline on a fake graph."""
    monkeypatch.setattr(
        AcceleratorTarget,
        "from_device",
        classmethod(lambda _cls, _device: AcceleratorTarget("cuda", "sm120")),
    )
    nvfp4_compile_pass(graph, is_inference=True)
    convrot_nvfp4_compile_pass(graph, is_inference=True)
    nvfp4_sparse_compile_pass(graph, is_inference=True)
