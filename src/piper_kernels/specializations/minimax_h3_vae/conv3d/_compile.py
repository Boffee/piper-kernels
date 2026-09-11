"""Fold H3 framewise normalization and padding into explicit quantized convolutions."""

import operator
from collections.abc import Mapping

import torch
from torch._C._nn import pad as torch_pad
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch.nn import functional

from piper_kernels.conv3d.convrot.int8 import _ops, _policy, _validation, reference

_CONV = torch.ops.piper_kernels.convrot_int8_conv3d.default
_FUSED = torch.ops.piper_kernels.convrot_int8_group_norm_silu_conv3d.default

type _NormSiluInput = tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, int, float]


def _value(node: object) -> torch.Tensor | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("example_value", node.meta.get("val"))
    return value if isinstance(value, torch.Tensor) else None


def _method(node: object, name: str) -> torch.fx.Node | None:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_method"
        and node.target == name
        and not node.kwargs
        and len(node.users) == 1
        and node.args
        and isinstance(node.args[0], torch.fx.Node)
    ):
        return node.args[0]
    return None


def _unpermute(node: object) -> torch.fx.Node | None:
    source = _method(node, "permute")
    if source is not None:
        assert isinstance(node, torch.fx.Node)
        if node.args[1:] == (0, 2, 1, 3, 4):
            return source
    return None


def _framewise_norm_silu(node: object) -> _NormSiluInput | None:  # noqa: PLR0911
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not functional.silu
        or len(node.users) != 1
        or len(node.args) != 1
        or node.kwargs.keys() - {"inplace"}
        or node.kwargs.get("inplace", False) is not False
    ):
        return None
    restored = _method(node.args[0], "contiguous")
    unfolded = _unpermute(restored)
    norm = _method(unfolded, "view")
    if (
        norm is None
        or norm.target is not functional.group_norm
        or len(norm.users) != 1
        or len(norm.args) < 2
        or len(norm.args) > 5
        or norm.kwargs.keys() - {"weight", "bias", "eps"}
    ):
        return None
    folded = norm.args[0]
    contiguous = _method(folded, "view")
    permuted = _method(contiguous, "contiguous")
    source = _unpermute(permuted)
    source_value = _value(source)
    folded_value = _value(folded)
    unfolded_value = _value(unfolded)
    if source_value is None or folded_value is None or unfolded_value is None:
        return None
    if source_value.ndim != 5 or any(type(size) is not int for size in source_value.shape):
        return None
    batch, channels, frames, height, width = source_value.shape
    if tuple(folded_value.shape) != (batch * frames, channels, 1, height, width):
        return None
    if tuple(unfolded_value.shape) != (batch, frames, channels, height, width):
        return None
    assert source is not None
    norm_args = norm.args
    weight = norm_args[2] if len(norm_args) > 2 else norm.kwargs.get("weight")
    bias = norm_args[3] if len(norm_args) > 3 else norm.kwargs.get("bias")
    epsilon = norm_args[4] if len(norm_args) > 4 else norm.kwargs.get("eps", 1e-5)
    groups = norm_args[1]
    if (
        not isinstance(weight, torch.fx.Node)
        or not isinstance(bias, torch.fx.Node)
        or type(groups) is not int
        or type(epsilon) is not float
    ):
        return None
    weight_value, bias_value = _value(weight), _value(bias)
    if weight_value is None or bias_value is None:
        return None
    try:
        _validation._validate_norm(source_value, weight_value, bias_value, groups, epsilon)
    except (ValueError, RuntimeError):
        return None
    return source, weight, bias, groups, epsilon


def _padding(node: object) -> tuple[torch.fx.Node, bool, bool] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.target not in (functional.pad, torch_pad)
        or len(node.users) != 1
        or len(node.args) != 4
        or node.kwargs
        or node.args[2] != "reflect"
        or node.args[3] is not None
        or not isinstance(node.args[0], torch.fx.Node)
    ):
        return None
    pads = node.args[1]
    if pads in ((1, 1, 1, 1, 0, 0), (1, 1, 1, 1)):
        return node.args[0], True, False
    if pads in ((0, 1, 0, 1, 0, 0), (0, 1, 0, 1)):
        return node.args[0], False, True
    return None


def _residual(node: torch.fx.Node) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    if len(node.users) != 1:
        return None
    consumer = next(iter(node.users))
    if (
        consumer.target not in (operator.add, torch.add)
        or len(consumer.args) != 2
        or consumer.kwargs.keys() - {"alpha"}
        or consumer.kwargs.get("alpha", 1) != 1
    ):
        return None
    other = consumer.args[1] if consumer.args[0] is node else consumer.args[0]
    value, other_value = _value(node), _value(other)
    if (
        not isinstance(other, torch.fx.Node)
        or other is node
        or value is None
        or other_value is None
        or any(type(size) is not int for size in (*value.shape, *other_value.shape))
        or value.shape != other_value.shape
        or value.dtype != other_value.dtype
        or value.device != other_value.device
    ):
        return None
    return consumer, other


def fuse_conv3d(graph: torch.fx.Graph) -> None:
    """Rewrite only explicit ConvRot operators with recognized inference boundaries."""
    if torch.is_grad_enabled():
        return
    changed = False
    for node in list(graph.nodes):
        if node.target != _CONV or node.kwargs or len(node.args) != 10:
            continue
        args = list(node.args)
        symmetric, right = args[7:9]
        padding = _padding(args[0]) if symmetric is False and right is False else None
        if padding is not None:
            args[0], args[7], args[8] = padding
        norm = _framewise_norm_silu(args[0])
        target = _CONV
        if norm is not None:
            args = [*norm, *args[1:]]
            target = _FUSED
        residual = _residual(node) if args[-1] is None else None
        output = node
        if residual is not None:
            output, args[-1] = residual
        if padding is None and target == _CONV and residual is None:
            continue
        # A residual may have been produced after the convolution in graph order.
        with graph.inserting_before(output):
            replacement = graph.call_function(target, args=tuple(args))
        replacement.meta = output.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        output.replace_all_uses_with(replacement)
        graph.erase_node(output)
        if output is not node:
            graph.erase_node(node)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()


class _CompilePass(CustomGraphPass):
    def __call__(self, graph: torch.fx.Graph) -> None:
        fuse_conv3d(graph)

    def uuid(self) -> bytes:
        files = [
            __file__,
            _ops.__file__,
            _policy.__file__,
            _validation.__file__,
            reference.__file__,
        ]
        if _ops.triton_backend is not None:
            files.append(_ops.triton_backend.__file__)
        return get_hash_for_files(
            tuple(files),
            extra="minimax-h3-convrot-int8-conv3d-v1",
        )


compile_pass = _CompilePass()


def minimax_h3_vae_convrot_int8_conv3d_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install encoder fusion without loading decoder linear specializations."""
    combined = dict(options) if options is not None else {}
    existing = combined.get("pre_grad_custom_pass")
    passes = (
        tuple(existing)
        if isinstance(existing, (list, tuple))
        else (() if existing is None else (existing,))
    )
    if not any(item is compile_pass for item in passes):
        combined["pre_grad_custom_pass"] = (*passes, compile_pass)
    return combined


__all__ = ["minimax_h3_vae_convrot_int8_conv3d_compile_options"]
