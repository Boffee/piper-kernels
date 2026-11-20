"""Fold explicit zero/spatial reflection padding into quantized convolutions.

This optional pre-grad Inductor pass lowers standard ``F.conv3d`` calls on
``ConvRotInt8Tensor`` weights and absorbs exclusive chains of zero padding and
spatial reflection without altering ordinary floating-point convolutions:

- Weight storage (``qdata``, ``scale``, ``act_per_tensor_scale``) is read
  through runtime ``getattr`` nodes rather than captured values so compiled
  graphs observe offload refills and activation-scale replacement.
- Padding is represented in ``F.pad`` order ``(W_l, W_r, H_t, H_b, T_f, T_b)``.
  Zero pads compose by addition; one spatial reflection combines with temporal
  zeros, while shared pads, nonzero constants, cropping, temporal reflection,
  and mixed spatial zero/reflection borders remain separate operations.
- Autocast FP32->FP16 casts commute with zero/reflection padding and stay in
  place so padding-only fusion preserves eager autocast rounding.
- Static and symbolic shapes are supported when reflection bounds are
  statically proven without introducing shape guards or runtime asserts.
"""

from collections.abc import Iterator
from typing import cast

import torch
from torch._C._nn import pad as torch_pad
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.nn import functional

from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

from . import _functional
from ._policy import Padding

_CONV = torch.ops.piper_kernels.convrot_int8_conv3d.default


def get_tensor_metadata(node: object) -> torch.Tensor | None:
    """Read tensor metadata from a Dynamo or FX node without evaluating it."""
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("example_value", node.meta.get("val"))
    return value if isinstance(value, torch.Tensor) else None


def _autocast_contexts(
    graph: torch.fx.Graph,
) -> Iterator[tuple[torch.fx.Node, dict[str, torch.dtype | None]]]:
    """Track autocast scopes retained in Dynamo's graph as well as outer scopes."""
    current = {
        device: torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else None
        for device in ("cpu", "cuda")
    }
    stack = []
    for node in list(graph.nodes):
        if node.target is torch.amp.autocast_mode._enter_autocast:
            device, dtype, enabled, _ = node.args
            assert isinstance(device, str)
            assert dtype is None or isinstance(dtype, torch.dtype)
            stack.append(current)
            current = {
                **current,
                device: (dtype or torch.get_autocast_dtype(device)) if enabled else None,
            }
        elif node.target is torch.amp.autocast_mode._exit_autocast:
            current = stack.pop()
        else:
            yield node, current


def lower_standard_convolutions(graph: torch.fx.Graph) -> None:
    """Expose quantized storage while it is still a subclass in the pre-grad graph.

    Read storage through runtime getattr nodes, never captured checkpoint values,
    so compiled calls observe offload refill and weight/scale replacement.
    """
    for node, autocast in _autocast_contexts(graph):
        if node.target is not functional.conv3d:
            continue
        bound = _functional.bind_conv3d_arguments(functional.conv3d, node.args, dict(node.kwargs))
        weight = bound.weight
        weight_value = get_tensor_metadata(weight)
        if not isinstance(weight_value, ConvRotInt8Tensor):
            continue
        stride, padding = _functional.convolution_parameters(bound)
        _functional._weight_storage(weight_value)
        input_node, bias = bound.input, bound.bias
        input_value = get_tensor_metadata(input_node)
        if (
            input_value is None
            or not isinstance(input_node, torch.fx.Node)
            or not isinstance(weight, torch.fx.Node)
            or (bias is not None and not isinstance(bias, torch.fx.Node))
        ):
            continue
        with graph.inserting_before(node):
            # Match the functional boundary's supported autocast conversion.
            device_type = input_value.device.type
            dtype = autocast.get(device_type)
            if dtype is not None:
                if dtype is not torch.float16:
                    continue
                if input_value.dtype is torch.float32:
                    input_node = graph.call_method("to", (input_node,), {"dtype": dtype})
                    input_node.meta["example_value"] = input_value.to(dtype=dtype)
                bias_value = get_tensor_metadata(bias)
                if bias_value is not None and bias_value.dtype is torch.float32:
                    bias = graph.call_method("to", (bias,), {"dtype": dtype})
                    bias.meta["example_value"] = bias_value.to(dtype=dtype)
            storage = []
            for name in ("qdata", "scale", "act_per_tensor_scale"):
                field = graph.call_function(getattr, (weight, name))
                field.meta["example_value"] = getattr(weight_value, name)
                storage.append(field)
            replacement = graph.call_function(
                _CONV,
                (
                    input_node,
                    storage[0],
                    storage[1],
                    bias,
                    weight_value.group_size,
                    storage[2],
                    list(stride),
                    list(padding),
                    False,
                    None,
                ),
            )
        replacement.meta = node.meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)


def _pad(node: object) -> tuple[torch.fx.Node, Padding, bool] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or len(node.users) != 1
        or node.target not in (functional.pad, torch_pad)
    ):
        return None
    arguments: dict[str, object] = dict(
        zip(("input", "pad", "mode", "value"), node.args, strict=False)
    )
    arguments.update(node.kwargs)
    source, pads = arguments.get("input"), arguments.get("pad")
    mode, value = arguments.get("mode", "constant"), arguments.get("value")
    if not isinstance(source, torch.fx.Node) or not isinstance(pads, (tuple, list)):
        return None
    if len(pads) not in (2, 4, 6) or any(type(p) is not int or p < 0 for p in pads):
        return None
    resolved = cast(Padding, (*pads, *((0,) * (6 - len(pads)))))
    front, back = resolved[4:]
    if mode == "constant" and value not in (None, 0, 0.0):
        return None
    if mode not in ("constant", "reflect") or (mode == "reflect" and (front or back)):
        return None
    return source, resolved, mode == "reflect"


def fp16_cast_source(node: object) -> torch.fx.Node | None:
    """Recognize an exclusive FP32-to-FP16 cast at the convolution boundary."""
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_method"
        or node.target != "to"
        or len(node.users) != 1
        or len(node.args) != 1
        or node.kwargs != {"dtype": torch.float16}
    ):
        return None
    source = node.args[0]
    value = get_tensor_metadata(source)
    return (
        source
        if isinstance(source, torch.fx.Node) and value is not None and value.dtype is torch.float32
        else None
    )


def fuse_padding(graph: torch.fx.Graph) -> None:
    """Absorb exclusive, representable pad chains; leave all other graphs intact.

    Zero pads compose by addition. One spatial reflection can combine with
    temporal zeros; mixed zero/reflection spatial borders or nested reflection
    remain separate because their indexing is not represented by this kernel.
    """
    if torch.is_grad_enabled():
        return
    lower_standard_convolutions(graph)
    changed = False
    for node in list(graph.nodes):
        if node.target != _CONV or node.kwargs or len(node.args) != 10:
            continue
        args = list(node.args)
        while True:
            cast_source = fp16_cast_source(args[0])
            previous_pad = cast_source if cast_source is not None else args[0]
            matched = _pad(previous_pad)
            if matched is None:
                break
            source, pads, reflection = matched
            existing, existing_reflection = args[7:9]
            assert isinstance(existing, (tuple, list))
            assert isinstance(existing_reflection, bool)
            if any(pads[:4]) and any(existing[:4]) and (reflection or existing_reflection):
                break
            merged = [a + b for a, b in zip(existing, pads, strict=True)]
            use_reflection = (reflection and any(pads[:4])) or (
                existing_reflection and any(existing[:4])
            )
            source_value = get_tensor_metadata(source)
            if source_value is None or source_value.ndim != 5:
                break
            if use_reflection and (
                not statically_known_true(max(merged[:2]) < source_value.shape[4])
                or not statically_known_true(max(merged[2:4]) < source_value.shape[3])
            ):
                break
            if cast_source is not None:
                cast_node = args[0]
                assert isinstance(cast_node, torch.fx.Node)
                cast_node.args = (source,)
                cast_node.meta["example_value"] = source_value.to(dtype=torch.float16)
                # Casting commutes with zero/reflection padding. Keep the cast
                # so padding-only fusion preserves ordinary autocast rounding.
            else:
                args[0] = source
            args[7], args[8] = merged, use_reflection
            node.args = tuple(args)
            node.meta.pop("eager_input_vals", None)
            assert isinstance(previous_pad, torch.fx.Node)
            graph.erase_node(previous_pad)
            changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()


__all__ = ["fp16_cast_source", "fuse_padding", "get_tensor_metadata", "lower_standard_convolutions"]
