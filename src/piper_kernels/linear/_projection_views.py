"""Normalize row reshapes around semantic projections before fusion matching."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ._preparation_sharing import tensor_metadata

_RESHAPE_OPS = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
)
_PROJECTION_NAMES = {
    "piper_kernels::convrot_int8_linear",
    "piper_kernels::nvfp4_linear",
    "piper_kernels::convrot_nvfp4_linear",
}


def _reshape_input(node: torch.fx.Node) -> torch.fx.Node | None:
    if (
        node.op == "call_function"
        and node.target in _RESHAPE_OPS
        and not node.kwargs
        and len(node.args) == 2
        and isinstance(node.args[0], torch.fx.Node)
    ):
        return node.args[0]
    return None


def _same_shape(left: Sequence[int | torch.SymInt], right: Sequence[int | torch.SymInt]) -> bool:
    return len(left) == len(right) and all(
        statically_known_true(a == b) for a, b in zip(left, right, strict=True)
    )


def _compatible_projection_views(
    source: torch.fx.Node,
    input_view: torch.fx.Node,
    linear: torch.fx.Node,
    output_view: torch.fx.Node,
) -> bool:
    """Prove that the views only regroup rows of a contiguous projection."""
    original = tensor_metadata(source)
    reshaped = tensor_metadata(input_view)
    projected = tensor_metadata(linear)
    restored = tensor_metadata(output_view)
    if original is None or reshaped is None or projected is None or restored is None:
        return False
    return (
        all(
            value.ndim > 0
            and value.layout is torch.strided
            and value.is_contiguous()
            and value.dtype == original.dtype
            and value.device == original.device
            for value in (original, reshaped, projected, restored)
        )
        and _same_shape(original.shape[-1:], reshaped.shape[-1:])
        and _same_shape(reshaped.shape[:-1], projected.shape[:-1])
        and _same_shape(original.shape[:-1], restored.shape[:-1])
        and _same_shape(projected.shape[-1:], restored.shape[-1:])
        and statically_known_true(original.numel() == reshaped.numel())
        and statically_known_true(projected.numel() == restored.numel())
    )


def _normalize_projection_view(restore: torch.fx.Node) -> bool:
    linear = _reshape_input(restore)
    if (
        linear is None
        or linear.op != "call_function"
        or not isinstance(linear.target, torch._ops.OpOverload)
        or linear.target._schema.name not in _PROJECTION_NAMES
        or linear.kwargs
        or not linear.args
        or not isinstance(linear.args[0], torch.fx.Node)
        or len(linear.users) != 1
    ):
        return False
    input_view = linear.args[0]
    source = _reshape_input(input_view)
    if source is None or not _compatible_projection_views(source, input_view, linear, restore):
        return False

    # Only the restore consumes the old projection. Other consumers of either
    # the input view or restored result keep their original shapes, and the
    # fusion matchers still see every external use of the projected values.
    linear.args = (source, *linear.args[1:])
    linear.meta = restore.meta.copy()
    linear.meta.pop("eager_input_vals", None)
    restore.replace_all_uses_with(linear)
    linear.graph.erase_node(restore)
    if not input_view.users:
        linear.graph.erase_node(input_view)
    return True


def normalize_projection_views(graph: torch.fx.Graph) -> bool:
    """Lift feature-preserving row reshapes through Piper semantic linears.

    DTensor's local linear decomposition flattens leading dimensions and then
    restores them. Rebuild the batched linear before preparation or fusion so
    FFN and attention matchers can use the same graph as ordinary local inputs.
    Attention head reshapes and communication operators remain in place.
    """
    changed = False
    for node in list(graph.nodes):
        changed = _normalize_projection_view(node) or changed
    if changed:
        graph.lint()
    return changed


__all__ = ["normalize_projection_views"]
