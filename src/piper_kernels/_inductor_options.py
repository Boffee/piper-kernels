"""Unified phase-ordered Inductor compiler pipeline for Piper Kernels.

Public callers configure compilation through :func:`piper_kernels.compile_options`,
which installs at most one post-grad pass (:class:`_PiperPostGradPass`) and one
pre-grad pass (:class:`_PiperPreGradPass`) without mutating caller options or
duplicating passes across repeated calls. Model-specific pre-grad specializations attach
via :func:`add_pre_grad_hook` so core stays model-agnostic. Every pass hashes both the
package source tree and its active configuration tuple into ``uuid()``, and imports
rewrite modules lazily inside ``__call__`` only when matching operators appear in the FX
graph.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch._inductor.codecache import build_code_hash
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    CustomInferenceAwareGraphPass,
    CustomPassBase,
)

_PRE_GRAD_PASS = "pre_grad_custom_pass"
_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"

_FFN_OPS = frozenset(
    {
        "piper_kernels::convrot_int8_linear",
        "piper_kernels::nvfp4_linear",
        "piper_kernels::convrot_nvfp4_linear",
        "piper_kernels::convrot_int8_ffn",
        "piper_kernels::nvfp4_ffn",
    }
)
_DENSE_ATTN_OPS = frozenset(
    {
        "piper_kernels::piper_attention",
        "piper_kernels::piper_attention_from_quantized_query",
    }
)
_DENSE_OUT_OPS = frozenset(
    {
        "piper_kernels::piper_attention_from_quantized",
        "piper_kernels::convrot_int8_piper_projected_query_attention_output",
    }
)
_SPARSE_ATTN_OPS = frozenset(
    {
        "piper_kernels::sparse_piper_attention",
        "piper_kernels::sparse_piper_attention_from_quantized",
    }
)
_SPARSE_INT8_OPS = _SPARSE_ATTN_OPS | frozenset(
    {
        "piper_kernels::convrot_int8_sparse_piper_project_key",
        "piper_kernels::convrot_int8_sparse_piper_projected_query_attention_output",
    }
)
_NVFP4_LINEAR_OPS = frozenset(
    {
        "piper_kernels::nvfp4_linear_prepared",
        "piper_kernels::nvfp4_linear",
        "piper_kernels::convrot_nvfp4_linear",
    }
)


@dataclass(frozen=True, slots=True)
class GraphPassHook:
    """One keyed extension rewrite attached to a unified Piper compiler pass."""

    key: str
    apply: Callable[[torch.fx.Graph], object] = field(compare=False, repr=False)


def _graph_op_names(graph: torch.fx.Graph) -> set[str]:
    return {
        node.target._schema.name
        for node in graph.nodes
        if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload)
    }


def _dedupe_hooks(hooks: tuple[GraphPassHook, ...]) -> tuple[GraphPassHook, ...]:
    by_key = {hook.key: hook for hook in hooks}
    return tuple(by_key[key] for key in sorted(by_key))


def compute_compile_pass_uuid(
    compiler_pass: CustomPassBase,
    *,
    config: object = None,
) -> bytes:
    """Return the Inductor FX cache key for a Piper pass and its active config."""
    pass_class = type(compiler_pass)
    hasher = hashlib.sha256(_compute_package_source_hash())
    hasher.update(f"{pass_class.__module__}.{pass_class.__qualname__}".encode())
    effective_config = (
        config if config is not None else getattr(compiler_pass, "_config_key", lambda: None)()
    )
    if effective_config is not None:
        hasher.update(b"\0")
        hasher.update(repr(effective_config).encode())
    return hasher.digest()


@functools.cache
def _compute_package_source_hash() -> bytes:
    """Hash every package module's name and source once per process."""
    hasher = hashlib.sha256()
    build_code_hash([str(Path(__file__).parent)], "piper_kernels.", hasher)
    return hasher.digest()


@dataclass(frozen=True)
class _PiperPostGradPass(CustomInferenceAwareGraphPass):
    """Single phase-ordered post-grad Inductor pass covering all Piper rewrites.

    Phase 0: residual updates; Phase 1: semantic FFN and ConvRot INT8 attention;
    Phase 2: linear preparation sharing and quantized all-gather hoisting;
    Phase 3: NVFP4 sparse attention and opt-in dense attention output.
    """

    linear: bool = True
    ffn: bool = False
    attention: bool = False
    attention_output: bool = False
    residual_updates: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention", self.attention or self.attention_output)

    def _config_key(self) -> tuple[bool, bool, bool, bool, bool]:
        return (
            self.linear,
            self.ffn,
            self.attention,
            self.attention_output,
            self.residual_updates,
        )

    def merge(
        self,
        *,
        linear: bool = False,
        ffn: bool = False,
        attention: bool = False,
        attention_output: bool = False,
        residual_updates: bool = False,
    ) -> _PiperPostGradPass:
        return _PiperPostGradPass(
            linear=self.linear or linear,
            ffn=self.ffn or ffn,
            attention=self.attention or attention,
            attention_output=self.attention_output or attention_output,
            residual_updates=self.residual_updates or residual_updates,
        )

    def _run_phase1(self, graph: torch.fx.Graph, op_names: set[str]) -> bool:
        changed = False
        if self.ffn and (op_names & _FFN_OPS):
            from piper_kernels.fusions.ffn._compile import fold_ffns  # noqa: PLC0415

            changed = fold_ffns(graph) or changed
        if not self.attention:
            return changed
        if (op_names & _DENSE_ATTN_OPS) and "piper_kernels::convrot_int8_linear" in op_names:
            from piper_kernels.fusions.convrot_int8_piper._compile import (  # noqa: PLC0415
                fold_attention_projections,
            )

            changed = fold_attention_projections(graph) or changed
        if (op_names & _SPARSE_INT8_OPS) and any(
            name.startswith("piper_kernels::convrot_int8_") for name in op_names
        ):
            from piper_kernels.fusions.convrot_int8_sparse_piper._compile import (  # noqa: PLC0415
                fold_sparse_attention,
            )

            changed = fold_sparse_attention(graph) or changed
        return changed

    def _run_phase2(self, graph: torch.fx.Graph, op_names: set[str]) -> bool:
        if not self.linear:
            return False
        changed = False
        if op_names & {
            "piper_kernels::convrot_int8_linear",
            "piper_kernels::convrot_int8_prepare_input",
        }:
            from piper_kernels.linear.convrot.int8._compile import fold_linear  # noqa: PLC0415

            changed = fold_linear(graph) or changed
        if "piper_kernels::nvfp4_linear" in op_names:
            from piper_kernels.linear.nvfp4._compile import (  # noqa: PLC0415
                fold_linear as fold_nvfp4,
            )

            changed = fold_nvfp4(graph) or changed
        if "piper_kernels::convrot_nvfp4_linear" in op_names:
            from piper_kernels.linear.convrot.nvfp4._compile import (  # noqa: PLC0415
                fold_linear as fold_convrot_nvfp4,
            )

            changed = fold_convrot_nvfp4(graph) or changed
        return changed

    def _run_phase3(self, graph: torch.fx.Graph, op_names: set[str]) -> None:
        if self.attention and (
            ((op_names & _SPARSE_ATTN_OPS) and (op_names & _NVFP4_LINEAR_OPS))
            or "piper_kernels::nvfp4_sparse_piper_projected_query_attention_output" in op_names
        ):
            from piper_kernels.fusions.nvfp4_sparse_piper._compile import (  # noqa: PLC0415
                fold_sparse_attention,
            )

            fold_sparse_attention(graph)
        if self.attention_output and (op_names & _DENSE_OUT_OPS):
            from piper_kernels.fusions.convrot_int8_piper._compile import (  # noqa: PLC0415
                fold_attention_output,
            )

            fold_attention_output(graph)

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            return
        op_names = _graph_op_names(graph)
        if self.residual_updates and "aten::copy_" in op_names:
            from piper_kernels.fusions.residual_updates._compile import (  # noqa: PLC0415
                fold_residual_updates,
            )

            fold_residual_updates(graph)
        if self._run_phase1(graph, op_names):
            op_names = _graph_op_names(graph)
        if self._run_phase2(graph, op_names):
            op_names = _graph_op_names(graph)
        self._run_phase3(graph, op_names)

    def uuid(self) -> bytes:
        return compute_compile_pass_uuid(self)


@dataclass(frozen=True)
class _PiperPreGradPass(CustomGraphPass):
    """Single pre-grad Inductor pass covering Conv3D padding and specialization hooks."""

    conv3d: bool = False
    hooks: tuple[GraphPassHook, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hooks", _dedupe_hooks(self.hooks))

    def _config_key(self) -> tuple[bool, tuple[str, ...]]:
        return (self.conv3d, tuple(hook.key for hook in self.hooks))

    def merge(
        self, *, conv3d: bool = False, hooks: tuple[GraphPassHook, ...] = ()
    ) -> _PiperPreGradPass:
        return _PiperPreGradPass(conv3d=self.conv3d or conv3d, hooks=(*self.hooks, *hooks))

    def __call__(self, graph: torch.fx.Graph) -> None:
        if torch.is_grad_enabled():
            return
        if self.conv3d and _has_conv3d_ops(graph):
            from piper_kernels.conv3d.convrot.int8._compile import fuse_padding  # noqa: PLC0415

            fuse_padding(graph)
        for hook in self.hooks:
            hook.apply(graph)

    def uuid(self) -> bytes:
        return compute_compile_pass_uuid(self)


def _has_conv3d_ops(graph: torch.fx.Graph) -> bool:
    return any(
        node.op == "call_function"
        and (
            node.target is torch.nn.functional.conv3d
            or (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target._schema.name == "piper_kernels::convrot_int8_conv3d"
            )
        )
        for node in graph.nodes
    )


def _install_or_merge[PassT: (_PiperPostGradPass, _PiperPreGradPass)](
    combined: dict[str, object],
    option_key: str,
    pass_type: type[PassT],
    merge_fn: Callable[[PassT | None], PassT | None],
) -> PassT | None:
    raw = combined.get(option_key)
    passes = () if raw is None else tuple(raw) if isinstance(raw, (list, tuple)) else (raw,)
    existing_index = next(
        (idx for idx, item in enumerate(passes) if isinstance(item, pass_type)), None
    )
    existing = passes[existing_index] if existing_index is not None else None
    assert existing is None or isinstance(existing, pass_type)
    merged = merge_fn(existing)
    if merged is None:
        return None
    if existing_index is None:
        combined[option_key] = (*passes, merged)
    else:
        combined[option_key] = tuple(
            merged if idx == existing_index else item
            for idx, item in enumerate(passes)
            if idx == existing_index or not isinstance(item, pass_type)
        )
    return merged


def compile_options(
    options: Mapping[str, object] | None = None,
    *,
    linear: bool = True,
    ffn: bool = False,
    attention: bool = False,
    attention_output: bool = False,
    residual_updates: bool = False,
    conv3d: bool = False,
) -> dict[str, object]:
    """Return Inductor compile options with Piper Kernels' unified compiler pipeline.

    Repeated calls merge (union) enabled flags into one phase-ordered post-grad
    pass and one pre-grad pass without mutating ``options`` or duplicating
    passes, preserving unrelated caller passes and options.
    """
    flags = {
        "linear": linear,
        "ffn": ffn,
        "attention": attention,
        "attention_output": attention_output,
        "residual_updates": residual_updates,
        "conv3d": conv3d,
    }
    for name, value in flags.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")

    combined = dict(options) if options is not None else {}
    has_post = linear or ffn or attention or attention_output or residual_updates
    merged_post = _install_or_merge(
        combined,
        _POST_GRAD_PRE_PASS,
        _PiperPostGradPass,
        lambda existing: (
            existing.merge(
                linear=linear,
                ffn=ffn,
                attention=attention,
                attention_output=attention_output,
                residual_updates=residual_updates,
            )
            if existing is not None
            else (
                _PiperPostGradPass(
                    linear=linear,
                    ffn=ffn,
                    attention=attention,
                    attention_output=attention_output,
                    residual_updates=residual_updates,
                )
                if has_post
                else None
            )
        ),
    )
    if merged_post is not None and merged_post.residual_updates:
        combined["allow_buffer_reuse"] = False
    _install_or_merge(
        combined,
        _PRE_GRAD_PASS,
        _PiperPreGradPass,
        lambda existing: (
            existing.merge(conv3d=conv3d)
            if existing is not None
            else (_PiperPreGradPass(conv3d=True) if conv3d else None)
        ),
    )
    return combined


def add_pre_grad_hook(
    options: Mapping[str, object] | None,
    hook: GraphPassHook,
) -> dict[str, object]:
    """Attach a pre-grad hook to the unified pre-grad pass without mutating options."""
    combined = dict(options) if options is not None else {}
    _install_or_merge(
        combined,
        _PRE_GRAD_PASS,
        _PiperPreGradPass,
        lambda existing: (
            existing.merge(hooks=(hook,))
            if existing is not None
            else _PiperPreGradPass(hooks=(hook,))
        ),
    )
    return combined


__all__ = [
    "GraphPassHook",
    "_PiperPostGradPass",
    "_PiperPreGradPass",
    "add_pre_grad_hook",
    "compile_options",
    "compute_compile_pass_uuid",
]
