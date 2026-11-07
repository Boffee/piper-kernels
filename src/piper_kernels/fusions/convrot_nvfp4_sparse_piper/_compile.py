"""ConvRot NVFP4 projection folding for sparse Piper attention."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass

from piper_kernels import _inductor_options as inductor_options
from piper_kernels.fusions.nvfp4_sparse_piper import _compile as nvfp4_sparse_compile
from piper_kernels.linear.convrot.nvfp4 import _compile as convrot_nvfp4_compile

from . import _output_compile


class _CompilePass(CustomInferenceAwareGraphPass):
    """Reuse NVFP4 sparse projections and fold ConvRot NVFP4 output preparation."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            nvfp4_sparse_compile._fold_projection(graph)
            _output_compile._fold_attention_output(graph)

    def uuid(self) -> bytes:
        return inductor_options.compute_compile_pass_uuid(self)


compile_pass = _CompilePass()


def convrot_nvfp4_sparse_piper_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install sparse fusion immediately after ConvRot NVFP4 normalization."""
    return inductor_options.add_ordered_post_grad_passes(
        options,
        (convrot_nvfp4_compile.compile_pass, compile_pass),
    )


__all__ = ["convrot_nvfp4_sparse_piper_compile_options"]
