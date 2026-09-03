"""ConvRot NVFP4 projection folding for sparse Piper attention."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_hash_for_files,
)

from piper_kernels.fusions.nvfp4_sparse_piper import _compile as nvfp4_sparse_compile
from piper_kernels.linear import _preparation_sharing as preparation_sharing
from piper_kernels.linear.convrot import _rotation as convrot_rotation
from piper_kernels.linear.convrot.nvfp4 import _compile as convrot_nvfp4_compile
from piper_kernels.linear.convrot.nvfp4 import _compile_fx as convrot_nvfp4_compile_fx
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4_triton

from . import _output_compile, output

_COMPILE_PASS_VERSION = "convrot-nvfp4-sparse-piper-compile-v3"


class _CompilePass(CustomInferenceAwareGraphPass):
    """Reuse NVFP4 sparse projections and fold ConvRot NVFP4 output preparation."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            nvfp4_sparse_compile._fold_projection(graph)
            _output_compile._fold_attention_output(graph)

    def uuid(self) -> bytes:
        source_files = (
            __file__,
            _output_compile.__file__,
            output.__file__,
            convrot_rotation.__file__,
            convrot_nvfp4_compile_fx.__file__,
            convrot_nvfp4_triton.__file__,
            *nvfp4_sparse_compile._source_files(),
        )
        return get_hash_for_files(
            tuple(file_name for file_name in source_files if file_name is not None),
            extra=_COMPILE_PASS_VERSION,
        )


compile_pass = _CompilePass()


def convrot_nvfp4_sparse_piper_compile_options(
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Install sparse fusion immediately after ConvRot NVFP4 normalization."""
    return preparation_sharing.add_ordered_post_grad_passes(
        options,
        (convrot_nvfp4_compile.compile_pass, compile_pass),
    )


__all__ = ["convrot_nvfp4_sparse_piper_compile_options"]
