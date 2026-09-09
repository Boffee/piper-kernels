"""Global dynamic scaling and storage reuse across both NVFP4 output formats."""

from __future__ import annotations

import pytest
import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale

from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
)
from piper_kernels.fusions.convrot_nvfp4_sparse_piper import output as convrot_output
from piper_kernels.fusions.nvfp4_sparse_piper import output
from piper_kernels.linear.convrot.nvfp4 import _ops as convrot_ops
from piper_kernels.linear.nvfp4 import _ops

from .._accuracy import assert_fusion_output_close
from ._helpers import exact_sm120_available
from .test_output import _arguments


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", [None, 16, 256])
@pytest.mark.parametrize(
    ("output_features", "weight_global_scale"),
    [(128, True), (256, False), (320, True)],
    ids=["narrower", "equal-blockwise", "wider"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dynamic_output_reuses_storage_and_replays(
    group_size: int | None,
    output_features: int,
    weight_global_scale: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(1021)
    arguments, _ = _arguments(sequence_length=193, bias=False)
    # Two batches with different value means detect per-batch scaling and unsafe
    # write order when compacting a narrower output into the attention allocation.
    attention_args = [tensor.repeat(2, *([1] * (tensor.ndim - 1))) for tensor in arguments[:10]]
    attention_args[9][1].add_(3)
    attention_args.extend(arguments[10:14])
    dense_weight = torch.randn((output_features, 256), device="cuda", dtype=torch.bfloat16)
    weight = NVFP4Tensor.to_nvfp4(
        dense_weight,
        per_tensor_scale=(
            per_tensor_amax_to_scale(dense_weight.abs().amax()) if weight_global_scale else None
        ),
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    # Exercise both nibble orders without multiplying the whole test matrix.
    high_first = dtype is torch.float16
    qdata = ((weight.qdata & 15) << 4) | (weight.qdata >> 4) if high_first else weight.qdata
    bias = torch.randn(output_features, device="cuda", dtype=torch.float32)
    projection_args = (qdata, weight.scale, weight.per_tensor_scale, None, bias)
    op = output._attention_output_op if group_size is None else convrot_output._attention_output_op
    op_args = (
        *attention_args,
        *projection_args,
        *((group_size,) if group_size is not None else ()),
        128,
    )
    op_kwargs = {"dynamic_activation_scale": True, "output_dtype": dtype, "high_first": high_first}

    def reference() -> torch.Tensor:
        attended = _sparse_piper_attention_from_quantized_op(*attention_args, output_dtype=dtype)
        if group_size is None:
            return _ops.linear(attended.flatten(2), *projection_args, True, high_first)
        return convrot_ops.linear(
            attended.flatten(2), *projection_args, True, group_size, high_first
        )

    with torch.no_grad():
        expected = reference()
        actual = op(*op_args, **op_kwargs)
        assert_fusion_output_close(actual, expected)
        assert actual.is_contiguous()
        retained_features = max(256, output_features)
        assert (
            actual.untyped_storage().nbytes() == 2 * 193 * retained_features * actual.element_size()
        )
        # Warm before capture, then change the input so a stale global scale fails.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(*op_args, **op_kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = op(*op_args, **op_kwargs)
        attention_args[9].mul_(4)
        graph.replay()
        # Replay must reproduce eager execution of this same fused operation.
        eager = op(*op_args, **op_kwargs)
        torch.testing.assert_close(replayed, eager, atol=0, rtol=0)
        assert_fusion_output_close(replayed, reference())
        result = torch.library.opcheck(
            op,
            op_args,
            op_kwargs,
            test_utils=("test_faketensor", "test_aot_dispatch_dynamic"),
        )
        assert all(value == "SUCCESS" for value in result.values())
