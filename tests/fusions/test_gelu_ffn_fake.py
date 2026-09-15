"""Fake/meta contract tests shared by bounded GELU FFN formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from piper_kernels.fusions.convrot_int8_gelu_ffn.triton import (
    _chunked_gelu_ffn_gated_updates_op as convrot_int8_gated_op,
)
from piper_kernels.fusions.convrot_int8_gelu_ffn.triton import (
    _chunked_gelu_ffn_op as convrot_int8_op,
)
from piper_kernels.fusions.convrot_nvfp4_gelu_ffn.triton import (
    _chunked_gelu_ffn_gated_updates_op as convrot_nvfp4_gated_op,
)
from piper_kernels.fusions.convrot_nvfp4_gelu_ffn.triton import (
    _chunked_gelu_ffn_op as convrot_nvfp4_op,
)
from piper_kernels.fusions.nvfp4_gelu_ffn.triton import (
    _chunked_gelu_ffn_gated_updates_op as nvfp4_gated_op,
)
from piper_kernels.fusions.nvfp4_gelu_ffn.triton import (
    _chunked_gelu_ffn_op as nvfp4_op,
)
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

_FORMATS = ("convrot_int8", "nvfp4", "convrot_nvfp4")


@dataclass(frozen=True, slots=True)
class _Case:
    operation: Callable[..., torch.Tensor]
    gated_operation: Callable[..., None]
    arguments: tuple[object, ...]
    down_scale_index: int
    gated_prefix: int


def _nvfp4_weight(output_features: int, input_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(
            output_features,
            input_features // 2,
            device="meta",
            dtype=torch.uint8,
        ),
        torch.empty(
            nvfp4_layout.scale_shape(output_features, input_features),
            device="meta",
            dtype=torch.float8_e4m3fn,
        ),
    )


def _case(format_name: str) -> _Case:
    input = torch.empty(2, 3, 256, device="meta", dtype=torch.bfloat16)  # noqa: A001
    if format_name == "convrot_int8":
        arguments = (
            input,
            torch.empty(64, 256, device="meta", dtype=torch.int8),
            torch.empty(64, 1, device="meta", dtype=torch.float32),
            None,
            64,
            torch.empty(32, 64, device="meta", dtype=torch.int8),
            torch.empty(32, 1, device="meta", dtype=torch.float32),
            None,
            64,
            128,
            None,
            None,
        )
        return _Case(convrot_int8_op, convrot_int8_gated_op, arguments, 6, 9)

    up_qdata, up_scale = _nvfp4_weight(64, 256)
    down_qdata, down_scale = _nvfp4_weight(32, 64)
    if format_name == "nvfp4":
        arguments = (
            input,
            up_qdata,
            up_scale,
            None,
            None,
            None,
            True,
            False,
            down_qdata,
            down_scale,
            None,
            None,
            None,
            True,
            False,
            128,
        )
        return _Case(nvfp4_op, nvfp4_gated_op, arguments, 9, 15)
    arguments = (
        input,
        up_qdata,
        up_scale,
        None,
        None,
        None,
        True,
        16,
        False,
        down_qdata,
        down_scale,
        None,
        None,
        None,
        True,
        64,
        False,
        128,
    )
    return _Case(convrot_nvfp4_op, convrot_nvfp4_gated_op, arguments, 10, 17)


@pytest.mark.parametrize("format_name", _FORMATS)
def test_fake_gelu_ffn_validates_projection_metadata(format_name: str) -> None:
    case = _case(format_name)

    result = case.operation(*case.arguments)
    assert result.shape == (2, 3, 32)
    assert result.device.type == "meta"

    invalid = list(case.arguments)
    invalid[case.down_scale_index] = invalid[case.down_scale_index].to(torch.bfloat16)
    with pytest.raises(ValueError, match="scale"):
        case.operation(*invalid)

    invalid = list(case.arguments)
    invalid[case.gated_prefix] = 0
    with pytest.raises(ValueError, match="chunk_rows"):
        case.operation(*invalid)


@pytest.mark.parametrize("format_name", _FORMATS)
def test_fake_gelu_ffn_rejects_noncontiguous_input(format_name: str) -> None:
    case = _case(format_name)
    invalid = list(case.arguments)
    invalid[0] = torch.empty(2, 3, 256, device="meta", dtype=torch.bfloat16).transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous"):
        case.operation(*invalid)


@pytest.mark.parametrize("format_name", _FORMATS)
def test_fake_gated_gelu_ffn_validates_update_metadata(format_name: str) -> None:
    case = _case(format_name)
    base = torch.empty(2, 3, 32, device="meta", dtype=torch.bfloat16)
    reusable_update = torch.empty_like(base)
    update_gate = torch.empty(4, 32, device="meta", dtype=torch.bfloat16)
    ffn_gate = torch.empty_like(update_gate)
    gate_indices = torch.empty(6, device="meta", dtype=torch.int64)
    prefix = case.arguments[: case.gated_prefix]
    suffix = case.arguments[case.gated_prefix :]
    arguments = (
        *prefix,
        base,
        reusable_update,
        update_gate,
        ffn_gate,
        gate_indices,
        False,
        *suffix,
    )

    assert case.gated_operation(*arguments) is None
    invalid = list(arguments)
    invalid[case.gated_prefix + 1] = reusable_update.to(torch.float16)
    with pytest.raises(ValueError, match="reusable update"):
        case.gated_operation(*invalid)
