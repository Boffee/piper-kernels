"""Shared fake/meta contracts for bounded FFN operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from piper_kernels.fusions.convrot_int8_gelu_ffn import triton as int8_gelu
from piper_kernels.fusions.convrot_nvfp4_gelu_ffn import triton as convrot_gelu
from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn import triton as convrot_swiglu
from piper_kernels.fusions.nvfp4_gelu_ffn import triton as nvfp4_gelu
from piper_kernels.fusions.nvfp4_swiglu_ffn import triton as nvfp4_swiglu
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

_OPERATIONS = {
    "convrot_int8_gelu": (
        int8_gelu._chunked_gelu_ffn_op,
        int8_gelu._chunked_gelu_ffn_gated_updates_op,
    ),
    "nvfp4_gelu": (
        nvfp4_gelu._chunked_gelu_ffn_op,
        nvfp4_gelu._chunked_gelu_ffn_gated_updates_op,
    ),
    "convrot_nvfp4_gelu": (
        convrot_gelu._chunked_gelu_ffn_op,
        convrot_gelu._chunked_gelu_ffn_gated_updates_op,
    ),
    "nvfp4_swiglu": (
        nvfp4_swiglu._chunked_swiglu_ffn_op,
        nvfp4_swiglu._chunked_swiglu_ffn_gated_updates_op,
    ),
    "convrot_nvfp4_swiglu": (
        convrot_swiglu._chunked_swiglu_ffn_op,
        convrot_swiglu._chunked_swiglu_ffn_gated_updates_op,
    ),
}


@dataclass(frozen=True, slots=True)
class _Case:
    operation: Callable[..., torch.Tensor | None]
    arguments: dict[str, object]


def _case(
    name: str,
    *,
    gated: bool = False,
    input_features: int = 256,
    intermediate_features: int = 64,
) -> _Case:
    arguments: dict[str, object] = {
        "input": torch.empty(2, 3, input_features, device="meta", dtype=torch.bfloat16),
        "chunk_rows": 128,
    }
    sources = ("gate", "value") if name.endswith("swiglu") else ("up",)
    for prefix in (*sources, "down"):
        width = intermediate_features if prefix == "down" else input_features
        height = 32 if prefix == "down" else intermediate_features
        operands: dict[str, object]
        if name == "convrot_int8_gelu":
            operands = {
                "weight_qdata": torch.empty(height, width, device="meta", dtype=torch.int8),
                "weight_scale": torch.empty(height, 1, device="meta", dtype=torch.float32),
                "bias": None,
                "group_size": 64,
                "input_scale": None,
            }
        else:
            operands = {
                "weight_qdata": torch.empty(height, width // 2, device="meta", dtype=torch.uint8),
                "weight_scale": torch.empty(
                    nvfp4_layout.scale_shape(height, width),
                    device="meta",
                    dtype=torch.float8_e4m3fn,
                ),
                "weight_per_tensor_scale": None,
                "activation_per_tensor_scale": None,
                "bias": None,
                "dynamic_activation_scale": True,
                "high_first": False,
            }
            if name.startswith("convrot"):
                operands["group_size"] = 64
        arguments.update({f"{prefix}_{key}": value for key, value in operands.items()})
    if gated:
        arguments.update(
            base=torch.empty(2, 3, 32, device="meta", dtype=torch.bfloat16),
            reusable_update=torch.empty(2, 3, 32, device="meta", dtype=torch.bfloat16),
            update_gate=torch.empty(4, 32, device="meta", dtype=torch.bfloat16),
            ffn_gate=torch.empty(4, 32, device="meta", dtype=torch.bfloat16),
            gate_indices=torch.empty(6, device="meta", dtype=torch.int64),
            python_indexing=False,
        )
    return _Case(_OPERATIONS[name][int(gated)], arguments)


@pytest.mark.parametrize("name", _OPERATIONS)
@pytest.mark.parametrize("gated", [False, True])
def test_fake_ffn_validates_projection_metadata(name: str, gated: bool) -> None:
    case = _case(name, gated=gated)
    result = case.operation(**case.arguments)
    if gated:
        assert result is None
    else:
        assert result is not None
        assert result.shape == (2, 3, 32)
        assert result.device.type == "meta"

    scale = case.arguments["down_weight_scale"]
    assert isinstance(scale, torch.Tensor)
    with pytest.raises(ValueError, match="scale"):
        case.operation(**{**case.arguments, "down_weight_scale": scale.to(torch.bfloat16)})
    with pytest.raises(ValueError, match="chunk_rows"):
        case.operation(**{**case.arguments, "chunk_rows": 0})


@pytest.mark.parametrize("name", _OPERATIONS)
@pytest.mark.parametrize("gated", [False, True])
def test_fake_ffn_rejects_noncontiguous_input(name: str, gated: bool) -> None:
    case = _case(name, gated=gated)
    input = torch.empty(2, 3, 256, device="meta", dtype=torch.bfloat16).transpose(0, 1)  # noqa: A001
    with pytest.raises(ValueError, match="contiguous"):
        case.operation(**{**case.arguments, "input": input})


@pytest.mark.parametrize("name", ["convrot_nvfp4_gelu", "convrot_nvfp4_swiglu"])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    ("input_features", "intermediate_features", "match"),
    [(80, 64, "source features"), (256, 80, "activated features")],
)
def test_fake_ffn_validates_rotation_divisibility(
    name: str,
    gated: bool,
    input_features: int,
    intermediate_features: int,
    match: str,
) -> None:
    case = _case(
        name,
        gated=gated,
        input_features=input_features,
        intermediate_features=intermediate_features,
    )
    with pytest.raises(ValueError, match=match):
        case.operation(**case.arguments)


@pytest.mark.parametrize("name", _OPERATIONS)
def test_fake_gated_ffn_validates_update_metadata(name: str) -> None:
    case = _case(name, gated=True)
    invalid_update = torch.empty(2, 3, 32, device="meta", dtype=torch.float16)
    with pytest.raises(ValueError, match="reusable update"):
        case.operation(**{**case.arguments, "reusable_update": invalid_update})
