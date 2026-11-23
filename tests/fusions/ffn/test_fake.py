"""Portable fake/meta, chunk-row policy, and lazy-import isolation tests for FFN ops."""

from __future__ import annotations

import math
import subprocess
import sys
from typing import Literal, cast

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from piper_kernels.fusions.ffn import _convrot_int8 as int8_ffn
from piper_kernels.fusions.ffn import _nvfp4 as nvfp4_ffn
from piper_kernels.weights.nvfp4 import _layout as nvfp4_layout

type _Format = Literal["convrot_int8", "nvfp4", "convrot_nvfp4"]
type _Activation = Literal["gelu_tanh", "swiglu"]
type _Mode = Literal["ffn", "residual_add", "gated_updates"]

_FORMATS: tuple[_Format, ...] = ("convrot_int8", "nvfp4", "convrot_nvfp4")
_ACTIVATIONS: tuple[_Activation, ...] = ("gelu_tanh", "swiglu")
_MODES: tuple[_Mode, ...] = ("ffn", "residual_add", "gated_updates")


def _case(
    fmt: _Format,
    activation: _Activation,
    *,
    mode: _Mode = "ffn",
    input_features: int = 256,
    intermediate_features: int = 64,
    output_features: int = 32,
) -> tuple[object, dict[str, object]]:
    args: dict[str, object] = {
        "input": torch.empty(2, 3, input_features, device="meta", dtype=torch.bfloat16),
        "activation_fn": activation,
        "chunk_rows": 128,
    }
    prefixes = ("up", "gate", "down") if activation == "swiglu" else ("up", "down")
    for prefix in prefixes:
        width = intermediate_features if prefix == "down" else input_features
        height = output_features if prefix == "down" else intermediate_features
        if fmt == "convrot_int8":
            operands: dict[str, object] = {
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
                "group_size": 64 if fmt == "convrot_nvfp4" else None,
                "high_first": False,
            }
        args.update({f"{prefix}_{key}": value for key, value in operands.items()})

    if mode != "ffn":
        args.update(
            output=torch.empty(2, 3, output_features, device="meta", dtype=torch.bfloat16),
            ffn_gate=torch.empty(4, output_features, device="meta", dtype=torch.bfloat16),
            gate_indices=torch.empty(6, device="meta", dtype=torch.int64),
            python_indexing=False,
        )
    if mode == "gated_updates":
        args.update(
            base=torch.empty(2, 3, output_features, device="meta", dtype=torch.bfloat16),
            update_gate=torch.empty(4, output_features, device="meta", dtype=torch.bfloat16),
        )

    if fmt == "convrot_int8":
        op = int8_ffn.convrot_int8_ffn_op if mode == "ffn" else int8_ffn.convrot_int8_ffn_update_op
    else:
        op = nvfp4_ffn.nvfp4_ffn_op if mode == "ffn" else nvfp4_ffn.nvfp4_ffn_update_op
    return op, args


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize("mode", _MODES)
def test_fake_ffn_validates_projection_metadata(
    fmt: _Format, activation: _Activation, mode: _Mode
) -> None:
    op, args = _case(fmt, activation, mode=mode)
    result = op(**args)  # type: ignore[operator]
    if mode != "ffn":
        assert result is None
    else:
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3, 32)
        assert result.device.type == "meta"

    scale = args["down_weight_scale"]
    assert isinstance(scale, torch.Tensor)
    with pytest.raises(ValueError, match="scale"):
        op(**{**args, "down_weight_scale": scale.to(torch.bfloat16)})  # type: ignore[operator]
    with pytest.raises(ValueError, match="chunk_rows"):
        op(**{**args, "chunk_rows": 0})  # type: ignore[operator]
    with pytest.raises(ValueError, match="activation_fn"):
        op(**{**args, "activation_fn": "silu"})  # type: ignore[operator]


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize("mode", _MODES)
def test_fake_ffn_rejects_noncontiguous_input(
    fmt: _Format, activation: _Activation, mode: _Mode
) -> None:
    op, args = _case(fmt, activation, mode=mode)
    noncontiguous = torch.empty(2, 3, 256, device="meta", dtype=torch.bfloat16).transpose(0, 1)
    with pytest.raises(ValueError, match="contiguous"):
        op(**{**args, "input": noncontiguous})  # type: ignore[operator]


@pytest.mark.parametrize("fmt", ["convrot_int8", "nvfp4"])
@pytest.mark.parametrize("mode", _MODES)
def test_fake_ffn_validates_activation_gate_pairing(fmt: _Format, mode: _Mode) -> None:
    gelu_op, gelu_args = _case(fmt, "gelu_tanh", mode=mode)
    with pytest.raises(ValueError, match="gate"):
        gelu_op(  # type: ignore[operator]
            **{**gelu_args, "gate_weight_qdata": gelu_args["up_weight_qdata"]}
        )

    swiglu_op, swiglu_args = _case(fmt, "swiglu", mode=mode)
    with pytest.raises(ValueError, match="gate"):
        swiglu_op(**{**swiglu_args, "gate_weight_qdata": None})  # type: ignore[operator]


@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    ("input_features", "intermediate_features", "match"),
    [(80, 64, "source features"), (256, 80, "activated features")],
)
def test_fake_convrot_nvfp4_validates_rotation_divisibility(
    activation: _Activation,
    mode: _Mode,
    input_features: int,
    intermediate_features: int,
    match: str,
) -> None:
    op, args = _case(
        "convrot_nvfp4",
        activation,
        mode=mode,
        input_features=input_features,
        intermediate_features=intermediate_features,
    )
    with pytest.raises(ValueError, match=match):
        op(**args)  # type: ignore[operator]


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
def test_fake_update_validates_output_base_and_pairing(
    fmt: _Format, activation: _Activation
) -> None:
    op, gated_args = _case(fmt, activation, mode="gated_updates")
    invalid = torch.empty(2, 3, 32, device="meta", dtype=torch.float16)
    with pytest.raises(ValueError, match="reusable update"):
        op(**{**gated_args, "output": invalid})  # type: ignore[operator]
    with pytest.raises(ValueError, match="base"):
        op(**{**gated_args, "base": invalid})  # type: ignore[operator]
    with pytest.raises(ValueError, match="both be provided or both omitted"):
        op(**{**gated_args, "update_gate": None})  # type: ignore[operator]

    op, residual_args = _case(fmt, activation, mode="residual_add")
    with pytest.raises(ValueError, match="base"):
        op(**{**residual_args, "output": invalid})  # type: ignore[operator]


@pytest.mark.parametrize(
    ("plain_op", "update_op"),
    [
        (int8_ffn.convrot_int8_ffn_op, int8_ffn.convrot_int8_ffn_update_op),
        (nvfp4_ffn.nvfp4_ffn_op, nvfp4_ffn.nvfp4_ffn_update_op),
    ],
)
def test_update_op_mutates_only_output_and_shares_ffn_argument_names(
    plain_op: object, update_op: object
) -> None:
    plain_schema = plain_op._opoverload._schema  # type: ignore[attr-defined]
    update_schema = update_op._opoverload._schema  # type: ignore[attr-defined]
    plain_names = {arg.name for arg in plain_schema.arguments}
    update_names = {arg.name for arg in update_schema.arguments}
    assert plain_names <= update_names
    assert update_names - plain_names == {
        "output",
        "base",
        "update_gate",
        "ffn_gate",
        "gate_indices",
        "python_indexing",
    }
    assert [
        arg.name
        for arg in update_schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ] == ["output"]


@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize(
    ("output_features", "adds_residual"),
    [(5_120, False), (16_384, True), (32_768, True)],
)
def test_convrot_int8_chunk_rows_bounds_workspace(
    activation: _Activation, output_features: int, adds_residual: bool
) -> None:
    source = torch.empty(100_000, 5_120, dtype=torch.bfloat16, device="meta")
    up = torch.empty(13_824, 5_120, dtype=torch.int8, device="meta")
    down = torch.empty(output_features, 13_824, dtype=torch.int8, device="meta")

    rows = int8_ffn.default_chunk_rows(source, up, down, activation, adds_residual=adds_residual)
    projection_features = (2 if activation == "swiglu" else 1) * 13_824
    projected_output_bytes = (
        output_features * source.element_size()
        if adds_residual and output_features > projection_features
        else 0
    )
    bytes_per_row = (
        projection_features * source.element_size()
        + 13_824
        + torch.float32.itemsize
        + projected_output_bytes
    )
    limit = 512 * 1_024**2
    assert rows % 128 == 0
    assert rows * bytes_per_row <= limit
    assert (rows + 128) * bytes_per_row > limit

    # Ultra-wide intermediates clamp cleanly to the 128-row granularity floor.
    ultra_wide_up = torch.empty(2_097_152, 5_120, dtype=torch.int8, device="meta")
    ultra_wide_down = torch.empty(5_120, 2_097_152, dtype=torch.int8, device="meta")
    assert (
        int8_ffn.default_chunk_rows(
            source, ultra_wide_up, ultra_wide_down, activation, adds_residual=False
        )
        == 128
    )

    # Symbolic feature widths fall back to DEFAULT_CHUNK_ROWS.
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        sym_width = shape_env.create_unbacked_symint()
        sym_source = torch.empty(1_024, sym_width, dtype=torch.bfloat16, device="cpu")
        sym_up = torch.empty(1_024, sym_width, dtype=torch.int8, device="cpu")
        sym_down = torch.empty(sym_width, 1_024, dtype=torch.int8, device="cpu")
        assert (
            int8_ffn.default_chunk_rows(
                sym_source, sym_up, sym_down, activation, adds_residual=False
            )
            == int8_ffn.DEFAULT_CHUNK_ROWS
        )


def _nvfp4_prepared_bytes_per_row(features: int) -> int:
    rows = nvfp4_layout.SCALE_ROW_TILE
    qdata_shape = cast(tuple[int, int], nvfp4_layout.qdata_shape(rows, features))
    scale_shape = cast(tuple[int, int], nvfp4_layout.scale_shape(rows, features))
    return (math.prod(qdata_shape) + math.prod(scale_shape)) // rows


@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize(
    ("input_features", "intermediate_features", "output_features", "adds_residual"),
    [
        (5_120, 13_824, 5_120, False),
        (80, 144, 192, True),
        (80, 144, 320, True),
        (512, 256, 256, False),
    ],
)
def test_nvfp4_chunk_rows_bounds_workspace(
    activation: _Activation,
    input_features: int,
    intermediate_features: int,
    output_features: int,
    adds_residual: bool,
) -> None:
    source = torch.empty(100_000, input_features, dtype=torch.bfloat16, device="meta")
    up = torch.empty(intermediate_features, input_features // 2, dtype=torch.uint8, device="meta")
    down = torch.empty(
        output_features, intermediate_features // 2, dtype=torch.uint8, device="meta"
    )

    rows = nvfp4_ffn.default_chunk_rows(source, up, down, activation, adds_residual=adds_residual)
    projection_features = (2 if activation == "swiglu" else 1) * intermediate_features
    projected_output_bytes = (
        output_features * source.element_size()
        if adds_residual and output_features > projection_features
        else 0
    )
    bytes_per_row = (
        max(
            _nvfp4_prepared_bytes_per_row(input_features),
            _nvfp4_prepared_bytes_per_row(intermediate_features),
        )
        + projection_features * source.element_size()
        + projected_output_bytes
    )
    limit = 256 * 1_024**2
    assert rows % 128 == 0
    assert rows * bytes_per_row <= limit
    assert (rows + 128) * bytes_per_row > limit

    # Ultra-wide intermediates clamp cleanly to the 128-row granularity floor.
    ultra_wide_up = torch.empty(2_097_152, input_features // 2, dtype=torch.uint8, device="meta")
    ultra_wide_down = torch.empty(output_features, 2_097_152 // 2, dtype=torch.uint8, device="meta")
    assert (
        nvfp4_ffn.default_chunk_rows(
            source, ultra_wide_up, ultra_wide_down, activation, adds_residual=False
        )
        == 128
    )

    # Symbolic feature widths fall back to DEFAULT_CHUNK_ROWS.
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        sym_half = shape_env.create_unbacked_symint()
        sym_source = torch.empty(1_024, 2 * sym_half, dtype=torch.bfloat16, device="cpu")
        sym_up = torch.empty(1_024, sym_half, dtype=torch.uint8, device="cpu")
        sym_down = torch.empty(2 * sym_half, 512, dtype=torch.uint8, device="cpu")
        assert (
            nvfp4_ffn.default_chunk_rows(
                sym_source, sym_up, sym_down, activation, adds_residual=False
            )
            == nvfp4_ffn.DEFAULT_CHUNK_ROWS
        )


def test_ffn_package_and_compile_options_are_triton_and_extra_free() -> None:
    script = """
import builtins
import importlib.util
import sys

original_import = builtins.__import__
original_find_spec = importlib.util.find_spec
blocked = ("triton", "torchao")

def guarded_import(name, *args, **kwargs):
    if any(name == pkg or name.startswith(pkg + ".") for pkg in blocked):
        raise ModuleNotFoundError(f"{name} intentionally unavailable", name=name)
    return original_import(name, *args, **kwargs)

def guarded_find_spec(name, *args, **kwargs):
    if any(name == pkg or name.startswith(pkg + ".") for pkg in blocked):
        return None
    return original_find_spec(name, *args, **kwargs)

builtins.__import__ = guarded_import
importlib.util.find_spec = guarded_find_spec

import piper_kernels
import piper_kernels.fusions.ffn as ffn_pkg
assert "piper_kernels.fusions.ffn._compile" not in sys.modules
assert "piper_kernels.fusions.ffn._convrot_int8" not in sys.modules
assert "piper_kernels.fusions.ffn._nvfp4" not in sys.modules
assert "piper_kernels.fusions.ffn.triton" not in sys.modules

options = piper_kernels.compile_options(ffn=True)
assert "post_grad_custom_pre_pass" in options
assert "piper_kernels.fusions.ffn._compile" not in sys.modules
assert "piper_kernels.fusions.ffn._convrot_int8" not in sys.modules
assert "piper_kernels.fusions.ffn._nvfp4" not in sys.modules
assert "piper_kernels.fusions.ffn.triton" not in sys.modules
assert "piper_kernels.linear.convrot" not in sys.modules
assert "piper_kernels.linear.nvfp4" not in sys.modules
assert "triton" not in sys.modules
assert "torchao" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


def test_plain_nvfp4_graph_folding_does_not_register_convrot() -> None:
    script = """
import sys
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from piper_kernels import compile_options
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops  # noqa: F401
from piper_kernels.weights.nvfp4 import _layout

options = compile_options(ffn=True)
compiler_pass = options["post_grad_custom_pre_pass"][0]
assert not hasattr(torch.ops.piper_kernels, "convrot_nvfp4_linear")

with FakeTensorMode():
    graph = torch.fx.Graph()
    def ph(name, shape, dtype):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=dtype, device="meta")
        return node

    def linear(name, src, in_f, out_f):
        wq = ph(f"{name}_q", (out_f, in_f // 2), torch.uint8)
        ws = ph(f"{name}_s", _layout.scale_shape(out_f, in_f), torch.float8_e4m3fn)
        node = graph.call_function(
            torch.ops.piper_kernels.nvfp4_linear.default,
            (src, wq, ws, None, None, None, True, False),
        )
        node.meta["val"] = torch.empty(16, out_f, dtype=torch.bfloat16, device="meta")
        return node

    x = ph("x", (16, 64), torch.bfloat16)
    gate = linear("gate", x, 64, 128)
    val = linear("val", x, 64, 128)
    silu = graph.call_function(torch.ops.aten.silu.default, (gate,))
    silu.meta["val"] = torch.empty(16, 128, dtype=torch.bfloat16, device="meta")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (val, silu))
    mul.meta["val"] = torch.empty(16, 128, dtype=torch.bfloat16, device="meta")
    down = linear("down", mul, 128, 64)
    graph.output(down)

    compiler_pass(graph, is_inference=True)
    targets = [n.target for n in graph.nodes if n.op == "call_function"]
    assert torch.ops.piper_kernels.nvfp4_ffn.default in targets

assert not hasattr(torch.ops.piper_kernels, "convrot_nvfp4_linear")
assert not hasattr(torch.ops.piper_kernels, "convrot_int8_linear")
assert "piper_kernels.linear.convrot" not in sys.modules
assert "piper_kernels.fusions.ffn._convrot_int8" not in sys.modules
assert "piper_kernels.fusions.ffn.triton" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
