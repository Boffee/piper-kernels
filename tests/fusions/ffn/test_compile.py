"""Parametrized compiler-folding tests for ConvRot INT8, NVFP4, and ConvRot NVFP4 FFNs."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch
from _accelerators import (
    ensure_convrot_int8_linear_ops,
    ensure_nvfp4_linear_ops,
    exact_sm120_available,
)
from _compile_capture import TargetCapturePass
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels import compile_options

from ._helpers import (
    Activation,
    Int8GatedUpdates,
    Int8GeluFfn,
    Int8SwiGluFfn,
    Nvfp4FfnModule,
    Nvfp4GatedUpdates,
    assert_copy_back_computes_ffn_inside_its_add,
    make_int8_gated_update_arguments,
    make_nvfp4_gated_update_arguments,
    make_nvfp4_operands,
    materialized_nvfp4,
    relative_l2,
)

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"
_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
_SM120 = pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")


def _run_ffn_pass(graph: torch.fx.Graph) -> None:
    passes = compile_options(linear=False, ffn=True)[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    for compiler_pass in passes:
        compiler_pass(graph, is_inference=True)


def test_compile_options_install_idempotent_and_preserve_unrelated_options() -> None:
    options = compile_options({"max_autotune": True}, ffn=True)
    assert options["max_autotune"] is True
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    assert len(passes) == 1
    assert compile_options(options, ffn=True) == options


@pytest.mark.parametrize("ffn_first", [False, True])
def test_compile_options_compose_with_linear_attention_and_residual_updates(
    ffn_first: bool,
) -> None:
    combined = (
        compile_options(compile_options(ffn=True), attention=True, residual_updates=True)
        if ffn_first
        else compile_options(compile_options(attention=True, residual_updates=True), ffn=True)
    )
    assert combined == compile_options(ffn=True, attention=True, residual_updates=True)
    assert combined["allow_buffer_reuse"] is False


def test_compile_options_preserve_unrelated_pass_order() -> None:
    before = object()
    after = object()
    linear_pass = compile_options()[_POST_GRAD_PRE_PASS][0]  # type: ignore[index]
    options = compile_options({_POST_GRAD_PRE_PASS: (before, linear_pass, after)}, ffn=True)
    expected = compile_options(ffn=True)[_POST_GRAD_PRE_PASS][0]  # type: ignore[index]
    assert options[_POST_GRAD_PRE_PASS] == (before, expected, after)


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale_mode", ["dynamic", "source-static", "down-static", "static"])
def test_convrot_int8_compile_folds_ffn(
    activation: Activation,
    bias_dtype: torch.dtype | None,
    dtype: torch.dtype,
    scale_mode: str,
) -> None:
    torch.manual_seed(511)
    model: Int8GeluFfn | Int8SwiGluFfn = (
        Int8GeluFfn(dtype=dtype, bias_dtype=bias_dtype).eval()
        if activation == "gelu_tanh"
        else Int8SwiGluFfn(
            promote_gate=scale_mode in ("source-static", "static"),
            reverse_multiply=scale_mode in ("down-static", "static"),
            dtype=dtype,
            bias_dtype=bias_dtype,
        ).eval()
    )
    if scale_mode in ("source-static", "static"):
        scale = torch.tensor(0.02, device="cuda")
        if isinstance(model, Int8GeluFfn):
            model.up.weight.act_per_tensor_scale = scale
        else:
            model.gate.weight.act_per_tensor_scale = scale
            model.value.weight.act_per_tensor_scale = scale
    if scale_mode in ("down-static", "static"):
        model.down.weight.act_per_tensor_scale = torch.tensor(0.04, device="cuda")

    source = torch.randn(2, 257, model.input_features, dtype=dtype, device="cuda")
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=compile_options())(source)
        torch._dynamo.reset()
        actual = torch.compile(
            model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
        )(source)

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype is dtype
    assert relative_l2(actual, expected) < 0.01
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_ffn.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16])
def test_convrot_int8_compile_folds_autocast_swiglu_ffn(
    bias_dtype: torch.dtype | None,
) -> None:
    torch.manual_seed(512)
    model = Int8SwiGluFfn(dtype=torch.bfloat16, bias_dtype=bias_dtype).eval()
    source = torch.randn(2, 257, model.input_features, dtype=torch.bfloat16, device="cuda")
    capture = TargetCapturePass()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=compile_options())(source)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(compile_options(ffn=True)),
        )(source)

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype is torch.float16
    assert relative_l2(actual, expected) < 0.01
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_ffn.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize(
    "failure", ["projection-escapes", "noncontiguous", "empty", "different-input"]
)
def test_convrot_int8_compile_fails_closed(activation: Activation, failure: str) -> None:
    if activation == "gelu_tanh" and failure == "different-input":
        pytest.skip("gelu_tanh has a single source projection")
    torch.manual_seed(513)
    model: Int8GeluFfn | Int8SwiGluFfn = (
        Int8GeluFfn(expose_up=failure == "projection-escapes").eval()
        if activation == "gelu_tanh"
        else Int8SwiGluFfn(expose_gate=failure == "projection-escapes").eval()
    )
    if failure == "noncontiguous":
        storage = torch.randn(257, 2 * model.input_features, dtype=torch.bfloat16, device="cuda")
        arguments: tuple[torch.Tensor, ...] = (storage[:, ::2],)
    elif failure == "empty":
        arguments = (torch.empty(0, model.input_features, dtype=torch.bfloat16, device="cuda"),)
    elif failure == "different-input":
        source = torch.randn(257, model.input_features, dtype=torch.bfloat16, device="cuda")
        arguments = (source, torch.randn_like(source))
    else:
        arguments = (torch.randn(257, model.input_features, dtype=torch.bfloat16, device="cuda"),)

    capture = TargetCapturePass()
    ordinary = compile_options()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=ordinary)(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(compile_options(ordinary, ffn=True)),
        )(*arguments)

    expected_values = expected if isinstance(expected, tuple) else (expected,)
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    assert all(
        torch.equal(left, right) for left, right in zip(actual_values, expected_values, strict=True)
    )
    assert "piper_kernels.convrot_int8_ffn.default" not in capture.target_names


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("python_indexing", [False, True])
def test_convrot_int8_compile_folds_gated_updates_and_residual_add(
    activation: Activation, python_indexing: bool
) -> None:
    torch.manual_seed(517)
    model = Int8GatedUpdates(activation=activation, python_indexing=python_indexing).eval()
    arguments = make_int8_gated_update_arguments(model, 257)
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=compile_options())(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
        )(*arguments)

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert relative_l2(actual, expected) < 0.01
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_ffn_update_.default) == 1
    assert "piper_kernels.convrot_int8_ffn.default" not in capture.target_names

    assert_copy_back_computes_ffn_inside_its_add(
        model,
        arguments,
        "convrot_int8_ffn_update_",
        compile_options(),
        compile_options(ffn=True),
        0.01,
    )


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("expose", ["ffn", "hidden"])
def test_convrot_int8_gated_updates_fail_closed_when_intermediate_escapes(
    expose: Literal["ffn", "hidden"],
) -> None:
    torch.manual_seed(216)
    model = Int8GatedUpdates(activation="swiglu", expose=expose).eval()
    arguments = make_int8_gated_update_arguments(model, 257)
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=compile_options())(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
        )(*arguments)

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)
    assert all(
        relative_l2(left, right) < 0.01 for left, right in zip(actual, expected, strict=True)
    )
    assert "piper_kernels.convrot_int8_ffn_update_.default" not in capture.target_names
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_ffn.default) == 1


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("update_mode", ["direct", "alias"])
def test_convrot_int8_gated_updates_do_not_mutate_caller_input(
    update_mode: Literal["direct", "alias"],
) -> None:
    torch.manual_seed(217)
    model = Int8GatedUpdates(activation="swiglu", update_mode=update_mode).eval()
    arguments = make_int8_gated_update_arguments(model, 257)
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=compile_options())(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
        )(*arguments)

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert relative_l2(actual, expected) < 0.01
    assert "piper_kernels.convrot_int8_ffn_update_.default" not in capture.target_names
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_ffn.default) == 1


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("mode", ["shared", "distinct", "mixed", "static-down", "dynamic-down"])
def test_convrot_int8_compiled_static_scales_update_without_recompilation(mode: str) -> None:
    def _scales(scale_mode: str) -> tuple[torch.Tensor | None, ...]:
        scale = torch.tensor(0.02, device="cuda")
        down = torch.tensor(0.25, device="cuda")
        mapping: dict[str, tuple[torch.Tensor | None, ...]] = {
            "shared": (scale, scale, down),
            "distinct": (scale, torch.tensor(0.04, device="cuda"), down),
            "mixed": (scale, None, down),
            "static-down": (None, None, down),
            "dynamic-down": (scale, scale, None),
        }
        return mapping[scale_mode]

    torch._dynamo.reset()
    torch.manual_seed(1072)
    model = Int8SwiGluFfn().eval()
    scales = _scales(mode)
    for projection, scale in zip((model.gate, model.value, model.down), scales, strict=True):
        projection.weight.act_per_tensor_scale = scale
    source = torch.randn(129, model.input_features, device="cuda", dtype=torch.bfloat16)
    capture = TargetCapturePass()
    compiled = torch.compile(
        model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
    )
    ordinary = torch.compile(model, fullgraph=True, options=compile_options())
    with torch.inference_mode():
        first = compiled(source)
        assert relative_l2(first, ordinary(source)) < 0.01
        next(scale for scale in scales if scale is not None).mul_(0.5)
        second = compiled(source)
        assert relative_l2(second, ordinary(source)) < 0.01
    assert not torch.equal(first, second)
    assert capture.calls == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_ffn.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


@pytest.mark.gpu
@_GPU
def test_convrot_int8_static_ffn_feeds_unfused_projection_and_cache_isolation() -> None:
    class FfnWithTail(Int8SwiGluFfn):
        output_features = 512

        def __init__(self) -> None:
            super().__init__()
            from ._helpers import _make_int8_module_linear  # noqa: PLC0415

            self.tail = _make_int8_module_linear(64, self.output_features, None, torch.bfloat16)

        def forward(self, activation: torch.Tensor) -> torch.Tensor:
            out = super().forward(activation)
            assert isinstance(out, torch.Tensor)
            return self.tail(out)

    torch._dynamo.reset()
    torch.manual_seed(1075)
    model = FfnWithTail().eval()
    model.gate.weight.act_per_tensor_scale = torch.tensor(0.02, device="cuda")
    model.value.weight.act_per_tensor_scale = None
    model.down.weight.act_per_tensor_scale = torch.tensor(0.25, device="cuda")
    source = torch.randn(129, model.input_features, device="cuda", dtype=torch.bfloat16)
    capture = TargetCapturePass()
    with torch.inference_mode():
        gate, value = model.gate(source), model.value(source)
        weight = model.down.weight
        ensure_convrot_int8_linear_ops()
        down = torch.ops.piper_kernels.convrot_int8_linear(
            torch.cat((value, gate), dim=-1),
            weight.qdata,
            weight.scale,
            model.down.bias,
            weight.group_size,
            "swiglu",
            weight.act_per_tensor_scale,
        )
        expected = model.tail(down)
        actual = torch.compile(
            model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
        )(source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_ffn.default) == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_linear.default) == 1

    # Verify FX cache does not confuse shared vs distinct static scale aliases.
    cache_capture = TargetCapturePass(cache_key=b"ffn-int8-scale-cache-collisions")
    cache_options = cache_capture.wrap_options(compile_options(ffn=True))
    for mode in ("shared", "distinct", "mixed", "shared"):
        torch._dynamo.reset()
        torch.manual_seed(1074)
        m = Int8SwiGluFfn().eval()
        s = torch.tensor(0.02, device="cuda")
        d = torch.tensor(0.25, device="cuda")
        scales = (
            (s, s, d)
            if mode == "shared"
            else (s, torch.tensor(0.04, device="cuda"), d)
            if mode == "distinct"
            else (s, None, d)
        )
        for projection, sc in zip((m.gate, m.value, m.down), scales, strict=True):
            projection.weight.act_per_tensor_scale = sc
        act = torch.randn(131, m.input_features, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            assert (
                relative_l2(torch.compile(m, fullgraph=True, options=cache_options)(act), m(act))
                < 0.01
            )


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize(
    ("up_group_size", "down_group_size"),
    [(None, None), (16, 64), (16, None), (None, 64)],
    ids=["standard", "convrot", "convrot-standard", "standard-convrot"],
)
@pytest.mark.parametrize(
    ("up_dynamic", "down_dynamic", "bias_dtype", "high_first"),
    [
        (False, False, None, False),
        (False, True, torch.float16, True),
        (True, False, torch.bfloat16, False),
        (True, True, torch.float32, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_compile_folds_standard_convrot_and_mixed_ffn(
    activation: Activation,
    up_group_size: int | None,
    down_group_size: int | None,
    up_dynamic: bool,
    down_dynamic: bool,
    bias_dtype: torch.dtype | None,
    high_first: bool,
    dtype: torch.dtype,
) -> None:
    operands = make_nvfp4_operands(
        activation,
        rows=258,
        up_dynamic=up_dynamic,
        down_dynamic=down_dynamic,
        up_group_size=up_group_size,
        down_group_size=down_group_size,
        bias_dtype=bias_dtype,
        up_high_first=high_first,
        down_high_first=not high_first,
        dtype=dtype,
        seed=1009 + up_dynamic + 10 * down_dynamic,
    )
    source = operands.input.reshape(2, 129, -1)
    model = Nvfp4FfnModule(
        operands,
        promote_gate=down_dynamic,
        reverse_multiply=up_dynamic,
        explicit_promotion=high_first,
    ).eval()
    ordinary = compile_options()
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=ordinary)(source)
        torch._dynamo.reset()
        actual = torch.compile(
            model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
        )(source)

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype is dtype
    assert relative_l2(actual, expected) < (0.1 if down_dynamic else 0.07)
    if up_group_size is None and down_group_size is None:
        reference = materialized_nvfp4(operands).reshape_as(actual)
        assert relative_l2(actual, reference) < (0.1 if down_dynamic else 0.08)
    assert capture.target_names.count("piper_kernels.nvfp4_ffn.default") == 1
    assert "piper_kernels.nvfp4_linear.default" not in capture.target_names
    assert "piper_kernels.convrot_nvfp4_linear.default" not in capture.target_names


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("group_size", [None, 16], ids=["standard", "convrot"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("high_first", [False, True])
def test_nvfp4_shared_projection_weights_preserve_distinct_biases(
    group_size: int | None,
    dynamic: bool,
    high_first: bool,
) -> None:
    operands = make_nvfp4_operands(
        "swiglu",
        rows=129,
        up_dynamic=dynamic,
        down_dynamic=dynamic,
        up_group_size=group_size,
        down_group_size=group_size,
        up_high_first=high_first,
        down_high_first=high_first,
        seed=911,
    )
    assert operands.gate is not None
    operands = replace(operands, up=replace(operands.up, weight=operands.gate.weight))
    assert operands.gate.bias is not operands.up.bias
    model = Nvfp4FfnModule(operands).eval()
    assert model.gate is not None
    model.up.weight = model.gate.weight
    capture = TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        actual = torch.compile(
            model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
        )(operands.input)
        expected = torch.ops.piper_kernels.nvfp4_ffn.default(**operands.arguments(1536))
    assert "piper_kernels.nvfp4_ffn.default" in capture.target_names
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("group_size", [None, 16], ids=["standard", "convrot"])
@pytest.mark.parametrize("failure", ["projection-escapes", "noncontiguous", "different-input"])
def test_nvfp4_compile_fails_closed(
    activation: Activation, group_size: int | None, failure: str
) -> None:
    if activation == "gelu_tanh" and failure == "different-input":
        pytest.skip("gelu_tanh has a single source projection")
    operands = make_nvfp4_operands(
        activation,
        rows=258,
        up_dynamic=False,
        down_dynamic=False,
        up_group_size=group_size,
        down_group_size=group_size,
        seed=1013,
    )
    model = Nvfp4FfnModule(operands, expose_intermediate=failure == "projection-escapes").eval()
    if failure == "noncontiguous":
        arguments: tuple[torch.Tensor, ...] = (operands.input.view(2, 129, -1).transpose(0, 1),)
    elif failure == "different-input":
        arguments = (operands.input, torch.randn_like(operands.input))
    else:
        arguments = (operands.input,)

    ordinary = compile_options()
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=ordinary)(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model, fullgraph=True, options=capture.wrap_options(compile_options(ordinary, ffn=True))
        )(*arguments)

    expected_values = expected if isinstance(expected, tuple) else (expected,)
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    assert all(
        torch.equal(left, right) for left, right in zip(actual_values, expected_values, strict=True)
    )
    assert "piper_kernels.nvfp4_ffn.default" not in capture.target_names


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
@pytest.mark.parametrize("group_size", [None, 16], ids=["standard", "convrot"])
@pytest.mark.parametrize("python_indexing", [False, True])
def test_nvfp4_compile_folds_gated_updates_and_residual_add(
    activation: Activation, group_size: int | None, python_indexing: bool
) -> None:
    operands = make_nvfp4_operands(
        activation,
        rows=257,
        output_features=384,
        up_dynamic=False,
        down_dynamic=False,
        up_group_size=group_size,
        down_group_size=group_size,
        bias_dtype=torch.float32,
        seed=1021,
    )
    model = Nvfp4GatedUpdates(operands, python_indexing=python_indexing).eval()
    arguments = make_nvfp4_gated_update_arguments(257, operands.down.weight.shape[0])
    ordinary = compile_options()
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=ordinary)(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
        )(*arguments)

    assert relative_l2(actual, expected) < 0.07
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_ffn_update_.default) == 1
    assert "piper_kernels.nvfp4_ffn.default" not in capture.target_names

    assert_copy_back_computes_ffn_inside_its_add(
        model,
        arguments,
        "nvfp4_ffn_update_",
        ordinary,
        compile_options(ffn=True),
        0.07,
    )


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize("fmt", ["convrot_int8", "nvfp4", "convrot_nvfp4"])
@pytest.mark.parametrize("activation", ["gelu_tanh", "swiglu"])
def test_compiled_ffn_reuses_dynamic_graph_and_replays_in_cuda_graph(
    fmt: str, activation: Activation
) -> None:
    if fmt != "convrot_int8" and not exact_sm120_available():
        pytest.skip("requires exact NVIDIA SM120")
    torch.manual_seed(701)
    if fmt == "convrot_int8":
        model: torch.nn.Module = (
            Int8GeluFfn().eval() if activation == "gelu_tanh" else Int8SwiGluFfn().eval()
        )
        width = 256
        out_width = 384
        expected_op = "piper_kernels.convrot_int8_ffn.default"
    else:
        group = 16 if fmt == "convrot_nvfp4" else None
        operands = make_nvfp4_operands(
            activation,
            rows=257,
            up_dynamic=False,
            down_dynamic=False,
            up_group_size=group,
            down_group_size=group,
            seed=701,
        )
        model = Nvfp4FfnModule(operands).eval()
        width = operands.input.shape[-1]
        out_width = operands.down.weight.shape[0]
        expected_op = "piper_kernels.nvfp4_ffn.default"

    first = torch.randn(257, width, device="cuda", dtype=torch.bfloat16)
    second = torch.randn(385, width, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(first, 0)
    torch._dynamo.mark_dynamic(second, 0)
    capture = TargetCapturePass()
    torch._dynamo.reset()
    compiled = torch.compile(
        model, fullgraph=True, options=capture.wrap_options(compile_options(ffn=True))
    )
    with torch.inference_mode():
        assert compiled(first).shape == (257, out_width)
        assert compiled(second).shape == (385, out_width)
    assert capture.calls == 1
    assert capture.target_names.count(expected_op) == 1

    # Explicit CUDA graph capture & replay with updated input values.
    static_x = torch.randn(256, width, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.reset()
    compiled_static = torch.compile(
        model, fullgraph=True, options=compile_options({"triton.cudagraphs": False}, ffn=True)
    )
    with torch.inference_mode():
        warmup_out = compiled_static(static_x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out = compiled_static(static_x)
        fresh = torch.randn_like(static_x)
        static_x.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        expected_replay = compiled_static(fresh)
    assert warmup_out.shape == captured_out.shape == (256, out_width)
    torch.testing.assert_close(captured_out, expected_replay, rtol=0, atol=0)


@pytest.mark.gpu
@_SM120
@pytest.mark.parametrize("register_convrot", [False, True])
def test_nvfp4_cuda_compile_in_fresh_process_respects_convrot_boundary(
    register_convrot: bool,
) -> None:
    script = """
import sys
sys.path.insert(0, sys.argv[2])
import torch
from _compile_capture import TargetCapturePass
from piper_kernels import compile_options
from tests.fusions.ffn._helpers import Nvfp4FfnModule, make_nvfp4_operands

assert not hasattr(torch.ops.piper_kernels, "convrot_nvfp4_linear")
register_convrot = sys.argv[1] == "True"
if register_convrot:
    import piper_kernels.linear.convrot.nvfp4._ops  # noqa: F401

torch.set_num_threads(1)
operands = make_nvfp4_operands(
    "swiglu", rows=256, output_features=256, up_dynamic=True, down_dynamic=True
)
model = Nvfp4FfnModule(operands).eval()
capture = TargetCapturePass()
options = capture.wrap_options(compile_options(ffn=True))
options.update({"triton.cudagraphs": False, "compile_threads": 1})
with torch.no_grad():
    actual = torch.compile(model, fullgraph=True, options=options)(operands.input)
    expected = torch.ops.piper_kernels.nvfp4_ffn.default(**operands.arguments(1536))

torch.testing.assert_close(actual, expected, rtol=0, atol=0)
assert capture.targets.count(torch.ops.piper_kernels.nvfp4_ffn.default) == 1
assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets
assert hasattr(torch.ops.piper_kernels, "convrot_nvfp4_linear") == register_convrot
"""
    subprocess.run(
        [sys.executable, "-c", script, str(register_convrot), str(Path(__file__).parents[2])],
        env={**os.environ, "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"},
        check=True,
        timeout=180,
    )


def _add_int8_linear_node(
    graph: torch.fx.Graph,
    name: str,
    src: torch.fx.Node,
    rows: int,
    in_features: int,
    out_features: int,
    device: str,
) -> torch.fx.Node:
    ensure_convrot_int8_linear_ops()
    weight = graph.placeholder(f"{name}_w")
    weight.meta["val"] = torch.empty(out_features, in_features, dtype=torch.int8, device=device)
    scale = graph.placeholder(f"{name}_s")
    scale.meta["val"] = torch.empty(out_features, 1, dtype=torch.float32, device=device)
    node = graph.call_function(
        torch.ops.piper_kernels.convrot_int8_linear.default,
        kwargs={
            "input": src,
            "weight_qdata": weight,
            "weight_scale": scale,
            "bias": None,
            "group_size": 16,
        },
    )
    node.meta["val"] = torch.empty(rows, out_features, dtype=torch.bfloat16, device=device)
    return node


def _add_nvfp4_meta_linear_node(
    graph: torch.fx.Graph,
    name: str,
    src: torch.fx.Node,
    rows: int,
    features: int,
) -> torch.fx.Node:
    ensure_nvfp4_linear_ops()
    weight = graph.placeholder(f"{name}_q")
    weight.meta["val"] = torch.empty(features, features // 2, dtype=torch.uint8, device="meta")
    scale = graph.placeholder(f"{name}_s")
    # Swizzled NVFP4 scale layout for 128x128 is (32, 32).
    scale.meta["val"] = torch.empty(32, 32, dtype=torch.float8_e4m3fn, device="meta")
    node = graph.call_function(
        torch.ops.piper_kernels.nvfp4_linear.default,
        kwargs={
            "input": src,
            "weight_qdata": weight,
            "weight_scale": scale,
            "weight_per_tensor_scale": None,
            "activation_per_tensor_scale": None,
            "bias": None,
            "dynamic_activation_scale": True,
            "high_first": False,
        },
    )
    node.meta["val"] = torch.empty(rows, features, dtype=torch.bfloat16, device="meta")
    return node


def _fx_call(
    graph: torch.fx.Graph,
    target: torch._ops.OpOverload,
    shape: tuple[int, ...],
    val_dtype: torch.dtype,
    device: str,
    /,
    **kwargs: object,
) -> torch.fx.Node:
    node = graph.call_function(target, kwargs=kwargs)  # type: ignore[arg-type]
    node.meta["val"] = torch.empty(shape, dtype=val_dtype, device=device)
    return node


@pytest.mark.parametrize(
    ("case", "alpha", "should_fold"),
    [
        ("gelu_tanh", 1, True),
        ("gelu_tanh", 1.0, True),
        ("gelu_tanh", 2, False),
        ("swiglu_decomposed", 1, True),
        ("swiglu_decomposed", 1.0, True),
        ("swiglu_decomposed", 2, False),
    ],
)
def test_schema_bound_activation_matching_accepts_kwargs_and_checks_alpha(
    case: str,
    alpha: int | float,
    should_fold: bool,
) -> None:
    """Keyword activation operands and explicit alpha=1 must match; alpha!=1 must not."""
    aten = torch.ops.aten
    prims = torch.ops.prims
    rows, features = 16, 128
    shape = (rows, features)
    with FakeTensorMode():
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        if case == "gelu_tanh":
            up = _add_nvfp4_meta_linear_node(graph, "up", x, rows, features)
            up_fp32 = _fx_call(
                graph,
                prims.convert_element_type.default,
                shape,
                torch.float32,
                "meta",
                a=up,
                dtype=torch.float32,
            )
            half = _fx_call(
                graph, aten.mul.Tensor, shape, torch.float32, "meta", self=up_fp32, other=0.5
            )
            sq = _fx_call(
                graph, aten.mul.Tensor, shape, torch.float32, "meta", self=up_fp32, other=up_fp32
            )
            cube = _fx_call(
                graph, aten.mul.Tensor, shape, torch.float32, "meta", self=sq, other=up_fp32
            )
            cubic = _fx_call(
                graph, aten.mul.Tensor, shape, torch.float32, "meta", self=cube, other=0.044715
            )
            inner = _fx_call(
                graph,
                aten.add.Tensor,
                shape,
                torch.float32,
                "meta",
                self=up_fp32,
                other=cubic,
                alpha=1,
            )
            scaled = _fx_call(
                graph,
                aten.mul.Tensor,
                shape,
                torch.float32,
                "meta",
                self=inner,
                other=math.sqrt(2.0 / math.pi),
            )
            tanh_node = _fx_call(
                graph, aten.tanh.default, shape, torch.float32, "meta", self=scaled
            )
            shifted = _fx_call(
                graph,
                aten.add.Tensor,
                shape,
                torch.float32,
                "meta",
                self=tanh_node,
                other=1.0,
                alpha=alpha,
            )
            prod = _fx_call(
                graph, aten.mul.Tensor, shape, torch.float32, "meta", self=half, other=shifted
            )
            activated = _fx_call(
                graph,
                prims.convert_element_type.default,
                shape,
                torch.bfloat16,
                "meta",
                a=prod,
                dtype=torch.bfloat16,
            )
        else:
            val = _add_nvfp4_meta_linear_node(graph, "val", x, rows, features)
            gate = _add_nvfp4_meta_linear_node(graph, "gate", x, rows, features)
            neg = _fx_call(graph, aten.neg.default, shape, torch.bfloat16, "meta", self=gate)
            exp = _fx_call(graph, aten.exp.default, shape, torch.bfloat16, "meta", self=neg)
            denom = _fx_call(
                graph,
                aten.add.Tensor,
                shape,
                torch.bfloat16,
                "meta",
                self=exp,
                other=1.0,
                alpha=alpha,
            )
            silu = _fx_call(
                graph, aten.div.Tensor, shape, torch.bfloat16, "meta", self=gate, other=denom
            )
            activated = _fx_call(
                graph, aten.mul.Tensor, shape, torch.bfloat16, "meta", self=val, other=silu
            )

        down = _add_nvfp4_meta_linear_node(graph, "down", activated, rows, features)
        graph.output(down)
        _run_ffn_pass(graph)
        targets = [node.target for node in graph.nodes if node.op == "call_function"]

    ffn_op = getattr(torch.ops.piper_kernels, "nvfp4_ffn", None)
    folded = ffn_op is not None and ffn_op.default in targets
    assert folded is should_fold
    assert (torch.ops.piper_kernels.nvfp4_linear.default in targets) is not should_fold


@pytest.mark.gpu
@_GPU
@pytest.mark.parametrize(
    ("hidden_alpha", "output_alpha", "should_fold_update"),
    [
        (1, 1.0, True),
        (2, 1, False),
        (1, 2, False),
    ],
)
def test_schema_bound_gated_updates_accept_kwargs_and_check_alpha(
    hidden_alpha: int | float,
    output_alpha: int | float,
    should_fold_update: bool,
) -> None:
    """Keyword gated-update adds with alpha=1 must fold; alpha!=1 must reject update fusion."""
    aten = torch.ops.aten
    rows, width, hidden = 16, 64, 128
    with FakeTensorMode():
        graph = torch.fx.Graph()
        base = graph.placeholder("base")
        base.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        src = graph.placeholder("src")
        src.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        update_gate = graph.placeholder("update_gate")
        update_gate.meta["val"] = torch.empty(4, width, dtype=torch.bfloat16, device="cuda")
        ffn_gate = graph.placeholder("ffn_gate")
        ffn_gate.meta["val"] = torch.empty(4, width, dtype=torch.bfloat16, device="cuda")
        indices = graph.placeholder("indices")
        indices.meta["val"] = torch.empty(rows, dtype=torch.int64, device="cuda")

        reusable = graph.call_function(aten.relu.default, (src,))
        reusable.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        sel_update = graph.call_function(aten.index_select.default, (update_gate, 0, indices))
        sel_update.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        gated_update = graph.call_function(
            aten.mul.Tensor, kwargs={"self": sel_update, "other": reusable}
        )
        gated_update.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        hidden_node = graph.call_function(
            aten.add.Tensor,
            kwargs={"self": base, "other": gated_update, "alpha": hidden_alpha},
        )
        hidden_node.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")

        ffn_in = graph.call_function(aten.relu.default, (hidden_node,))
        ffn_in.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        val = _add_int8_linear_node(graph, "val", ffn_in, rows, width, hidden, "cuda")
        gate = _add_int8_linear_node(graph, "gate", ffn_in, rows, width, hidden, "cuda")
        silu = graph.call_function(aten.silu.default, kwargs={"self": gate})
        silu.meta["val"] = torch.empty(rows, hidden, dtype=torch.bfloat16, device="cuda")
        activated = graph.call_function(aten.mul.Tensor, kwargs={"self": val, "other": silu})
        activated.meta["val"] = torch.empty(rows, hidden, dtype=torch.bfloat16, device="cuda")
        down = _add_int8_linear_node(graph, "down", activated, rows, hidden, width, "cuda")

        sel_ffn = graph.call_function(aten.index_select.default, (ffn_gate, 0, indices))
        sel_ffn.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        gated_ffn = graph.call_function(aten.mul.Tensor, kwargs={"self": sel_ffn, "other": down})
        gated_ffn.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        output = graph.call_function(
            aten.add.Tensor,
            kwargs={"self": hidden_node, "other": gated_ffn, "alpha": output_alpha},
        )
        output.meta["val"] = torch.empty(rows, width, dtype=torch.bfloat16, device="cuda")
        graph.output(output)
        _run_ffn_pass(graph)
        target_names = [str(node.target) for node in graph.nodes if node.op == "call_function"]

    assert ("piper_kernels.convrot_int8_ffn_update_.default" in target_names) is should_fold_update
    assert ("piper_kernels.convrot_int8_ffn.default" in target_names) is not should_fold_update


@pytest.mark.parametrize("external_consumer", [False, True])
def test_nvfp4_swiglu_coalesces_duplicate_autocast_inputs(external_consumer: bool) -> None:
    """Identical input casts feeding NVFP4 SwiGLU projections must coalesce and fold FFN."""
    aten = torch.ops.aten
    prims = torch.ops.prims
    rows, features = 16, 128
    shape = (rows, features)
    with FakeTensorMode():
        graph = torch.fx.Graph()
        raw = graph.placeholder("raw")
        raw.meta["val"] = torch.empty(shape, dtype=torch.float32, device="meta")
        cast_val = _fx_call(
            graph,
            prims.convert_element_type.default,
            shape,
            torch.bfloat16,
            "meta",
            a=raw,
            dtype=torch.bfloat16,
        )
        cast_gate = _fx_call(
            graph,
            prims.convert_element_type.default,
            shape,
            torch.bfloat16,
            "meta",
            a=raw,
            dtype=torch.bfloat16,
        )
        val = _add_nvfp4_meta_linear_node(graph, "val", cast_val, rows, features)
        gate = _add_nvfp4_meta_linear_node(graph, "gate", cast_gate, rows, features)
        silu = _fx_call(graph, aten.silu.default, shape, torch.bfloat16, "meta", self=gate)
        activated = _fx_call(
            graph, aten.mul.Tensor, shape, torch.bfloat16, "meta", self=val, other=silu
        )
        down = _add_nvfp4_meta_linear_node(graph, "down", activated, rows, features)
        graph.output((down, cast_val) if external_consumer else down)
        _run_ffn_pass(graph)
        graph.lint()
        target_names = [str(node.target) for node in graph.nodes if node.op == "call_function"]

    assert target_names.count("prims.convert_element_type.default") == (
        2 if external_consumer else 1
    )
    assert target_names.count("piper_kernels.nvfp4_ffn.default") == (0 if external_consumer else 1)
    assert target_names.count("piper_kernels.nvfp4_linear.default") == (
        3 if external_consumer else 0
    )
