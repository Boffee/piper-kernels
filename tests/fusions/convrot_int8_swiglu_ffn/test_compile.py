"""Tests for automatic semantic ConvRot INT8 SwiGLU FFN folding."""

from typing import Literal

import pytest
import torch
from _compile_capture import TargetCapturePass

from piper_kernels.fusions.convrot_int8_sparse_piper import (
    convrot_int8_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_int8_sparse_piper._compile import (
    compile_pass as sparse_piper_compile_pass,
)
from piper_kernels.fusions.convrot_int8_swiglu_ffn import convrot_int8_swiglu_ffn_compile_options
from piper_kernels.fusions.convrot_int8_swiglu_ffn._compile import (
    compile_pass as fusion_compile_pass,
)
from piper_kernels.linear.convrot import convrot_int8_compile_options
from piper_kernels.linear.convrot.int8._compile import compile_pass as convrot_int8_compile_pass

from ._helpers import GatedUpdates, SwiGluFfn, make_gated_update_arguments, relative_l2

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


def test_compile_options_install_fusion_before_convrot() -> None:
    options = convrot_int8_swiglu_ffn_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (fusion_compile_pass, convrot_int8_compile_pass)


def test_compile_options_reapply_without_duplication() -> None:
    options = convrot_int8_swiglu_ffn_compile_options(convrot_int8_compile_options())

    assert options[_POST_GRAD_PRE_PASS] == (fusion_compile_pass, convrot_int8_compile_pass)
    assert convrot_int8_swiglu_ffn_compile_options(options) == options


def test_compile_options_preserve_unrelated_pass_order() -> None:
    before_convrot = object()
    after_convrot = object()
    options = convrot_int8_swiglu_ffn_compile_options(
        {_POST_GRAD_PRE_PASS: (before_convrot, convrot_int8_compile_pass, after_convrot)}
    )

    assert options[_POST_GRAD_PRE_PASS] == (
        before_convrot,
        fusion_compile_pass,
        convrot_int8_compile_pass,
        after_convrot,
    )


@pytest.mark.parametrize("ffn_first", [False, True])
def test_compile_options_compose_with_sparse_piper(ffn_first: bool) -> None:
    options = (
        convrot_int8_sparse_piper_compile_options(convrot_int8_swiglu_ffn_compile_options())
        if ffn_first
        else convrot_int8_swiglu_ffn_compile_options(convrot_int8_sparse_piper_compile_options())
    )

    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    assert passes.count(fusion_compile_pass) == 1
    assert passes.count(sparse_piper_compile_pass) == 1
    assert passes.count(convrot_int8_compile_pass) == 1
    assert passes.index(fusion_compile_pass) < passes.index(convrot_int8_compile_pass)
    assert passes.index(sparse_piper_compile_pass) < passes.index(convrot_int8_compile_pass)


def test_fusion_compiler_pass_uuid_is_versioned_and_stable() -> None:
    assert fusion_compile_pass.uuid() == fusion_compile_pass.uuid()
    assert fusion_compile_pass.uuid()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize(
    ("promote_gate", "reverse_multiply", "bias_dtype"),
    [
        (False, False, None),
        (False, True, torch.float16),
        (False, True, torch.bfloat16),
        (True, False, torch.float32),
        (True, True, None),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compile_options_fold_semantic_swiglu_ffn(
    promote_gate: bool,
    reverse_multiply: bool,
    bias_dtype: torch.dtype | None,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(220 + promote_gate + 10 * reverse_multiply)
    model = SwiGluFfn(
        promote_gate=promote_gate,
        reverse_multiply=reverse_multiply,
        bias_dtype=bias_dtype,
        dtype=dtype,
    ).eval()
    activation = torch.randn(
        2,
        257,
        model.input_features,
        dtype=dtype,
        device="cuda",
    )
    capture = TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            activation
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(convrot_int8_swiglu_ffn_compile_options()),
        )(activation)

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype is dtype
    assert relative_l2(actual, expected) < 0.01
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("failure", ["projection-escapes", "different-input", "noncontiguous"])
def test_compile_options_fail_closed(failure: str) -> None:
    torch.manual_seed(225)
    model = SwiGluFfn(expose_gate=failure == "projection-escapes").eval()
    activation = torch.randn(257, model.input_features, dtype=torch.bfloat16, device="cuda")
    if failure == "different-input":
        arguments = activation, torch.randn_like(activation)
    elif failure == "noncontiguous":
        storage = torch.randn(
            257,
            2 * model.input_features,
            dtype=torch.bfloat16,
            device="cuda",
        )
        arguments = (storage[:, ::2],)
    else:
        arguments = (activation,)
    capture = TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            *arguments
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(convrot_int8_swiglu_ffn_compile_options()),
        )(*arguments)

    expected_values = expected if isinstance(expected, tuple) else (expected,)
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    assert all(
        torch.equal(left, right) for left, right in zip(actual_values, expected_values, strict=True)
    )
    assert torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_compiled_ffn_reuses_one_dynamic_row_graph() -> None:
    torch.manual_seed(227)
    model = SwiGluFfn().eval()
    first = torch.randn(257, model.input_features, dtype=torch.bfloat16, device="cuda")
    second = torch.randn(385, model.input_features, dtype=torch.bfloat16, device="cuda")
    torch._dynamo.mark_dynamic(first, 0)
    torch._dynamo.mark_dynamic(second, 0)
    capture = TargetCapturePass()
    torch._dynamo.reset()
    compiled = torch.compile(
        model,
        fullgraph=True,
        options=capture.wrap_options(convrot_int8_swiglu_ffn_compile_options()),
    )

    with torch.no_grad():
        first_output = compiled(first)
        second_output = compiled(second)

    assert first_output.shape == (257, model.output_features)
    assert second_output.shape == (385, model.output_features)
    assert capture.calls == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("python_indexing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compile_options_fold_h3_style_gated_updates(
    python_indexing: bool, dtype: torch.dtype
) -> None:
    torch.manual_seed(224)
    model = GatedUpdates(python_indexing=python_indexing, dtype=dtype).eval()
    arguments = make_gated_update_arguments(model, 257)
    capture = TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            *arguments
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(convrot_int8_swiglu_ffn_compile_options()),
        )(*arguments)

    assert actual.dtype is dtype
    assert relative_l2(actual, expected) < 0.01
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_swiglu_ffn_gated_updates_.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("expose", ["ffn", "hidden"])
def test_gated_updates_fail_closed_when_intermediate_escapes(
    expose: Literal["ffn", "hidden"],
) -> None:
    torch.manual_seed(216)
    model = GatedUpdates(expose=expose).eval()
    arguments = make_gated_update_arguments(model, 257)
    capture = TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            *arguments
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(convrot_int8_swiglu_ffn_compile_options()),
        )(*arguments)

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)
    assert all(
        relative_l2(left, right) < 0.01 for left, right in zip(actual, expected, strict=True)
    )
    assert (
        torch.ops.piper_kernels.convrot_int8_swiglu_ffn_gated_updates_.default
        not in capture.targets
    )
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("update_mode", ["direct", "alias"])
def test_gated_updates_do_not_mutate_caller_input(
    update_mode: Literal["direct", "alias"],
) -> None:
    torch.manual_seed(217)
    model = GatedUpdates(update_mode=update_mode).eval()
    arguments = make_gated_update_arguments(model, 257)
    capture = TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            *arguments
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(convrot_int8_swiglu_ffn_compile_options()),
        )(*arguments)

    assert relative_l2(actual, expected) < 0.01
    assert (
        torch.ops.piper_kernels.convrot_int8_swiglu_ffn_gated_updates_.default
        not in capture.targets
    )
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1
