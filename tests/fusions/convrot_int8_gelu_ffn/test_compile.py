"""Tests for automatic semantic ConvRot INT8 GELU FFN folding."""

import pytest
import torch
from _compile_capture import TargetCapturePass
from torch.nn import functional as F  # noqa: N812

from piper_kernels.fusions.convrot_int8_gelu_ffn import convrot_int8_gelu_ffn_compile_options
from piper_kernels.fusions.convrot_int8_gelu_ffn._compile import (
    compile_pass as gelu_compile_pass,
)
from piper_kernels.fusions.convrot_int8_swiglu_ffn import (
    convrot_int8_swiglu_ffn_compile_options,
)
from piper_kernels.fusions.convrot_int8_swiglu_ffn._compile import (
    compile_pass as swiglu_compile_pass,
)
from piper_kernels.linear.convrot import convrot_int8_compile_options
from piper_kernels.linear.convrot.int8._compile import compile_pass as convrot_int8_compile_pass
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


class _GeluFfn(torch.nn.Module):
    input_features = 256
    intermediate_features = 512
    output_features = 384

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.bfloat16,
        bias_dtype: torch.dtype | None = torch.bfloat16,
        expose_up: bool = False,
    ) -> None:
        super().__init__()
        self.expose_up = expose_up
        self.up = self._linear(self.intermediate_features, self.input_features, bias_dtype, dtype)
        self.down = self._linear(
            self.output_features,
            self.intermediate_features,
            bias_dtype,
            dtype,
        )

    @staticmethod
    def _linear(
        out_features: int,
        in_features: int,
        bias_dtype: torch.dtype | None,
        dtype: torch.dtype,
    ) -> torch.nn.Linear:
        weight = ConvRotInt8Tensor.from_quantized(
            torch.randint(
                -127,
                128,
                (out_features, in_features),
                dtype=torch.int8,
                device="cuda",
            ),
            torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01,
            group_size=256,
            logical_dtype=dtype,
        )
        linear = torch.nn.Linear(
            in_features,
            out_features,
            bias=bias_dtype is not None,
            dtype=dtype,
            device="cuda",
        )
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias_dtype is not None:
            assert linear.bias is not None
            linear.bias = torch.nn.Parameter(linear.bias.to(bias_dtype), requires_grad=False)
        return linear

    def forward(self, activation: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        up = self.up(activation)
        output = self.down(F.gelu(up, approximate="tanh"))
        return (output, up) if self.expose_up else output


class _GatedUpdates(torch.nn.Module):
    def __init__(self, *, python_indexing: bool = False, expose_ffn: bool = False) -> None:
        super().__init__()
        self.ffn = _GeluFfn()
        self.python_indexing = python_indexing
        self.expose_ffn = expose_ffn
        self.update = torch.nn.Linear(
            self.ffn.output_features,
            self.ffn.output_features,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.update.weight.requires_grad_(False)

    def forward(
        self,
        base: torch.Tensor,
        update_source: torch.Tensor,
        update_gate: torch.Tensor,
        ffn_gate: torch.Tensor,
        gate_indices: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        reusable_update = self.update(update_source)
        if self.python_indexing:
            selected_update_gate = update_gate[gate_indices]
            selected_ffn_gate = ffn_gate[gate_indices]
        else:
            selected_update_gate = update_gate.index_select(0, gate_indices)
            selected_ffn_gate = ffn_gate.index_select(0, gate_indices)
        hidden = base + selected_update_gate * reusable_update
        ffn = self.ffn(hidden[..., : self.ffn.input_features].contiguous())
        assert isinstance(ffn, torch.Tensor)
        output = hidden + selected_ffn_gate * ffn
        return (output, ffn) if self.expose_ffn else output


def _capturing_options(capture: TargetCapturePass) -> dict[str, object]:
    options = convrot_int8_gelu_ffn_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    return options


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual.float() - expected.float()).norm() / expected.float().norm()


def _gated_update_arguments(rows: int) -> tuple[torch.Tensor, ...]:
    features = _GeluFfn.output_features
    base = torch.randn(rows, features, dtype=torch.bfloat16, device="cuda")
    update_source = torch.randn_like(base)
    update_gate = torch.randn(7, features, dtype=torch.bfloat16, device="cuda")
    ffn_gate = torch.randn(7, features, dtype=torch.bfloat16, device="cuda")
    gate_indices = torch.randint(0, 7, (rows,), dtype=torch.int64, device="cuda")
    return base, update_source, update_gate, ffn_gate, gate_indices


def test_compile_options_install_fusion_before_convrot() -> None:
    options = convrot_int8_gelu_ffn_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (gelu_compile_pass, convrot_int8_compile_pass)


@pytest.mark.parametrize("gelu_first", [False, True])
def test_compile_options_compose_with_swiglu(gelu_first: bool) -> None:
    options = (
        convrot_int8_gelu_ffn_compile_options(convrot_int8_swiglu_ffn_compile_options())
        if gelu_first
        else convrot_int8_swiglu_ffn_compile_options(convrot_int8_gelu_ffn_compile_options())
    )

    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    assert passes.count(gelu_compile_pass) == 1
    assert passes.count(swiglu_compile_pass) == 1
    assert passes.count(convrot_int8_compile_pass) == 1
    assert passes.index(gelu_compile_pass) < passes.index(convrot_int8_compile_pass)
    assert passes.index(swiglu_compile_pass) < passes.index(convrot_int8_compile_pass)


def test_compile_pass_uuid_is_versioned_and_stable() -> None:
    assert gelu_compile_pass.uuid() == gelu_compile_pass.uuid()
    assert gelu_compile_pass.uuid()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale_mode", ["dynamic", "up-static", "down-static", "static"])
def test_compile_options_fold_semantic_gelu_ffn(bias_dtype, dtype, scale_mode) -> None:
    torch.manual_seed(511)
    model = _GeluFfn(dtype=dtype, bias_dtype=bias_dtype).eval()
    if scale_mode in ("up-static", "static"):
        model.up.weight.act_per_tensor_scale = torch.tensor(0.02, device="cuda")
    if scale_mode in ("down-static", "static"):
        model.down.weight.act_per_tensor_scale = torch.tensor(0.04, device="cuda")
    activation = torch.randn(2, 257, model.input_features, dtype=dtype, device="cuda")
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            activation
        )
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=_capturing_options(capture))(
            activation
        )

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype is dtype
    assert _relative_l2(actual, expected) < 0.01
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_gelu_ffn.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("failure", ["projection-escapes", "noncontiguous", "empty"])
def test_compile_options_fail_closed(failure: str) -> None:
    torch.manual_seed(513)
    model = _GeluFfn(expose_up=failure == "projection-escapes").eval()
    if failure == "noncontiguous":
        storage = torch.randn(
            257,
            2 * model.input_features,
            dtype=torch.bfloat16,
            device="cuda",
        )
        activation = storage[:, ::2]
    elif failure == "empty":
        activation = torch.empty(0, model.input_features, dtype=torch.bfloat16, device="cuda")
    else:
        activation = torch.randn(257, model.input_features, dtype=torch.bfloat16, device="cuda")
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            activation
        )
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=_capturing_options(capture))(
            activation
        )

    expected_values = expected if isinstance(expected, tuple) else (expected,)
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    assert all(
        torch.equal(left, right) for left, right in zip(actual_values, expected_values, strict=True)
    )
    assert torch.ops.piper_kernels.convrot_int8_gelu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_compiled_ffn_reuses_one_dynamic_row_graph() -> None:
    torch.manual_seed(515)
    model = _GeluFfn().eval()
    first = torch.randn(257, model.input_features, dtype=torch.bfloat16, device="cuda")
    second = torch.randn(385, model.input_features, dtype=torch.bfloat16, device="cuda")
    torch._dynamo.mark_dynamic(first, 0)
    torch._dynamo.mark_dynamic(second, 0)
    capture = TargetCapturePass()
    torch._dynamo.reset()
    compiled = torch.compile(model, fullgraph=True, options=_capturing_options(capture))

    with torch.inference_mode():
        assert compiled(first).shape == (257, model.output_features)
        assert compiled(second).shape == (385, model.output_features)

    assert capture.calls == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_gelu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize("python_indexing", [False, True])
def test_compile_options_fold_indexed_gated_updates(python_indexing: bool) -> None:
    torch.manual_seed(517)
    model = _GatedUpdates(python_indexing=python_indexing).eval()
    arguments = _gated_update_arguments(257)
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            *arguments
        )
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=_capturing_options(capture))(
            *arguments
        )

    assert _relative_l2(actual, expected) < 0.01
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_int8_gelu_ffn_gated_updates_.default)
        == 1
    )
    assert torch.ops.piper_kernels.convrot_int8_gelu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_gated_updates_fail_closed_when_ffn_escapes() -> None:
    torch.manual_seed(519)
    model = _GatedUpdates(expose_ffn=True).eval()
    arguments = _gated_update_arguments(257)
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            *arguments
        )
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=_capturing_options(capture))(
            *arguments
        )

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)
    assert all(
        _relative_l2(left, right) < 0.01 for left, right in zip(actual, expected, strict=True)
    )
    assert (
        torch.ops.piper_kernels.convrot_int8_gelu_ffn_gated_updates_.default not in capture.targets
    )
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_gelu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
def test_compiled_static_scales_are_runtime_values() -> None:
    torch.manual_seed(521)
    model = _GeluFfn().eval()
    model.up.weight.act_per_tensor_scale = torch.tensor(0.02, device="cuda")
    model.down.weight.act_per_tensor_scale = torch.tensor(0.04, device="cuda")
    activation = torch.randn(129, model.input_features, dtype=torch.bfloat16, device="cuda")
    capture = TargetCapturePass()
    compiled = torch.compile(model, fullgraph=True, options=_capturing_options(capture))
    ordinary = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())

    with torch.inference_mode():
        first = compiled(activation)
        assert _relative_l2(first, ordinary(activation)) < 0.01
        model.down.weight.act_per_tensor_scale.mul_(0.5)
        second = compiled(activation)
        assert _relative_l2(second, ordinary(activation)) < 0.01

    assert not torch.equal(first, second)
    assert capture.calls == 1
