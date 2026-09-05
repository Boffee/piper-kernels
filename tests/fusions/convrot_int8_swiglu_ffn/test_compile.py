"""Tests for automatic semantic ConvRot INT8 SwiGLU FFN folding."""

import uuid
from typing import Literal

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812

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
from piper_kernels.linear.convrot import ConvRotInt8Tensor, convrot_int8_compile_options
from piper_kernels.linear.convrot.int8._compile import compile_pass as convrot_int8_compile_pass

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


class _SwiGluFfn(torch.nn.Module):
    input_features = 256
    intermediate_features = 512
    output_features = 384

    def __init__(
        self,
        *,
        promote_gate: bool = False,
        reverse_multiply: bool = False,
        bias_dtype: torch.dtype | None = torch.bfloat16,
        expose_gate: bool = False,
    ) -> None:
        super().__init__()
        self.promote_gate = promote_gate
        self.reverse_multiply = reverse_multiply
        self.expose_gate = expose_gate
        self.gate = self._linear(self.intermediate_features, self.input_features, bias_dtype)
        self.value = self._linear(self.intermediate_features, self.input_features, bias_dtype)
        self.down = self._linear(self.output_features, self.intermediate_features, bias_dtype)

    @staticmethod
    def _linear(
        out_features: int,
        in_features: int,
        bias_dtype: torch.dtype | None,
    ) -> torch.nn.Linear:
        qdata = torch.randint(
            -127,
            128,
            (out_features, in_features),
            dtype=torch.int8,
            device="cuda",
        )
        scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
        weight = ConvRotInt8Tensor.from_quantized(qdata, scale, group_size=256)
        linear = torch.nn.Linear(
            in_features,
            out_features,
            bias=bias_dtype is not None,
            dtype=torch.bfloat16,
            device="cuda",
        )
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias_dtype is not None:
            assert linear.bias is not None
            linear.bias = torch.nn.Parameter(linear.bias.to(bias_dtype), requires_grad=False)
        return linear

    def forward(
        self,
        activation: torch.Tensor,
        value_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(activation)
        value = self.value(activation if value_input is None else value_input)
        activated_gate = F.silu(gate.float()).to(gate.dtype) if self.promote_gate else F.silu(gate)
        activated = activated_gate * value if self.reverse_multiply else value * activated_gate
        output = self.down(activated)
        return (output, gate) if self.expose_gate else output


class _GatedUpdates(torch.nn.Module):
    def __init__(
        self,
        *,
        expose: Literal["none", "ffn", "hidden"] = "none",
        update_mode: Literal["materialized", "direct", "alias"] = "materialized",
        python_indexing: bool = False,
    ) -> None:
        super().__init__()
        self.ffn = _SwiGluFfn(promote_gate=True, reverse_multiply=True)
        self.expose = expose
        self.update_mode = update_mode
        self.python_indexing = python_indexing
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
        if self.update_mode == "materialized":
            reusable_update = self.update(update_source)
        elif self.update_mode == "alias":
            reusable_update = update_source[1:]
        else:
            reusable_update = update_source
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
        if self.expose == "ffn":
            return output, ffn
        if self.expose == "hidden":
            return output, hidden
        return output


def _gated_update_arguments(
    model: _GatedUpdates,
    rows: int,
) -> tuple[torch.Tensor, ...]:
    features = model.ffn.output_features
    base = torch.randn(rows, features, dtype=torch.bfloat16, device="cuda")
    update_source = torch.randn(
        rows + int(model.update_mode == "alias"),
        features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate_storage = torch.randn(7, 6 * features, dtype=torch.bfloat16, device="cuda")
    update_gate = gate_storage[:, 2 * features : 3 * features]
    ffn_gate = gate_storage[:, 5 * features :]
    gate_indices = torch.randint(0, 7, (rows,), dtype=torch.int64, device="cuda")
    return base, update_source, update_gate, ffn_gate, gate_indices


class _TargetCapturePass(CustomInferenceAwareGraphPass):
    def __init__(self) -> None:
        self.targets: list[object] = []
        self.calls = 0
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.calls += 1
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid


def _capturing_options(capture: _TargetCapturePass) -> dict[str, object]:
    options = convrot_int8_swiglu_ffn_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    return options


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual.float() - expected.float()).norm() / expected.float().norm()


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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("promote_gate", "reverse_multiply", "bias_dtype"),
    [
        (False, False, None),
        (False, True, torch.bfloat16),
        (True, False, torch.float32),
        (True, True, None),
    ],
)
def test_cuda_compile_options_fold_semantic_swiglu_ffn(
    promote_gate: bool,
    reverse_multiply: bool,
    bias_dtype: torch.dtype | None,
) -> None:
    torch.manual_seed(220 + promote_gate + 10 * reverse_multiply)
    model = _SwiGluFfn(
        promote_gate=promote_gate,
        reverse_multiply=reverse_multiply,
        bias_dtype=bias_dtype,
    ).eval()
    activation = torch.randn(
        2,
        257,
        model.input_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    capture = _TargetCapturePass()
    with torch.no_grad():
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
    assert _relative_l2(actual, expected) < 0.01
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("failure", ["projection-escapes", "different-input", "noncontiguous"])
def test_cuda_compile_options_fail_closed(failure: str) -> None:
    torch.manual_seed(225)
    model = _SwiGluFfn(expose_gate=failure == "projection-escapes").eval()
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
    capture = _TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            *arguments
        )
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=_capturing_options(capture))(
            *arguments
        )

    expected_values = expected if isinstance(expected, tuple) else (expected,)
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    assert all(
        torch.equal(left, right) for left, right in zip(actual_values, expected_values, strict=True)
    )
    assert torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_compiled_ffn_reuses_one_dynamic_row_graph() -> None:
    torch.manual_seed(227)
    model = _SwiGluFfn().eval()
    first = torch.randn(257, model.input_features, dtype=torch.bfloat16, device="cuda")
    second = torch.randn(385, model.input_features, dtype=torch.bfloat16, device="cuda")
    torch._dynamo.mark_dynamic(first, 0)
    torch._dynamo.mark_dynamic(second, 0)
    capture = _TargetCapturePass()
    torch._dynamo.reset()
    compiled = torch.compile(model, fullgraph=True, options=_capturing_options(capture))

    with torch.no_grad():
        first_output = compiled(first)
        second_output = compiled(second)

    assert first_output.shape == (257, model.output_features)
    assert second_output.shape == (385, model.output_features)
    assert capture.calls == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("python_indexing", [False, True])
def test_cuda_compile_options_fold_h3_style_gated_updates(python_indexing: bool) -> None:
    torch.manual_seed(224)
    model = _GatedUpdates(python_indexing=python_indexing).eval()
    arguments = _gated_update_arguments(model, 257)
    capture = _TargetCapturePass()
    with torch.no_grad():
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
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_swiglu_ffn_gated_updates_.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("expose", ["ffn", "hidden"])
def test_cuda_gated_updates_fail_closed_when_intermediate_escapes(
    expose: Literal["ffn", "hidden"],
) -> None:
    torch.manual_seed(216)
    model = _GatedUpdates(expose=expose).eval()
    arguments = _gated_update_arguments(model, 257)
    capture = _TargetCapturePass()
    with torch.no_grad():
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
        torch.ops.piper_kernels.convrot_int8_swiglu_ffn_gated_updates_.default
        not in capture.targets
    )
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("update_mode", ["direct", "alias"])
def test_cuda_gated_updates_do_not_mutate_caller_input(
    update_mode: Literal["direct", "alias"],
) -> None:
    torch.manual_seed(217)
    model = _GatedUpdates(update_mode=update_mode).eval()
    arguments = _gated_update_arguments(model, 257)
    capture = _TargetCapturePass()
    with torch.no_grad():
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
        torch.ops.piper_kernels.convrot_int8_swiglu_ffn_gated_updates_.default
        not in capture.targets
    )
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1
