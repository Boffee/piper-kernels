"""Tests for automatic bounded-workspace ConvRot SwiGLU FFN folding."""

import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812

from piper_kernels.fusions.convrot_sparse_piper import (
    convrot_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_sparse_piper._compile import (
    compile_pass as sparse_piper_compile_pass,
)
from piper_kernels.fusions.convrot_swiglu_ffn import (
    convrot_swiglu_ffn_compile_options,
)
from piper_kernels.fusions.convrot_swiglu_ffn._compile import (
    compile_pass as fusion_compile_pass,
)
from piper_kernels.linear.convrot import ConvRotInt8Tensor, convrot_int8_compile_options
from piper_kernels.linear.convrot.int8._compile import compile_pass as convrot_compile_pass

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


class _SwiGluFfn(torch.nn.Module):
    input_features = 256
    intermediate_features = 512
    output_features = 384

    def __init__(self, *, expose_packed: bool = False) -> None:
        super().__init__()
        self.expose_packed = expose_packed
        self.up = self._linear(2 * self.intermediate_features, self.input_features)
        self.down = self._linear(self.output_features, self.intermediate_features)

    @staticmethod
    def _linear(out_features: int, in_features: int) -> torch.nn.Linear:
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
            bias=True,
            dtype=torch.bfloat16,
            device="cuda",
        )
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        linear.bias.requires_grad_(False)
        return linear

    def forward(self, activation: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        packed = self.up(activation)
        up, gate = packed.chunk(2, dim=-1)
        output = self.down(up * F.silu(gate))
        return (output, packed) if self.expose_packed else output


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
    options = convrot_swiglu_ffn_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    return options


def test_compile_options_install_fusion_after_convrot() -> None:
    options = convrot_swiglu_ffn_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (convrot_compile_pass, fusion_compile_pass)


def test_compile_options_reapply_without_duplication() -> None:
    base_options = convrot_int8_compile_options()
    options = convrot_swiglu_ffn_compile_options(base_options)

    assert options[_POST_GRAD_PRE_PASS] == (convrot_compile_pass, fusion_compile_pass)
    assert convrot_swiglu_ffn_compile_options(options) == options


def test_compile_options_preserve_unrelated_pass_order() -> None:
    before_convrot = object()
    after_convrot = object()
    options = convrot_swiglu_ffn_compile_options(
        {
            _POST_GRAD_PRE_PASS: (
                before_convrot,
                convrot_compile_pass,
                after_convrot,
            )
        }
    )

    assert options[_POST_GRAD_PRE_PASS] == (
        before_convrot,
        convrot_compile_pass,
        fusion_compile_pass,
        after_convrot,
    )


@pytest.mark.parametrize("ffn_first", [False, True])
def test_compile_options_compose_with_sparse_piper(ffn_first: bool) -> None:
    if ffn_first:
        options = convrot_sparse_piper_compile_options(convrot_swiglu_ffn_compile_options())
    else:
        options = convrot_swiglu_ffn_compile_options(convrot_sparse_piper_compile_options())

    assert options[_POST_GRAD_PRE_PASS] == (
        sparse_piper_compile_pass,
        convrot_compile_pass,
        fusion_compile_pass,
    )


def test_fusion_compiler_pass_uuid_is_versioned_and_stable() -> None:
    assert fusion_compile_pass.uuid() == fusion_compile_pass.uuid()
    assert fusion_compile_pass.uuid()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_compile_options_fold_complete_swiglu_ffn() -> None:
    torch.manual_seed(211)
    model = _SwiGluFfn().eval()
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
        expected = torch.compile(
            model,
            fullgraph=True,
            options=convrot_int8_compile_options(),
        )(activation)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(activation)

    assert torch.equal(actual, expected)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_swiglu_ffn.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets
    assert torch.ops.piper_kernels.convrot_int8_prepare_input.default not in capture.targets
    assert torch.ops.piper_kernels.convrot_int8_linear_prepared.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_compiled_ffn_reuses_one_dynamic_row_graph() -> None:
    torch.manual_seed(212)
    model = _SwiGluFfn().eval()
    capture = _TargetCapturePass()
    baseline = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
        options=convrot_int8_compile_options(),
    )
    compiled = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
        options=_capturing_options(capture),
    )

    with torch.no_grad():
        for rows in (257, 385):
            activation = torch.randn(
                rows,
                model.input_features,
                dtype=torch.bfloat16,
                device="cuda",
            )
            assert torch.equal(compiled(activation), baseline(activation))

    assert capture.calls == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_swiglu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_compile_options_fail_closed_when_packed_output_escapes() -> None:
    torch.manual_seed(213)
    model = _SwiGluFfn(expose_packed=True).eval()
    activation = torch.randn(
        257,
        model.input_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    capture = _TargetCapturePass()
    with torch.no_grad():
        expected = model(activation)
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(activation)

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)
    assert all(
        torch.equal(actual_value, expected_value)
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )
    assert torch.ops.piper_kernels.convrot_swiglu_ffn.default not in capture.targets
