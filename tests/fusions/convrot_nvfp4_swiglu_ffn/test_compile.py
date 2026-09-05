"""Tests for automatic ConvRot NVFP4 SwiGLU FFN folding."""

import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812

from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn import (
    convrot_nvfp4_swiglu_ffn_compile_options,
)
from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn._compile import (
    compile_pass as fusion_compile_pass,
)
from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn.triton import (
    _DEFAULT_CHUNK_ROWS,
    _chunked_swiglu_ffn_op,
)
from piper_kernels.linear.convrot.nvfp4 import convrot_nvfp4_compile_options
from piper_kernels.linear.convrot.nvfp4._compile import (
    compile_pass as convrot_nvfp4_compile_pass,
)

from ._helpers import Linear, Operands, down_affine_reference, make_operands

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


class _SwiGluFfn(torch.nn.Module):
    def __init__(self, operands: Operands, *, expose_packed: bool = False) -> None:
        super().__init__()
        self.expose_packed = expose_packed
        self.up = self._linear(operands.up)
        self.down = self._linear(operands.down)

    @staticmethod
    def _linear(operands: Linear) -> torch.nn.Linear:
        out_features, in_features = operands.weight.shape
        linear = torch.nn.Linear(
            in_features,
            out_features,
            bias=operands.bias is not None,
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear.weight = torch.nn.Parameter(operands.weight, requires_grad=False)
        if operands.bias is not None:
            linear.bias = torch.nn.Parameter(operands.bias, requires_grad=False)
        return linear

    def forward(self, input: torch.Tensor):  # noqa: A002
        packed = self.up(input)
        up, gate = packed.chunk(2, dim=-1)
        output = self.down(up * F.silu(gate))
        return (output, packed) if self.expose_packed else output


class _SwiGluFfnGatedUpdates(torch.nn.Module):
    def __init__(self, operands: Operands) -> None:
        super().__init__()
        self.ffn = _SwiGluFfn(operands)
        output_features = operands.down.weight.shape[0]
        self.input_features = operands.up.weight.shape[1]
        self.update = torch.nn.Linear(
            output_features,
            output_features,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.update.weight.requires_grad_(False)

    def forward(
        self,
        base: torch.Tensor,
        update_source: torch.Tensor,
        update_gate: torch.Tensor,
        ffn_gate: torch.Tensor,
        gate_indices: torch.Tensor,
    ) -> torch.Tensor:
        reusable_update = self.update(update_source)
        hidden = base + update_gate.index_select(0, gate_indices) * reusable_update
        ffn = self.ffn(hidden[..., : self.input_features].contiguous())
        assert isinstance(ffn, torch.Tensor)
        return hidden + ffn_gate.index_select(0, gate_indices) * ffn


class _TargetCapturePass(CustomInferenceAwareGraphPass):
    def __init__(self) -> None:
        self.targets: list[object] = []
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid


def _capturing_options(capture: _TargetCapturePass) -> dict[str, object]:
    options = convrot_nvfp4_swiglu_ffn_compile_options()
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*passes, capture)
    return options


def test_compile_options_install_fusion_after_convrot_nvfp4() -> None:
    options = convrot_nvfp4_swiglu_ffn_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (
        convrot_nvfp4_compile_pass,
        fusion_compile_pass,
    )


def test_compile_options_reapply_without_duplication() -> None:
    options = convrot_nvfp4_swiglu_ffn_compile_options(convrot_nvfp4_compile_options())

    assert options[_POST_GRAD_PRE_PASS] == (
        convrot_nvfp4_compile_pass,
        fusion_compile_pass,
    )
    assert convrot_nvfp4_swiglu_ffn_compile_options(options) == options


def test_fusion_compiler_pass_uuid_is_versioned_and_stable() -> None:
    assert fusion_compile_pass.uuid() == fusion_compile_pass.uuid()
    assert fusion_compile_pass.uuid()


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("high_first", [False, True])
def test_cuda_compile_options_fold_complete_swiglu_ffn(
    dynamic: bool,
    high_first: bool,
) -> None:
    operands = make_operands(
        rows=258,
        dynamic=dynamic,
        high_first=high_first,
        seed=941 + dynamic + 10 * high_first,
    )
    activation = operands.input.reshape(2, 129, -1)
    model = _SwiGluFfn(operands).eval()
    capture = _TargetCapturePass()
    with torch.no_grad():
        expected = (
            _chunked_swiglu_ffn_op(*operands.arguments(_DEFAULT_CHUNK_ROWS))
            if dynamic
            else down_affine_reference(operands)
        ).reshape(*activation.shape[:-1], operands.down.weight.shape[0])
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(activation)

    assert torch.equal(actual, expected)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_nvfp4_swiglu_ffn.default) == 1
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default not in capture.targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compile_options_fail_closed_when_packed_projection_escapes() -> None:
    operands = make_operands(rows=129, dynamic=False, seed=943)
    model = _SwiGluFfn(operands, expose_packed=True).eval()
    capture = _TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(
            model,
            fullgraph=True,
            options=convrot_nvfp4_compile_options(),
        )(operands.input)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(operands.input)

    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
    assert torch.ops.piper_kernels.convrot_nvfp4_swiglu_ffn.default not in capture.targets
    assert torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default in capture.targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compile_options_fold_gated_updates() -> None:
    rows = 257
    operands = make_operands(rows=rows, dynamic=False, seed=944)
    model = _SwiGluFfnGatedUpdates(operands).eval()
    output_features = operands.down.weight.shape[0]
    base = torch.randn(rows, output_features, device="cuda", dtype=torch.bfloat16)
    update_source = torch.randn_like(base)
    gate_storage = torch.randn(7, 6 * output_features, device="cuda", dtype=torch.bfloat16)
    update_gate = gate_storage[:, 2 * output_features : 3 * output_features]
    ffn_gate = gate_storage[:, 5 * output_features :]
    gate_indices = torch.randint(0, 7, (rows,), device="cuda", dtype=torch.int64)
    arguments = base, update_source, update_gate, ffn_gate, gate_indices
    capture = _TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(
            model,
            fullgraph=True,
            options=convrot_nvfp4_compile_options(),
        )(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(*arguments)

    assert torch.equal(actual, expected)
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_nvfp4_swiglu_ffn_gated_updates_.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.convrot_nvfp4_swiglu_ffn.default not in capture.targets
