"""Tests for automatic bounded-workspace NVFP4 SwiGLU FFN folding."""

import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812

from piper_kernels.fusions.nvfp4_sparse_piper import nvfp4_sparse_piper_compile_options
from piper_kernels.fusions.nvfp4_sparse_piper._compile import (
    compile_pass as sparse_piper_compile_pass,
)
from piper_kernels.fusions.nvfp4_swiglu_ffn import (
    nvfp4_swiglu_ffn_compile_options,
)
from piper_kernels.fusions.nvfp4_swiglu_ffn._compile import (
    compile_pass as fusion_compile_pass,
)
from piper_kernels.linear.nvfp4 import nvfp4_compile_options
from piper_kernels.linear.nvfp4._compile import compile_pass as nvfp4_compile_pass

from ._helpers import Operands, down_affine_reference, make_operands

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
    def _linear(operands) -> torch.nn.Linear:
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
        self.calls = 0
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.calls += 1
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid


def _capturing_options(capture: _TargetCapturePass) -> dict[str, object]:
    options = nvfp4_swiglu_ffn_compile_options()
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*passes, capture)
    return options


def test_compile_options_install_fusion_after_nvfp4() -> None:
    options = nvfp4_swiglu_ffn_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (nvfp4_compile_pass, fusion_compile_pass)


def test_compile_options_reapply_without_duplication() -> None:
    options = nvfp4_swiglu_ffn_compile_options(nvfp4_compile_options())

    assert options[_POST_GRAD_PRE_PASS] == (nvfp4_compile_pass, fusion_compile_pass)
    assert nvfp4_swiglu_ffn_compile_options(options) == options


@pytest.mark.parametrize("ffn_first", [False, True])
def test_compile_options_compose_with_sparse_piper(ffn_first: bool) -> None:
    if ffn_first:
        options = nvfp4_sparse_piper_compile_options(nvfp4_swiglu_ffn_compile_options())
    else:
        options = nvfp4_swiglu_ffn_compile_options(nvfp4_sparse_piper_compile_options())

    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    assert passes.count(nvfp4_compile_pass) == 1
    assert passes.count(fusion_compile_pass) == 1
    assert passes.count(sparse_piper_compile_pass) == 1
    assert passes.index(nvfp4_compile_pass) < passes.index(fusion_compile_pass)
    assert passes.index(nvfp4_compile_pass) < passes.index(sparse_piper_compile_pass)


def test_fusion_compiler_pass_uuid_is_versioned_and_stable() -> None:
    assert fusion_compile_pass.uuid() == fusion_compile_pass.uuid()
    assert fusion_compile_pass.uuid()


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("dynamic", [False, True])
def test_cuda_compile_options_fold_complete_swiglu_ffn(dynamic: bool) -> None:
    operands = make_operands(rows=258, dynamic=dynamic, seed=911 + dynamic)
    activation = operands.input.reshape(2, 129, -1)
    model = _SwiGluFfn(operands).eval()
    capture = _TargetCapturePass()
    with torch.no_grad():
        if dynamic:
            torch._dynamo.reset()
            expected = torch.compile(
                model,
                fullgraph=True,
                options=nvfp4_compile_options(),
            )(activation)
        else:
            expected = down_affine_reference(operands).reshape(
                *activation.shape[:-1],
                operands.down.weight.shape[0],
            )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(activation)

    assert torch.equal(actual, expected)
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_swiglu_ffn.default) == 1
    assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.nvfp4_prepare_input.default not in capture.targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compiled_ffn_reuses_one_dynamic_row_graph() -> None:
    operands = make_operands(rows=257, dynamic=False, seed=915)
    model = _SwiGluFfn(operands).eval()
    capture = _TargetCapturePass()
    first = operands.input
    second = torch.randn(385, first.shape[-1], device="cuda", dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(first, 0)
    torch._dynamo.mark_dynamic(second, 0)
    torch._dynamo.reset()
    compiled = torch.compile(
        model,
        fullgraph=True,
        options=_capturing_options(capture),
    )

    with torch.no_grad():
        first_output = compiled(first)
        second_output = compiled(second)

    assert first_output.shape == (257, operands.down.weight.shape[0])
    assert second_output.shape == (385, operands.down.weight.shape[0])
    assert capture.calls == 1
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_swiglu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compile_options_fail_closed_when_packed_projection_escapes() -> None:
    operands = make_operands(rows=129, dynamic=False, seed=913)
    model = _SwiGluFfn(operands, expose_packed=True).eval()
    capture = _TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(
            model,
            fullgraph=True,
            options=nvfp4_compile_options(),
        )(operands.input)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(operands.input)

    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
    assert torch.ops.piper_kernels.nvfp4_swiglu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compile_options_fold_gated_updates() -> None:
    rows = 257
    operands = make_operands(
        rows=rows,
        output_features=384,
        dynamic=False,
        seed=914,
    )
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
            options=nvfp4_compile_options(),
        )(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(*arguments)

    assert torch.equal(actual, expected)
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_swiglu_ffn_gated_updates_.default) == 1
    )
    assert torch.ops.piper_kernels.nvfp4_swiglu_ffn.default not in capture.targets
