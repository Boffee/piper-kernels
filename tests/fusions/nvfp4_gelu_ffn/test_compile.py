"""Tests for automatic standard NVFP4 GELU FFN folding."""

from __future__ import annotations

import pytest
import torch
from _compile_capture import TargetCapturePass
from torch.nn import functional as F  # noqa: N812

from piper_kernels.fusions.nvfp4_gelu_ffn import nvfp4_gelu_ffn_compile_options
from piper_kernels.fusions.nvfp4_gelu_ffn._compile import compile_pass as gelu_compile_pass
from piper_kernels.fusions.nvfp4_swiglu_ffn import nvfp4_swiglu_ffn_compile_options
from piper_kernels.fusions.nvfp4_swiglu_ffn._compile import compile_pass as swiglu_compile_pass
from piper_kernels.linear.nvfp4 import nvfp4_compile_options
from piper_kernels.linear.nvfp4._compile import compile_pass as nvfp4_compile_pass

from ._helpers import Linear, Operands, make_operands

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


class GeluFfn(torch.nn.Module):
    """Two semantic linears around promoted tanh-GELU."""

    def __init__(
        self,
        operands: Operands,
        *,
        expose_up: bool = False,
        explicit_promotion: bool = False,
    ) -> None:
        super().__init__()
        self.expose_up = expose_up
        self.explicit_promotion = explicit_promotion
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
            dtype=operands.weight.dtype,
        )
        linear.weight = torch.nn.Parameter(operands.weight, requires_grad=False)
        if operands.bias is not None:
            linear.bias = torch.nn.Parameter(operands.bias, requires_grad=False)
        return linear

    def forward(
        self,
        activation: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        up = self.up(activation)
        activated = F.gelu(
            up.float() if self.explicit_promotion else up,
            approximate="tanh",
        )
        if self.explicit_promotion:
            activated = activated.to(up.dtype)
        output = self.down(activated)
        return (output, up) if self.expose_up else output


class GatedUpdates(torch.nn.Module):
    """H3-style indexed updates containing the GELU FFN."""

    def __init__(self, operands: Operands, *, python_indexing: bool = False) -> None:
        super().__init__()
        self.ffn = GeluFfn(operands)
        self.python_indexing = python_indexing
        self.input_features = operands.up.weight.shape[1]
        output_features = operands.down.weight.shape[0]
        self.update = torch.nn.Linear(
            output_features,
            output_features,
            bias=False,
            device="cuda",
            dtype=operands.input.dtype,
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
        if self.python_indexing:
            selected_update_gate = update_gate[gate_indices]
            selected_ffn_gate = ffn_gate[gate_indices]
        else:
            selected_update_gate = update_gate.index_select(0, gate_indices)
            selected_ffn_gate = ffn_gate.index_select(0, gate_indices)
        hidden = base + selected_update_gate * reusable_update
        ffn = self.ffn(hidden[..., : self.input_features].contiguous())
        assert isinstance(ffn, torch.Tensor)
        return hidden + selected_ffn_gate * ffn


def capturing_options(capture: TargetCapturePass) -> dict[str, object]:
    options = nvfp4_gelu_ffn_compile_options()
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*passes, capture)
    return options


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual.float() - expected.float()).norm() / expected.float().norm()


def gated_update_arguments(
    rows: int,
    features: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    base = torch.randn(rows, features, dtype=dtype, device="cuda")
    update_source = torch.randn_like(base)
    update_gate = torch.randn(7, features, dtype=dtype, device="cuda")
    ffn_gate = torch.randn(7, features, dtype=dtype, device="cuda")
    gate_indices = torch.randint(0, 7, (rows,), dtype=torch.int64, device="cuda")
    return base, update_source, update_gate, ffn_gate, gate_indices


def test_compile_options_install_fusion_before_nvfp4() -> None:
    options = nvfp4_gelu_ffn_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (gelu_compile_pass, nvfp4_compile_pass)


@pytest.mark.parametrize("gelu_first", [False, True])
def test_compile_options_compose_with_swiglu(gelu_first: bool) -> None:
    options = (
        nvfp4_gelu_ffn_compile_options(nvfp4_swiglu_ffn_compile_options())
        if gelu_first
        else nvfp4_swiglu_ffn_compile_options(nvfp4_gelu_ffn_compile_options())
    )

    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    assert passes.count(gelu_compile_pass) == 1
    assert passes.count(swiglu_compile_pass) == 1
    assert passes.count(nvfp4_compile_pass) == 1
    assert passes.index(gelu_compile_pass) < passes.index(nvfp4_compile_pass)
    assert passes.index(swiglu_compile_pass) < passes.index(nvfp4_compile_pass)


def test_compile_pass_uuid_is_versioned_and_stable() -> None:
    assert gelu_compile_pass.uuid() == gelu_compile_pass.uuid()
    assert gelu_compile_pass.uuid()


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("up_dynamic", "down_dynamic", "bias_dtype", "high_first", "explicit_promotion"),
    [
        (False, False, None, False, False),
        (False, False, torch.float32, True, True),
        (False, True, torch.float16, True, False),
        (True, False, torch.bfloat16, False, False),
        (True, True, torch.float32, True, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compile_options_fold_semantic_gelu_ffn(
    up_dynamic: bool,
    down_dynamic: bool,
    bias_dtype: torch.dtype | None,
    high_first: bool,
    explicit_promotion: bool,
    dtype: torch.dtype,
) -> None:
    operands = make_operands(
        rows=258,
        up_dynamic=up_dynamic,
        down_dynamic=down_dynamic,
        bias_dtype=bias_dtype,
        up_high_first=high_first,
        down_high_first=not high_first,
        dtype=dtype,
        seed=1009 + up_dynamic + 10 * down_dynamic,
    )
    activation = operands.input.reshape(2, 129, -1)
    model = GeluFfn(operands, explicit_promotion=explicit_promotion).eval()
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=nvfp4_compile_options())(activation)
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=capturing_options(capture))(
            activation
        )

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype is dtype
    assert relative_l2(actual, expected) < (0.08 if down_dynamic else 0.05)
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_gelu_ffn.default) == 1
    assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_compile_options_fail_closed_when_projection_escapes() -> None:
    operands = make_operands(rows=129, up_dynamic=False, down_dynamic=False, seed=1013)
    model = GeluFfn(operands, expose_up=True).eval()
    activation = operands.input
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=nvfp4_compile_options())(activation)
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=capturing_options(capture))(
            activation
        )

    expected_values = expected if isinstance(expected, tuple) else (expected,)
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    assert all(
        torch.equal(left, right) for left, right in zip(actual_values, expected_values, strict=True)
    )
    assert torch.ops.piper_kernels.nvfp4_gelu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_compile_options_fail_closed_on_noncontiguous_input() -> None:
    operands = make_operands(rows=258, up_dynamic=False, down_dynamic=False, seed=1015)
    model = GeluFfn(operands).eval()
    activation = operands.input.view(2, 129, -1).transpose(0, 1)
    assert not activation.is_contiguous()
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=nvfp4_compile_options())(activation)
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=capturing_options(capture))(
            activation
        )

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.ops.piper_kernels.nvfp4_gelu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_compiled_ffn_reuses_one_dynamic_row_graph() -> None:
    operands = make_operands(rows=257, up_dynamic=False, down_dynamic=False, seed=1019)
    model = GeluFfn(operands).eval()
    first = operands.input
    second = torch.randn(385, first.shape[-1], device="cuda", dtype=first.dtype)
    torch._dynamo.mark_dynamic(first, 0)
    torch._dynamo.mark_dynamic(second, 0)
    capture = TargetCapturePass()
    torch._dynamo.reset()
    compiled = torch.compile(model, fullgraph=True, options=capturing_options(capture))

    with torch.inference_mode():
        assert compiled(first).shape == (257, operands.down.weight.shape[0])
        assert compiled(second).shape == (385, operands.down.weight.shape[0])

    assert capture.calls == 1
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_gelu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("python_indexing", [False, True])
def test_compile_options_fold_indexed_gated_updates(python_indexing: bool) -> None:
    operands = make_operands(
        rows=257,
        up_dynamic=False,
        down_dynamic=False,
        output_features=384,
        seed=1021,
    )
    model = GatedUpdates(operands, python_indexing=python_indexing).eval()
    arguments = gated_update_arguments(257, operands.down.weight.shape[0])
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=nvfp4_compile_options())(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capturing_options(capture),
        )(*arguments)

    assert relative_l2(actual, expected) < 0.05
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_gelu_ffn_gated_updates_.default) == 1
    assert torch.ops.piper_kernels.nvfp4_gelu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_compiled_static_scales_are_runtime_values() -> None:
    operands = make_operands(rows=129, up_dynamic=False, down_dynamic=False, seed=1027)
    model = GeluFfn(operands).eval()
    activation = operands.input
    capture = TargetCapturePass()
    compiled = torch.compile(model, fullgraph=True, options=capturing_options(capture))
    ordinary = torch.compile(model, fullgraph=True, options=nvfp4_compile_options())

    with torch.inference_mode():
        first = compiled(activation)
        assert relative_l2(first, ordinary(activation)) < 0.05
        assert model.down.weight.act_per_tensor_scale is not None
        model.down.weight.act_per_tensor_scale.mul_(0.5)
        second = compiled(activation)
        assert relative_l2(second, ordinary(activation)) < 0.05

    assert not torch.equal(first, second)
    assert capture.calls == 1


__all__ = [
    "GatedUpdates",
    "GeluFfn",
    "TargetCapturePass",
    "capturing_options",
    "gated_update_arguments",
    "relative_l2",
]
