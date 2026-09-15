"""Tests for automatic mixed standard/ConvRot NVFP4 GELU FFN folding."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from piper_kernels.fusions.convrot_nvfp4_gelu_ffn import (
    convrot_nvfp4_gelu_ffn_compile_options,
)
from piper_kernels.fusions.convrot_nvfp4_gelu_ffn._compile import (
    compile_pass as convrot_gelu_compile_pass,
)
from piper_kernels.fusions.nvfp4_gelu_ffn import nvfp4_gelu_ffn_compile_options
from piper_kernels.fusions.nvfp4_gelu_ffn._compile import (
    compile_pass as standard_gelu_compile_pass,
)
from piper_kernels.linear.convrot.nvfp4 import convrot_nvfp4_compile_options
from piper_kernels.linear.convrot.nvfp4._compile import (
    compile_pass as convrot_nvfp4_compile_pass,
)
from piper_kernels.linear.nvfp4 import nvfp4_compile_options
from piper_kernels.linear.nvfp4._compile import compile_pass as nvfp4_compile_pass

from ..nvfp4_gelu_ffn.test_compile import (
    GatedUpdates,
    TargetCapturePass,
    gated_update_arguments,
    relative_l2,
)
from ._helpers import Linear, Operands, make_operands

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


class GeluFfn(torch.nn.Module):
    def __init__(self, operands: Operands, *, expose_up: bool = False) -> None:
        super().__init__()
        self.expose_up = expose_up
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
        activated = F.gelu(up, approximate="tanh")
        output = self.down(activated)
        return (output, up) if self.expose_up else output


def _capturing_options(capture: TargetCapturePass) -> dict[str, object]:
    options = convrot_nvfp4_gelu_ffn_compile_options(nvfp4_gelu_ffn_compile_options())
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*passes, capture)
    return options


def _ordinary_options() -> dict[str, object]:
    return convrot_nvfp4_compile_options(nvfp4_compile_options())


def test_compile_options_install_fusion_before_linear_passes() -> None:
    options = convrot_nvfp4_gelu_ffn_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (
        convrot_gelu_compile_pass,
        nvfp4_compile_pass,
        convrot_nvfp4_compile_pass,
    )


def test_compile_options_compose_with_standard_gelu() -> None:
    options = convrot_nvfp4_gelu_ffn_compile_options(nvfp4_gelu_ffn_compile_options())

    assert options[_POST_GRAD_PRE_PASS] == (
        standard_gelu_compile_pass,
        convrot_gelu_compile_pass,
        nvfp4_compile_pass,
        convrot_nvfp4_compile_pass,
    )


def test_compile_pass_uuid_is_versioned_and_stable() -> None:
    assert convrot_gelu_compile_pass.uuid() == convrot_gelu_compile_pass.uuid()
    assert convrot_gelu_compile_pass.uuid()


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("up_group_size", "down_group_size"),
    [(16, 64), (16, None), (None, 64)],
    ids=["convrot", "convrot-standard", "standard-convrot"],
)
@pytest.mark.parametrize(
    ("up_dynamic", "down_dynamic"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["static", "down-dynamic", "up-dynamic", "dynamic"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compile_options_fold_semantic_gelu_ffn(
    up_group_size: int | None,
    down_group_size: int | None,
    up_dynamic: bool,
    down_dynamic: bool,
    dtype: torch.dtype,
) -> None:
    operands = make_operands(
        rows=258,
        up_dynamic=up_dynamic,
        down_dynamic=down_dynamic,
        up_group_size=up_group_size,
        down_group_size=down_group_size,
        dtype=dtype,
        bias_dtype=torch.float32,
        up_high_first=down_dynamic,
        down_high_first=up_dynamic,
        seed=1031 + up_dynamic + 10 * down_dynamic,
    )
    activation = operands.input.reshape(2, 129, -1)
    model = GeluFfn(operands).eval()
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=_ordinary_options())(activation)
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=_capturing_options(capture))(
            activation
        )

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    assert actual.dtype is dtype
    assert relative_l2(actual, expected) < (0.1 if down_dynamic else 0.07)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_nvfp4_gelu_ffn.default) == 1
    assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_compile_options_fail_closed_when_projection_escapes() -> None:
    operands = make_operands(rows=129, up_dynamic=False, down_dynamic=False, seed=1033)
    model = GeluFfn(operands, expose_up=True).eval()
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=_ordinary_options())(operands.input)
        torch._dynamo.reset()
        actual = torch.compile(model, fullgraph=True, options=_capturing_options(capture))(
            operands.input
        )

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
    assert torch.ops.piper_kernels.convrot_nvfp4_gelu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_compile_options_fold_indexed_gated_updates() -> None:
    operands = make_operands(
        rows=257,
        up_dynamic=False,
        down_dynamic=False,
        output_features=384,
        seed=1039,
    )
    model = GatedUpdates(operands).eval()  # type: ignore[arg-type]
    arguments = gated_update_arguments(257, operands.down.weight.shape[0])
    capture = TargetCapturePass()
    with torch.inference_mode():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=_ordinary_options())(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(*arguments)

    assert relative_l2(actual, expected) < 0.07
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_nvfp4_gelu_ffn_gated_updates_.default)
        == 1
    )
    assert torch.ops.piper_kernels.convrot_nvfp4_gelu_ffn.default not in capture.targets
