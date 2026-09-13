"""Static and mixed input scales retain complete bounded FFN fusion."""

from unittest.mock import Mock

import pytest
import torch

from piper_kernels.fusions.convrot_int8_swiglu_ffn.triton import _chunked_swiglu_ffn_op
from piper_kernels.linear.convrot import convrot_int8_compile_options
from piper_kernels.linear.convrot.int8 import _backend, _ops

from .test_compile import (
    _capturing_options,
    _gated_update_arguments,
    _GatedUpdates,
    _relative_l2,
    _SwiGluFfn,
    _TargetCapturePass,
)
from .test_triton import _operands

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm"),
]


def _scales(mode):
    scale = torch.tensor(0.02, device="cuda")
    down = torch.tensor(0.25, device="cuda")
    if mode == "shared":
        return scale, scale, down
    if mode == "alias":
        return scale, scale.view(()), down
    if mode == "distinct":
        return scale, torch.tensor(0.04, device="cuda"), down
    if mode == "mixed":
        return scale, None, down
    if mode == "static-down":
        return None, None, down
    if mode == "dynamic-down":
        return scale, scale, None
    raise AssertionError(mode)


@pytest.mark.parametrize(
    "mode", ["shared", "alias", "distinct", "mixed", "static-down", "dynamic-down"]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_static_ffn_preserves_math_and_shares_only_compatible_inputs(monkeypatch, mode, dtype):
    torch.manual_seed(1071)
    operands = _operands(rows=129, dtype=dtype)
    scales = _scales(mode)
    gate = _ops.linear(operands.input, *operands.gate.arguments(), None, scales[0])
    value = _ops.linear(operands.input, *operands.value.arguments(), None, scales[1])
    expected = _ops.linear(
        torch.cat((value, gate), dim=-1), *operands.down.arguments(), "swiglu", scales[2]
    )
    backend = _backend.require_linear_backend(operands.input)
    prepare = Mock(wraps=backend.prepare_input)
    project = Mock(wraps=backend.linear_prepared)
    monkeypatch.setattr(backend, "prepare_input", prepare)
    monkeypatch.setattr(backend, "linear_prepared", project)
    actual = _chunked_swiglu_ffn_op(*operands.arguments(64), *scales)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    shared = mode not in ("distinct", "mixed")
    assert prepare.call_count == (6 if shared else 9)
    assert project.call_count == prepare.call_count
    assert sum(
        call.kwargs.get("second_projection") is not None for call in project.call_args_list
    ) == (3 if shared else 0)


@pytest.mark.parametrize("mode", ["shared", "mixed"])
def test_static_ffn_custom_op_passes_opcheck(mode):
    operands = _operands(rows=65)
    with torch.inference_mode():
        results = torch.library.opcheck(
            _chunked_swiglu_ffn_op, (*operands.arguments(64), *_scales(mode))
        )
    assert set(results.values()) == {"SUCCESS"}


@pytest.mark.parametrize("mode", ["shared", "distinct", "mixed", "static-down", "dynamic-down"])
def test_compiled_static_ffn_observes_scale_changes_without_recompilation(mode):
    torch._dynamo.reset()
    torch.manual_seed(1072)
    model = _SwiGluFfn().eval()
    scales = _scales(mode)
    for projection, scale in zip((model.gate, model.value, model.down), scales, strict=True):
        projection.weight.act_per_tensor_scale = scale
    activation = torch.randn(129, model.input_features, device="cuda", dtype=torch.bfloat16)
    capture = _TargetCapturePass()
    compiled = torch.compile(model, fullgraph=True, options=_capturing_options(capture))
    ordinary = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())
    with torch.inference_mode():
        first = compiled(activation)
        assert _relative_l2(first, ordinary(activation)) < 0.01
        next(scale for scale in scales if scale is not None).mul_(0.5)
        second = compiled(activation)
        assert _relative_l2(second, ordinary(activation)) < 0.01
    assert not torch.equal(first, second)
    assert capture.calls == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


def test_static_ffn_gated_updates_remain_fused():
    torch._dynamo.reset()
    torch.manual_seed(1073)
    model = _GatedUpdates().eval()
    scales = _scales("distinct")
    for projection, scale in zip(
        (model.ffn.gate, model.ffn.value, model.ffn.down), scales, strict=True
    ):
        projection.weight.act_per_tensor_scale = scale
    arguments = _gated_update_arguments(model, 129)
    capture = _TargetCapturePass()
    with torch.inference_mode():
        expected = torch.compile(model, fullgraph=True, options=convrot_int8_compile_options())(
            *arguments
        )
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


def test_static_ffn_can_feed_an_unfused_projection():
    class FfnWithTail(_SwiGluFfn):
        output_features = 512

        def __init__(self):
            super().__init__()
            self.tail = self._linear(64, self.output_features, None, torch.bfloat16)

        def forward(self, activation):
            return self.tail(super().forward(activation))

    torch._dynamo.reset()
    torch.manual_seed(1075)
    model = FfnWithTail().eval()
    for projection, scale in zip(
        (model.gate, model.value, model.down), _scales("mixed"), strict=True
    ):
        projection.weight.act_per_tensor_scale = scale
    activation = torch.randn(129, model.input_features, device="cuda", dtype=torch.bfloat16)
    capture = _TargetCapturePass()
    with torch.inference_mode():
        # Keep the fused FP32 SwiGLU boundary while preparing each projection
        # independently; an eager BF16 activation chain adds a lossy round trip.
        gate, value = model.gate(activation), model.value(activation)
        weight = model.down.weight
        down = _ops.linear(
            torch.cat((value, gate), dim=-1),
            weight.qdata,
            weight.scale,
            model.down.bias,
            weight.group_size,
            "swiglu",
            weight.act_per_tensor_scale,
        )
        expected = model.tail(down)
        actual = torch.compile(model, fullgraph=True, options=_capturing_options(capture))(
            activation
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_swiglu_ffn.default) == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_linear.default) == 1


@pytest.mark.parametrize("fused", [False, True])
def test_compiler_cache_does_not_confuse_shared_and_independent_scales(fused):
    # Reuse identical compiler options across models: a fresh capture UUID for each
    # model would hide AOT cache collisions involving aliases inside weight wrappers.
    capture = _TargetCapturePass()
    options = _capturing_options(capture) if fused else convrot_int8_compile_options()
    for mode in ("shared", "distinct", "mixed", "shared"):
        torch._dynamo.reset()
        torch.manual_seed(1074)
        model = _SwiGluFfn().eval()
        scales = _scales(mode)
        for projection, scale in zip((model.gate, model.value, model.down), scales, strict=True):
            projection.weight.act_per_tensor_scale = scale
        activation = torch.randn(131, model.input_features, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = model(activation)
            actual = torch.compile(model, fullgraph=True, options=options)(activation)
        assert _relative_l2(actual, expected) < 0.01
