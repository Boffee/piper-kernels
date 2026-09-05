"""Tests for automatic bounded-workspace ConvRot SwiGLU FFN folding."""

import uuid
from typing import Literal

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

    def __init__(
        self,
        *,
        expose_packed: bool = False,
        bias_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.expose_packed = expose_packed
        self.up = self._linear(
            2 * self.intermediate_features,
            self.input_features,
            bias_dtype,
        )
        self.down = self._linear(
            self.output_features,
            self.intermediate_features,
            bias_dtype,
        )

    @staticmethod
    def _linear(
        out_features: int,
        in_features: int,
        bias_dtype: torch.dtype,
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
            bias=True,
            dtype=torch.bfloat16,
            device="cuda",
        )
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        assert linear.bias is not None
        linear.bias = torch.nn.Parameter(linear.bias.to(bias_dtype), requires_grad=False)
        return linear

    def forward(self, activation: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        packed = self.up(activation)
        up, gate = packed.chunk(2, dim=-1)
        output = self.down(up * F.silu(gate))
        return (output, packed) if self.expose_packed else output


class _SwiGluFfnGatedUpdates(torch.nn.Module):
    def __init__(
        self,
        *,
        expose: Literal["none", "ffn", "hidden"] = "none",
        update_mode: Literal["materialized", "direct", "alias"] = "materialized",
        python_indexing: bool = False,
    ) -> None:
        super().__init__()
        self.ffn = _SwiGluFfn()
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
    model: _SwiGluFfnGatedUpdates,
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
    gate_storage = torch.randn(
        7,
        6 * features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    update_gate = gate_storage[:, 2 * features : 3 * features]
    ffn_gate = gate_storage[:, 5 * features :]
    assert update_gate.stride() == (6 * features, 1)
    assert ffn_gate.stride() == (6 * features, 1)
    gate_indices = torch.randint(0, 7, (rows,), dtype=torch.int64, device="cuda")
    return (
        base,
        update_source,
        update_gate,
        ffn_gate,
        gate_indices,
    )


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
    model = _SwiGluFfn(bias_dtype=torch.float32).eval()
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
def test_cuda_compile_options_fail_closed_for_noncontiguous_input() -> None:
    torch.manual_seed(218)
    model = _SwiGluFfn().eval()
    storage = torch.randn(
        257,
        2 * model.input_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    activation = storage[:, ::2]
    assert not activation.is_contiguous()
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
    assert torch.ops.piper_kernels.convrot_swiglu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_compile_options_fold_gated_updates() -> None:
    torch.manual_seed(214)
    model = _SwiGluFfnGatedUpdates().eval()
    rows = 257
    capture = _TargetCapturePass()
    arguments = _gated_update_arguments(model, rows)
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(
            model,
            fullgraph=True,
            options=convrot_int8_compile_options(),
        )(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(*arguments)

    assert torch.equal(actual, expected)
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_swiglu_ffn_gated_updates_.default)
        == 1
    )
    assert torch.ops.piper_kernels.convrot_swiglu_ffn.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_compile_options_preserve_negative_python_indices() -> None:
    torch.manual_seed(219)
    model = _SwiGluFfnGatedUpdates(python_indexing=True).eval()
    capture = _TargetCapturePass()
    arguments = list(_gated_update_arguments(model, 257))
    arguments[-1] = torch.arange(257, dtype=torch.int64, device="cuda").remainder(7) - 7
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(
            model,
            fullgraph=True,
            options=convrot_int8_compile_options(),
        )(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(*arguments)

    assert torch.equal(actual, expected)
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_swiglu_ffn_gated_updates_.default)
        == 1
    )


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
def test_cuda_compiled_gated_updates_reuse_one_dynamic_row_graph() -> None:
    torch.manual_seed(215)
    model = _SwiGluFfnGatedUpdates().eval()
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
            arguments = _gated_update_arguments(model, rows)
            assert torch.equal(compiled(*arguments), baseline(*arguments))

    assert capture.calls == 1
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_swiglu_ffn_gated_updates_.default)
        == 1
    )


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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("expose", ["ffn", "hidden"])
def test_cuda_gated_updates_fail_closed_when_intermediate_escapes(
    expose: Literal["ffn", "hidden"],
) -> None:
    torch.manual_seed(216)
    model = _SwiGluFfnGatedUpdates(expose=expose).eval()
    rows = 257
    arguments = _gated_update_arguments(model, rows)
    capture = _TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(
            model,
            fullgraph=True,
            options=convrot_int8_compile_options(),
        )(*arguments)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(*arguments)

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)
    assert all(
        torch.equal(actual_value, expected_value)
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )
    assert torch.ops.piper_kernels.convrot_swiglu_ffn_gated_updates_.default not in capture.targets
    assert capture.targets.count(torch.ops.piper_kernels.convrot_swiglu_ffn.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("update_mode", ["direct", "alias"])
def test_cuda_gated_updates_do_not_mutate_caller_input(
    update_mode: Literal["direct", "alias"],
) -> None:
    torch.manual_seed(217)
    model = _SwiGluFfnGatedUpdates(update_mode=update_mode).eval()
    arguments = _gated_update_arguments(model, 257)
    capture = _TargetCapturePass()
    with torch.no_grad():
        expected = torch.compile(
            model,
            fullgraph=True,
            options=convrot_int8_compile_options(),
        )(*arguments)
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_capturing_options(capture),
        )(*arguments)

    assert torch.equal(actual, expected)
    assert torch.ops.piper_kernels.convrot_swiglu_ffn_gated_updates_.default not in capture.targets
    assert capture.targets.count(torch.ops.piper_kernels.convrot_swiglu_ffn.default) == 1
    if update_mode == "alias":
        assert torch.ops.aten.slice.Tensor in capture.targets
