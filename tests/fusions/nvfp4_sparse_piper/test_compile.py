"""Tests for automatic NVFP4-to-sparse-Piper graph fusion."""

from __future__ import annotations

import operator
import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from piper_kernels import SparsePiperAttention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.fusions.nvfp4_sparse_piper import (
    nvfp4_sparse_piper_compile_options,
)
from piper_kernels.fusions.nvfp4_sparse_piper._compile import compile_pass
from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor
from piper_kernels.linear.nvfp4._compile import compile_pass as nvfp4_compile_pass

from ._helpers import exact_sm120_available

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


def _placeholder(
    graph: torch.fx.Graph,
    name: str,
    value: torch.Tensor,
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def _semantic_linear(
    graph: torch.fx.Graph,
    input: torch.fx.Node,  # noqa: A002
    prefix: str,
    activation_scale: torch.fx.Node | None,
    *,
    dynamic: bool,
) -> torch.fx.Node:
    weight_qdata = _placeholder(
        graph,
        f"{prefix}_weight_qdata",
        torch.empty((256, 128), device="cuda", dtype=torch.uint8),
    )
    weight_scale = _placeholder(
        graph,
        f"{prefix}_weight_scale",
        torch.empty((64, 64), device="cuda", dtype=torch.float8_e4m3fn),
    )
    weight_global_scale = _placeholder(
        graph,
        f"{prefix}_weight_global_scale",
        torch.empty((), device="cuda", dtype=torch.float32),
    )
    projected = graph.call_function(
        torch.ops.piper_kernels.nvfp4_linear.default,
        args=(
            input,
            weight_qdata,
            weight_scale,
            weight_global_scale,
            activation_scale,
            None,
            dynamic,
        ),
    )
    projected.meta["val"] = torch.empty(
        (1, 192, 256),
        device="cuda",
        dtype=torch.bfloat16,
    )
    return projected


def _normalized_rope(
    graph: torch.fx.Graph,
    projected: torch.fx.Node,
    norm: torch.fx.Node,
    cos: torch.fx.Node,
    sin: torch.fx.Node,
) -> torch.fx.Node:
    reshaped = graph.call_function(
        torch.ops.aten.reshape.default,
        args=(projected, (1, 192, 2, 128)),
    )
    promoted = graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(reshaped, torch.float32),
    )
    squared = graph.call_function(torch.ops.aten.pow.Tensor_Scalar, args=(promoted, 2))
    mean = graph.call_function(torch.ops.aten.mean.dim, args=(squared, [3], True))
    variance = graph.call_function(torch.ops.aten.add.Scalar, args=(mean, 1e-5))
    inverse_rms = graph.call_function(torch.ops.aten.rsqrt.default, args=(variance,))
    normalized = graph.call_function(torch.ops.aten.mul.Tensor, args=(promoted, inverse_rms))
    scaled = graph.call_function(torch.ops.aten.mul.Tensor, args=(normalized, norm))
    rounded = graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(scaled, torch.bfloat16),
    )
    rotary = graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(rounded, 3, 0, 96),
    )
    split = graph.call_function(torch.ops.aten.split.Tensor, args=(rotary, 48, -1))
    first = graph.call_function(operator.getitem, args=(split, 0))
    second = graph.call_function(operator.getitem, args=(split, 1))
    cos_bf16 = graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(cos, torch.bfloat16),
    )
    cos_bf16 = graph.call_function(torch.ops.aten.unsqueeze.default, args=(cos_bf16, 0))
    cos_bf16 = graph.call_function(torch.ops.aten.unsqueeze.default, args=(cos_bf16, 2))
    direct = graph.call_function(torch.ops.aten.mul.Tensor, args=(rotary, cos_bf16))
    negated = graph.call_function(torch.ops.aten.neg.default, args=(second,))
    rotated = graph.call_function(torch.ops.aten.cat.default, args=([negated, first], -1))
    sin_bf16 = graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(sin, torch.bfloat16),
    )
    sin_bf16 = graph.call_function(torch.ops.aten.unsqueeze.default, args=(sin_bf16, 0))
    sin_bf16 = graph.call_function(torch.ops.aten.unsqueeze.default, args=(sin_bf16, 2))
    rotated = graph.call_function(torch.ops.aten.mul.Tensor, args=(rotated, sin_bf16))
    rotary_output = graph.call_function(torch.ops.aten.add.Tensor, args=(direct, rotated))
    passthrough = graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(rounded, 3, 96, torch.iinfo(torch.int64).max),
    )
    return graph.call_function(
        torch.ops.aten.cat.default,
        args=([rotary_output, passthrough], -1),
    )


def _semantic_attention_graph(*, dynamic: bool) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    with FakeTensorMode():
        input = _placeholder(  # noqa: A001
            graph,
            "input",
            torch.empty((1, 192, 256), device="cuda", dtype=torch.bfloat16),
        )
        activation_scales = tuple(
            None
            if dynamic
            else _placeholder(
                graph,
                f"{prefix}_activation_scale",
                torch.empty((), device="cuda", dtype=torch.float32),
            )
            for prefix in ("q", "k", "v")
        )
        projected = tuple(
            _semantic_linear(
                graph,
                input,
                prefix,
                activation_scale,
                dynamic=dynamic,
            )
            for prefix, activation_scale in zip(("q", "k", "v"), activation_scales, strict=True)
        )
        q_norm = _placeholder(
            graph,
            "q_norm",
            torch.empty(128, device="cuda", dtype=torch.bfloat16),
        )
        k_norm = _placeholder(
            graph,
            "k_norm",
            torch.empty(128, device="cuda", dtype=torch.bfloat16),
        )
        cos = _placeholder(
            graph,
            "cos",
            torch.empty((192, 96), device="cuda", dtype=torch.float32),
        )
        sin = _placeholder(
            graph,
            "sin",
            torch.empty((192, 96), device="cuda", dtype=torch.float32),
        )
        query = _normalized_rope(graph, projected[0], q_norm, cos, sin)
        key = _normalized_rope(graph, projected[1], k_norm, cos, sin)
        value = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(projected[2], (1, 192, 2, 128)),
        )
        output = graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention.default,
            args=(query, key, value, [5000, 10000], 2, 128**-0.5),
        )
        output.meta["val"] = torch.empty(
            (1, 192, 2, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        graph.output(output)
    torch.fx.GraphModule({}, graph)
    return graph


class _SparseProjectionAttention(torch.nn.Module):
    """Small canonical H3 attention region used to exercise the fusion pass."""

    sequence_length = 192
    sparse_key_blocks = 2
    input_features = 256
    heads = 2
    head_dim = 128
    rotary_dim = 96

    def __init__(self, *, batch: int = 1, dynamic: bool = False) -> None:
        super().__init__()
        self.batch = batch
        calibration = torch.full(
            (1, self.sequence_length, self.input_features),
            3.0,
            device="cuda",
            dtype=torch.bfloat16,
        )
        base_scale = per_tensor_amax_to_scale(calibration.abs().amax())
        quantization = QuantizeTensorToNVFP4Kwargs(
            block_size=16,
            is_swizzled_scales=True,
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=dynamic,
        )
        projections = []
        for index in range(3):
            dense = torch.randn(
                (self.heads * self.head_dim, self.input_features),
                device="cuda",
                dtype=torch.bfloat16,
            )
            torchao_weight = TorchAONVFP4Tensor.to_nvfp4(
                dense,
                per_tensor_scale=per_tensor_amax_to_scale(dense.abs().amax()),
                act_per_tensor_scale=(None if dynamic else base_scale * (1.0 + index * 0.01)),
                is_swizzled_scales=True,
                act_quant_kwargs=quantization,
            )
            projection = torch.nn.Linear(
                self.input_features,
                self.heads * self.head_dim,
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
            )
            projection.weight = torch.nn.Parameter(
                PiperNVFP4Tensor.from_torchao(torchao_weight),
                requires_grad=False,
            )
            projections.append(projection)
        self.query, self.key, self.value = projections
        self.query_norm = torch.nn.Parameter(
            torch.rand(self.head_dim, device="cuda", dtype=torch.float32).add_(0.5).bfloat16(),
            requires_grad=False,
        )
        self.key_norm = torch.nn.Parameter(
            torch.rand(self.head_dim, device="cuda", dtype=torch.float32).add_(0.5).bfloat16(),
            requires_grad=False,
        )
        angles = torch.rand(
            (self.sequence_length, self.rotary_dim),
            device="cuda",
            dtype=torch.float32,
        ).mul_(2 * torch.pi)
        self.register_buffer("cos", angles.cos().contiguous())
        self.register_buffer("sin", angles.sin().contiguous())
        self.sparse_attention = SparsePiperAttention((0.5, 1.0))

    def _norm_rope(self, projected: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(
            projected.view(
                self.batch,
                self.sequence_length,
                self.heads,
                self.head_dim,
            ),
            (self.head_dim,),
            norm,
            1e-5,
        )
        rotary = normalized[..., : self.rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        cos = self.cos.to(torch.bfloat16)[None, :, None, :]
        sin = self.sin.to(torch.bfloat16)[None, :, None, :]
        rotary = rotary * cos + rotated * sin
        return torch.cat((rotary, normalized[..., self.rotary_dim :]), dim=-1).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._norm_rope(self.query(hidden_states), self.query_norm)
        key = self._norm_rope(self.key(hidden_states), self.key_norm)
        value = self.value(hidden_states).view(
            self.batch,
            self.sequence_length,
            self.heads,
            self.head_dim,
        )
        return self.sparse_attention(
            query,
            key,
            value,
            sparse_key_blocks=self.sparse_key_blocks,
        )


class _TargetCapturePass(CustomInferenceAwareGraphPass):
    def __init__(self) -> None:
        self.targets: list[object] = []
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid


def _options_with_capture(capture: _TargetCapturePass) -> dict[str, object]:
    options = nvfp4_sparse_piper_compile_options()
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*passes, capture)
    return options


def test_compile_options_install_versioned_idempotent_passes() -> None:
    options = nvfp4_sparse_piper_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    assert passes == (nvfp4_compile_pass, compile_pass)
    assert nvfp4_sparse_piper_compile_options(options) == options
    assert compile_pass.uuid() == compile_pass.uuid()
    assert compile_pass.uuid()


@pytest.mark.parametrize(("dynamic", "preparation_count"), [(False, 3), (True, 1)])
def test_prepared_projection_family_fuses_without_materializing_linears(
    monkeypatch: pytest.MonkeyPatch,
    dynamic: bool,
    preparation_count: int,
) -> None:
    graph = _semantic_attention_graph(dynamic=dynamic)
    monkeypatch.setattr(
        AcceleratorTarget,
        "from_device",
        classmethod(lambda _cls, _device: AcceleratorTarget("cuda", "sm120")),
    )

    nvfp4_compile_pass(graph, is_inference=True)
    prepared_targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert (
        prepared_targets.count(torch.ops.piper_kernels.nvfp4_prepare_input.default)
        == preparation_count
    )
    assert prepared_targets.count(torch.ops.piper_kernels.nvfp4_linear_prepared.default) == 3

    compile_pass(graph, is_inference=True)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(torch.ops.piper_kernels.nvfp4_prepare_input.default) == preparation_count
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_linear_mean.default) == 1
    assert targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default) == 1
    assert torch.ops.piper_kernels.nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in targets
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in targets
    graph.lint()


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(("dynamic", "preparation_count"), [(False, 3), (True, 1)])
def test_cuda_compile_fuses_nvfp4_sparse_projection_region(
    dynamic: bool,
    preparation_count: int,
) -> None:
    torch.manual_seed(823)
    model = _SparseProjectionAttention(dynamic=dynamic).eval()
    hidden_states = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True)(hidden_states)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=_options_with_capture(capture),
        )(hidden_states)

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.025
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_prepare_input.default)
        == preparation_count
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default) == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    )
    assert capture.targets.count(torch.ops.piper_kernels.nvfp4_linear_mean.default) == 1
    assert (
        capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default)
        == 1
    )
    assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compile_fails_closed_for_batch_two() -> None:
    torch.manual_seed(827)
    model = _SparseProjectionAttention(batch=2).eval()
    hidden_states = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()

    with torch.no_grad():
        torch.compile(
            model,
            fullgraph=True,
            options=_options_with_capture(capture),
        )(hidden_states)

    assert torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default not in capture.targets
    assert capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention.default) == 1
