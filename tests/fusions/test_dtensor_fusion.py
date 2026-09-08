"""Ordinary DTensor boundaries retain the same local FFN and attention fusions."""

import copy
import uuid
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

from piper_kernels.fusions.convrot_int8_sparse_piper import (
    convrot_int8_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_int8_swiglu_ffn import convrot_int8_swiglu_ffn_compile_options
from piper_kernels.fusions.convrot_nvfp4_sparse_piper import (
    convrot_nvfp4_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_nvfp4_swiglu_ffn import convrot_nvfp4_swiglu_ffn_compile_options
from piper_kernels.fusions.nvfp4_sparse_piper import nvfp4_sparse_piper_compile_options
from piper_kernels.fusions.nvfp4_swiglu_ffn import nvfp4_swiglu_ffn_compile_options
from piper_kernels.linear.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor
from piper_kernels.linear.nvfp4._layout import swap_packed_pairs

from .convrot_int8_sparse_piper.test_compile import _MeanPoolSparseProjectionAttentionOutput
from .convrot_nvfp4_sparse_piper.test_compile import _ConvRotSparseProjectionAttentionOutput
from .nvfp4_sparse_piper.test_compile import _SparseProjectionAttentionOutput

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
        reason="requires an SM120 CUDA device",
    ),
]

_FFN_OPTIONS = {
    "convrot_int8": convrot_int8_swiglu_ffn_compile_options,
    "nvfp4": nvfp4_swiglu_ffn_compile_options,
    "convrot_nvfp4": convrot_nvfp4_swiglu_ffn_compile_options,
}
_ATTENTION_OPTIONS = {
    "convrot_int8": convrot_int8_sparse_piper_compile_options,
    "nvfp4": nvfp4_sparse_piper_compile_options,
    "convrot_nvfp4": convrot_nvfp4_sparse_piper_compile_options,
}
_FFN_PROJECTIONS = ("gate", "value", "down")
_ATTENTION_PROJECTIONS = ("query", "key", "value", "output")


@pytest.fixture(scope="module")
def mesh():
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires distributed Gloo support")
    dist.init_process_group(
        "gloo", store=dist.HashStore(), rank=0, world_size=1, timeout=timedelta(seconds=120)
    )
    try:
        yield DeviceMesh("cuda", [0])
    finally:
        dist.destroy_process_group()


class _TargetCapturePass(CustomInferenceAwareGraphPass):
    def __init__(self):
        self.targets = []
        self.calls = 0
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph, is_inference):
        assert is_inference
        self.calls += 1
        self.targets = [str(node.target) for node in graph.nodes if node.op == "call_function"]

    def uuid(self):
        return self._uuid


def _compile_with_capture(model, options_fn):
    capture = _TargetCapturePass()
    options = options_fn({"triton.cudagraphs": False, "compile_threads": 1})
    options["post_grad_custom_pre_pass"] = (*options["post_grad_custom_pre_pass"], capture)
    torch.compiler.reset()
    return torch.compile(model, fullgraph=True, options=options), capture


def _replicate(value, mesh):
    return DTensor.from_local(value, mesh, [Replicate()], run_check=False)


def _distribute_parameters(layer, mesh, placement=None):
    placement = Replicate() if placement is None else placement
    for name in ("weight", "bias"):
        value = getattr(layer, name)
        if value is not None:
            parameter_placement = placement
            if name == "bias" and placement == Shard(1):
                parameter_placement = Replicate()
            distributed = DTensor.from_local(
                value.detach(), mesh, [parameter_placement], run_check=False
            )
            setattr(layer, name, torch.nn.Parameter(distributed, requires_grad=False))


class _ProjectionBoundary(torch.nn.Module):
    def __init__(self, layer, mesh, placement=None):
        super().__init__()
        self.layer = layer
        self.mesh = mesh
        self.input_placement = Shard(-1) if placement == Shard(1) else Replicate()
        _distribute_parameters(layer, mesh, placement)

    def forward(self, value):
        distributed = DTensor.from_local(value, self.mesh, [self.input_placement], run_check=False)
        return self.layer(distributed).to_local()


def _distributed_ffn(model, mesh, *, sharded=False):
    distributed = copy.deepcopy(model)
    for name in _FFN_PROJECTIONS:
        placement = Shard(1 if name == "down" else 0) if sharded else Replicate()
        _distribute_parameters(getattr(distributed, name), mesh, placement)
    return distributed


def _distributed_attention(model, mesh, *, projections=_ATTENTION_PROJECTIONS, sharded=False):
    distributed = copy.deepcopy(model)
    for name in projections:
        placement = Shard(1 if name == "output" else 0) if sharded else Replicate()
        layer = _ProjectionBoundary(getattr(distributed, name), mesh, placement)
        setattr(distributed, name, layer)
    return distributed


class _FFN(torch.nn.Module):
    def __init__(
        self,
        format_name,
        *,
        bias=False,
        high_first=False,
        expose_gate=False,
        intermediate_features=512,
    ):
        super().__init__()
        self.expose_gate = expose_gate
        for name, output_features, input_features in (
            ("gate", intermediate_features, 256),
            ("value", intermediate_features, 256),
            ("down", 256, intermediate_features),
        ):
            source = (
                torch.randn(output_features, input_features, device="cuda", dtype=torch.bfloat16)
                * 0.02
            )
            layer = torch.nn.Linear(input_features, output_features, bias=False, device="meta")
            layer.weight = torch.nn.Parameter(
                _weight(format_name, source, high_first=high_first), requires_grad=False
            )
            if bias:
                layer.bias = torch.nn.Parameter(
                    torch.randn(output_features, device="cuda", dtype=torch.float32) * 0.01,
                    requires_grad=False,
                )
            setattr(self, name, layer)

    def forward(self, value):
        gate = self.gate(value)
        result = self.down(F.silu(gate) * self.value(value))
        return (result, gate) if self.expose_gate else result


def _weight(format_name, source, *, high_first=False):
    if format_name == "convrot_int8":
        return ConvRotInt8Tensor.from_hp(source, group_size=64)
    cls = PiperNVFP4Tensor if format_name == "nvfp4" else ConvRotNVFP4Tensor
    weight = cls.from_hp(
        source,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(
            block_size=16,
            is_swizzled_scales=True,
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=True,
        ),
        **({"group_size": 64} if cls is ConvRotNVFP4Tensor else {}),
    )
    if high_first:
        weight.qdata = swap_packed_pairs(weight.qdata)
        weight.high_first = True
    return weight


def _ffn_target(format_name):
    return f"piper_kernels.{format_name}_swiglu_ffn.default"


def _assert_ffn_fused(capture, format_name):
    assert capture.targets.count(_ffn_target(format_name)) == 1
    _assert_no_separate_linears(capture)


def _assert_no_separate_linears(capture):
    assert not any(
        target.endswith(("_linear.default", "_linear_prepared.default"))
        for target in capture.targets
    )


@pytest.mark.parametrize("format_name", tuple(_FFN_OPTIONS))
@pytest.mark.parametrize("shape", [(192, 256), (1, 192, 256), (2, 3, 64, 256)])
def test_dtensor_ffn_selects_local_fused_operator(mesh, format_name, shape):
    torch.manual_seed(109)
    model = _FFN(format_name, bias=True, high_first=format_name != "convrot_int8").eval()
    distributed = _distributed_ffn(model, mesh)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        local, local_capture = _compile_with_capture(model, _FFN_OPTIONS[format_name])
        expected = local(value)
        _assert_ffn_fused(local_capture, format_name)
        compiled, capture = _compile_with_capture(distributed, _FFN_OPTIONS[format_name])
        actual = compiled(_replicate(value, mesh)).to_local()
    _assert_ffn_fused(capture, format_name)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("format_name", tuple(_FFN_OPTIONS))
def test_dtensor_ffn_reuses_symbolic_leading_dimensions(mesh, format_name):
    torch.manual_seed(110)
    model = _FFN(format_name).eval()
    distributed = _distributed_ffn(model, mesh)
    values = [torch.randn(2, rows, 256, device="cuda", dtype=torch.bfloat16) for rows in (129, 193)]
    with torch.no_grad():
        local, _ = _compile_with_capture(model, _FFN_OPTIONS[format_name])
        expected = [local(value) for value in values]
        compiled, capture = _compile_with_capture(distributed, _FFN_OPTIONS[format_name])
        for value, reference in zip(values, expected, strict=True):
            dx = _replicate(value, mesh)
            torch._dynamo.mark_dynamic(dx, 1)
            torch._dynamo.mark_dynamic(dx._local_tensor, 1)
            actual = compiled(dx).to_local()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    _assert_ffn_fused(capture, format_name)
    assert capture.calls == 1


@pytest.mark.parametrize("format_name", tuple(_FFN_OPTIONS))
def test_dtensor_ffn_preserves_externally_used_gate(mesh, format_name):
    torch.manual_seed(111)
    model = _FFN(format_name, expose_gate=True).eval()
    distributed = _distributed_ffn(model, mesh)
    value = torch.randn(1, 192, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        local, local_capture = _compile_with_capture(model, _FFN_OPTIONS[format_name])
        expected = local(value)
        compiled, capture = _compile_with_capture(distributed, _FFN_OPTIONS[format_name])
        actual = compiled(_replicate(value, mesh))
    assert _ffn_target(format_name) not in local_capture.targets
    assert _ffn_target(format_name) not in capture.targets
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output.to_local(), reference, rtol=0, atol=0)


def _attention(format_name):
    if format_name == "convrot_int8":
        return _MeanPoolSparseProjectionAttentionOutput().eval()
    if format_name == "nvfp4":
        return _SparseProjectionAttentionOutput(dynamic=True, routing="mean").eval()
    return _ConvRotSparseProjectionAttentionOutput(dynamic=True, routing="mean").eval()


def _assert_attention_fused(capture, format_name):
    assert (
        capture.targets.count(
            f"piper_kernels.{format_name}_sparse_piper_projected_query_attention_output.default"
        )
        == 1
    )
    projection_prefix = "convrot_int8" if format_name == "convrot_int8" else "nvfp4"
    for operation in ("project_key", "project_value"):
        assert (
            capture.targets.count(
                f"piper_kernels.{projection_prefix}_sparse_piper_{operation}.default"
            )
            == 1
        )
    _assert_no_separate_linears(capture)


@pytest.mark.parametrize("format_name", tuple(_ATTENTION_OPTIONS))
@pytest.mark.parametrize(
    "projections",
    [
        pytest.param(("query", "key", "value"), id="qkv"),
        pytest.param(("output",), id="output"),
        pytest.param(_ATTENTION_PROJECTIONS, id="both"),
    ],
)
def test_dtensor_attention_boundaries_preserve_complete_fusion(mesh, format_name, projections):
    torch.manual_seed(112)
    model = _attention(format_name)
    distributed = _distributed_attention(model, mesh, projections=projections)
    value = torch.randn(1, 192, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        local, local_capture = _compile_with_capture(model, _ATTENTION_OPTIONS[format_name])
        expected = local(value)
        _assert_attention_fused(local_capture, format_name)
        compiled, capture = _compile_with_capture(distributed, _ATTENTION_OPTIONS[format_name])
        actual = compiled(value)
    _assert_attention_fused(capture, format_name)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _DynamicAttention(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def _norm_rope(self, projected, norm, cos, sin):
        model = self.attention
        batch, rows, _ = projected.shape
        normalized = F.rms_norm(
            projected.view(batch, rows, model.heads, model.head_dim),
            (model.head_dim,),
            norm,
            1e-5,
        )
        rotary = normalized[..., : model.rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        rotary = rotary * cos.to(torch.bfloat16)[None, :, None, :]
        rotary = rotary + rotated * sin.to(torch.bfloat16)[None, :, None, :]
        return torch.cat((rotary, normalized[..., model.rotary_dim :]), dim=-1).contiguous()

    def forward(self, value, cos, sin):
        model = self.attention
        batch, rows, _ = value.shape
        query = self._norm_rope(model.query(value), model.query_norm, cos, sin)
        key = self._norm_rope(model.key(value), model.key_norm, cos, sin)
        projected_value = model.value(value).view(batch, rows, model.heads, model.head_dim)
        attention = model.sparse_attention(query, key, projected_value, sparse_key_blocks=2)
        return model.output(attention.flatten(2))


@pytest.mark.parametrize("format_name", tuple(_ATTENTION_OPTIONS))
def test_dtensor_attention_reuses_symbolic_sequence_length(mesh, format_name):
    torch.manual_seed(113)
    model = _DynamicAttention(_attention(format_name))
    distributed = _DynamicAttention(_distributed_attention(model.attention, mesh))
    inputs = []
    for rows in (192, 256):
        value = torch.randn(1, rows, 256, device="cuda", dtype=torch.bfloat16)
        angles = torch.rand(rows, 96, device="cuda")
        cos, sin = angles.cos(), angles.sin()
        torch._dynamo.mark_dynamic(value, 1)
        torch._dynamo.mark_dynamic(cos, 0)
        torch._dynamo.mark_dynamic(sin, 0)
        inputs.append((value, cos, sin))
    with torch.no_grad():
        local, local_capture = _compile_with_capture(model, _ATTENTION_OPTIONS[format_name])
        expected = [local(*values) for values in inputs]
        _assert_attention_fused(local_capture, format_name)
        compiled, capture = _compile_with_capture(distributed, _ATTENTION_OPTIONS[format_name])
        for values, reference in zip(inputs, expected, strict=True):
            actual = compiled(*values)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    _assert_attention_fused(capture, format_name)
    assert capture.calls == 1


def _check_partial_output(local, expected, cpu_mesh):
    # Two Gloo ranks share one GPU. Only dense partial outputs are communicated,
    # after the fused local computation, and all collectives operate on CPU.
    torch.testing.assert_close(local, expected, rtol=0, atol=0)
    full_expected = expected.cpu()
    dist.all_reduce(full_expected, op=dist.ReduceOp.SUM)
    actual = DTensor.from_local(local.cpu(), cpu_mesh, [Partial()], run_check=False).full_tensor()
    torch.testing.assert_close(actual, full_expected, rtol=0, atol=0)


def _check_sharded_ffn(format_name, mesh, cpu_mesh):
    model = _FFN(
        format_name, intermediate_features=256, high_first=format_name != "convrot_int8"
    ).eval()
    distributed = _distributed_ffn(model, mesh, sharded=True)
    # Each rank owns a different intermediate-width shard of the weights.
    torch.manual_seed(114)
    value = torch.randn(1, 192, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        local, local_capture = _compile_with_capture(model, _FFN_OPTIONS[format_name])
        expected = local(value)
        _assert_ffn_fused(local_capture, format_name)
        compiled, capture = _compile_with_capture(distributed, _FFN_OPTIONS[format_name])
        for streamed in (False, True):
            if streamed:
                for name in _FFN_PROJECTIONS:
                    layer = getattr(distributed, name)
                    # Stream fresh storage with the same quantization metadata.
                    weight = getattr(model, name).weight.detach().cpu().cuda()
                    placement = Shard(1) if name == "down" else Shard(0)
                    layer.weight = torch.nn.Parameter(
                        DTensor.from_local(weight, mesh, [placement], run_check=False),
                        requires_grad=False,
                    )
            result = compiled(_replicate(value, mesh))
            assert result.placements == (Partial(),)
            _check_partial_output(result.to_local(), expected, cpu_mesh)
    _assert_ffn_fused(capture, format_name)
    assert capture.calls == 1


def _check_sharded_attention(format_name, mesh, cpu_mesh):
    model = _attention(format_name)
    # Each rank owns complete local heads; output projection consumes only
    # those heads and produces a partial result in the shared output width.
    distributed = _distributed_attention(model, mesh, sharded=True)
    # DTensor divides the replicated output bias across partial products.
    with torch.no_grad():
        model.output.bias.div_(2)
    torch.manual_seed(115)
    value = torch.randn(1, 192, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        local, local_capture = _compile_with_capture(model, _ATTENTION_OPTIONS[format_name])
        expected = local(value)
        _assert_attention_fused(local_capture, format_name)
        compiled, capture = _compile_with_capture(distributed, _ATTENTION_OPTIONS[format_name])
        actual = compiled(value)
    _assert_attention_fused(capture, format_name)
    _check_partial_output(actual, expected, cpu_mesh)


def _two_rank_fusions(rank, store_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(store_path, 2),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    try:
        mesh = DeviceMesh("cuda", [0, 1])
        cpu_mesh = DeviceMesh("cpu", [0, 1])
        for format_name in _FFN_OPTIONS:
            for check in (_check_sharded_ffn, _check_sharded_attention):
                torch.manual_seed(116 + rank)
                try:
                    check(format_name, mesh, cpu_mesh)
                except Exception as error:
                    raise AssertionError(
                        f"rank={rank}, format={format_name}, check={check.__name__}: {error}"
                    ) from error
    finally:
        dist.destroy_process_group()


def test_two_rank_local_fusions_and_partial_output_sums(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires distributed Gloo support")
    mp.spawn(_two_rank_fusions, args=(str(tmp_path / "store"),), nprocs=2, join=True)
