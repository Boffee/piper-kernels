"""Tests for weight-only W4A16 NVFP4 and ConvRot NVFP4 linear execution."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest
import torch
import triton
from _accelerators import mock_cuda_target, w4a16_nvfp4_available
from _nvidia_resources import capture_cuda_spill_stores
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F  # noqa: N812
from triton.backends.compiler import GPUTarget

from piper_kernels import compile_options
from piper_kernels.fusions.nvfp4_sparse_piper import output as nvfp4_sparse_output
from piper_kernels.linear.nvfp4 import reference as nvfp4_reference
from piper_kernels.linear.nvfp4 import triton as nvfp4_triton
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor


def _hip_target_param(arch: str, warp_size: int) -> object:
    return pytest.param(
        GPUTarget("hip", arch, warp_size),
        marks=pytest.mark.skipif(sys.platform != "linux", reason="HIP compiler is Linux-only"),
    )


_PRIMARY_ISA_TARGETS = [
    GPUTarget("cuda", 80, 32),
    _hip_target_param("gfx942", 64),
    _hip_target_param("gfx1201", 32),
]

_ADDITIONAL_CUDA_TARGETS = [
    GPUTarget("cuda", 75, 32),
    GPUTarget("cuda", 86, 32),
    GPUTarget("cuda", 89, 32),
    GPUTarget("cuda", 120, 32),
]


def _tolerance(dtype: torch.dtype) -> float:
    if dtype is torch.float32:
        return 2e-4
    if dtype is torch.float16:
        return 6e-3
    return 2.5e-2


def _make_nvfp4_weight(
    out_features: int,
    in_features: int,
    *,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    high_first: bool = False,
    two_level: bool = True,
    group_size: int = 0,
    seed: int = 101,
) -> PiperNVFP4Tensor:
    torch.manual_seed(seed)
    dense = torch.randn(out_features, in_features, dtype=dtype) * (in_features**-0.5)
    if group_size > 0:
        base = ConvRotNVFP4Tensor.from_hp(
            dense,
            group_size=group_size,
            compute_per_tensor_scale=two_level,
            is_swizzled_scales=True,
            act_quant=None,
        )
    else:
        base = PiperNVFP4Tensor.from_hp(
            dense,
            compute_per_tensor_scale=two_level,
            is_swizzled_scales=True,
            act_quant=None,
        )
    if high_first:
        qdata = ((base.qdata & 0x0F) << 4) | (base.qdata >> 4)
        if isinstance(base, ConvRotNVFP4Tensor):
            base = ConvRotNVFP4Tensor(
                qdata,
                base.scale,
                base.block_size,
                base.dtype,
                base.group_size,
                base.per_tensor_scale,
                base.act_per_tensor_scale,
                base.is_swizzled_scales,
                base.act_quant,
                high_first=True,
            )
        else:
            base = PiperNVFP4Tensor(
                qdata,
                base.scale,
                base.block_size,
                base.dtype,
                base.per_tensor_scale,
                base.act_per_tensor_scale,
                base.is_swizzled_scales,
                base.act_quant,
                high_first=True,
            )
    return base.to(device)


def _run_pre_passes(module: torch.fx.GraphModule) -> None:
    passes = compile_options(linear=True, ffn=True)["post_grad_custom_pre_pass"]
    assert isinstance(passes, tuple)
    for compiler_pass in passes:
        compiler_pass(module.graph, is_inference=True)
    module.recompile()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group_size", [0, 16, 64, 256])
@pytest.mark.parametrize("high_first", [False, True])
@pytest.mark.parametrize("two_level", [False, True])
def test_reference_w4a16_matches_dequantized_dense_linear(
    dtype: torch.dtype,
    group_size: int,
    high_first: bool,
    two_level: bool,
) -> None:
    torch.manual_seed(700 + group_size)
    in_features = 256
    out_features = 80
    weight = _make_nvfp4_weight(
        out_features,
        in_features,
        dtype=dtype,
        high_first=high_first,
        two_level=two_level,
        group_size=group_size,
    )
    source = torch.randn(3, 5, in_features, dtype=dtype)
    bias_storage = torch.randn(out_features * 2, dtype=torch.float32)
    bias = bias_storage[::2]

    actual = nvfp4_reference.linear_w4a16(
        source,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        bias,
        high_first=high_first,
        group_size=group_size,
    )
    expected = F.linear(source.float(), weight.dequantize(torch.float32), bias).to(dtype)

    assert actual.dtype is dtype
    assert actual.shape == (3, 5, out_features)
    rtol = 1.6e-2 if dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=1e-3)


def test_reference_w4a16_handles_empty_shapes_and_zero_blocks() -> None:
    qdata = torch.zeros(32, 16, dtype=torch.uint8)
    scale = torch.zeros(32, 16, dtype=torch.float8_e4m3fn)
    empty_in = nvfp4_reference.linear_w4a16(
        torch.empty(0, 32, dtype=torch.bfloat16),
        qdata,
        scale,
        None,
        None,
    )
    assert empty_in.shape == (0, 32)
    assert empty_in.dtype is torch.bfloat16

    empty_out = nvfp4_reference.linear_w4a16(
        torch.ones(2, 32, dtype=torch.float16),
        torch.empty(0, 16, dtype=torch.uint8),
        scale,
        None,
        None,
    )
    assert empty_out.shape == (2, 0)
    assert empty_out.dtype is torch.float16

    zeros = nvfp4_reference.linear_w4a16(
        torch.randn(4, 32, dtype=torch.bfloat16),
        qdata,
        scale,
        torch.tensor(1.0),
        None,
        group_size=16,
    )
    assert torch.equal(zeros, torch.zeros(4, 32, dtype=torch.bfloat16))


@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_cpu_weight_only_convrot_nvfp4_linear_matches_reference(group_size: int) -> None:
    torch.manual_seed(710 + group_size)
    weight = _make_nvfp4_weight(64, 256, dtype=torch.bfloat16, group_size=group_size)
    assert isinstance(weight, ConvRotNVFP4Tensor)
    source = torch.randn(3, 256, dtype=torch.bfloat16)
    bias = torch.randn(64, dtype=torch.float32)

    actual = F.linear(source, weight, bias)
    expected = nvfp4_reference.linear_w4a16(
        source,
        weight.qdata,
        weight.scale,
        weight.per_tensor_scale,
        bias,
        group_size=group_size,
    )

    assert torch.equal(actual, expected)


def test_cpu_weight_only_convrot_nvfp4_validation_and_compile() -> None:
    torch.manual_seed(715)
    weight = _make_nvfp4_weight(64, 256, dtype=torch.bfloat16, group_size=64)
    source = torch.randn(3, 256, dtype=torch.bfloat16)
    bias = torch.randn(64, dtype=torch.float32)

    with pytest.raises(ValueError, match="ConvRot NVFP4 linear"):
        F.linear(torch.randn(3, 128, dtype=torch.bfloat16), weight)
    with pytest.raises(ValueError, match="bias"):
        F.linear(source, weight, torch.zeros(64, dtype=torch.int32))
    with pytest.raises(ValueError, match="contiguous"):
        F.linear(source, weight, torch.randn(128, dtype=torch.float32)[::2])

    torch._dynamo.reset()

    def projection(inp: torch.Tensor) -> torch.Tensor:
        return F.linear(inp, weight, bias)

    with torch.inference_mode():
        eager = projection(source)
        compiled = torch.compile(projection, fullgraph=True)(source)
    assert torch.equal(compiled, eager)


def test_cpu_weight_only_convrot_nvfp4_runs_without_triton() -> None:
    script = """
import builtins
import importlib.util
import sys

original_import = builtins.__import__
original_find_spec = importlib.util.find_spec

def without_triton(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        raise ModuleNotFoundError("Triton intentionally unavailable", name="triton")
    return original_import(name, *args, **kwargs)

def find_spec(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        return None
    return original_find_spec(name, *args, **kwargs)

builtins.__import__ = without_triton
importlib.util.find_spec = find_spec

import torch
from torch.nn import functional as F

from piper_kernels.linear.nvfp4 import reference as nvfp4_reference
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

torch.manual_seed(718)
dense = torch.randn(64, 256, dtype=torch.bfloat16) * (256**-0.5)
weight = ConvRotNVFP4Tensor.from_hp(
    dense,
    group_size=64,
    compute_per_tensor_scale=True,
    is_swizzled_scales=True,
    act_quant=None,
)
source = torch.randn(3, 256, dtype=torch.bfloat16)
bias = torch.randn(64, dtype=torch.float32)

actual = F.linear(source, weight, bias)
expected = nvfp4_reference.linear_w4a16(
    source,
    weight.qdata,
    weight.scale,
    weight.per_tensor_scale,
    bias,
    group_size=64,
)
assert torch.equal(actual, expected)

q_meta = torch.empty(128, 128, dtype=torch.uint8, device="meta")
s_meta = torch.empty(32, 64, dtype=torch.float8_e4m3fn, device="meta")
gs_meta = torch.empty((), device="meta")
x_meta = torch.empty(2, 256, dtype=torch.bfloat16, device="meta")
plain_meta = PiperNVFP4Tensor(q_meta, s_meta, 16, torch.bfloat16, gs_meta, None, True, None)
convrot_meta = ConvRotNVFP4Tensor(
    q_meta, s_meta, 16, torch.bfloat16, 64, gs_meta, None, True, None
)
assert F.linear(x_meta, plain_meta).shape == (2, 128)
assert F.linear(x_meta, convrot_meta).shape == (2, 128)
assert "triton" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("group_size", [0, 64])
@pytest.mark.parametrize("shape", [(256,), (2, 256), (3, 5, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_meta_weight_only_linear_supports_shapes_and_preserves_unsplit_graph(
    group_size: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    qdata = torch.empty(128, 128, dtype=torch.uint8, device="meta")
    scale = torch.empty(32, 64, dtype=torch.float8_e4m3fn, device="meta")
    global_scale = torch.empty((), device="meta")
    if group_size > 0:
        weight: PiperNVFP4Tensor = ConvRotNVFP4Tensor(
            qdata, scale, 16, dtype, group_size, global_scale, None, True, None
        )
    else:
        weight = PiperNVFP4Tensor(qdata, scale, 16, dtype, global_scale, None, True, None)
    input_value = torch.empty(shape, dtype=dtype, device="meta")
    bias = torch.empty(128, dtype=dtype, device="meta")

    eager_meta = F.linear(input_value, weight, bias)
    assert eager_meta.shape == (*shape[:-1], 128)
    assert eager_meta.dtype is dtype

    def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.linear(x, weight, bias), F.linear(x, weight, None)

    module = make_fx(fn)(input_value)
    _run_pre_passes(module)
    target_names = [str(node.target) for node in module.graph.nodes if node.op == "call_function"]
    expected_name = (
        "piper_kernels.convrot_nvfp4_linear.default"
        if group_size > 0
        else "piper_kernels.nvfp4_linear.default"
    )
    assert target_names.count(expected_name) == 2
    assert not any(
        blocked in name
        for name in target_names
        for blocked in ("prepare_input", "linear_prepared", "nvfp4_ffn")
    )


@pytest.mark.parametrize("group_size", [0, 64])
def test_meta_weight_only_ffn_is_not_folded_into_w4a4_ffn(group_size: int) -> None:
    def _meta_w(out_f: int, in_f: int) -> PiperNVFP4Tensor:
        q = torch.empty(out_f, in_f // 2, dtype=torch.uint8, device="meta")
        s = torch.empty(
            ((out_f + 127) // 128) * 32,
            ((in_f + 63) // 64) * 16,
            dtype=torch.float8_e4m3fn,
            device="meta",
        )
        gs = torch.empty((), device="meta")
        if group_size > 0:
            return ConvRotNVFP4Tensor(q, s, 16, torch.bfloat16, group_size, gs, None, True, None)
        return PiperNVFP4Tensor(q, s, 16, torch.bfloat16, gs, None, True, None)

    w_gate = _meta_w(512, 256)
    w_up = _meta_w(512, 256)
    w_down = _meta_w(256, 512)
    x = torch.empty(64, 256, dtype=torch.bfloat16, device="meta")

    def swiglu_ffn(inp: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(inp, w_gate)) * F.linear(inp, w_up), w_down)

    module = make_fx(swiglu_ffn)(x)
    _run_pre_passes(module)
    target_names = [str(node.target) for node in module.graph.nodes if node.op == "call_function"]
    assert not any(
        blocked in name
        for name in target_names
        for blocked in ("prepare_input", "linear_prepared", "nvfp4_ffn")
    )


@pytest.mark.parametrize("group_size", [0, 64])
@pytest.mark.parametrize("rows", [1, 8, 16, 17])
def test_meta_w4a4_linear_preserves_prepare_split_for_small_m(
    group_size: int,
    rows: int,
) -> None:
    q = torch.empty(128, 128, dtype=torch.uint8, device="meta")
    s = torch.empty(32, 64, dtype=torch.float8_e4m3fn, device="meta")
    gs = torch.empty((), device="meta")
    if group_size > 0:
        w: PiperNVFP4Tensor = ConvRotNVFP4Tensor(
            q, s, 16, torch.bfloat16, group_size, gs, None, True, "dynamic"
        )
    else:
        w = PiperNVFP4Tensor(q, s, 16, torch.bfloat16, gs, None, True, "dynamic")
    x = torch.empty(rows, 256, dtype=torch.bfloat16, device="meta")

    def repeated(inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.linear(inp, w), F.linear(inp, w)

    module = make_fx(repeated)(x)
    _run_pre_passes(module)
    targets = [node.target for node in module.graph.nodes if node.op == "call_function"]
    expected_linear = (
        torch.ops.piper_kernels.convrot_nvfp4_linear.default
        if group_size > 0
        else torch.ops.piper_kernels.nvfp4_linear.default
    )
    assert expected_linear not in targets
    assert targets.count(torch.ops.piper_kernels.nvfp4_prepare_input.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_linear_prepared.default) == 2


@pytest.mark.parametrize("group_size", [0, 64])
def test_meta_weight_only_rejects_activation_scales(group_size: int) -> None:
    q = torch.empty(128, 128, dtype=torch.uint8, device="meta")
    s = torch.empty(32, 64, dtype=torch.float8_e4m3fn, device="meta")
    gs = torch.empty((), device="meta")
    w: PiperNVFP4Tensor = (
        ConvRotNVFP4Tensor(q, s, 16, torch.bfloat16, group_size, gs, None, True, None)
        if group_size > 0
        else PiperNVFP4Tensor(q, s, 16, torch.bfloat16, gs, None, True, None)
    )
    x = torch.empty(4, 256, dtype=torch.bfloat16, device="meta")
    F.linear(x, w)

    act_scale = torch.empty((), dtype=torch.float32, device="meta")
    for dynamic, scale in ((True, None), (False, act_scale)):

        def _invoke(dynamic_flag: bool = dynamic, scale_arg: torch.Tensor | None = scale) -> None:
            if group_size > 0:
                torch.ops.piper_kernels.convrot_nvfp4_linear.default(
                    x, q, s, gs, scale_arg, None, dynamic_flag, group_size, False, True
                )
            else:
                torch.ops.piper_kernels.nvfp4_linear.default(
                    x, q, s, gs, scale_arg, None, dynamic_flag, False, True
                )

        with pytest.raises(
            ValueError, match="weight-only execution does not accept activation scales"
        ):
            _invoke()


@pytest.mark.parametrize("group_size", [0, 64])
@pytest.mark.parametrize("weight_only", [False, True])
def test_attention_output_pass_skips_weight_only_linear(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
    weight_only: bool,
) -> None:
    assert nvfp4_sparse_output.__name__.endswith(".output")
    q_meta = torch.empty(128, 128, dtype=torch.uint8, device="meta")
    s_meta = torch.empty(32, 64, dtype=torch.float8_e4m3fn, device="meta")
    gs_meta = torch.empty((), device="meta")
    meta_weight: PiperNVFP4Tensor = (
        ConvRotNVFP4Tensor(
            q_meta, s_meta, 16, torch.bfloat16, group_size, gs_meta, None, True, None
        )
        if group_size > 0
        else PiperNVFP4Tensor(q_meta, s_meta, 16, torch.bfloat16, gs_meta, None, True, None)
    )
    F.linear(torch.empty(2, 256, dtype=torch.bfloat16, device="meta"), meta_weight)

    mock_cuda_target(monkeypatch, (12, 0))
    graph = torch.fx.Graph()

    def _placeholder(name: str, tensor: torch.Tensor) -> torch.fx.Node:
        node = graph.placeholder(name)
        node.meta["val"] = tensor
        return node

    with FakeTensorMode():
        operand_specs = (
            ("query", (1, 2, 192, 128), torch.int8),
            ("query_scale", (1, 2, 6), torch.float32),
            ("query_summary", (1, 2, 3, 128), torch.float32),
            ("key", (1, 2, 192, 128), torch.int8),
            ("key_scale", (1, 2, 3), torch.float32),
            ("key_center", (1, 2, 3, 2, 128), torch.int8),
            ("key_center_scale", (1, 2, 3, 2), torch.float32),
            ("key_summary", (1, 2, 3, 128), torch.float32),
            ("key_aux", (1, 2, 3, 128), torch.float32),
            ("value", (1, 2, 128, 192), torch.int8),
            ("value_scale", (1, 2, 3, 1), torch.float32),
            ("value_mean", (1, 2, 3, 128), torch.bfloat16),
        )
        operands = tuple(
            _placeholder(name, torch.empty(shape, device="cuda", dtype=dtype))
            for name, shape, dtype in operand_specs
        )
        attention = graph.call_function(
            torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
            args=(*operands, [5_000, 10_000], 2, 192, 0),
        )
        attention.meta["val"] = torch.empty((1, 192, 2, 128), device="cuda", dtype=torch.bfloat16)
        reshaped = graph.call_function(
            torch.ops.aten.reshape.default,
            args=(attention, (1, 192, 256)),
        )
        reshaped.meta["val"] = torch.empty((1, 192, 256), device="cuda", dtype=torch.bfloat16)

        weight_qdata = _placeholder(
            "weight_qdata", torch.empty((320, 128), device="cuda", dtype=torch.uint8)
        )
        weight_scale = _placeholder(
            "weight_scale", torch.empty((96, 64), device="cuda", dtype=torch.float8_e4m3fn)
        )
        weight_global_scale = _placeholder(
            "weight_global_scale", torch.empty((), device="cuda", dtype=torch.float32)
        )
        activation_scale = (
            None
            if weight_only
            else _placeholder(
                "activation_scale", torch.empty((), device="cuda", dtype=torch.float32)
            )
        )
        if group_size > 0:
            linear_target = torch.ops.piper_kernels.convrot_nvfp4_linear.default
            linear_args: tuple[object, ...] = (
                reshaped,
                weight_qdata,
                weight_scale,
                weight_global_scale,
                activation_scale,
                None,
                False,
                group_size,
                False,
                weight_only,
            )
        else:
            linear_target = torch.ops.piper_kernels.nvfp4_linear.default
            linear_args = (
                reshaped,
                weight_qdata,
                weight_scale,
                weight_global_scale,
                activation_scale,
                None,
                False,
                False,
                weight_only,
            )
        projected = graph.call_function(linear_target, args=linear_args)
        projected.meta["val"] = torch.empty((1, 192, 320), device="cuda", dtype=torch.bfloat16)
        graph.output(projected)
    torch.fx.GraphModule({}, graph)

    passes = compile_options(linear=False, attention=True)["post_grad_custom_pre_pass"]
    assert isinstance(passes, tuple)
    for compiler_pass in passes:
        compiler_pass(graph, is_inference=True)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    fused_op = torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default
    if weight_only:
        assert linear_target in targets
        assert fused_op not in targets
    else:
        assert fused_op in targets
        assert linear_target not in targets


def _assert_clean_offline_compile(
    compiled: triton.compiler.CompiledKernel,
    target: GPUTarget,
    *,
    cuda_spill_stores: int | None = None,
) -> None:
    assert compiled.metadata.global_scratch_size == 0
    if target.backend == "cuda":
        assert compiled.asm.get("ptx")
        assert compiled.asm.get("cubin")
        assert cuda_spill_stores == 0, f"sm{target.arch} spilled {cuda_spill_stores} bytes"
        return
    assert compiled.asm.get("hsaco")
    amdgcn = compiled.asm.get("amdgcn", "")
    vgpr_spills = re.search(r"vgpr_spill[^:]*:\s*(\d+)", amdgcn)
    priv_size = re.search(r"\.private_segment_fixed_size:\s*(\d+)", amdgcn)
    if vgpr_spills is not None:
        assert int(vgpr_spills.group(1)) == 0
    if priv_size is not None:
        assert int(priv_size.group(1)) == 0


def _compile_w4a16_and_assert_clean(
    target: GPUTarget,
    **kwargs: object,
) -> dict[str, triton.compiler.CompiledKernel]:
    spills_by_hash, compile_fn = capture_cuda_spill_stores()
    compiled_kernels = nvfp4_triton.compile_w4a16_kernels(
        target,
        compile_fn=compile_fn,
        **kwargs,  # type: ignore[arg-type]
    )
    assert compiled_kernels
    for kernel in compiled_kernels.values():
        _assert_clean_offline_compile(
            kernel,
            target,
            cuda_spill_stores=spills_by_hash[kernel.hash] if target.backend == "cuda" else None,
        )
    return compiled_kernels


@pytest.mark.parametrize("target", _PRIMARY_ISA_TARGETS)
@pytest.mark.parametrize(
    (
        "rows",
        "out_features",
        "input_features",
        "input_dtype",
        "bias_dtype",
        "high_first",
        "two_level",
        "group_size",
    ),
    [
        (1, 1024, 5120, torch.bfloat16, torch.float32, False, True, 0),
        (1, 5120, 5120, torch.bfloat16, torch.float32, False, True, 0),
        (4, 5120, 5120, torch.float32, torch.float32, False, True, 0),
        (8, 5120, 5120, torch.float16, torch.float16, False, True, 16),
        (16, 5120, 5120, torch.bfloat16, torch.bfloat16, True, True, 0),
        (16, 5120, 5120, torch.float32, torch.float32, False, False, 0),
        (16, 5120, 5120, torch.float16, torch.float16, False, True, 16),
        (64, 5120, 5120, torch.float16, torch.float16, True, False, 0),
        (64, 5120, 5120, torch.float16, torch.float32, False, True, 0),
    ],
)
def test_w4a16_workloads_compile_without_vgpr_spills(
    target: GPUTarget,
    rows: int,
    out_features: int,
    input_features: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
    high_first: bool,
    two_level: bool,
    group_size: int,
) -> None:
    compiled = _compile_w4a16_and_assert_clean(
        target,
        rows=rows,
        out_features=out_features,
        input_features=input_features,
        input_dtype=input_dtype,
        bias_dtype=bias_dtype,
        group_size=group_size,
        high_first=high_first,
        has_weight_per_tensor_scale=two_level,
    )
    roles = set(compiled)
    assert roles <= {"gemv", "dequant", "reduce"}
    assert {"gemv", "dequant"} & roles


_SM120_W4A16_REGIMES = [
    (1, 1024, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (1, 1024, 5120, torch.float16, torch.float16, True, True, 0),
    (1, 4096, 4096, torch.bfloat16, torch.float32, False, True, 0),
    (1, 4096, 14336, torch.bfloat16, torch.float32, False, True, 0),
    (1, 8192, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (4, 5120, 5120, torch.float32, torch.float32, False, True, 0),
    (4, 5120, 5120, torch.float16, torch.float16, False, True, 16),
    (16, 5120, 5120, torch.bfloat16, torch.bfloat16, True, True, 0),
    (32, 1024, 5120, torch.float16, torch.float32, False, False, 0),
    (64, 1024, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (64, 14336, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (64, 5120, 5120, torch.float32, torch.float32, False, True, 0),
    (64, 5120, 5120, torch.float16, torch.float16, False, True, 16),
    (128, 1024, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (256, 4096, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (1024, 5120, 5120, torch.float16, None, True, False, 0),
    (1024, 1024, 8192, torch.bfloat16, torch.float32, False, True, 0),
]

_RDNA4_W4A16_REGIMES = [
    (1, 4096, 512, torch.bfloat16, torch.float32, False, True, 0),
    (1, 4096, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (1, 64, 5120, torch.bfloat16, torch.float16, True, True, 0),
    (2, 4096, 5120, torch.float16, torch.float16, False, True, 0),
    (4, 128, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (4, 4096, 5120, torch.float32, torch.float32, False, True, 0),
    (4, 4096, 5120, torch.bfloat16, torch.float32, False, False, 0),
    (16, 1024, 5120, torch.float16, torch.float32, False, True, 0),
    (16, 4096, 5120, torch.float32, torch.float32, False, True, 0),
    (16, 4096, 5120, torch.float16, torch.float16, False, True, 16),
    (32, 4096, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (64, 4096, 5120, torch.float16, torch.float32, False, True, 0),
    (128, 1024, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (128, 4096, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (256, 4096, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (256, 4096, 5120, torch.float16, None, True, True, 0),
    (1024, 4096, 5120, torch.bfloat16, torch.float32, False, True, 0),
    (1024, 4096, 5120, torch.float16, torch.float16, False, True, 0),
    (2048, 4096, 5120, torch.float16, torch.float16, False, True, 0),
]


@pytest.mark.parametrize(
    ("target", "num_sms", "regimes"),
    [
        pytest.param(GPUTarget("cuda", 120, 32), 170, _SM120_W4A16_REGIMES, id="sm120"),
        pytest.param(
            GPUTarget("hip", "gfx1201", 32),
            32,
            _RDNA4_W4A16_REGIMES,
            marks=pytest.mark.skipif(sys.platform != "linux", reason="HIP compiler is Linux-only"),
            id="gfx1201",
        ),
    ],
)
def test_w4a16_arch_regimes_compile_without_spills(
    target: GPUTarget,
    num_sms: int,
    regimes: list[tuple[int, int, int, torch.dtype, torch.dtype | None, bool, bool, int]],
) -> None:
    is_sm120 = target.backend == "cuda" and target.arch == 120
    seen_roles: set[str] = set()
    for (
        rows,
        out_features,
        input_features,
        input_dtype,
        bias_dtype,
        high_first,
        two_level,
        group_size,
    ) in regimes:
        compiled = _compile_w4a16_and_assert_clean(
            target,
            rows=rows,
            out_features=out_features,
            input_features=input_features,
            input_dtype=input_dtype,
            bias_dtype=bias_dtype,
            group_size=group_size,
            high_first=high_first,
            has_weight_per_tensor_scale=two_level,
            num_sms=num_sms,
        )
        roles = set(compiled)
        assert roles <= {"gemv", "dequant", "reduce"}
        seen_roles |= roles
        if is_sm120 and rows == 1:
            if (out_features, input_features) == (4096, 4096):
                assert "reduce" not in compiled
            elif (out_features, input_features) in ((1024, 5120), (4096, 14336)):
                assert "reduce" in compiled
            ptx = compiled["gemv"].asm["ptx"]
            assert "cvt.rn.f16x2.e2m1x2" in ptx
            assert "cvt.rn.bf16x2.e2m1x2" not in ptx
        if (
            is_sm120
            and rows == 64
            and (out_features, input_features) == (5120, 5120)
            and (
                input_dtype is torch.float32 or (input_dtype is torch.float16 and group_size == 16)
            )
        ):
            assert "dequant" in compiled
            ptx = compiled["dequant"].asm["ptx"]
            assert "cvt.rn.f16x2.e2m1x2" in ptx
            assert "cvt.rn.bf16x2.e2m1x2" not in ptx
    assert {"gemv", "dequant", "reduce"} <= seen_roles


@pytest.mark.parametrize(
    "target",
    [
        _hip_target_param("gfx1100", 32),
        _hip_target_param("gfx1151", 32),
        _hip_target_param("gfx1200", 32),
    ],
)
def test_w4a16_additional_amd_arches_compile_without_spills(target: GPUTarget) -> None:
    compiled = _compile_w4a16_and_assert_clean(
        target,
        rows=16,
        out_features=5120,
        input_features=5120,
        input_dtype=torch.float16,
        bias_dtype=torch.float32,
        group_size=0,
        high_first=False,
        has_weight_per_tensor_scale=True,
    )
    assert "gemv" in compiled
    if target.arch == "gfx1200":
        canary = _compile_w4a16_and_assert_clean(
            target,
            rows=256,
            out_features=4096,
            input_features=5120,
            input_dtype=torch.bfloat16,
            bias_dtype=torch.float32,
            group_size=0,
            high_first=False,
            has_weight_per_tensor_scale=True,
            num_sms=32,
        )
        roles = set(canary)
        assert roles <= {"gemv", "dequant", "reduce"}
        assert {"gemv", "dequant"} & roles


@pytest.mark.parametrize("target", _ADDITIONAL_CUDA_TARGETS)
@pytest.mark.parametrize(
    (
        "rows",
        "out_features",
        "input_features",
        "input_dtype",
        "bias_dtype",
        "high_first",
        "two_level",
        "group_size",
    ),
    [
        (1, 1024, 5120, torch.float16, torch.float16, False, True, 0),
        (16, 5120, 5120, torch.bfloat16, torch.float32, False, True, 0),
        (16, 5120, 5120, torch.float16, torch.float32, False, True, 0),
        (16, 5120, 5120, torch.float32, None, False, False, 0),
        (16, 5120, 5120, torch.float16, torch.float16, False, True, 16),
        (64, 5120, 5120, torch.bfloat16, None, True, True, 0),
    ],
)
def test_w4a16_workloads_compile_for_additional_cuda_arches(
    target: GPUTarget,
    rows: int,
    out_features: int,
    input_features: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
    high_first: bool,
    two_level: bool,
    group_size: int,
) -> None:
    _compile_w4a16_and_assert_clean(
        target,
        rows=rows,
        out_features=out_features,
        input_features=input_features,
        input_dtype=input_dtype,
        bias_dtype=bias_dtype,
        group_size=group_size,
        high_first=high_first,
        has_weight_per_tensor_scale=two_level,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("group_size", [0, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 3, 8, 16, 17, 32, 65, 130])
def test_gpu_w4a16_linear_matches_reference_across_rows_and_dtypes(
    group_size: int,
    dtype: torch.dtype,
    rows: int,
) -> None:
    torch.manual_seed(720 + rows + group_size)
    in_features, out_features = 512, 272
    weight = _make_nvfp4_weight(
        out_features,
        in_features,
        dtype=dtype,
        device="cuda",
        high_first=(rows % 2 == 1),
        two_level=(group_size != 64),
        group_size=group_size,
    )
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    bias_dtype = torch.float32 if rows in (3, 17) else dtype
    bias = None if rows == 8 else torch.randn(out_features, device="cuda", dtype=bias_dtype)

    actual = F.linear(x, weight, bias)
    expected = nvfp4_reference.linear_w4a16(
        x.cpu(),
        weight.qdata.cpu(),
        weight.scale.cpu(),
        None if weight.per_tensor_scale is None else weight.per_tensor_scale.cpu(),
        None if bias is None else bias.cpu(),
        high_first=weight.high_first,
        group_size=group_size,
    ).cuda()

    assert actual.dtype is dtype
    assert actual.shape == (rows, out_features)
    rel_l2 = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert rel_l2 < _tolerance(dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("group_size", [0, 16])
@pytest.mark.parametrize("high_first", [False, True])
def test_gpu_w4a16_handles_noncontiguous_3d_ragged_and_empty_inputs(
    group_size: int,
    high_first: bool,
) -> None:
    weight = _make_nvfp4_weight(
        80,
        144,
        dtype=torch.bfloat16,
        device="cuda",
        high_first=high_first,
        two_level=True,
        group_size=group_size,
        seed=730,
    )
    base = torch.randn(2, 3, 144, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    assert not base.is_contiguous()
    bias = torch.randn(80, device="cuda", dtype=torch.bfloat16)

    actual_3d = F.linear(base, weight, bias)
    expected_3d = nvfp4_reference.linear_w4a16(
        base.cpu(),
        weight.qdata.cpu(),
        weight.scale.cpu(),
        weight.per_tensor_scale.cpu() if weight.per_tensor_scale is not None else None,
        bias.cpu(),
        high_first=high_first,
        group_size=group_size,
    ).cuda()
    assert actual_3d.shape == (3, 2, 80)
    rel_l2 = ((actual_3d.float() - expected_3d.float()).norm() / expected_3d.float().norm()).item()
    assert rel_l2 < _tolerance(torch.bfloat16)

    vec = torch.randn(144, device="cuda", dtype=torch.bfloat16)
    actual_1d = F.linear(vec, weight, bias)
    assert actual_1d.shape == (80,)

    empty_rows = F.linear(torch.empty(0, 144, device="cuda", dtype=torch.bfloat16), weight, bias)
    assert empty_rows.shape == (0, 80)
    assert empty_rows.dtype is torch.bfloat16

    empty_w = (
        ConvRotNVFP4Tensor(
            weight.qdata[:0],
            weight.scale[:0],
            weight.block_size,
            weight.dtype,
            group_size,
            weight.per_tensor_scale,
            None,
            True,
            None,
            high_first=high_first,
        )
        if isinstance(weight, ConvRotNVFP4Tensor)
        else PiperNVFP4Tensor(
            weight.qdata[:0],
            weight.scale[:0],
            weight.block_size,
            weight.dtype,
            weight.per_tensor_scale,
            None,
            True,
            None,
            high_first=high_first,
        )
    )
    empty_out = F.linear(torch.randn(2, 144, device="cuda", dtype=torch.bfloat16), empty_w)
    assert empty_out.shape == (2, 0)
    assert empty_out.dtype is torch.bfloat16

    strided_bias = torch.randn(160, device="cuda", dtype=torch.float32)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        F.linear(torch.randn(4, 144, device="cuda", dtype=torch.bfloat16), weight, strided_bias)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("rows", [1, 3, 8, 16, 32, 64])
@pytest.mark.parametrize("group_size", [0, 256])
def test_gpu_w4a16_splitk_is_deterministic_and_accurate(rows: int, group_size: int) -> None:
    torch.manual_seed(740 + rows + group_size)
    weight = _make_nvfp4_weight(
        512,
        16_384,
        dtype=torch.bfloat16,
        device="cuda",
        two_level=True,
        group_size=group_size,
    )
    x = torch.randn(rows, 16_384, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(512, device="cuda", dtype=torch.float32)

    first = F.linear(x, weight, bias)
    second = F.linear(x, weight, bias)
    expected = nvfp4_reference.linear_w4a16(
        x.cpu(),
        weight.qdata.cpu(),
        weight.scale.cpu(),
        weight.per_tensor_scale.cpu() if weight.per_tensor_scale is not None else None,
        bias.cpu(),
        group_size=group_size,
    ).cuda()

    assert torch.equal(first, second)
    rel_l2 = ((first.float() - expected.float()).norm() / expected.float().norm()).item()
    assert rel_l2 < _tolerance(torch.bfloat16)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_w4a16_m1_square_shape_matches_reference() -> None:
    torch.manual_seed(745)
    weight = _make_nvfp4_weight(
        4096,
        4096,
        dtype=torch.bfloat16,
        device="cuda",
        two_level=True,
    )
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    actual = F.linear(x, weight, bias)
    expected = nvfp4_reference.linear_w4a16(
        x.cpu(),
        weight.qdata.cpu(),
        weight.scale.cpu(),
        weight.per_tensor_scale.cpu() if weight.per_tensor_scale is not None else None,
        bias.cpu(),
    ).cuda()

    assert actual.dtype is torch.bfloat16
    assert actual.shape == (1, 4096)
    rel_l2 = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert rel_l2 < _tolerance(torch.bfloat16)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("high_first", [False, True])
@pytest.mark.parametrize("two_level", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gpu_large_m_w4a16_identity_matches_dequantize(
    high_first: bool,
    two_level: bool,
    output_dtype: torch.dtype,
) -> None:
    weight = _make_nvfp4_weight(
        80,
        144,
        dtype=output_dtype,
        device="cuda",
        high_first=high_first,
        two_level=two_level,
        seed=750,
    )
    actual = F.linear(torch.eye(144, device="cuda", dtype=output_dtype), weight).T
    q_low = ((weight.qdata & 0x0F) << 4) | (weight.qdata >> 4) if high_first else weight.qdata
    expected = PiperNVFP4Tensor(
        q_low,
        weight.scale,
        16,
        output_dtype,
        per_tensor_scale=weight.per_tensor_scale,
        is_swizzled_scales=True,
    ).dequantize(output_dtype)
    assert actual.dtype is output_dtype
    rtol = 1.6e-2 if output_dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("rows", [1, 17, 65])
def test_gpu_w4a16_mixed_bias_avoids_pre_bias_fp16_overflow(rows: int) -> None:
    dense = torch.ones(16, 32, dtype=torch.float16)
    weight = PiperNVFP4Tensor.from_hp(
        dense,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_quant=None,
    ).to("cuda")
    x = torch.full((rows, 32), 3000.0, device="cuda", dtype=torch.float16)
    bias = torch.full((16,), -40000.0, device="cuda", dtype=torch.float32)

    actual = F.linear(x, weight, bias)
    expected = nvfp4_reference.linear_w4a16(
        x.cpu(),
        weight.qdata.cpu(),
        weight.scale.cpu(),
        weight.per_tensor_scale.cpu() if weight.per_tensor_scale is not None else None,
        bias.cpu(),
    ).cuda()
    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("rows", [1, 8, 17])
def test_gpu_w4a16_convrot_fp16_preserves_rotation_range(rows: int) -> None:
    pattern = (
        torch.tensor(
            [1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1],
            dtype=torch.float16,
        )
        * 20000.0
    )
    x = pattern.expand(rows, 16).contiguous().cuda()
    qdata = torch.full((16, 8), 0x22, dtype=torch.uint8, device="cuda")
    scale = torch.ones((32, 16), dtype=torch.float8_e4m3fn, device="cuda")
    per_tensor_scale = torch.tensor(1.0 / 16.0, dtype=torch.float32, device="cuda")
    weight = ConvRotNVFP4Tensor(
        qdata, scale, 16, torch.float16, 16, per_tensor_scale, None, True, None
    )

    actual = F.linear(x, weight)
    expected = nvfp4_reference.linear_w4a16(
        x.cpu(),
        qdata.cpu(),
        scale.cpu(),
        per_tensor_scale.cpu(),
        None,
        group_size=16,
    ).cuda()
    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_w4a16_large_m_preserves_weight_scale_and_gemm_range() -> None:
    # Case 1: large global weight scale with small FP16 activations.
    q_small = torch.full((16, 8), 0x22, dtype=torch.uint8, device="cuda")
    s_ones = torch.ones((32, 16), dtype=torch.float8_e4m3fn, device="cuda")
    gs_huge = torch.tensor(100_000.0, dtype=torch.float32, device="cuda")
    w_huge = PiperNVFP4Tensor(q_small, s_ones, 16, torch.float16, gs_huge, None, True, None)
    x_small = torch.full((17, 16), 0.001, device="cuda", dtype=torch.float16)
    actual_1 = F.linear(x_small, w_huge)
    expected_1 = nvfp4_reference.linear_w4a16(
        x_small.cpu(), q_small.cpu(), s_ones.cpu(), gs_huge.cpu(), None
    ).cuda()
    assert actual_1.dtype is torch.float16
    assert torch.isfinite(actual_1).all()
    torch.testing.assert_close(actual_1, expected_1, rtol=1e-2, atol=1.0)

    # Case 2: large unbiased FP16 dot product scaled down by fractional global scale.
    q_wide = torch.full((16, 16), 0x22, dtype=torch.uint8, device="cuda")
    gs_frac = torch.tensor(1.0 / 16.0, dtype=torch.float32, device="cuda")
    w_frac = PiperNVFP4Tensor(q_wide, s_ones, 16, torch.float16, gs_frac, None, True, None)
    x_large = torch.full((17, 32), 3000.0, device="cuda", dtype=torch.float16)
    actual_2 = F.linear(x_large, w_frac)
    expected_2 = nvfp4_reference.linear_w4a16(
        x_large.cpu(), q_wide.cpu(), s_ones.cpu(), gs_frac.cpu(), None
    ).cuda()
    assert actual_2.dtype is torch.float16
    assert torch.isfinite(actual_2).all()
    torch.testing.assert_close(actual_2, expected_2, rtol=1e-3, atol=1e-3)

    # Case 3: extreme BF16 product of block scale and global scale with tiny activations.
    s_max = torch.full((32, 16), 448.0, dtype=torch.float8_e4m3fn, device="cuda")
    gs_extreme = torch.tensor(1e37, dtype=torch.float32, device="cuda")
    w_extreme = PiperNVFP4Tensor(q_small, s_max, 16, torch.bfloat16, gs_extreme, None, True, None)
    x_tiny = torch.full((17, 16), 1e-37, device="cuda", dtype=torch.bfloat16)
    actual_3 = F.linear(x_tiny, w_extreme)
    expected_3 = nvfp4_reference.linear_w4a16(
        x_tiny.cpu(), q_small.cpu(), s_max.cpu(), gs_extreme.cpu(), None
    ).cuda()
    assert actual_3.dtype is torch.bfloat16
    assert torch.isfinite(actual_3).all()
    torch.testing.assert_close(actual_3, expected_3, rtol=1.6e-2, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("group_size", [0, 64])
@pytest.mark.parametrize("rows", [1, 8, 17, 65])
def test_gpu_w4a16_torch_compile_fullgraph(group_size: int, rows: int) -> None:
    torch._dynamo.reset()
    weight = _make_nvfp4_weight(
        128,
        256,
        dtype=torch.bfloat16,
        device="cuda",
        group_size=group_size,
        seed=760 + rows + group_size,
    )
    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)

    def projection(inp: torch.Tensor) -> torch.Tensor:
        return F.linear(inp, weight, bias)

    with torch.inference_mode():
        eager = projection(x)
        compiled = torch.compile(projection, fullgraph=True, options=compile_options())(x)
    assert torch.equal(compiled, eager)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("rows", [1, 2, 8, 17, 65, 512])
def test_gpu_w4a16_preserves_zero_and_tiny_global_scales(rows: int) -> None:
    q = torch.full((16, 8), 0x77, dtype=torch.uint8, device="cuda")
    s_ones = torch.ones((32, 16), dtype=torch.float8_e4m3fn, device="cuda")
    x_huge = torch.full((rows, 16), 1e37, device="cuda", dtype=torch.float32)

    w_zero = PiperNVFP4Tensor(
        q, s_ones, 16, torch.float32, torch.tensor(0.0, device="cuda"), None, True, None
    )
    assert torch.equal(F.linear(x_huge, w_zero), torch.zeros(rows, 16, device="cuda"))

    gs_tiny = torch.tensor(1e-30, device="cuda", dtype=torch.float32)
    w_tiny = PiperNVFP4Tensor(q, s_ones, 16, torch.float32, gs_tiny, None, True, None)
    actual_tiny = F.linear(x_huge, w_tiny)
    expected_tiny = nvfp4_reference.linear_w4a16(
        x_huge.cpu(), q.cpu(), s_ones.cpu(), gs_tiny.cpu(), None
    ).cuda()
    assert torch.isfinite(actual_tiny).all()
    torch.testing.assert_close(actual_tiny, expected_tiny, rtol=1e-4, atol=1e-4)

    # Scaling x rather than block scales preserves representable tiny weights (3 * 2^-149).
    s_rep = torch.full((32, 16), 2.0**-9, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    gs_rep = torch.tensor(2.0**-141, device="cuda", dtype=torch.float32)
    w_rep = PiperNVFP4Tensor(q, s_rep, 16, torch.float32, gs_rep, None, True, None)
    x_rep = torch.full((rows, 16), 3e38, device="cuda", dtype=torch.float32)
    actual_rep = F.linear(x_rep, w_rep)
    expected_rep = nvfp4_reference.linear_w4a16(
        x_rep.cpu(), q.cpu(), s_rep.cpu(), gs_rep.cpu(), None
    ).cuda()
    assert float(actual_rep.abs().min()) > 0.0
    torch.testing.assert_close(actual_rep, expected_rep, rtol=1e-4, atol=1e-6)

    if rows >= 2:
        gs_fp16 = torch.tensor(1e-8, device="cuda", dtype=torch.float32)
        s_max = torch.full((32, 16), 448.0, device="cuda", dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        w_fp16 = PiperNVFP4Tensor(q, s_max, 16, torch.float16, gs_fp16, None, True, None)
        x_fp16 = torch.ones((rows, 16), device="cuda", dtype=torch.float16)
        actual_fp16 = F.linear(x_fp16, w_fp16)
        expected_fp16 = nvfp4_reference.linear_w4a16(
            x_fp16.cpu(), q.cpu(), s_max.cpu(), gs_fp16.cpu(), None
        ).cuda()
        assert float(actual_fp16.abs().min()) > 0.0
        torch.testing.assert_close(actual_fp16, expected_fp16, rtol=1e-2, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(("rows", "out_features"), [(256, 4096), (512, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("high_first", [False, True])
def test_gpu_w4a16_wide_2d_gemm_matches_reference(
    rows: int,
    out_features: int,
    dtype: torch.dtype,
    high_first: bool,
) -> None:
    torch.manual_seed(770 + rows + int(high_first))
    in_features = 256
    weight = _make_nvfp4_weight(
        out_features,
        in_features,
        dtype=dtype,
        device="cuda",
        high_first=high_first,
        two_level=True,
        seed=771,
    )
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    bias = torch.randn(out_features, device="cuda", dtype=torch.float32)

    actual = F.linear(x, weight, bias)
    expected = nvfp4_reference.linear_w4a16(
        x.cpu(),
        weight.qdata.cpu(),
        weight.scale.cpu(),
        None if weight.per_tensor_scale is None else weight.per_tensor_scale.cpu(),
        bias.cpu(),
        high_first=high_first,
    ).cuda()
    assert actual.dtype is dtype
    assert actual.shape == (rows, out_features)
    rel_l2 = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert rel_l2 < _tolerance(dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("high_first", [False, True])
def test_gpu_w4a16_deep_k_bf16_matches_dequantized_linear(high_first: bool) -> None:
    torch.manual_seed(790 + int(high_first))
    rows, out_features, in_features = 1024, 1024, 8192
    weight = _make_nvfp4_weight(
        out_features,
        in_features,
        dtype=torch.bfloat16,
        device="cuda",
        high_first=high_first,
        two_level=True,
        seed=791,
    )
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(out_features, device="cuda", dtype=torch.float32)

    actual = F.linear(x, weight, bias)
    expected = F.linear(x.float(), weight.dequantize(torch.float32), bias.float())
    assert actual.dtype is torch.bfloat16
    assert actual.shape == (rows, out_features)
    rel_l2 = ((actual.float() - expected).norm() / expected.norm()).item()
    assert rel_l2 < _tolerance(torch.bfloat16)


def test_meta_nvfp4_linear_rejects_scalar_input() -> None:
    q = torch.empty(16, 8, dtype=torch.uint8, device="meta")
    s = torch.empty(32, 16, dtype=torch.float8_e4m3fn, device="meta")
    x0 = torch.empty((), dtype=torch.bfloat16, device="meta")

    with pytest.raises(ValueError, match="non-scalar"):
        torch.ops.piper_kernels.nvfp4_linear.default(x0, q, s, None, None, None, False, False, True)
    with pytest.raises(ValueError, match="non-scalar"):
        torch.ops.piper_kernels.convrot_nvfp4_linear.default(
            x0, q, s, None, None, None, False, 16, False, True
        )


def test_reference_w4a16_preserves_precision_under_autocast() -> None:
    pattern = (
        torch.tensor(
            [1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1],
            dtype=torch.float16,
        )
        * 20000.0
    )
    x = pattern.expand(4, 16).contiguous()
    qdata = torch.full((16, 8), 0x22, dtype=torch.uint8)
    scale = torch.ones((32, 16), dtype=torch.float8_e4m3fn)
    per_tensor_scale = torch.tensor(1.0 / 16.0, dtype=torch.float32)

    expected = nvfp4_reference.linear_w4a16(x, qdata, scale, per_tensor_scale, None, group_size=16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = nvfp4_reference.linear_w4a16(
            x, qdata, scale, per_tensor_scale, None, group_size=16
        )
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_w4a16_convrot_fp16_under_autocast_preserves_range() -> None:
    pattern = (
        torch.tensor(
            [1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1],
            dtype=torch.float16,
        )
        * 20000.0
    )
    x = pattern.expand(17, 16).contiguous().cuda()
    qdata = torch.full((16, 8), 0x22, dtype=torch.uint8, device="cuda")
    scale = torch.ones((32, 16), dtype=torch.float8_e4m3fn, device="cuda")
    per_tensor_scale = torch.tensor(1.0 / 16.0, dtype=torch.float32, device="cuda")
    weight = ConvRotNVFP4Tensor(
        qdata, scale, 16, torch.float16, 16, per_tensor_scale, None, True, None
    )

    with torch.autocast("cuda", dtype=torch.float16):
        actual = F.linear(x, weight)
    expected = nvfp4_reference.linear_w4a16(
        x.cpu(), qdata.cpu(), scale.cpu(), per_tensor_scale.cpu(), None, group_size=16
    ).cuda()
    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("input_features", [16, 48, 80])
@pytest.mark.parametrize("rows", [1, 16, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gpu_w4a16_ignores_unaligned_scale_padding(
    input_features: int,
    rows: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(910 + input_features + rows)
    out_features = 130
    weight = _make_nvfp4_weight(
        out_features,
        input_features,
        dtype=dtype,
        device="cuda",
        high_first=(rows % 2 == 1),
        seed=911 + input_features,
    )
    valid_r = torch.arange(out_features, device="cuda")[:, None]
    valid_c = torch.arange(input_features // 16, device="cuda")[None, :]
    scale_col_blocks = (input_features + 63) // 64
    valid_offsets = (
        (((valid_r // 128) * scale_col_blocks + (valid_c // 4)) * 32 + (valid_r % 128) % 32) * 16
        + ((valid_r % 128) // 32) * 4
        + (valid_c % 4)
    )
    flat_valid_offsets = valid_offsets.reshape(-1)
    flat_scale = weight.scale.view(torch.uint8).reshape(-1)
    valid_mask = torch.zeros(flat_scale.numel(), dtype=torch.bool, device="cuda")
    valid_mask[flat_valid_offsets] = True
    assert torch.all(flat_scale[~valid_mask] == 0)

    # Fill unused swizzled scale lanes with alternating FP8 NaN (0x7F) and max finite (0x7E).
    pad_u8 = torch.where(
        (torch.arange(flat_scale.numel(), device="cuda") & 1) == 0,
        0x7F,
        0x7E,
    ).to(torch.uint8)
    pad_u8[flat_valid_offsets] = flat_scale[flat_valid_offsets]
    padded_scale = pad_u8.view(weight.scale.shape).view(torch.float8_e4m3fn)
    padded_weight = PiperNVFP4Tensor(
        weight.qdata,
        padded_scale,
        weight.block_size,
        weight.dtype,
        weight.per_tensor_scale,
        weight.act_per_tensor_scale,
        weight.is_swizzled_scales,
        weight.act_quant,
        high_first=weight.high_first,
    )

    x = torch.randn(rows, input_features, device="cuda", dtype=dtype)
    actual = F.linear(x, padded_weight)
    expected = nvfp4_reference.linear_w4a16(
        x.cpu(),
        weight.qdata.cpu(),
        weight.scale.cpu(),
        None if weight.per_tensor_scale is None else weight.per_tensor_scale.cpu(),
        None,
        high_first=weight.high_first,
    ).cuda()
    assert actual.dtype is dtype
    assert actual.shape == (rows, out_features)
    assert torch.isfinite(actual).all()
    rel_l2 = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert rel_l2 < _tolerance(dtype)
