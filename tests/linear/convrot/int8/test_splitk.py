"""Contract and GPU runtime tests for small-M ConvRot INT8 W8A8 Split-K execution."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import pytest
import torch
import triton
from _accelerators import (
    exact_gfx1201_available,
    exact_sm120_available,
    sm8x_available,
)
from torch.nn import functional as F  # noqa: N812
from triton.compiler.compiler import LazyDict

import piper_kernels.linear.convrot  # noqa: F401 - register ConvRot custom ops
from piper_kernels import compile_options
from piper_kernels.linear.convrot.int8 import reference
from piper_kernels.linear.convrot.int8.triton import select_int8_split_k
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_TUNED_SPLITK_AVAILABLE = exact_gfx1201_available() or exact_sm120_available() or sm8x_available()


def _record_triton_launches(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, list[str]]:
    launched: list[str] = []

    def _record_launch(metadata: LazyDict) -> None:
        data = metadata.get()
        if isinstance(data, dict) and "name" in data:
            launched.append(str(data["name"]))

    triton.knobs.runtime.launch_enter_hook.add(_record_launch)
    try:
        output = fn()
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(_record_launch)
    return output, launched


def test_select_int8_splitk_contracts() -> None:
    # Batches above 16 rows must stay on the single unsplit GEMM.
    assert (
        select_int8_split_k(
            32,
            rows=17,
            out_features=512,
            in_features=16384,
            block_m=32,
            block_n=128,
            block_k=64,
        )
        == 1
    )
    assert (
        select_int8_split_k(
            32,
            rows=32,
            out_features=2048,
            in_features=4096,
            block_m=32,
            block_n=128,
            block_k=64,
        )
        == 1
    )
    # Nearly-full waves (> num_sms // 2 output tiles) must not split into two waves.
    assert (
        select_int8_split_k(
            32,
            rows=1,
            out_features=3840,
            in_features=16384,
            block_m=16,
            block_n=128,
            block_k=64,
        )
        == 1
    )
    # Uneven K partition on 170-SM SM120 selects 15 splits for 97 K-tiles (zero empty splits).
    assert (
        select_int8_split_k(
            170,
            rows=1,
            out_features=704,
            in_features=12416,
            block_m=32,
            block_n=64,
            block_k=128,
        )
        == 15
    )
    # Paired projections double column output tiles when sizing Split-K.
    assert (
        select_int8_split_k(
            32,
            rows=1,
            out_features=512,
            in_features=16384,
            block_m=16,
            block_n=64,
            block_k=256,
            paired=True,
        )
        == 2
    )
    # Extreme K depth exceeding the signed INT32 per-slice bound disables Split-K.
    assert (
        select_int8_split_k(
            170,
            rows=1,
            out_features=2,
            in_features=4_194_304,
            block_m=32,
            block_n=64,
            block_k=128,
        )
        == 1
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features"),
    [
        (1, 16384, 512),
        (1, 16384, 1024),
        (1, 16640, 512),
        (1, 16896, 512),
        (1, 17152, 512),
        (4, 16384, 512),
        (16, 16384, 512),
        (1, 16384, 515),
        (3, 16400, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [None, torch.float16, torch.bfloat16, torch.float32])
def test_gpu_splitk_linear_matches_reference(
    rows: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    bias_dtype: torch.dtype | None,
) -> None:
    torch.manual_seed(410 + rows + out_features)
    group_size = 256 if in_features % 256 == 0 else 16
    qdata = torch.randint(-127, 128, (out_features, in_features), device="cuda", dtype=torch.int8)
    scale = torch.rand((out_features, 1), device="cuda", dtype=torch.float32) * 0.01
    bias = (
        None
        if bias_dtype is None
        else torch.randn(out_features, device="cuda", dtype=bias_dtype) * 0.1
    )
    weight = ConvRotInt8Tensor.from_quantized(
        qdata, scale, group_size=group_size, logical_dtype=dtype
    )
    activation = torch.randn((rows, in_features), device="cuda", dtype=dtype)

    actual = F.linear(activation, weight, bias)
    prepared_q, prepared_scale = torch.ops.piper_kernels.convrot_int8_prepare_input(
        activation, group_size
    )
    expected = reference.linear_prepared(
        prepared_q.cpu(),
        prepared_scale.cpu(),
        qdata.cpu(),
        scale.cpu(),
        dtype,
        None if bias is None else bias.cpu(),
    ).cuda()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=2 * torch.finfo(dtype).eps,
        atol=1e-6,
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not _TUNED_SPLITK_AVAILABLE,
    reason="requires tuned INT8 Split-K target (gfx1201, SM8x, or SM120)",
)
def test_gpu_splitk_public_linear_launches_splitk_kernels() -> None:
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if sm_count < 8:
        pytest.skip("requires at least 8 SMs or WGPs for INT8 Split-K")
    torch.manual_seed(415)
    out_features, in_features = (512, 16384) if sm_count >= 16 else (256, 32768)
    qdata = torch.randint(-127, 128, (out_features, in_features), device="cuda", dtype=torch.int8)
    scale = torch.rand((out_features, 1), device="cuda", dtype=torch.float32) * 0.01
    weight = ConvRotInt8Tensor.from_quantized(
        qdata, scale, group_size=256, logical_dtype=torch.bfloat16
    )
    activation = torch.randn((1, in_features), device="cuda", dtype=torch.bfloat16)

    # Warm up compilation before recording a clean single execution.
    F.linear(activation, weight)
    actual, launched = _record_triton_launches(lambda: F.linear(activation, weight))

    assert "int8_matmul_splitk_kernel" in launched
    assert "int8_splitk_reduce_kernel" in launched

    prepared_q, prepared_scale = torch.ops.piper_kernels.convrot_int8_prepare_input(activation, 256)
    expected = reference.linear_prepared(
        prepared_q.cpu(),
        prepared_scale.cpu(),
        qdata.cpu(),
        scale.cpu(),
        torch.bfloat16,
        None,
    ).cuda()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=1e-6,
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not _TUNED_SPLITK_AVAILABLE,
    reason="requires tuned INT8 Split-K target (gfx1201, SM8x, or SM120)",
)
def test_gpu_splitk_exact_integer_sum_matches_reference() -> None:
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if sm_count < 8:
        pytest.skip("requires at least 8 SMs or WGPs for INT8 Split-K")
    # Use 256 columns (4 N-tiles) so every tuned device with >= 8 SMs splits K,
    # keeping [0..2048) within slice 0 (up to 16 splits) for +33,276,033 (> 2^24).
    in_features, out_features = 32768, 256
    input_q = torch.zeros((1, in_features), device="cuda", dtype=torch.int8)
    weight_q = torch.zeros((out_features, in_features), device="cuda", dtype=torch.int8)

    # Slice 0 [0..2048): +33,276,033 (odd > 2^24, rounds in FP32 partial summation)
    input_q[:, :2047] = -128
    weight_q[:, :2047] = -127
    input_q[:, 2047] = 1
    weight_q[:, 2047] = 1
    # Second half [16384..18431): -33,276,032
    input_q[:, 16384:18431] = -128
    weight_q[:, 16384:18431] = 127
    # Tail [32761..32768): +7
    input_q[:, 32761:32768] = 1
    weight_q[:, 32761:32768] = 1

    input_scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    weight_scale = torch.ones((out_features,), device="cuda", dtype=torch.float32)

    torch.ops.piper_kernels.convrot_int8_linear_prepared(
        input_q, input_scale, weight_q, weight_scale, None, torch.float32
    )
    actual, launched = _record_triton_launches(
        lambda: torch.ops.piper_kernels.convrot_int8_linear_prepared(
            input_q, input_scale, weight_q, weight_scale, None, torch.float32
        )
    )
    assert "int8_matmul_splitk_kernel" in launched
    assert "int8_splitk_reduce_kernel" in launched
    expected = reference.linear_prepared(
        input_q.cpu(), input_scale.cpu(), weight_q.cpu(), weight_scale.cpu(), torch.float32, None
    ).cuda()
    assert torch.equal(actual, torch.full_like(actual, 8.0))
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_gpu_splitk_torch_compile_fullgraph() -> None:
    torch._dynamo.reset()
    torch.manual_seed(420)
    in_features, out_features = 16384, 512
    qdata = torch.randint(-127, 128, (out_features, in_features), device="cuda", dtype=torch.int8)
    scale = torch.rand((out_features, 1), device="cuda", dtype=torch.float32) * 0.01
    bias = torch.randn(out_features, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = ConvRotInt8Tensor.from_quantized(
        qdata, scale, group_size=256, logical_dtype=torch.bfloat16
    )
    activation = torch.randn((1, in_features), device="cuda", dtype=torch.bfloat16)

    def projection(inp: torch.Tensor) -> torch.Tensor:
        return F.linear(inp, weight, bias)

    with torch.inference_mode():
        eager = projection(activation)
        compiled = torch.compile(projection, fullgraph=True, options=compile_options())(activation)
    assert torch.equal(compiled, eager)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two GPUs")
def test_gpu_splitk_noncurrent_device_preserves_caller_device() -> None:
    torch.manual_seed(430)
    in_features, out_features = 16384, 512
    with torch.cuda.device(0):
        target_device = torch.device("cuda", 1)
        qdata = torch.randint(
            -127, 128, (out_features, in_features), device=target_device, dtype=torch.int8
        )
        scale = torch.rand((out_features, 1), device=target_device, dtype=torch.float32) * 0.01
        weight = ConvRotInt8Tensor.from_quantized(
            qdata, scale, group_size=256, logical_dtype=torch.bfloat16
        )
        activation = torch.randn((1, in_features), device=target_device, dtype=torch.bfloat16)
        actual = F.linear(activation, weight)
        assert torch.cuda.current_device() == 0
        assert actual.device == target_device
        prepared_q, prepared_scale = torch.ops.piper_kernels.convrot_int8_prepare_input(
            activation, 256
        )
        expected = reference.linear_prepared(
            prepared_q.cpu(),
            prepared_scale.cpu(),
            qdata.cpu(),
            scale.cpu(),
            torch.bfloat16,
            None,
        ).to(target_device)
        assert torch.equal(actual, expected)


def test_cpu_convrot_int8_linear_runs_without_triton() -> None:
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

from piper_kernels.linear.convrot.int8 import reference
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

torch.manual_seed(440)
qdata = torch.randint(-127, 128, (64, 256), dtype=torch.int8)
scale = torch.rand((64, 1), dtype=torch.float32) * 0.01
bias = torch.randn(64, dtype=torch.bfloat16) * 0.1
weight = ConvRotInt8Tensor.from_quantized(
    qdata, scale, group_size=256, logical_dtype=torch.bfloat16
)
activation = torch.randn((3, 256), dtype=torch.bfloat16)
actual = F.linear(activation, weight, bias)
expected = reference.linear(activation, qdata, scale, 256, bias)
assert torch.equal(actual, expected)
assert "triton" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
