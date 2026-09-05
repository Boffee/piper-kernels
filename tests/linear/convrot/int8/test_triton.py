"""Shared custom-op tests and NVIDIA-only legacy low-level utility tests."""

from dataclasses import replace

import pytest
import torch
import triton
import triton.language as tl
from torch import nn

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot import (
    ConvRotInt8Tensor,
    convrot_int8_compile_options,
    convrot_int8_linear,
)
from piper_kernels.linear.convrot._rotation import rotate_groups
from piper_kernels.linear.convrot.int8 import triton as triton_backend
from piper_kernels.linear.convrot.int8._policy import select_execution_plan
from piper_kernels.linear.convrot.int8.reference import add_ as reference_add_
from piper_kernels.linear.convrot.int8.reference import (
    addmm_,
    linear,
)


@triton.jit
def _scaled_projection_epilogue_probe(  # noqa: PLR0913, PLR0917
    input_ptr,
    weight_ptr,
    input_scale_ptr,
    weight_scale_ptr,
    output_ptr,
    rows,
    out_features,
    in_features,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Exercise ConvRot's reusable scaled accumulator with a non-linear epilogue."""
    offsets_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    offsets_n = tl.program_id(1) * block_n + tl.arange(0, block_n)
    projected = triton_backend.scaled_int8_matmul(
        input_ptr,
        weight_ptr,
        input_scale_ptr,
        weight_scale_ptr,
        offsets_m,
        offsets_n,
        rows,
        out_features,
        in_features,
        block_m,
        block_n,
        block_k,
        False,
    )
    # A future attention projection substitutes normalization, RoPE, and
    # quantization here. This probe makes sure the reusable boundary returns an
    # accumulator rather than owning the ordinary BF16 store.
    projected = projected * projected + 0.25
    output_offsets = offsets_m[:, None] * out_features + offsets_n[None, :]
    tl.store(
        output_ptr + output_offsets,
        projected,
        mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < out_features),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_triton_linear_matches_gpu_reference(group_size: int) -> None:
    torch.manual_seed(9)
    in_features = 2 * group_size
    qdata = torch.randint(-127, 128, (96, in_features), dtype=torch.int8, device="cuda")
    scale = torch.rand(96, 1, dtype=torch.float32, device="cuda") * 0.01
    wrapped = ConvRotInt8Tensor.from_quantized(qdata, scale, group_size=group_size)
    activation = torch.randn(37, in_features, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(96, dtype=torch.bfloat16, device="cuda")

    expected = linear(activation, qdata, scale, group_size, bias)
    actual = torch.nn.functional.linear(activation, wrapped, bias)
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
def test_triton_linear_supports_mixed_precision_bias(
    bias_dtype: torch.dtype,
    compiled: bool,
) -> None:
    torch.manual_seed(154)
    in_features, out_features = 256, 96
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    weight = ConvRotInt8Tensor.from_quantized(qdata, scale, group_size=256)
    activation = torch.randn(37, in_features, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(out_features, dtype=bias_dtype, device="cuda")

    def projection(value: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, weight, offset)

    expected = linear(activation, qdata, scale, 256, bias)
    call = (
        torch.compile(projection, fullgraph=True, options=convrot_int8_compile_options())
        if compiled
        else projection
    )
    actual = call(activation, bias)

    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_scaled_projection_tile_supports_specialized_epilogues() -> None:
    torch.manual_seed(153)
    rows, out_features, in_features = 19, 23, 48
    input_qdata = torch.randint(
        -127,
        128,
        (rows, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    weight_qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    input_scale = torch.rand(rows, dtype=torch.float32, device="cuda") * 0.01
    weight_scale = torch.rand(out_features, dtype=torch.float32, device="cuda") * 0.01
    actual = torch.empty(rows, out_features, dtype=torch.float32, device="cuda")
    block_m, block_n, block_k = 16, 32, 32

    _scaled_projection_epilogue_probe[
        (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    ](
        input_qdata,
        weight_qdata,
        input_scale,
        weight_scale,
        actual,
        rows,
        out_features,
        in_features,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )

    # FP32 exactly represents this short INT8 reduction on CUDA.
    accumulated = input_qdata.float() @ weight_qdata.T.float()
    projected = accumulated * input_scale[:, None] * weight_scale[None, :]
    expected = projected * projected + 0.25
    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_factorized_h4_rotation_matches_gpu_reference(
    group_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(41 + group_size)
    activation = torch.randn(5, 2 * group_size, dtype=dtype, device="cuda")
    actual = torch.empty_like(activation)

    triton_backend.rotate_input(activation, actual, group_size, num_warps=4)
    expected = rotate_groups(activation, group_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_factorized_h4_rotation_handles_rounding_boundary_values(
    group_size: int,
    dtype: torch.dtype,
) -> None:
    one = torch.tensor(1.0, dtype=dtype, device="cuda")
    positive = torch.nextafter(one, torch.tensor(float("inf"), dtype=dtype, device="cuda"))
    negative = torch.nextafter(one, torch.tensor(float("-inf"), dtype=dtype, device="cuda"))
    pattern = torch.stack((one, positive, negative, -one, -positive, -negative, one, -one))
    width = 2 * group_size
    activation = pattern.repeat((width + pattern.numel() - 1) // pattern.numel())[:width]
    activation = activation.reshape(1, width)
    actual = torch.empty_like(activation)

    triton_backend.rotate_input(activation, actual, group_size, num_warps=4)
    expected = rotate_groups(activation, group_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "in_features",
    [512, 5_376, 7_168, 9_728, 14_336, 16_640, 28_672, 40_960, 49_152],
)
@pytest.mark.parametrize(
    ("dtype", "dtype_code"),
    [(torch.float16, 1), (torch.bfloat16, 2)],
)
@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA-only low-level utility")
def test_fused_rotation_quantization_matches_split_path_exactly(
    in_features: int,
    dtype: torch.dtype,
    dtype_code: int,
) -> None:
    torch.manual_seed(63)
    rows = 7
    activation = torch.randn(rows, in_features, dtype=dtype, device="cuda")
    rotated = torch.empty_like(activation)
    expected_qdata = torch.empty_like(activation, dtype=torch.int8)
    expected_scale = torch.empty(rows, dtype=torch.float32, device="cuda")
    triton_backend.rotate_input(activation, rotated, 256, num_warps=4)
    triton_backend.quantize_input(
        rotated,
        expected_qdata,
        expected_scale,
        dtype_code,
        num_warps=8,
    )

    actual_qdata = torch.empty_like(expected_qdata)
    actual_scale = torch.empty_like(expected_scale)
    triton_backend.fused_rotate_quantize_input(
        activation,
        actual_qdata,
        actual_scale,
        256,
        dtype_code,
        num_warps=select_execution_plan(
            AcceleratorTarget.from_device(activation.device), in_features=in_features
        ).fused_num_warps,
    )

    assert torch.equal(actual_qdata, expected_qdata)
    assert torch.equal(actual_scale, expected_scale)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "in_features",
    [512, 5_376, 9_728, 14_336, 16_640, 28_672, 40_960, 49_152],
)
@pytest.mark.parametrize(
    ("dtype", "dtype_code"),
    [(torch.float16, 1), (torch.bfloat16, 2)],
)
@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA-only low-level utility")
def test_fused_up_gate_swiglu_preparation_matches_materialized_path(
    in_features: int,
    dtype: torch.dtype,
    dtype_code: int,
) -> None:
    torch.manual_seed(75)
    rows = 7
    raw_activation = torch.randn(rows, 2 * in_features, dtype=dtype, device="cuda")
    up, gate = raw_activation.chunk(2, dim=-1)
    activation = up * torch.nn.functional.silu(gate)
    rotated = torch.empty_like(activation)
    expected_qdata = torch.empty_like(activation, dtype=torch.int8)
    expected_scale = torch.empty(rows, dtype=torch.float32, device="cuda")
    triton_backend.rotate_input(activation, rotated, 256, num_warps=4)
    triton_backend.quantize_input(
        rotated,
        expected_qdata,
        expected_scale,
        dtype_code,
        num_warps=8,
    )

    actual_qdata = torch.empty_like(expected_qdata)
    actual_scale = torch.empty_like(expected_scale)
    triton_backend.fused_rotate_quantize_input(
        raw_activation,
        actual_qdata,
        actual_scale,
        256,
        dtype_code,
        activation_fn="swiglu",
        num_warps=select_execution_plan(
            AcceleratorTarget.from_device(activation.device), in_features=in_features
        ).fused_num_warps,
    )

    qdata_error = (actual_qdata.to(torch.int16) - expected_qdata.to(torch.int16)).abs()
    assert qdata_error.max().item() <= 1
    torch.testing.assert_close(
        actual_scale,
        expected_scale,
        rtol=2 * torch.finfo(dtype).eps,
        atol=0,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "in_features",
    [512, 5_376, 14_336, 16_640, 28_672, 40_960],
)
@pytest.mark.parametrize(
    ("dtype", "dtype_code"),
    [(torch.float16, 1), (torch.bfloat16, 2), (torch.float32, 0)],
)
@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA-only low-level utility")
def test_fused_gelu_tanh_preparation_matches_materialized_path(
    in_features: int,
    dtype: torch.dtype,
    dtype_code: int,
) -> None:
    torch.manual_seed(76)
    rows = 7
    raw_input = torch.randn(rows, in_features, dtype=dtype, device="cuda")
    activated_input = torch.nn.functional.gelu(raw_input, approximate="tanh")
    rotated = torch.empty_like(activated_input)
    expected_qdata = torch.empty_like(activated_input, dtype=torch.int8)
    expected_scale = torch.empty(rows, dtype=torch.float32, device="cuda")
    triton_backend.rotate_input(activated_input, rotated, 256, num_warps=4)
    triton_backend.quantize_input(
        rotated,
        expected_qdata,
        expected_scale,
        dtype_code,
        num_warps=8,
    )

    actual_qdata = torch.empty_like(expected_qdata)
    actual_scale = torch.empty_like(expected_scale)
    triton_backend.fused_rotate_quantize_input(
        raw_input,
        actual_qdata,
        actual_scale,
        256,
        dtype_code,
        activation_fn="gelu_tanh",
        num_warps=select_execution_plan(
            AcceleratorTarget.from_device(raw_input.device), in_features=in_features
        ).fused_num_warps,
    )

    qdata_error = (actual_qdata.to(torch.int16) - expected_qdata.to(torch.int16)).abs()
    assert qdata_error.max().item() <= 1
    torch.testing.assert_close(
        actual_scale,
        expected_scale,
        rtol=max(2 * torch.finfo(dtype).eps, 2e-6),
        atol=0,
    )


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float32, 0),
        (torch.float16, 1),
        (torch.bfloat16, 2),
    ],
)
def test_dtype_code(dtype: torch.dtype, expected: int) -> None:
    assert triton_backend.dtype_code(dtype) == expected


def test_default_linear_execution_plan_accepts_explicit_target_for_meta_weight() -> None:
    qdata = torch.empty((96, 512), dtype=torch.int8, device="meta")

    plan = triton_backend.default_execution_plan(qdata, target=AcceleratorTarget("cuda", "sm120"))

    assert plan.fuse_rotation_quantization
    assert plan.matmul_block_m == 128
    assert plan.matmul_block_n == 256


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("activation_fn", [None, "gelu_tanh", "swiglu"])
@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA-only low-level utility")
def test_injected_linear_execution_plan_matches_reference(activation_fn: str | None) -> None:
    torch.manual_seed(129)
    rows, in_features, out_features = 17, 256, 96
    input_factor = 2 if activation_fn == "swiglu" else 1
    activation = torch.randn(
        rows,
        input_factor * in_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")
    production = triton_backend.default_execution_plan(qdata)
    candidate = replace(
        production,
        fuse_rotation_quantization=False,
        matmul_block_m=16,
        matmul_block_n=32,
        matmul_block_k=64,
        matmul_num_stages=2,
    )

    actual = triton_backend.run_linear(
        activation,
        qdata,
        scale,
        bias,
        256,
        activation_fn=activation_fn,
        execution_plan=candidate,
    )
    expected = linear(
        activation,
        qdata,
        scale,
        256,
        bias,
        activation_fn=activation_fn,
    )

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("activation_fn", [None, "swiglu"])
@pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA-only low-level utility")
def test_input_preparation_populates_caller_owned_storage(
    activation_fn: str | None,
    fused: bool,
) -> None:
    torch.manual_seed(131)
    rows, in_features = 17, 256
    input_factor = 2 if activation_fn == "swiglu" else 1
    activation = torch.randn(
        rows,
        input_factor * in_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    qdata = torch.empty((96, in_features), dtype=torch.int8, device="cuda")
    plan = replace(
        triton_backend.default_execution_plan(qdata),
        fuse_rotation_quantization=fused,
    )
    target = AcceleratorTarget.from_device(activation.device)
    expected_qdata, expected_scale = triton_backend._prepare_input(
        activation,
        in_features,
        256,
        activation_fn=activation_fn,
        execution_plan=plan,
        target=target,
    )
    qdata_storage = torch.full(
        (rows + 2, in_features),
        -128,
        dtype=torch.int8,
        device="cuda",
    )
    scale_storage = torch.full(
        (rows + 2,),
        -1.0,
        dtype=torch.float32,
        device="cuda",
    )

    qdata_out = qdata_storage[1:-1]
    scale_out = scale_storage[1:-1]
    actual = triton_backend._prepare_input(
        activation,
        in_features,
        256,
        activation_fn=activation_fn,
        execution_plan=plan,
        target=target,
        out=(qdata_out, scale_out),
    )

    assert actual[0] is qdata_out
    assert actual[1] is scale_out
    assert torch.equal(qdata_out, expected_qdata)
    assert torch.equal(scale_out, expected_scale)
    assert torch.all(qdata_storage[[0, -1]] == -128)
    assert torch.all(scale_storage[[0, -1]] == -1.0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA-only low-level utility")
def test_prepared_linear_populates_caller_owned_output(with_bias: bool) -> None:
    torch.manual_seed(132)
    rows, in_features, out_features = 129, 256, 257
    input_qdata = torch.randint(
        -127,
        128,
        (rows, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    input_scale = torch.rand(rows, dtype=torch.float32, device="cuda") * 0.01
    weight_qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    weight_scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda") if with_bias else None
    plan = triton_backend.default_execution_plan(weight_qdata)
    expected = triton_backend._execute_prepared_linear(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        torch.bfloat16,
        plan,
    )
    output_storage = torch.full(
        (rows + 2, out_features),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )

    output = output_storage[1:-1]
    actual = triton_backend._execute_prepared_linear(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        bias,
        torch.bfloat16,
        plan,
        out=output,
    )

    assert actual is output
    assert torch.equal(output, expected)
    assert torch.isnan(output_storage[[0, -1]]).all()


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.gpu
@pytest.mark.skipif(not _exact_sm120_available(), reason="exact SM120 is not available")
@pytest.mark.parametrize(
    ("rows", "out_features", "in_features", "group_size", "dtype"),
    [
        (512, 256, 256, 256, torch.bfloat16),
        (513, 256, 256, 256, torch.bfloat16),
        (513, 257, 256, 256, torch.bfloat16),
        (513, 257, 272, 16, torch.bfloat16),
        (2177, 257, 272, 16, torch.bfloat16),
        (513, 257, 64, 64, torch.bfloat16),
        (512, 256, 256, 256, torch.float32),
    ],
    ids=[
        "aligned",
        "ragged-m",
        "ragged-mn",
        "ragged-mnk",
        "second-m-group",
        "short-k",
        "float32",
    ],
)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.skipif(torch.version.hip is not None, reason="NVIDIA-only low-level utility")
def test_sm120_large_matmul_matches_reference(
    rows: int,
    out_features: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    with_bias: bool,
) -> None:
    torch.manual_seed(139)
    activation = torch.randn(rows, in_features, dtype=dtype, device="cuda")
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    bias = torch.randn(out_features, dtype=dtype, device="cuda") if with_bias else None

    plan = triton_backend.default_execution_plan(qdata)
    assert (
        plan.matmul_block_m,
        plan.matmul_block_n,
        plan.matmul_block_k,
        plan.matmul_num_warps,
    ) == (128, 256, 128, 8)

    actual = triton_backend.run_linear(
        activation,
        qdata,
        scale,
        bias,
        group_size,
        execution_plan=plan,
    )
    expected = linear(activation, qdata, scale, group_size, bias)

    if dtype is torch.bfloat16:
        if bias is None:
            assert torch.equal(actual, expected)
        else:
            torch.testing.assert_close(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


@pytest.mark.parametrize("activation_fn", [None, "gelu_tanh", "swiglu"])
def test_fused_preparation_validates_input_width_from_qdata(
    activation_fn: str | None,
) -> None:
    rows, in_features = 2, 16
    expected_width = in_features * (2 if activation_fn == "swiglu" else 1)
    activation = torch.empty(rows, expected_width + 1)
    input_qdata = torch.empty(rows, in_features, dtype=torch.int8)
    input_scale = torch.empty(rows, dtype=torch.float32)

    with pytest.raises(ValueError, match=f"must have shape \\({rows}, {expected_width}\\)"):
        triton_backend.fused_rotate_quantize_input(
            activation,
            input_qdata,
            input_scale,
            16,
            triton_backend.dtype_code(activation.dtype),
            activation_fn=activation_fn,  # type: ignore[arg-type]
            num_warps=4,
        )


def test_fused_preparation_rejects_unsupported_row_width() -> None:
    rows, in_features = 2, 49_408
    activation = torch.empty(rows, in_features)
    input_qdata = torch.empty(rows, in_features, dtype=torch.int8)
    input_scale = torch.empty(rows, dtype=torch.float32)

    with pytest.raises(ValueError, match=f"does not support row width {in_features}"):
        triton_backend.fused_rotate_quantize_input(
            activation,
            input_qdata,
            input_scale,
            256,
            triton_backend.dtype_code(activation.dtype),
            num_warps=4,
            target=AcceleratorTarget("cuda", "sm120"),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_fused_up_gate_swiglu_linear_matches_materialized_path(
    dtype: torch.dtype,
    with_bias: bool,
) -> None:
    torch.manual_seed(82)
    rows, in_features, out_features = 512, 512, 96
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    weight = ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=256,
        logical_dtype=dtype,
    )
    raw_activation = torch.randn(rows, 2 * in_features, dtype=dtype, device="cuda")
    bias = torch.randn(out_features, dtype=dtype, device="cuda") if with_bias else None
    up, gate = raw_activation.chunk(2, dim=-1)

    expected = linear(
        up * torch.nn.functional.silu(gate),
        qdata,
        scale,
        256,
        bias,
    )
    actual = convrot_int8_linear(raw_activation, weight, bias, activation_fn="swiglu")

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize(
    ("activation_fn", "input_factor", "with_bias"),
    [
        (None, 1, False),
        ("gelu_tanh", 1, True),
        ("swiglu", 2, True),
    ],
)
def test_semantic_linear_fake_kernel_traces_large_shapes_under_fullgraph_compile(
    activation_fn: str | None,
    input_factor: int,
    with_bias: bool,
) -> None:
    rows, in_features, out_features = 131_072, 14_336, 28_672
    activation = torch.empty(
        rows,
        input_factor * in_features,
        dtype=torch.bfloat16,
        device="meta",
    )
    qdata = torch.empty(out_features, in_features, dtype=torch.int8, device="meta")
    scale = torch.empty(out_features, 1, dtype=torch.float32, device="meta")
    bias = torch.empty(out_features, dtype=torch.bfloat16, device="meta") if with_bias else None

    def call(
        value: torch.Tensor,
        packed: torch.Tensor,
        weight_scale: torch.Tensor,
        linear_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return triton_backend.linear(
            value,
            packed,
            weight_scale,
            linear_bias,
            256,
            activation_fn,
        )

    actual = torch.compile(call, backend="eager", fullgraph=True)(
        activation,
        qdata,
        scale,
        bias,
    )

    assert actual.shape == (rows, out_features)
    assert actual.numel() > 2**31
    assert actual.dtype is activation.dtype
    assert actual.device.type == "meta"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize(
    ("activation_fn", "rows", "in_features", "group_size", "input_factor"),
    [
        (None, 17, 64, 64, 1),
        ("gelu_tanh", 512, 512, 256, 1),
        ("swiglu", 512, 512, 256, 2),
    ],
)
def test_cuda_semantic_linear_custom_ops_pass_opcheck(
    activation_fn: str | None,
    rows: int,
    in_features: int,
    group_size: int,
    input_factor: int,
) -> None:
    torch.manual_seed(124)
    out_features = 32
    activation = torch.randn(
        rows,
        input_factor * in_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    bias = torch.randn(out_features, dtype=torch.bfloat16, device="cuda")
    result = torch.library.opcheck(
        triton_backend.linear,
        (activation, qdata, scale, bias, group_size, activation_fn),
    )

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_cuda_semantic_addmm_custom_op_passes_opcheck() -> None:
    torch.manual_seed(126)
    qdata = torch.randint(-127, 128, (32, 64), dtype=torch.int8, device="cuda")
    scale = torch.rand(32, 1, dtype=torch.float32, device="cuda") * 0.01
    mat1 = torch.randn(32, 4, dtype=torch.bfloat16, device="cuda")
    mat2 = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")

    result = torch.library.opcheck(
        triton_backend.addmm_,
        (qdata, scale, mat1, mat2, 64, 0.5, 1.25, 123),
    )

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_cuda_semantic_add_custom_op_passes_opcheck() -> None:
    torch.manual_seed(127)
    qdata = torch.randint(-127, 128, (32, 64), dtype=torch.int8, device="cuda")
    scale = torch.rand(32, 1, dtype=torch.float32, device="cuda") * 0.01
    update = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")

    result = torch.library.opcheck(
        triton_backend.add_,
        (qdata, scale, update, 64, 1.25, 123),
    )

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("input_activation", [None, "gelu_tanh", "swiglu"])
def test_cuda_linear_accepts_noncontiguous_vector_bias(
    input_activation: str | None,
) -> None:
    torch.manual_seed(125)
    rows, in_features, out_features = 512, 512, 96
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    weight = ConvRotInt8Tensor.from_quantized(qdata, scale, group_size=256)
    input_factor = 2 if input_activation == "swiglu" else 1
    activation = torch.randn(
        rows,
        input_factor * in_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    bias = torch.randn(2 * out_features, dtype=torch.bfloat16, device="cuda")[::2]
    assert not bias.is_contiguous()

    if input_activation is None:
        expected = torch.nn.functional.linear(activation, weight, bias.contiguous())
        actual = torch.nn.functional.linear(activation, weight, bias)
    else:
        expected = convrot_int8_linear(
            activation,
            weight,
            bias.contiguous(),
            activation_fn=input_activation,
        )
        actual = convrot_int8_linear(
            activation,
            weight,
            bias,
            activation_fn=input_activation,
        )

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_triton_linear_runs_under_fullgraph_compile_with_noncontiguous_input() -> None:
    module = nn.Linear(64, 96, bias=True, device="meta", dtype=torch.bfloat16)
    module.weight = nn.Parameter(
        ConvRotInt8Tensor.from_quantized(
            torch.randint(-127, 128, (96, 64), dtype=torch.int8, device="cuda"),
            torch.rand(96, 1, dtype=torch.float32, device="cuda") * 0.01,
            group_size=64,
        ),
        requires_grad=False,
    )
    module.bias = nn.Parameter(
        torch.randn(96, dtype=torch.bfloat16, device="cuda"),
        requires_grad=False,
    )
    activation = torch.randn(2, 17, 64, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    assert not activation.is_contiguous()
    expected = module(activation)
    actual = torch.compile(module, fullgraph=True)(activation)
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_cuda_fp16_autocast_normalizes_semantic_linear(compiled: bool) -> None:
    torch.manual_seed(126)
    input = torch.randn(17, 256, device="cuda", dtype=torch.float32) * 0.01  # noqa: A001
    source = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.01
    weight = ConvRotInt8Tensor.from_hp(source, group_size=16)
    bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.01

    def projection(value: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, weight, offset)

    call = (
        torch.compile(projection, fullgraph=True, options=convrot_int8_compile_options())
        if compiled
        else projection
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        actual = call(input, bias)
    with torch.no_grad():
        expected = torch.nn.functional.linear(
            input.half(),
            weight.to(dtype=torch.float16),
            bias.half(),
        )

    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("rows", [17, 512], ids=["split", "fused"])
def test_swiglu_linear_runs_under_fullgraph_compile_with_noncontiguous_input(rows: int) -> None:
    torch.manual_seed(97)
    in_features, out_features = 512, 96
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    weight = ConvRotInt8Tensor.from_quantized(qdata, scale, group_size=256)
    activation_storage = torch.randn(
        rows,
        2,
        2 * in_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    raw_activation = activation_storage[:, 0, :]
    assert not raw_activation.is_contiguous()

    def call(value: torch.Tensor) -> torch.Tensor:
        return convrot_int8_linear(value, weight, activation_fn="swiglu")

    expected = call(raw_activation)
    actual = torch.compile(call, fullgraph=True)(raw_activation)

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("beta", "alpha"), [(0.25, 1.5), (0, 1.5), (0.25, 0)])
def test_triton_addmm_matches_gpu_reference(
    dtype: torch.dtype,
    beta: float,
    alpha: float,
) -> None:
    torch.manual_seed(18)
    weight = torch.randn(96, 64, dtype=dtype, device="cuda")
    mat1 = torch.randn(96, 8, dtype=dtype, device="cuda")
    mat2 = torch.randn(8, 64, dtype=dtype, device="cuda")
    actual = ConvRotInt8Tensor.from_hp(weight, group_size=64)
    expected = actual.clone()

    addmm_(expected.qdata, expected.scale, mat1, mat2, 64, beta, alpha)
    result = actual.addmm_(mat1, mat2, beta=beta, alpha=alpha)

    assert result is actual
    qdata_error = (actual.qdata.to(torch.int16) - expected.qdata.to(torch.int16)).abs()
    assert qdata_error.max().item() <= 2
    assert torch.allclose(
        actual.scale,
        expected.scale,
        rtol=2 * torch.finfo(dtype).eps,
        atol=1e-7,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("alpha", [1.5, -0.25])
def test_triton_add_matches_gpu_reference(dtype: torch.dtype, alpha: float) -> None:
    torch.manual_seed(19)
    weight = torch.randn(96, 64, dtype=dtype, device="cuda")
    update = torch.randn_like(weight)
    actual = ConvRotInt8Tensor.from_hp(weight, group_size=64)
    expected = actual.clone()

    reference_add_(expected.qdata, expected.scale, update, 64, alpha)
    result = actual.add_(update, alpha=alpha)

    assert result is actual
    qdata_error = (actual.qdata.to(torch.int16) - expected.qdata.to(torch.int16)).abs()
    assert qdata_error.max().item() <= 2
    assert torch.allclose(
        actual.scale,
        expected.scale,
        rtol=2 * torch.finfo(dtype).eps,
        atol=1e-7,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_triton_addmm_runs_under_torch_compile() -> None:
    torch.manual_seed(30)
    weight = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
    mat1 = torch.randn(32, 4, dtype=torch.bfloat16, device="cuda")
    mat2 = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    expected = ConvRotInt8Tensor.from_hp(weight, group_size=64)
    actual = expected.clone()
    expected.addmm_(mat1, mat2, beta=0.5, alpha=1.25, rounding_seed=123)

    def merge(
        target: ConvRotInt8Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> ConvRotInt8Tensor:
        return target.addmm_(
            left,
            right,
            beta=0.5,
            alpha=1.25,
            rounding_seed=123,
        )

    result = torch.compile(merge, fullgraph=True)(actual, mat1, mat2)

    assert result is actual
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_triton_add_runs_under_torch_compile() -> None:
    torch.manual_seed(31)
    weight = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
    update = torch.randn_like(weight)
    expected = ConvRotInt8Tensor.from_hp(weight, group_size=64)
    actual = expected.clone()
    expected.add_(update, alpha=1.25)

    def merge(
        target: ConvRotInt8Tensor,
        dense_update: torch.Tensor,
    ) -> ConvRotInt8Tensor:
        return target.add_(dense_update, alpha=1.25)

    result = torch.compile(merge, fullgraph=True)(actual, update)

    assert result is actual
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_triton_addmm_stochastic_rounding_replays_and_is_unbiased() -> None:
    rows, cols = 32, 8192
    rotated_update = torch.ones(rows, cols, dtype=torch.bfloat16, device="cuda")
    rotated_update[:, -1] = 2.0
    mat1 = torch.eye(rows, dtype=torch.bfloat16, device="cuda")
    mat2 = rotate_groups(rotated_update, 256)

    def make_weight() -> ConvRotInt8Tensor:
        return ConvRotInt8Tensor.from_quantized(
            torch.zeros(rows, cols, dtype=torch.int8, device="cuda"),
            torch.ones(rows, 1, dtype=torch.float32, device="cuda"),
            group_size=256,
            logical_dtype=torch.bfloat16,
        )

    first = make_weight()
    replay = make_weight()
    other = make_weight()
    first.addmm_(mat1, mat2, beta=0, rounding_seed=(1 << 64) - 1)
    replay.addmm_(mat1, mat2, beta=0, rounding_seed=(1 << 64) - 1)
    other.addmm_(mat1, mat2, beta=0, rounding_seed=(1 << 64) - 2)

    assert torch.equal(first.qdata, replay.qdata)
    assert torch.equal(first.scale, replay.scale)
    assert not torch.equal(first.qdata, other.qdata)
    assert torch.equal(first.scale, other.scale)
    samples = first.qdata[:, :-1]
    assert bool(((samples == 63) | (samples == 64)).all())
    assert samples.to(torch.float32).mean().item() == pytest.approx(63.5, abs=0.01)
    assert bool((first.qdata[:, -1] == 127).all())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_triton_addmm_handles_underflowing_float16_scale() -> None:
    rotated_update = torch.zeros(1, 16, dtype=torch.float16, device="cuda")
    rotated_update[0, 0] = 1e-6
    mat1 = torch.ones(1, 1, dtype=torch.float16, device="cuda")
    mat2 = rotate_groups(rotated_update, 16)
    qdata = torch.zeros(1, 16, dtype=torch.int8, device="cuda")
    scale = torch.ones(1, 1, dtype=torch.float32, device="cuda")
    actual = ConvRotInt8Tensor.from_quantized(
        qdata.clone(),
        scale.clone(),
        group_size=16,
        logical_dtype=torch.float16,
    )
    expected = actual.clone()

    addmm_(expected.qdata, expected.scale, mat1, mat2, 16, 0, 1)
    actual.addmm_(mat1, mat2, beta=0)

    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    assert actual.qdata[0, 0] == 127
    assert torch.count_nonzero(actual.qdata[0, 1:]) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
def test_triton_linear_handles_underflowing_float16_input_scale() -> None:
    rotated_activation = torch.zeros(1, 16, dtype=torch.float16, device="cuda")
    rotated_activation[0, 0] = 1e-6
    activation = rotate_groups(rotated_activation, 16)
    qdata = torch.arange(-8, 8, dtype=torch.int8, device="cuda").reshape(1, 16)
    scale = torch.ones(1, 1, dtype=torch.float32, device="cuda")
    weight = ConvRotInt8Tensor.from_quantized(
        qdata,
        scale,
        group_size=16,
        logical_dtype=torch.float16,
    )

    expected = linear(activation, qdata, scale, 16)
    actual = torch.nn.functional.linear(activation, weight)

    assert torch.equal(actual, expected)
