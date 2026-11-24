"""AMD support boundaries and on-device coverage of the shared INT8 contract."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _accelerators import exact_gfx1201_available, mock_rocm_target, mock_triton_driver

from piper_kernels import compile_options
from piper_kernels._input_activations import apply_input_activation
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot import convrot_int8_linear
from piper_kernels.linear.convrot.int8 import _backend, _generic, reference
from piper_kernels.linear.convrot.int8._amd import policy
from piper_kernels.linear.convrot.int8._amd import triton as amd_triton
from piper_kernels.linear.convrot.int8._generic import mean as generic_mean
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.int8 import _gguf as int8_gguf
from piper_kernels.weights.convrot.int8 import _quantization as int8_quantization
from piper_kernels.weights.convrot.int8 import _update as int8_updates
from piper_kernels.weights.convrot.int8 import triton as int8_weight_triton

pytestmark = pytest.mark.skipif(
    sys.platform not in ("linux", "win32"), reason="requires Linux or Windows ROCm"
)

_gpu = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires a supported ROCm GPU",
)
amd = _backend._amd_backend


@pytest.mark.parametrize("architecture", ["gfx942", "gfx1100", "gfx1151", "gfx1200", "gfx1201"])
@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_amd_selection_and_independent_auxiliary_support(monkeypatch, architecture, platform):
    monkeypatch.setattr(sys, "platform", platform)
    mock_rocm_target(monkeypatch, architecture)
    mock_triton_driver(monkeypatch, backend="hip", arch=architecture)
    value = SimpleNamespace(device=torch.device("cuda"), shape=(2, 256), numel=lambda: 512)
    expected = amd if platform in ("linux", "win32") else None
    assert _backend.select_linear_backend(value) is expected
    assert int8_updates._use_triton(value)
    assert int8_gguf.select_gguf_converter(value) is int8_weight_triton.convert_gguf_out
    assert _backend.select_dequantized_mean(value) is generic_mean.dequantized_input_mean


@pytest.mark.parametrize(
    "target",
    [
        AcceleratorTarget("hip"),
        AcceleratorTarget("hip", "gfx1036"),
        AcceleratorTarget("hip", "gfx9999"),
        AcceleratorTarget("cuda", "sm120"),
        AcceleratorTarget("cpu"),
    ],
)
def test_amd_policy_rejects_unvalidated_targets(target):
    assert not policy.supports_target(target)
    with pytest.raises(ValueError, match="no optimized policy"):
        policy.select_execution_plan(target, in_features=256)


def test_amd_missing_runtime_falls_back(monkeypatch):
    monkeypatch.setattr(_backend, "_amd_backend", None)
    mock_rocm_target(monkeypatch, "gfx1201")
    value = SimpleNamespace(device=torch.device("cuda"), shape=(2, 256), numel=lambda: 512)
    assert _backend.select_linear_backend(value) is None
    assert _backend.select_preparation_backend(value) is _generic
    assert int8_updates._use_triton(value)


@pytest.mark.parametrize("width", [256, 2048, 4096, 5376, 8192, 9216, 12288, 16384, 32768])
def test_rdna4_preparation_policy(width):
    plan = policy.select_execution_plan(AcceleratorTarget("hip", "gfx1201"), in_features=width)
    blocks = policy.preparation_blocks(width)
    assert len(blocks) == 3
    assert sum(blocks) >= width
    active = [block for block in blocks if block > 0]
    assert active
    assert blocks == (*active, *([0] * (3 - len(active))))
    assert all((block & (block - 1)) == 0 for block in active)
    assert 1 <= plan.fused_num_warps <= 16
    assert (plan.fused_num_warps & (plan.fused_num_warps - 1)) == 0
    assert plan.fuse_rotation_quantization is (width <= 16384)
    assert plan.matmul_block_m > 0
    assert plan.matmul_block_n > 0
    assert plan.matmul_block_k > 0
    assert plan.matmul_num_stages >= 1


@pytest.mark.parametrize(
    ("architecture", "rows", "outputs", "width", "expected_nk"),
    [
        ("gfx1201", 4096, 6144, 16384, (128, 128)),
        ("gfx1201", 4096, 6144, 16448, (256, 64)),
        ("gfx1201", 2048, 2048, 8192, (128, 128)),
        ("gfx1201", 4096, 16384, 8192, (128, 128)),
        ("gfx1201", 4096, 2048, 24576, (128, 128)),
        ("gfx1201", 1024, 1024, 16384, (256, 64)),
        ("gfx1200", 4096, 6144, 16384, (256, 64)),
    ],
)
def test_amd_matmul_selects_wide_k_for_aligned_strides(
    architecture, rows, outputs, width, expected_nk
):
    plan = policy.select_execution_plan(
        AcceleratorTarget("hip", architecture), in_features=width, rows=rows, out_features=outputs
    )
    assert (plan.matmul_block_n, plan.matmul_block_k) == expected_nk
    assert policy.matmul_group_m(plan) == 8


def test_gfx1201_short_batch_matmul_schedules():
    gfx1201 = AcceleratorTarget("hip", "gfx1201")
    gfx1200 = AcceleratorTarget("hip", "gfx1200")
    short_plans = {
        rows: policy.select_execution_plan(gfx1201, in_features=5376, rows=rows, out_features=14336)
        for rows in (1, 16, 17, 32, 33, 64)
    }
    for rows, plan in short_plans.items():
        # Every short batch up to 64 rows stays on one ungrouped M tile so weights stream once.
        assert plan.matmul_block_m >= rows
        assert policy.matmul_group_m(plan) == 0

    wide_plan = policy.select_execution_plan(gfx1201, in_features=5376, rows=65, out_features=14336)
    assert wide_plan.matmul_block_m == 128
    assert policy.matmul_group_m(wide_plan) == 8
    assert (
        short_plans[16].matmul_block_m
        < short_plans[17].matmul_block_m
        == short_plans[32].matmul_block_m
        < short_plans[33].matmul_block_m
        == short_plans[64].matmul_block_m
        < wide_plan.matmul_block_m
    )

    for rows, outputs in ((128, 5376), (128, 8192), (129, 5376)):
        plan = policy.select_execution_plan(
            gfx1201, in_features=8192, rows=rows, out_features=outputs
        )
        assert plan.matmul_block_m == 128
        assert policy.matmul_group_m(plan) == 8

    assert policy.select_execution_plan(
        gfx1200, in_features=5376, rows=32, out_features=5376
    ) == policy.select_execution_plan(gfx1200, in_features=5376)


def test_optional_amd_compiler_option_compatibility(monkeypatch):
    target = AcceleratorTarget("hip", "gfx1201")
    monkeypatch.setattr(amd_triton.importlib, "import_module", Mock(return_value=object()))
    assert amd_triton._amd_matmul_compiler_options(target) == {}


def test_gfx1201_accumulator_seed_fits_every_int8_partial_sum():
    target = AcceleratorTarget("hip", "gfx1201")
    seed = policy.matmul_accumulator_seed(target, 4096, 64)
    assert seed > 0
    assert policy.matmul_accumulator_seed(target, 4096, 63) == 0
    last_safe_width = (2**31 - 1 - seed) // (128 * 128)
    assert policy.matmul_accumulator_seed(target, last_safe_width, 64) == seed
    assert policy.matmul_accumulator_seed(target, last_safe_width + 1, 64) == 0
    assert policy.matmul_accumulator_seed(AcceleratorTarget("hip", "gfx1200"), 4096, 64) == 0


@pytest.mark.parametrize(
    ("arch", "dtype", "rows", "outputs", "width", "stride", "expected"),
    [
        ("gfx1201", torch.bfloat16, 4096, 14336, 5376, 28672, True),
        ("gfx1201", torch.bfloat16, 2816, 16384, 6144, 32768, True),
        ("gfx1201", torch.bfloat16, 20480, 8192, 2048, 16384, True),
        ("gfx1201", torch.bfloat16, 20480, 14336, 5376, 28672, True),
        ("gfx1201", torch.float32, 20480, 14336, 5376, 28672, False),
        ("gfx1200", torch.bfloat16, 4096, 14336, 5376, 28672, False),
        ("gfx1201", torch.bfloat16, 4096, 6144, 8192, 12288, False),
        ("gfx1201", torch.bfloat16, 4096, 320, 512, 640, False),
        ("gfx1201", torch.bfloat16, 4096, 512, 224, 1024, False),
        ("gfx1201", torch.bfloat16, 100000, 14336, 5376, 28672, False),
    ],
)
def test_gfx1201_paired_matmul_selection_boundaries(
    arch, dtype, rows, outputs, width, stride, expected
):
    target = AcceleratorTarget("hip", arch)
    plan = policy.select_execution_plan(target, in_features=width, rows=rows, out_features=outputs)
    assert (
        policy.supports_paired_matmul(
            target,
            plan,
            dtype,
            rows=rows,
            in_features=width,
            out_features=outputs,
            output_row_stride=stride,
        )
        is expected
    )


def test_gfx1201_paired_matmul_respects_custom_plan():
    target = AcceleratorTarget("hip", "gfx1201")
    plan = replace(
        policy.select_execution_plan(target, in_features=5376, rows=4096, out_features=14336),
        matmul_block_n=128,
    )
    assert not policy.supports_paired_matmul(
        target,
        plan,
        torch.bfloat16,
        rows=4096,
        in_features=5376,
        out_features=14336,
        output_row_stride=28672,
    )


@pytest.mark.parametrize(
    ("arch", "dtype", "rows", "outputs", "width", "expected"),
    [
        ("gfx1201", torch.bfloat16, None, 14336, 5376, True),
        ("gfx1201", torch.float16, None, 16384, 6144, True),
        ("gfx1201", torch.bfloat16, 65, 14336, 5376, True),
        ("gfx1201", torch.bfloat16, 64, 14336, 5376, False),
        ("gfx1201", torch.bfloat16, 200000, 14336, 5376, False),
        ("gfx1201", torch.float32, None, 14336, 5376, False),
        ("gfx1200", torch.bfloat16, None, 14336, 5376, False),
        ("gfx1201", torch.bfloat16, 4096, 6144, 8192, False),
        ("gfx1201", torch.bfloat16, 4096, 320, 512, False),
        ("gfx1201", torch.bfloat16, 4096, 512, 224, False),
    ],
)
def test_gfx1201_swiglu_epilogue_selection_boundaries(arch, dtype, rows, outputs, width, expected):
    target = AcceleratorTarget("hip", arch)
    plan = policy.select_execution_plan(
        target, in_features=width, rows=4096 if rows is None else rows, out_features=outputs
    )
    assert (
        policy.supports_swiglu_epilogue(
            target,
            plan,
            dtype,
            in_features=width,
            out_features=outputs,
            rows=rows,
        )
        is expected
    )


@pytest.mark.gpu
@_gpu
def test_amd_projection_recovers_integer_sum_before_fp32_rounding():
    # 2**24 - 1 is exact in FP32, but its seeded sum crosses a binade.
    width = 4096
    a = torch.zeros(1024, width, device="cuda", dtype=torch.int8)
    b = torch.zeros(2048, width, device="cuda", dtype=torch.int8)
    a[:, :2048] = -128
    b[:, :2048] = -64
    a[:, 2048] = 1
    b[:, 2048] = -1
    result = amd.linear_prepared(
        a, torch.ones(1024, device="cuda"), b, torch.ones(2048, device="cuda"), None, torch.float32
    )
    assert torch.equal(result, torch.full_like(result, 2**24 - 1))


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize(
    ("rows", "width", "columns"),
    [
        (1023, 256, 335),
        (1024, 256, 335),
        (1025, 256, 335),
        (2049, 256, 335),
        (1025, 4096, 1024),
        (1025, 8192, 2049),
    ],
)
@pytest.mark.parametrize("paired", [False, True])
def test_amd_projection_crosses_cache_group_and_tail_boundaries(rows, width, columns, paired):
    torch.manual_seed(893)
    qdata = torch.randint(-127, 128, (rows, width), device="cuda", dtype=torch.int8)
    scale = torch.rand(rows, device="cuda") * 0.01
    weights = [
        (
            torch.randint(-127, 128, (columns, width), device="cuda", dtype=torch.int8),
            torch.rand(columns, device="cuda") * 0.01,
            # Zero bias keeps this indexing regression bitwise-checkable against
            # separate PyTorch ops (nonzero bias may be fused into an FP32 FMA).
            torch.zeros(columns, device="cuda"),
        )
        for _ in range(2 if paired else 1)
    ]
    expected = []
    for weight, weight_scale, bias in weights:
        padded_weight = torch.nn.functional.pad(weight, (0, 0, 0, (-columns) % 8))
        value = torch._int_mm(qdata, padded_weight.T)[:, :columns].float()
        value = value * scale[:, None] * weight_scale[None, :]
        expected.append((value + bias).bfloat16())
    storage = torch.full(
        (rows, columns * len(weights) + 7), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    output = storage[:, : columns * len(weights)]
    actual = amd.linear_prepared(
        qdata,
        scale,
        weights[0][0],
        weights[0][1],
        weights[0][2],
        torch.bfloat16,
        out=output,
        second_projection=weights[1] if paired else None,
    )
    assert actual is output
    assert torch.equal(actual, torch.cat(expected, dim=-1))
    assert storage[:, columns * len(weights) :].isnan().all()


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [256, 2048, 4096, 5376, 8192, 9216, 12288, 16384, 32768])
@pytest.mark.parametrize("activation", [None, "gelu_tanh", "swiglu"])
def test_amd_preparation_matches_rotation_and_populates_storage(dtype, width, activation):
    torch.manual_seed(721)
    raw_width = width * (2 if activation == "swiglu" else 1)
    # Noncontiguous input and multidimensional leading shape.
    value = torch.randn(2, 3, raw_width * 2, device="cuda", dtype=dtype)[..., ::2]
    output = (
        torch.empty(2, 3, width, device="cuda", dtype=torch.int8),
        torch.empty(2, 3, device="cuda"),
    )
    implementation = _backend.require_linear_backend(value)
    assert implementation is amd
    actual = implementation.prepare_input(value, 256, activation, out=output)
    plan = amd.default_execution_plan(output[0].reshape(6, width))
    activated = apply_input_activation(value, activation)
    rotated = rotate_groups(activated.float(), 256)
    if not plan.fuse_rotation_quantization:
        # The split path stores a compact materialized rotation.
        rotated = rotated.to(activated.dtype)
    _, expected_scale = int8_quantization.dynamic_quantize_rows(rotated)
    assert actual is output
    # Nearest rounding has half a quantization step of error. Allow a small
    # margin for FP32 activation/reduction ordering, not intermediate BF16 casts.
    decoded = actual[0].float() * actual[1][..., None]
    assert ((decoded - rotated.float()).abs() <= 0.51 * actual[1][..., None] + 1e-7).all()
    torch.testing.assert_close(
        actual[1], expected_scale.squeeze(-1), rtol=3e-7 if activation is None else 2e-5, atol=1e-7
    )


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", [None, "gelu_tanh", "swiglu"])
def test_amd_public_linear_matches_reference(group_size, dtype, activation):
    torch.manual_seed(83)
    k, n = 3 * group_size, 79
    qdata = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    scale = torch.rand(n, 1, device="cuda") * 0.01
    bias = torch.randn(n * 2, device="cuda", dtype=torch.float32)[::2]
    value = torch.randn(129, k * (2 if activation == "swiglu" else 1), device="cuda", dtype=dtype)
    weight = ConvRotInt8Tensor.from_quantized(
        qdata, scale, group_size=group_size, logical_dtype=dtype
    )
    actual = convrot_int8_linear(value, weight, bias, activation_fn=activation)
    prepared = amd.prepare_input(value, group_size, activation)
    expected = reference.linear_prepared(*prepared, qdata, scale, dtype, bias)
    torch.testing.assert_close(actual, expected, rtol=2 * torch.finfo(dtype).eps, atol=1e-6)
    rotated = rotate_groups(apply_input_activation(value, activation).float(), group_size)
    _, expected_scale = int8_quantization.dynamic_quantize_rows(rotated)
    decoded = prepared[0].float() * prepared[1][..., None]
    assert ((decoded - rotated).abs() <= 0.51 * prepared[1][..., None] + 1e-7).all()
    torch.testing.assert_close(
        prepared[1],
        expected_scale.squeeze(-1),
        rtol=3e-7 if activation is None else 2e-5,
        atol=1e-7,
    )


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [256, 9216, 16384, 32768])
@pytest.mark.parametrize("magnitude", [0.0, 1e-6, 1e-31])
def test_amd_preparation_zero_and_tiny_scales(dtype, width, magnitude):
    value = torch.zeros(2, width, device="cuda", dtype=dtype)
    value[:, 0] = magnitude
    actual = amd.prepare_input(value, 256)
    rotated = rotate_groups(value.float(), 256)
    if not amd.default_execution_plan(actual[0]).fuse_rotation_quantization:
        rotated = rotated.to(dtype)
    expected_qdata, expected_scale = int8_quantization.dynamic_quantize_rows(rotated)
    expected = expected_qdata, expected_scale.squeeze(-1)
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("width", [256, 9216, 16384, 32768])
def test_amd_scale_matches_pytorch_scalar_reciprocal_rounding(width):
    # 3.25 / 127 and 3.25 * float32(1 / 127) differ by one FP32 ULP.
    # A regular normalized Hadamard leaves this constant input unchanged.
    value = torch.full((2, width), 3.25, dtype=torch.bfloat16, device="cuda")
    actual = amd.prepare_input(value, 256)
    expected = reference.prepare_input(value, 256)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.gpu
@_gpu
def test_amd_compiled_shared_preparation(monkeypatch):
    torch.manual_seed(294)
    weight = ConvRotInt8Tensor.from_hp(
        torch.randn(79, 256, device="cuda", dtype=torch.bfloat16), group_size=256
    )
    other = ConvRotInt8Tensor.from_hp(
        torch.randn(63, 256, device="cuda", dtype=torch.bfloat16), group_size=256
    )
    value = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)

    def project(value):
        return torch.nn.functional.linear(value, weight), torch.nn.functional.linear(value, other)

    expected = project(value)
    prepare = Mock(wraps=amd.prepare_input)
    monkeypatch.setattr(amd, "prepare_input", prepare)
    call = torch.compile(project, fullgraph=True, options=compile_options())
    call(value)
    prepare.reset_mock()
    actual = call(value)
    assert prepare.call_count == 1
    for result, reference_result in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference_result, rtol=0, atol=0)


@pytest.mark.gpu
@_gpu
@pytest.mark.parametrize("rows", [1, 32, 33, 63, 64, 65, 128])
@pytest.mark.parametrize("paired", [False, True])
def test_amd_short_batch_projections_match_reference(rows, paired):
    torch.manual_seed(913 + rows)
    width, columns = 512, 320
    dtype = torch.bfloat16
    x = torch.randn((rows, width), device="cuda", dtype=dtype)
    qdata, scale = amd.prepare_input(x, 256)
    weights = [
        (
            torch.randint(-127, 128, (columns, width), device="cuda", dtype=torch.int8),
            torch.rand(columns, device="cuda") * 0.02,
            torch.randn(columns, device="cuda", dtype=dtype),
        )
        for _ in range(2 if paired else 1)
    ]
    actual = amd.linear_prepared(
        qdata,
        scale,
        weights[0][0],
        weights[0][1],
        weights[0][2],
        dtype,
        second_projection=weights[1] if paired else None,
    )
    expected = torch.cat(
        [reference.linear_prepared(qdata, scale, w, ws, dtype, b) for w, ws, b in weights],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, rtol=2 * torch.finfo(dtype).eps, atol=1e-5)
    if not paired:
        weight = ConvRotInt8Tensor.from_quantized(
            weights[0][0], weights[0][1][:, None], group_size=256, logical_dtype=dtype
        )
        end_to_end = convrot_int8_linear(x, weight, weights[0][2])
        torch.testing.assert_close(end_to_end, expected, rtol=2 * torch.finfo(dtype).eps, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not exact_gfx1201_available(), reason="requires ROCm gfx1201")
@pytest.mark.parametrize(
    ("rows", "width", "columns", "expected_fused"),
    [
        (64, 256, 2560, False),
        (65, 256, 2560, True),
        (1024, 8192, 2048, False),
    ],
)
def test_gfx1201_swiglu_fallback_boundaries_match_reference(rows, width, columns, expected_fused):
    torch.manual_seed(917 + rows)
    dtype = torch.bfloat16
    qdata = torch.randint(-127, 128, (rows, width), device="cuda", dtype=torch.int8)
    scale = torch.rand(rows, device="cuda") * 0.01
    weights = [
        (
            torch.randint(-127, 128, (columns, width), device="cuda", dtype=torch.int8),
            torch.rand(columns, device="cuda") * 0.01,
            torch.randn(columns, device="cuda", dtype=torch.float32),
        )
        for _ in range(2)
    ]
    assert (
        amd.supports_prepared_swiglu(weights[0][0], weights[1][0], dtype, rows=rows)
        is expected_fused
    )
    args = (qdata, scale, weights[0][0], weights[0][1], weights[0][2])
    second = weights[1]
    paired = amd.linear_prepared(*args, dtype, second_projection=second)
    up_part, gate_part = paired.chunk(2, dim=-1)
    expected = (up_part.float() * torch.nn.functional.silu(gate_part.float())).to(dtype)
    actual = amd.linear_prepared_swiglu(*args, second, dtype)
    if expected_fused:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2 * torch.finfo(dtype).eps,
            atol=torch.finfo(dtype).smallest_normal,
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
