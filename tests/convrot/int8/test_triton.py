"""Tests for the Triton ConvRot backend."""

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels.convrot import ConvRotInt8Tensor, convrot_linear
from piper_kernels.convrot._rotation import rotate_groups
from piper_kernels.convrot.int8 import _policy as convrot_policy
from piper_kernels.convrot.int8 import dispatch as convrot_dispatch
from piper_kernels.convrot.int8 import triton as triton_backend
from piper_kernels.convrot.int8.reference import reference_addmm_, reference_linear


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
def test_triton_linear_matches_gpu_reference(group_size: int) -> None:
    torch.manual_seed(9)
    in_features = 2 * group_size
    qdata = torch.randint(-127, 128, (96, in_features), dtype=torch.int8, device="cuda")
    scale = torch.rand(96, 1, dtype=torch.float32, device="cuda") * 0.01
    wrapped = ConvRotInt8Tensor.from_packed(qdata, scale, group_size=group_size)
    activation = torch.randn(37, in_features, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(96, dtype=torch.bfloat16, device="cuda")

    expected = reference_linear(activation, qdata, scale, group_size, bias)
    actual = torch.nn.functional.linear(activation, wrapped, bias)
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_factorized_h4_rotation_matches_gpu_reference(
    group_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(41 + group_size)
    activation = torch.randn(5, 2 * group_size, dtype=dtype, device="cuda")
    actual = torch.empty_like(activation)

    triton_backend._rotate_activations(activation, actual, group_size)
    expected = rotate_groups(activation, group_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
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

    triton_backend._rotate_activations(activation, actual, group_size)
    expected = rotate_groups(activation, group_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("in_features", [512, 5_376, 7_168, 14_336])
@pytest.mark.parametrize(
    ("dtype", "dtype_code"),
    [(torch.float16, 1), (torch.bfloat16, 2)],
)
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
    triton_backend._rotate_activations(activation, rotated, 256)
    triton_backend._quantize_activations(
        rotated,
        expected_qdata,
        expected_scale,
        dtype_code,
    )

    actual_qdata = torch.empty_like(expected_qdata)
    actual_scale = torch.empty_like(expected_scale)
    triton_backend._fused_rotate_quantize_activations(
        activation,
        actual_qdata,
        actual_scale,
        256,
        dtype_code,
    )

    assert torch.equal(actual_qdata, expected_qdata)
    assert torch.equal(actual_scale, expected_scale)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("in_features", [512, 5_376, 7_168, 14_336])
@pytest.mark.parametrize(
    ("dtype", "dtype_code"),
    [(torch.float16, 1), (torch.bfloat16, 2)],
)
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
    triton_backend._rotate_activations(activation, rotated, 256)
    triton_backend._quantize_activations(
        rotated,
        expected_qdata,
        expected_scale,
        dtype_code,
    )

    actual_qdata = torch.empty_like(expected_qdata)
    actual_scale = torch.empty_like(expected_scale)
    triton_backend._fused_rotate_quantize_activations(
        raw_activation,
        actual_qdata,
        actual_scale,
        256,
        dtype_code,
        input_activation_code=1,
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
    weight = ConvRotInt8Tensor.from_packed(qdata, scale, group_size=256, dtype=dtype)
    raw_activation = torch.randn(rows, 2 * in_features, dtype=dtype, device="cuda")
    bias = torch.randn(out_features, dtype=dtype, device="cuda") if with_bias else None
    up, gate = raw_activation.chunk(2, dim=-1)

    expected = reference_linear(
        up * torch.nn.functional.silu(gate),
        qdata,
        scale,
        256,
        bias,
    )
    actual = convrot_linear(raw_activation, weight, bias, input_activation="swiglu")

    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_convrot_swiglu_linear_handles_empty_rows(with_bias: bool) -> None:
    in_features, out_features = 512, 96
    qdata = torch.empty(out_features, in_features, dtype=torch.int8, device="cuda")
    scale = torch.empty(out_features, 1, dtype=torch.float32, device="cuda")
    weight = ConvRotInt8Tensor.from_packed(qdata, scale, group_size=256)
    raw_activation = torch.empty(2, 0, 2 * in_features, dtype=torch.bfloat16, device="cuda")
    bias = torch.empty(out_features, dtype=torch.bfloat16, device="cuda") if with_bias else None

    actual = convrot_linear(raw_activation, weight, bias, input_activation="swiglu")

    assert actual.shape == (2, 0, out_features)
    assert actual.dtype is torch.bfloat16
    assert actual.device == raw_activation.device


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("prefix", [(), (2, 3), (2, 0)])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
def test_cuda_linear_handles_zero_input_features(
    prefix: tuple[int, ...],
    with_bias: bool,
    input_activation: str | None,
) -> None:
    out_features = 3
    qdata = torch.empty(out_features, 0, dtype=torch.int8, device="cuda")
    scale = torch.ones(out_features, 1, dtype=torch.float32, device="cuda")
    weight = ConvRotInt8Tensor.from_packed(
        qdata,
        scale,
        group_size=16,
        dtype=torch.bfloat16,
    )
    activation = torch.empty((*prefix, 0), dtype=torch.bfloat16, device="cuda")
    bias = torch.arange(out_features, dtype=torch.bfloat16, device="cuda") if with_bias else None

    if input_activation is None:
        result = torch.nn.functional.linear(activation, weight, bias)
    else:
        result = convrot_linear(
            activation,
            weight,
            bias,
            input_activation="swiglu",
        )

    expected = activation.new_zeros((*prefix, out_features))
    if bias is not None:
        expected += bias
    assert torch.equal(result, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("prefix", [(2, 3), (2, 0)])
@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
def test_cuda_linear_preserves_zero_output_and_row_dimensions(
    prefix: tuple[int, ...],
    input_activation: str | None,
) -> None:
    in_features = 16
    weight = ConvRotInt8Tensor.from_packed(
        torch.empty(0, in_features, dtype=torch.int8, device="cuda"),
        torch.empty(0, 1, dtype=torch.float32, device="cuda"),
        group_size=16,
    )
    input_factor = 1 if input_activation is None else 2
    activation = torch.empty(
        (*prefix, input_factor * in_features),
        dtype=torch.bfloat16,
        device="cuda",
    )

    if input_activation is None:
        result = torch.nn.functional.linear(activation, weight)
    else:
        result = convrot_linear(activation, weight, input_activation="swiglu")

    assert result.shape == (*prefix, 0)
    assert result.dtype is activation.dtype
    assert result.device == activation.device


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_zero_input_features_run_under_fullgraph_compile() -> None:
    weight = ConvRotInt8Tensor.from_packed(
        torch.empty(3, 0, dtype=torch.int8, device="cuda"),
        torch.ones(3, 1, dtype=torch.float32, device="cuda"),
        group_size=16,
    )
    activation = torch.empty(2, 4, 0, dtype=torch.bfloat16, device="cuda")
    bias = torch.arange(3, dtype=torch.bfloat16, device="cuda")

    def apply_both(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.nn.functional.linear(value, weight, bias),
            convrot_linear(value, weight, bias, input_activation="swiglu"),
        )

    expected = apply_both(activation)
    actual = torch.compile(apply_both, fullgraph=True)(activation)

    assert all(
        torch.equal(item, reference) for item, reference in zip(actual, expected, strict=True)
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_swiglu_zero_width_validates_raw_input_width() -> None:
    activation = torch.empty(2, 16, dtype=torch.bfloat16, device="cuda")
    qdata = torch.empty(3, 0, dtype=torch.int8, device="cuda")
    scale = torch.ones(3, 1, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="expected 0"):
        triton_backend.triton_convrot_int8_swiglu_linear(
            activation,
            qdata,
            scale,
            None,
            16,
        )


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((12, 0), True),
        ((11, 0), False),
        ((12, 1), False),
        ((13, 0), False),
    ],
)
def test_sm120_detection_is_an_exact_architecture_guard(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected: bool,
) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)

    assert convrot_policy.is_sm120(torch.device("cuda")) is expected
    assert triton_backend._is_sm120(torch.device("cuda")) is expected


@pytest.mark.parametrize(
    ("rows", "in_features", "group_size", "dtype", "is_sm120", "expected"),
    [
        (512, 512, 256, torch.float16, True, True),
        (512, 14_336, 256, torch.bfloat16, True, True),
        (511, 512, 256, torch.float16, True, False),
        (512, 512, 64, torch.float16, True, False),
        (512, 512, 256, torch.float32, True, False),
        (512, 16_640, 256, torch.bfloat16, True, False),
        (512, 512, 256, torch.float16, False, False),
    ],
)
def test_fused_rotation_quantization_guard(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    is_sm120: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(triton_backend, "_is_sm120", lambda _device: is_sm120)

    assert (
        triton_backend._can_fuse_rotation_quantization(
            rows,
            in_features,
            group_size,
            dtype,
            torch.device("cuda"),
        )
        is expected
    )


@pytest.mark.parametrize(
    ("rows", "in_features", "group_size", "dtype", "capability", "expected"),
    [
        (512, 512, 256, torch.float16, (12, 0), True),
        (512, 14_336, 256, torch.bfloat16, (12, 0), True),
        (511, 512, 256, torch.float16, (12, 0), False),
        (512, 512, 64, torch.float16, (12, 0), False),
        (512, 512, 256, torch.float32, (12, 0), False),
        (512, 16_640, 256, torch.bfloat16, (12, 0), False),
        (512, 512, 256, torch.float16, (12, 1), False),
    ],
)
def test_swiglu_dispatch_guard(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    in_features: int,
    group_size: int,
    dtype: torch.dtype,
    capability: tuple[int, int],
    expected: bool,
) -> None:
    monkeypatch.setattr(convrot_dispatch, "_triton_swiglu_linear", object())
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    with FakeTensorMode():
        activation = torch.empty(
            rows,
            2 * in_features,
            dtype=dtype,
            device="cuda",
        )
        qdata = torch.empty(96, in_features, dtype=torch.int8, device="cuda")

        assert convrot_dispatch._can_use_triton_swiglu(activation, qdata, group_size) is expected


@pytest.mark.parametrize(
    ("operator_name", "input_factor", "with_bias"),
    [
        ("_convrot_int8_linear_op", 1, False),
        ("_convrot_int8_swiglu_linear_op", 2, True),
    ],
)
def test_semantic_linear_fake_kernel_traces_large_shapes_under_fullgraph_compile(
    operator_name: str,
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
    operator = getattr(convrot_dispatch, operator_name)

    def call(
        value: torch.Tensor,
        packed: torch.Tensor,
        weight_scale: torch.Tensor,
        linear_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return operator(value, packed, weight_scale, linear_bias, 256)

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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("operator_name", "rows", "in_features", "group_size", "input_factor"),
    [
        ("_convrot_int8_linear_op", 17, 64, 64, 1),
        ("_convrot_int8_swiglu_linear_op", 512, 512, 256, 2),
    ],
)
def test_cuda_semantic_linear_custom_ops_pass_opcheck(
    operator_name: str,
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
    operator = getattr(convrot_dispatch, operator_name)

    result = torch.library.opcheck(operator, (activation, qdata, scale, bias, group_size))

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_semantic_addmm_custom_op_passes_opcheck() -> None:
    torch.manual_seed(126)
    qdata = torch.randint(-127, 128, (32, 64), dtype=torch.int8, device="cuda")
    scale = torch.rand(32, 1, dtype=torch.float32, device="cuda") * 0.01
    mat1 = torch.randn(32, 4, dtype=torch.bfloat16, device="cuda")
    mat2 = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")

    result = torch.library.opcheck(
        convrot_dispatch._convrot_int8_addmm_op,
        (qdata, scale, mat1, mat2, 64, 0.5, 1.25),
    )

    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("input_activation", [None, "swiglu"], ids=["ordinary", "swiglu"])
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
    weight = ConvRotInt8Tensor.from_packed(qdata, scale, group_size=256)
    input_factor = 1 if input_activation is None else 2
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
        expected = convrot_linear(
            activation,
            weight,
            bias.contiguous(),
            input_activation="swiglu",
        )
        actual = convrot_linear(
            activation,
            weight,
            bias,
            input_activation="swiglu",
        )

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_linear_runs_under_fullgraph_torch_compile() -> None:
    module = nn.Linear(64, 96, bias=True, device="meta", dtype=torch.bfloat16)
    module.weight = nn.Parameter(
        ConvRotInt8Tensor.from_packed(
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
    activation = torch.randn(17, 64, dtype=torch.bfloat16, device="cuda")
    expected = module(activation)
    actual = torch.compile(module, fullgraph=True)(activation)
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_swiglu_linear_runs_under_fullgraph_torch_compile() -> None:
    torch.manual_seed(97)
    rows, in_features, out_features = 512, 512, 96
    qdata = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="cuda",
    )
    scale = torch.rand(out_features, 1, dtype=torch.float32, device="cuda") * 0.01
    weight = ConvRotInt8Tensor.from_packed(qdata, scale, group_size=256)
    raw_activation = torch.randn(
        rows,
        2 * in_features,
        dtype=torch.bfloat16,
        device="cuda",
    )

    def call(value: torch.Tensor) -> torch.Tensor:
        return convrot_linear(value, weight, input_activation="swiglu")

    expected = call(raw_activation)
    actual = torch.compile(call, fullgraph=True)(raw_activation)

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
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

    reference_addmm_(expected.qdata, expected.scale, mat1, mat2, 64, beta, alpha)
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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_addmm_runs_under_torch_compile() -> None:
    torch.manual_seed(30)
    weight = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
    mat1 = torch.randn(32, 4, dtype=torch.bfloat16, device="cuda")
    mat2 = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    expected = ConvRotInt8Tensor.from_hp(weight, group_size=64)
    actual = expected.clone()
    expected.addmm_(mat1, mat2, beta=0.5, alpha=1.25)

    def merge(
        target: ConvRotInt8Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> ConvRotInt8Tensor:
        return target.addmm_(left, right, beta=0.5, alpha=1.25)

    result = torch.compile(merge, fullgraph=True)(actual, mat1, mat2)

    assert result is actual
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("out_features", [0, 3])
def test_triton_addmm_handles_zero_feature_weight(out_features: int) -> None:
    weight = ConvRotInt8Tensor.from_packed(
        torch.empty(out_features, 0, dtype=torch.int8, device="cuda"),
        torch.ones(out_features, 1, dtype=torch.float32, device="cuda"),
        group_size=16,
    )
    mat1 = torch.randn(out_features, 5, dtype=torch.bfloat16, device="cuda")
    mat2 = torch.empty(5, 0, dtype=torch.bfloat16, device="cuda")

    result = weight.addmm_(mat1, mat2, beta=0.5, alpha=1.25)

    assert result is weight
    assert weight.qdata.shape == (out_features, 0)
    assert weight.scale.shape == (out_features, 1)
    assert torch.all(weight.scale == 1e-30)
    assert weight.dequantize().shape == (out_features, 0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_addmm_handles_underflowing_float16_scale() -> None:
    rotated_update = torch.zeros(1, 16, dtype=torch.float16, device="cuda")
    rotated_update[0, 0] = 1e-6
    mat1 = torch.ones(1, 1, dtype=torch.float16, device="cuda")
    mat2 = rotate_groups(rotated_update, 16)
    qdata = torch.zeros(1, 16, dtype=torch.int8, device="cuda")
    scale = torch.ones(1, 1, dtype=torch.float32, device="cuda")
    actual = ConvRotInt8Tensor.from_packed(
        qdata.clone(),
        scale.clone(),
        group_size=16,
        dtype=torch.float16,
    )
    expected = actual.clone()

    reference_addmm_(expected.qdata, expected.scale, mat1, mat2, 16, 0, 1)
    actual.addmm_(mat1, mat2, beta=0)

    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    assert actual.qdata[0, 0] == 127
    assert torch.count_nonzero(actual.qdata[0, 1:]) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_linear_handles_underflowing_float16_activation_scale() -> None:
    rotated_activation = torch.zeros(1, 16, dtype=torch.float16, device="cuda")
    rotated_activation[0, 0] = 1e-6
    activation = rotate_groups(rotated_activation, 16)
    qdata = torch.arange(-8, 8, dtype=torch.int8, device="cuda").reshape(1, 16)
    scale = torch.ones(1, 1, dtype=torch.float32, device="cuda")
    weight = ConvRotInt8Tensor.from_packed(
        qdata,
        scale,
        group_size=16,
        dtype=torch.float16,
    )

    expected = reference_linear(activation, qdata, scale, 16)
    actual = torch.nn.functional.linear(activation, weight)

    assert torch.equal(actual, expected)
