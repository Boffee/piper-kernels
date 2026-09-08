"""Same-shape views preserve quantized weight semantics and shared storage."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

from piper_kernels.linear.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor


def _weight(wrapper, rows=32):
    if wrapper is ConvRotInt8Tensor:
        return wrapper.from_quantized(
            torch.zeros(rows, 64, dtype=torch.int8),
            torch.ones(rows, 1),
            group_size=64,
            logical_dtype=torch.float16,
        )
    return wrapper(
        torch.zeros(rows, 32, dtype=torch.uint8),
        torch.ones(rows, 4, dtype=torch.float8_e4m3fn),
        16,
        torch.float16,
        **({"group_size": 64} if wrapper is ConvRotNVFP4Tensor else {}),
        per_tensor_scale=torch.tensor(0.25),
        act_per_tensor_scale=torch.tensor(0.5),
        is_swizzled_scales=False,
        use_triton_kernel=True,
        act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(
            block_size=16,
            is_swizzled_scales=True,
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=True,
        ),
        high_first=True,
    )


@pytest.fixture(params=[ConvRotInt8Tensor, PiperNVFP4Tensor, ConvRotNVFP4Tensor])
def wrapper(request):
    return request.param


def _assert_weight_view(actual, expected):
    assert actual is not expected
    assert type(actual) is type(expected)
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.dtype is expected.dtype
    assert actual.device == expected.device
    assert torch._C._is_alias_of(actual, expected)
    names, metadata = expected.__tensor_flatten__()
    assert actual.__tensor_flatten__() == (names, metadata)
    for name in names:
        source = getattr(expected, name)
        viewed = getattr(actual, name)
        assert viewed.shape == source.shape
        assert viewed.stride() == source.stride()
        assert viewed.storage_offset() == source.storage_offset()
        assert viewed.dtype is source.dtype
        assert torch._C._is_alias_of(viewed, source)


@pytest.mark.parametrize("operation", ["view", "view_as", "infer_first", "infer_last", "keyword"])
def test_same_shape_view_preserves_wrapper_metadata_and_aliases(wrapper, operation):
    weight = _weight(wrapper)
    if operation == "view":
        actual = weight.view(weight.shape)
    elif operation == "view_as":
        actual = weight.view_as(weight)
    elif operation == "infer_first":
        actual = weight.view(-1, 64)
    elif operation == "infer_last":
        actual = weight.view(32, -1)
    else:
        actual = torch.ops.aten.view.default(self=weight, size=weight.shape)

    _assert_weight_view(actual, weight)
    actual.qdata[0, 0] = 77
    assert weight.qdata[0, 0].item() == 77


@pytest.mark.parametrize("shape", [(64, 32), (16, 128), (2048,), (1, 32, 64)])
def test_shape_changing_view_is_rejected(wrapper, shape):
    weight = _weight(wrapper)
    with pytest.raises(NotImplementedError, match="only supports same-shape views"):
        weight.view(shape)


@pytest.mark.parametrize("shape", [(-1, -1), (31, 64), (-2, 64)])
def test_invalid_view_shape_is_rejected(wrapper, shape):
    weight = _weight(wrapper)
    with pytest.raises(RuntimeError):
        weight.view(shape)


def test_empty_same_shape_view_preserves_wrapper(wrapper):
    weight = _weight(wrapper, rows=0)
    _assert_weight_view(weight.view(0, 64), weight)
    _assert_weight_view(weight.view(-1, 64), weight)
    with pytest.raises(RuntimeError, match="ambiguous"):
        weight.view(0, -1)


@pytest.mark.parametrize("offset", [None, 0])
def test_same_layout_as_strided_preserves_wrapper(wrapper, offset):
    weight = _weight(wrapper)
    _assert_weight_view(weight.as_strided(weight.shape, weight.stride(), offset), weight)


@pytest.mark.parametrize(
    ("shape", "strides", "offset"),
    [((16, 64), (64, 1), 0), ((32, 64), (1, 32), 0), ((32, 64), (64, 1), 1)],
)
def test_as_strided_rejects_changed_layout(wrapper, shape, strides, offset):
    weight = _weight(wrapper)
    with pytest.raises(NotImplementedError, match="unchanged shape, strides, and storage offset"):
        weight.as_strided(shape, strides, storage_offset=offset)


@pytest.fixture(scope="module")
def mesh():
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires distributed Gloo support")
    dist.init_process_group(
        "gloo", store=dist.HashStore(), rank=0, world_size=1, timeout=timedelta(seconds=30)
    )
    try:
        yield DeviceMesh("cpu", [0])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("inference", [False, True])
def test_dtensor_local_round_trip_preserves_wrapper_and_aliases(wrapper, mesh, inference):
    weight = _weight(wrapper)
    with torch.inference_mode(inference):
        distributed = DTensor.from_local(weight, mesh, [Replicate()], run_check=False)
        local = distributed.to_local()
    _assert_weight_view(local, weight)


def test_compiled_same_shape_view_preserves_wrapper_and_aliases(wrapper):
    weight = _weight(wrapper)

    def view(source):
        return source.view(source.shape)

    compiled = torch.compile(view, backend="aot_eager", fullgraph=True)
    _assert_weight_view(compiled(weight), weight)
