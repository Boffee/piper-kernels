"""DTensor linear uses the same quantized interpretation as local Piper linear."""

from datetime import timedelta
from itertools import product

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs

from piper_kernels.linear.convrot import convrot_int8_compile_options
from piper_kernels.linear.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.linear.convrot.nvfp4 import ConvRotNVFP4Tensor, convrot_nvfp4_compile_options
from piper_kernels.linear.nvfp4 import PiperNVFP4Tensor, nvfp4_compile_options
from piper_kernels.linear.nvfp4._layout import swap_packed_pairs

_FORMATS = {
    "int8": (ConvRotInt8Tensor, convrot_int8_compile_options),
    "nvfp4": (PiperNVFP4Tensor, nvfp4_compile_options),
    "convrot_nvfp4": (ConvRotNVFP4Tensor, convrot_nvfp4_compile_options),
}


def _weight(format_name, source, *, high_first=False, dynamic=True):
    cls, _ = _FORMATS[format_name]
    if cls is ConvRotInt8Tensor:
        return cls.from_hp(source, group_size=64)
    result = cls.from_hp(
        source,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        act_per_tensor_scale=None if dynamic else torch.tensor(0.002, device=source.device),
        act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(
            block_size=16,
            is_swizzled_scales=True,
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=dynamic,
        ),
        **({"group_size": 64} if cls is ConvRotNVFP4Tensor else {}),
    )
    if high_first:
        result.qdata = swap_packed_pairs(result.qdata)
        result.high_first = True
    return result


@pytest.fixture(scope="module")
def process_group():
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires distributed Gloo support")
    dist.init_process_group(
        "gloo",
        store=dist.HashStore(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=60),
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _replicated(value, mesh):
    return DTensor.from_local(value, mesh, [Replicate()], run_check=False)


def _linear(format_name, device_type, compiled):
    if not compiled:
        return F.linear
    torch.compiler.reset()
    if device_type == "cpu":
        return torch.compile(F.linear, fullgraph=True, backend="aot_eager")
    _, compile_options = _FORMATS[format_name]
    return torch.compile(
        F.linear,
        fullgraph=True,
        options=compile_options({"triton.cudagraphs": False}),
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_cpu_dtensor_int8_linear(process_group, compiled, bias):
    torch.manual_seed(106)
    weight = ConvRotInt8Tensor.from_hp(torch.randn(7, 64), group_size=64)
    activation = torch.randn(3, 64)
    bias = torch.randn(7) if bias else None
    mesh = DeviceMesh("cpu", [0])
    dw, dx = _replicated(weight, mesh), _replicated(activation, mesh)
    db = None if bias is None else _replicated(bias, mesh)
    linear = _linear("int8", "cpu", compiled)
    with torch.no_grad():
        actual = linear(dx, dw, db).to_local()
        expected = F.linear(activation, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _sharded_inputs(rank, format_name, placement, mesh, with_bias):
    torch.manual_seed(106)
    device_type = mesh.device_type
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
    source = torch.randn(384, 512, dtype=dtype) * 0.02
    activation = torch.randn(32, 512, dtype=dtype)
    bias = torch.randn(384, dtype=dtype) if with_bias else None
    if isinstance(placement, Shard):
        source = source.chunk(2, dim=placement.dim)[rank].contiguous()
        if placement.dim == 0 and bias is not None:
            bias = bias.chunk(2)[rank]
        elif placement.dim == 1:
            activation = activation.chunk(2, dim=1)[rank].contiguous()
    weight = _weight(format_name, source.to(device_type), high_first=format_name != "int8")
    activation = activation.to(device_type)
    bias = None if bias is None else bias.to(device_type)

    # Map weight placement to activation, bias, and output placements.
    if placement == Shard(0):
        input_placement, bias_placement, output_placement = Replicate(), Shard(0), Shard(1)
    elif placement == Shard(1):
        input_placement, bias_placement, output_placement = Shard(1), Replicate(), Partial()
    else:
        input_placement, bias_placement, output_placement = Replicate(), Replicate(), Replicate()
    dx = DTensor.from_local(activation, mesh, [input_placement], run_check=False)
    dw = DTensor.from_local(weight, mesh, [placement], run_check=False)
    db = None if bias is None else DTensor.from_local(bias, mesh, [bias_placement], run_check=False)
    # A replicated bias contributes once when partial products are summed;
    # DTensor divides it between the two ranks.
    local_bias = bias / 2 if bias is not None and placement == Shard(1) else bias
    return (activation, weight, local_bias), (dx, dw, db), output_placement


def _gather_reference(local, placement):
    if isinstance(placement, Replicate):
        return local
    gathered = [torch.empty_like(local) for _ in range(2)]
    dist.all_gather(gathered, local)
    if isinstance(placement, Partial):
        return gathered[0] + gathered[1]
    return torch.cat(gathered, dim=placement.dim)


def _check_sharded_linear(local_args, distributed_args, output_placement, cpu_mesh, linear):
    with torch.no_grad():
        expected = F.linear(*local_args).cpu()
        result = linear(*distributed_args)
    assert result.placements == (output_placement,), (
        f"expected {(output_placement,)}, got {result.placements}"
    )
    local = result.to_local().cpu()
    torch.testing.assert_close(local, expected, rtol=0, atol=0)
    full_expected = _gather_reference(expected, output_placement)
    # Gloo can reduce/gather dense CPU outputs even when two simulated CUDA
    # ranks share one GPU.
    full_actual = DTensor.from_local(
        local,
        cpu_mesh,
        result.placements,
        run_check=False,
    ).full_tensor()
    torch.testing.assert_close(full_actual, full_expected, rtol=0, atol=0)


def _two_rank_linear(rank, store_path, device_type):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(store_path, 2),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        mesh = DeviceMesh(device_type, [0, 1])
        cpu_mesh = DeviceMesh("cpu", [0, 1])
        formats = tuple(_FORMATS) if device_type == "cuda" else ("int8",)
        cases = product(formats, (Replicate(), Shard(0), Shard(1)), (False, True), (False, True))
        for format_name, placement, with_bias, compiled in cases:
            label = (
                f"rank={rank}, device={device_type}, format={format_name}, "
                f"weight_placement={placement}, bias={with_bias}, compiled={compiled}"
            )
            try:
                local_args, distributed_args, output_placement = _sharded_inputs(
                    rank,
                    format_name,
                    placement,
                    mesh,
                    with_bias,
                )
                _check_sharded_linear(
                    local_args,
                    distributed_args,
                    output_placement,
                    cpu_mesh,
                    _linear(format_name, device_type, compiled),
                )
            except Exception as error:
                raise AssertionError(f"{label}: {error}") from error
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_two_rank_replicated_and_sharded_linear(tmp_path, device_type):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires distributed Gloo support")
    if device_type == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0)
    ):
        pytest.skip("requires an SM120 CUDA device")
    mp.spawn(_two_rank_linear, args=(str(tmp_path / "store"), device_type), nprocs=2, join=True)


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("format_name", "case"),
    [
        (format_name, case)
        for format_name in _FORMATS
        for case in ("plain", "bias", "mixed_bias", "batched", "high_first", "static")
        if format_name != "int8" or case not in ("high_first", "static")
    ],
)
def test_cuda_dtensor_linear_matches_local_quantized_weight(
    process_group,
    format_name,
    compiled,
    case,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires an SM120 CUDA device")
    torch.manual_seed(19)
    weight = _weight(
        format_name,
        torch.randn(192, 256, device="cuda", dtype=torch.bfloat16) * 0.02,
        high_first=case == "high_first",
        dynamic=case != "static",
    )
    shape = (2, 16, 256) if case == "batched" else (32, 256)
    activation = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    bias = None
    if case in ("bias", "mixed_bias"):
        bias_dtype = torch.float32 if case == "mixed_bias" else torch.bfloat16
        bias = torch.randn(192, device="cuda", dtype=bias_dtype)
    mesh = DeviceMesh("cuda", [0])
    dw, dx = _replicated(weight, mesh), _replicated(activation, mesh)
    db = None if bias is None else _replicated(bias, mesh)
    linear = _linear(format_name, "cuda", compiled)
    with torch.no_grad():
        transposed = dw.t().to_local()
        assert type(transposed) is type(weight)
        assert transposed.transposed
        if hasattr(weight, "high_first"):
            assert transposed.high_first == weight.high_first
        if hasattr(weight, "group_size"):
            assert transposed.group_size == weight.group_size
        expected = F.linear(activation, weight, bias)
        actual = linear(dx, dw, db).to_local()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
