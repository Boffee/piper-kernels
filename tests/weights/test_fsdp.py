"""Unit, CPU Gloo FSDP2, and 2-GPU RX 9070 XT FSDP2 integration tests."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _compile_capture import FunctionalTargetCapturePass, TargetCapturePass
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn import functional as F  # noqa: N812

from piper_kernels import compile_options, piper_attention
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor
from piper_kernels.weights.sharding import shard_quantized_weight

_FORMATS = [(ConvRotInt8Tensor, False, False)] + [
    (cls, swizzled, high_first)
    for cls in (PiperNVFP4Tensor, ConvRotNVFP4Tensor)
    for swizzled in (False, True)
    for high_first in (False, True)
]

_HIDDEN = 256
_INTERMEDIATE = 512
_HEADS = 2
_HEAD_DIM = 128
_ROTARY_DIM = 64


def _swap_nibbles(qdata: torch.Tensor) -> torch.Tensor:
    return ((qdata & 0x0F) << 4) | (qdata >> 4)


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = float(expected.float().norm().item())
    if denom == 0.0:
        return float(actual.float().norm().item())
    return float((actual.float() - expected.float()).norm().item()) / denom


def _make_weight(
    cls: type[Any],
    *,
    rows: int = 32,
    features: int = 128,
    group_size: int = 64,
    swizzled: bool = False,
    high_first: bool = False,
    with_global_scales: bool = True,
    logical_dtype: torch.dtype = torch.bfloat16,
    seed: int = 101,
    device: str | torch.device = "cpu",
) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randn(rows, features, generator=generator, dtype=logical_dtype) * 0.05
    if cls is ConvRotInt8Tensor:
        act_scale = torch.tensor(0.025, dtype=torch.float32) if with_global_scales else None
        weight = cls.from_hp(source, group_size=group_size, act_per_tensor_scale=act_scale)
    else:
        options: dict[str, Any] = {"is_swizzled_scales": swizzled}
        if cls is ConvRotNVFP4Tensor:
            options["group_size"] = group_size
        if with_global_scales:
            options["compute_per_tensor_scale"] = True
            options["act_per_tensor_scale"] = torch.tensor(0.015, dtype=torch.float32)
            options["act_quant"] = "static"
        weight = cls.from_hp(source, **options)
        if high_first:
            weight.qdata = _swap_nibbles(weight.qdata)
            weight.high_first = True
    return weight.to(device) if torch.device(device).type != "cpu" else weight


def _make_meta_like(weight: Any) -> Any:
    names, attrs = weight.__tensor_flatten__()
    meta_tensors = {
        name: torch.empty(
            getattr(weight, name).shape,
            dtype=getattr(weight, name).dtype,
            device="meta",
        )
        for name in names
    }
    return type(weight).__tensor_unflatten__(meta_tensors, attrs, None, None)


def _assert_weight_close(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    assert actual.shape == expected.shape
    assert actual.dtype is expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(
        actual.dequantize(torch.float32),
        expected.dequantize(torch.float32),
        rtol=0,
        atol=0,
    )
    if getattr(expected, "per_tensor_scale", None) is not None:
        torch.testing.assert_close(
            actual.per_tensor_scale, expected.per_tensor_scale, rtol=0, atol=0
        )
    else:
        assert getattr(actual, "per_tensor_scale", None) is None
    if getattr(expected, "act_per_tensor_scale", None) is not None:
        torch.testing.assert_close(
            actual.act_per_tensor_scale, expected.act_per_tensor_scale, rtol=0, atol=0
        )
    else:
        assert getattr(actual, "act_per_tensor_scale", None) is None


# ---------------------------------------------------------------------------
# Single-process CPU unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "swizzled", "high_first"), _FORMATS)
@pytest.mark.parametrize("rows", [32, 256])
def test_split_contract_across_formats(cls, swizzled, high_first, rows):
    weight = _make_weight(cls, rows=rows, swizzled=swizzled, high_first=high_first)
    reference = weight.dequantize(torch.float32)

    half = rows // 2
    chunks = torch.split(weight, half, dim=0)
    neg_dim_chunks = weight.split(half, dim=-2)
    assert isinstance(chunks, tuple)
    assert len(chunks) == 2
    assert len(neg_dim_chunks) == 2
    for c0, c1 in zip(neg_dim_chunks, chunks, strict=True):
        _assert_weight_close(c0, c1)

    zero_copy_scale = (not swizzled) or (half % 128 == 0)
    for idx, chunk in enumerate(chunks):
        assert type(chunk) is type(weight)
        assert chunk.shape == (half, weight.shape[1])
        torch.testing.assert_close(
            chunk.dequantize(torch.float32),
            reference.narrow(0, idx * half, half),
            rtol=0,
            atol=0,
        )
        if zero_copy_scale:
            assert torch._C._is_alias_of(chunk, weight)
            assert torch._C._is_alias_of(chunk.qdata, weight.qdata)
            assert torch._C._is_alias_of(chunk.scale, weight.scale)
        else:
            assert not torch._C._is_alias_of(chunk.scale, weight.scale)

    # Full or oversized split returns one full-extent alias chunk.
    single = weight.split(rows * 2, dim=0)
    assert len(single) == 1
    assert torch._C._is_alias_of(single[0], weight)
    assert torch._C._is_alias_of(single[0].scale, weight.scale)


def test_split_swizzled_flat_scale_and_meta_device():
    full = _make_weight(ConvRotNVFP4Tensor, rows=32, swizzled=True, high_first=True)
    full.scale = full.scale.flatten()
    chunks = full.split(16, dim=0)
    assert len(chunks) == 2
    expected = full.dequantize(torch.float32)
    torch.testing.assert_close(chunks[0].dequantize(torch.float32), expected[:16], rtol=0, atol=0)
    torch.testing.assert_close(chunks[1].dequantize(torch.float32), expected[16:], rtol=0, atol=0)

    meta = _make_meta_like(full)
    meta_chunks = meta.split(16, dim=0)
    assert len(meta_chunks) == 2
    assert all(c.device.type == "meta" and c.shape == (16, 128) for c in meta_chunks)


@pytest.mark.parametrize(("cls", "swizzled", "high_first"), _FORMATS)
def test_split_rejections(cls, swizzled, high_first):
    weight = _make_weight(cls, rows=32, swizzled=swizzled, high_first=high_first)
    with pytest.raises(NotImplementedError, match="dim=0"):
        weight.split(64, dim=1)
    with pytest.raises(NotImplementedError, match="even dim=0"):
        weight.split(10, dim=0)
    with pytest.raises(ValueError, match="positive"):
        weight.split(0, dim=0)
    with pytest.raises(IndexError, match="out of range"):
        weight.split(16, dim=2)
    with pytest.raises(NotImplementedError, match="transposed"):
        weight.t().split(16, dim=0)


def test_split_rejects_higher_rank_weights():
    conv3d = ConvRotInt8Tensor.from_quantized(
        torch.zeros(4, 3, 3, 3, 64, dtype=torch.int8),
        torch.ones(4, 1),
        group_size=64,
        logical_dtype=torch.bfloat16,
    )
    moe = PiperNVFP4Tensor.from_hp(torch.randn(2, 32, 64, dtype=torch.bfloat16))
    for weight in (conv3d, moe):
        with pytest.raises(NotImplementedError, match="2-D"):
            weight.split(1, dim=0)


@pytest.mark.parametrize(("cls", "swizzled", "high_first"), _FORMATS)
@pytest.mark.parametrize("flat_scales", [False, True])
@pytest.mark.parametrize("with_global_scales", [False, True])
def test_new_zeros_and_copy_round_trip(cls, swizzled, high_first, flat_scales, with_global_scales):
    src = _make_weight(
        cls,
        rows=32,
        swizzled=swizzled,
        high_first=high_first,
        with_global_scales=with_global_scales,
        logical_dtype=torch.float16,
    )
    if flat_scales and cls is not ConvRotInt8Tensor:
        src.scale = src.scale.flatten()

    assert src.is_pinned() is False

    dst = src.new_zeros((16, src.shape[1]))
    assert type(dst) is type(src)
    assert dst.shape == (16, src.shape[1])
    assert dst.dtype is torch.float16
    assert torch.count_nonzero(dst.qdata) == 0

    full_dst = src.new_zeros(src.shape, dtype=torch.bfloat16)
    assert full_dst.dtype is torch.bfloat16
    assert not torch._C._is_alias_of(full_dst, src)
    ret = full_dst.copy_(src)
    assert ret is full_dst
    assert full_dst.dtype is torch.bfloat16
    torch.testing.assert_close(
        full_dst.dequantize(torch.float32),
        src.dequantize(torch.float32),
        rtol=0,
        atol=0,
    )
    for name in src.__tensor_flatten__()[0]:
        assert not torch._C._is_alias_of(getattr(full_dst, name), getattr(src, name))

    meta_dst = src.new_zeros(src.shape, device="meta")
    assert meta_dst.device.type == "meta"
    assert meta_dst.shape == src.shape


def test_copy_and_new_zeros_rejections_preserve_dst():
    dst = _make_weight(ConvRotNVFP4Tensor, rows=32, swizzled=True, high_first=True)
    snap_q = dst.qdata.clone()
    snap_s = dst.scale.clone()
    snap_pts = dst.per_tensor_scale.clone()
    snap_apts = dst.act_per_tensor_scale.clone()

    no_globals_same_act_quant = _make_weight(
        ConvRotNVFP4Tensor,
        rows=32,
        swizzled=True,
        high_first=True,
        with_global_scales=False,
    )
    no_globals_same_act_quant.act_quant = dst.act_quant

    for bad_src, err in (
        (torch.zeros(32, 128, dtype=torch.bfloat16), NotImplementedError),
        (
            _make_weight(PiperNVFP4Tensor, rows=32, swizzled=True, high_first=True),
            NotImplementedError,
        ),
        (_make_weight(ConvRotNVFP4Tensor, rows=16, swizzled=True, high_first=True), RuntimeError),
        (
            _make_weight(ConvRotNVFP4Tensor, rows=32, swizzled=True, high_first=True).t(),
            NotImplementedError,
        ),
        (
            _make_weight(
                ConvRotNVFP4Tensor, rows=32, group_size=16, swizzled=True, high_first=True
            ),
            NotImplementedError,
        ),
        (
            _make_weight(ConvRotNVFP4Tensor, rows=32, swizzled=False, high_first=True),
            NotImplementedError,
        ),
        (
            _make_weight(ConvRotNVFP4Tensor, rows=32, swizzled=True, high_first=False),
            NotImplementedError,
        ),
        (
            _make_weight(
                ConvRotNVFP4Tensor,
                rows=32,
                swizzled=True,
                high_first=True,
                with_global_scales=False,
            ),
            NotImplementedError,
        ),
        (no_globals_same_act_quant, ValueError),
    ):
        with pytest.raises(err):
            dst.copy_(bad_src)

    torch.testing.assert_close(dst.qdata, snap_q, rtol=0, atol=0)
    torch.testing.assert_close(
        dst.scale.view(torch.uint8), snap_s.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(dst.per_tensor_scale, snap_pts, rtol=0, atol=0)
    torch.testing.assert_close(dst.act_per_tensor_scale, snap_apts, rtol=0, atol=0)

    with pytest.raises(NotImplementedError):
        dst.new_zeros((16, 64))
    with pytest.raises(NotImplementedError):
        dst.new_zeros((16, 128, 1))
    with pytest.raises(NotImplementedError):
        dst.new_zeros((-1, 128))
    with pytest.raises(TypeError):
        dst.new_zeros((16, 128), dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        dst.new_zeros((16, 128), layout=torch.sparse_coo)


def test_fsdp2_single_rank_end_to_end() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires distributed Gloo support")
    dist.init_process_group(
        "gloo", store=dist.HashStore(), rank=0, world_size=1, timeout=timedelta(seconds=30)
    )
    try:
        mesh_one = DeviceMesh("cpu", [0])
        for cls, swizzled in (
            (ConvRotInt8Tensor, False),
            (PiperNVFP4Tensor, True),
            (ConvRotNVFP4Tensor, True),
        ):
            ref_layer, ref_weight = _build_quantized_linear(
                cls,
                rows=32,
                features=128,
                swizzled=swizzled,
                with_global_scales=True,
                bias=False,
            )
            inputs, meta = ref_weight.fsdp_pre_all_gather(mesh_one)
            assert len(inputs) == 1
            unsharded, extra = ref_weight.fsdp_post_all_gather(inputs, meta, torch.bfloat16)
            assert len(extra) == 1
            _assert_weight_close(unsharded, ref_weight)

            fsdp_layer, _ = _build_quantized_linear(
                cls,
                rows=32,
                features=128,
                swizzled=swizzled,
                with_global_scales=True,
                bias=False,
            )
            fully_shard(fsdp_layer, mesh=mesh_one, reshard_after_forward=True)
            x = torch.randn(3, 128, dtype=torch.bfloat16)
            torch.testing.assert_close(fsdp_layer(x), ref_layer(x), rtol=0, atol=0)
            torch.testing.assert_close(fsdp_layer(x), ref_layer(x), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_convrot_int8_fsdp_does_not_import_nvfp4() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires distributed Gloo support")
    script = """
import sys
from datetime import timedelta

class _BlockOptionalNVFP4:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("piper_kernels.weights.nvfp4"):
            raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
        return None

sys.meta_path.insert(0, _BlockOptionalNVFP4())

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

weight = ConvRotInt8Tensor.from_hp(
    torch.randn(32, 128, dtype=torch.bfloat16) * 0.05,
    group_size=64,
    act_per_tensor_scale=torch.tensor(0.025, dtype=torch.float32),
)
chunks = weight.split(16, dim=0)
assert len(chunks) == 2
copy_dst = weight.new_zeros(weight.shape)
copy_dst.copy_(weight)
assert weight.is_pinned() is False
assert weight.view(-1).shape == (weight.numel(),)
assert weight[:].shape == weight.shape

dist.init_process_group(
    "gloo", store=dist.HashStore(), rank=0, world_size=1, timeout=timedelta(seconds=30)
)
try:
    mesh = DeviceMesh("cpu", [0])
    inputs, meta = chunks[0].fsdp_pre_all_gather(mesh)
    assert len(inputs) == 1
    unsharded, extra = chunks[0].fsdp_post_all_gather(inputs, meta, torch.bfloat16)
    assert len(extra) == 1
    chunks[0].fsdp_post_all_gather(inputs, meta, torch.bfloat16, out=unsharded)

    layer = nn.Linear(128, 32, bias=False, dtype=torch.bfloat16)
    layer.weight = nn.Parameter(weight, requires_grad=False)
    fully_shard(layer, mesh=mesh, reshard_after_forward=True)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    _ = layer(x)
    _ = layer(x)
finally:
    dist.destroy_process_group()

assert "piper_kernels.weights.nvfp4" not in sys.modules
"""
    env = os.environ.copy()
    env.setdefault("HIP_VISIBLE_DEVICES", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_fsdp_post_all_gather_validation_preserves_out() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires distributed Gloo support")
    dist.init_process_group(
        "gloo", store=dist.HashStore(), rank=0, world_size=1, timeout=timedelta(seconds=30)
    )
    try:
        mesh_one = DeviceMesh("cpu", [0])
        int8_w = _make_weight(ConvRotInt8Tensor, rows=32, features=128)
        nvfp4_w = _make_weight(
            ConvRotNVFP4Tensor, rows=32, features=128, swizzled=True, high_first=True, seed=201
        )
        nvfp4_inputs, nvfp4_meta = nvfp4_w.fsdp_pre_all_gather(mesh_one)
        assert len(nvfp4_inputs) == 1

        with pytest.raises(TypeError, match="does not match metadata"):
            int8_w.fsdp_post_all_gather(nvfp4_inputs, nvfp4_meta, torch.bfloat16)
        with pytest.raises(ValueError, match="expected 1 all-gather output"):
            nvfp4_w.fsdp_post_all_gather(
                (nvfp4_inputs[0], nvfp4_w.scale), nvfp4_meta, torch.bfloat16
            )
        with pytest.raises(ValueError, match="qdata mismatch"):
            nvfp4_w.fsdp_post_all_gather(
                (nvfp4_inputs[0].to(torch.int16),), nvfp4_meta, torch.bfloat16
            )
        with pytest.raises(ValueError, match="qdata mismatch"):
            nvfp4_w.fsdp_post_all_gather(
                (nvfp4_inputs[0].t().contiguous().t(),),
                nvfp4_meta,
                torch.bfloat16,
            )

        out_w, _ = nvfp4_w.fsdp_post_all_gather(nvfp4_inputs, nvfp4_meta, torch.bfloat16)
        snap_q = out_w.qdata.clone()
        snap_s = out_w.scale.clone()
        snap_pts = out_w.per_tensor_scale.clone()
        snap_apts = out_w.act_per_tensor_scale.clone()

        alt_w = _make_weight(
            ConvRotNVFP4Tensor, rows=32, features=128, swizzled=True, high_first=True, seed=202
        )
        alt_w.per_tensor_scale.fill_(9.0)
        alt_w.act_per_tensor_scale.fill_(7.0)
        _, alt_meta = alt_w.fsdp_pre_all_gather(mesh_one)

        with pytest.raises(RuntimeError, match=r"expected out\.qdata to alias"):
            nvfp4_w.fsdp_post_all_gather(
                (nvfp4_inputs[0].clone(),),
                alt_meta,
                torch.bfloat16,
                out=out_w,
            )
        with pytest.raises(ValueError, match="dtype mismatch"):
            nvfp4_w.fsdp_post_all_gather(
                nvfp4_inputs,
                alt_meta,
                torch.float16,
                out=out_w,
            )

        out_w.scale = out_w.scale[:, :16].contiguous()
        snap_bad_s = out_w.scale.clone()
        with pytest.raises(ValueError, match="target scale mismatch"):
            nvfp4_w.fsdp_post_all_gather(
                nvfp4_inputs,
                alt_meta,
                torch.bfloat16,
                out=out_w,
            )

        torch.testing.assert_close(out_w.qdata, snap_q, rtol=0, atol=0)
        torch.testing.assert_close(
            out_w.scale.view(torch.uint8), snap_bad_s.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(out_w.per_tensor_scale, snap_pts, rtol=0, atol=0)
        torch.testing.assert_close(out_w.act_per_tensor_scale, snap_apts, rtol=0, atol=0)
        _ = snap_s
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 2-rank CPU Gloo FSDP2 tests
# ---------------------------------------------------------------------------


class _CPUDequantLinear(nn.Linear):
    """Run F.linear on dequantized unsharded NVFP4 weights on CPU (no FP4 scaled_mm kernel)."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return F.linear(input, self.weight.dequantize(self.weight.dtype), self.bias)


def _linear_module_cls(weight_cls: type[Any]) -> type[nn.Linear]:
    return nn.Linear if weight_cls is ConvRotInt8Tensor else _CPUDequantLinear


def _build_quantized_linear(
    cls: type[Any],
    *,
    rows: int,
    features: int,
    swizzled: bool = False,
    high_first: bool = False,
    with_global_scales: bool = True,
    flat_scales: bool = False,
    bias: bool = True,
    logical_dtype: torch.dtype = torch.bfloat16,
    seed: int = 200,
) -> tuple[nn.Linear, Any]:
    weight = _make_weight(
        cls,
        rows=rows,
        features=features,
        swizzled=swizzled,
        high_first=high_first,
        with_global_scales=with_global_scales,
        logical_dtype=logical_dtype,
        seed=seed,
    )
    if flat_scales and cls is not ConvRotInt8Tensor:
        weight.scale = weight.scale.flatten()
    layer = _linear_module_cls(cls)(features, rows, bias=bias, dtype=logical_dtype)
    layer.weight = nn.Parameter(weight, requires_grad=False)
    if bias:
        gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        layer.bias = nn.Parameter(
            torch.randn(rows, generator=gen, dtype=logical_dtype) * 0.1, requires_grad=False
        )
    return layer, weight


def _load_meta_fsdp2_linear(
    cls: type[Any],
    mesh: DeviceMesh,
    rank: int,
    world: int,
    full_weight: Any,
    full_bias: torch.Tensor,
    *,
    reshard_after_forward: bool,
    mp_policy: MixedPrecisionPolicy,
) -> nn.Linear:
    rows, features = full_weight.shape
    layer = _linear_module_cls(cls)(features, rows, bias=True, device="meta", dtype=torch.bfloat16)
    layer.weight = nn.Parameter(_make_meta_like(full_weight), requires_grad=False)
    fully_shard(
        layer,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=mp_policy,
    )
    local_rows = rows // world
    local_w = shard_quantized_weight(full_weight, dim=0, start=rank * local_rows, length=local_rows)
    local_b = full_bias.narrow(0, rank * local_rows, local_rows).clone()
    layer.load_state_dict(
        {
            "weight": DTensor.from_local(
                local_w,
                mesh,
                [Shard(0)],
                run_check=False,
                shape=full_weight.shape,
                stride=full_weight.stride(),
            ),
            "bias": DTensor.from_local(
                local_b,
                mesh,
                [Shard(0)],
                run_check=False,
                shape=full_bias.shape,
                stride=full_bias.stride(),
            ),
        },
        assign=True,
    )
    return layer


def _check_gloo_fsdp2_linear(
    mesh: DeviceMesh,
    rank: int,
    world: int,
    *,
    cls: type[Any],
    rows: int,
    features: int,
    seed: int = 300,
    options: dict[str, Any] | None = None,
) -> None:
    cfg = dict(options or {})
    meta_init = cfg.pop("meta_init", False)
    inference_mode = cfg.pop("inference_mode", False)
    reshard_after_forward = cfg.pop("reshard_after_forward", True)
    mp_param_dtype = cfg.pop("mp_param_dtype", None)
    build_kwargs = {
        "rows": rows,
        "features": features,
        "swizzled": cfg.pop("swizzled", False),
        "high_first": cfg.pop("high_first", False),
        "with_global_scales": cfg.pop("with_global_scales", True),
        "flat_scales": cfg.pop("flat_scales", False),
        "seed": seed,
    }
    assert not cfg

    ref_layer, full_weight = _build_quantized_linear(cls, **build_kwargs)
    assert ref_layer.bias is not None
    ref_weight = full_weight.to(mp_param_dtype) if mp_param_dtype is not None else full_weight
    ref_bias = ref_layer.bias.to(mp_param_dtype) if mp_param_dtype is not None else ref_layer.bias

    def _ref_forward(x: torch.Tensor) -> torch.Tensor:
        w = ref_weight if cls is ConvRotInt8Tensor else ref_weight.dequantize(ref_weight.dtype)
        return F.linear(x, w, ref_bias)

    mp_policy = MixedPrecisionPolicy(param_dtype=mp_param_dtype)
    ctx = torch.inference_mode(inference_mode) if inference_mode else torch.no_grad()
    with ctx:
        if meta_init:
            layer = _load_meta_fsdp2_linear(
                cls,
                mesh,
                rank,
                world,
                full_weight,
                ref_layer.bias,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )
        else:
            layer, _ = _build_quantized_linear(cls, **build_kwargs)
            fully_shard(
                layer,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )

        assert isinstance(layer.weight, DTensor)
        assert type(layer.weight.to_local()) is cls

        expected_dtype = mp_param_dtype or full_weight.dtype
        for step in range(3):
            x_gen = torch.Generator(device="cpu").manual_seed(seed + 100 * step + rank)
            x = torch.randn(4, features, generator=x_gen, dtype=expected_dtype)
            actual = layer(x)
            assert actual.dtype is expected_dtype
            torch.testing.assert_close(actual, _ref_forward(x), rtol=0, atol=0)

        if not reshard_after_forward:
            layer.reshard()
        layer.unshard()
        assert type(layer.weight.data) is cls
        _assert_weight_close(layer.weight.data, ref_weight)
        layer.reshard()
        x = torch.randn(2, features, dtype=expected_dtype)
        torch.testing.assert_close(layer(x), _ref_forward(x), rtol=0, atol=0)


def _check_gloo_nvfp4_scale_paths(mesh: DeviceMesh, rank: int) -> None:
    # Verify 2-rank NVFP4 scale path: zero-copy for local_rows % 128 == 0 vs uint8 unswizzle.
    aligned_shard = _make_weight(PiperNVFP4Tensor, rows=128, features=128, swizzled=True)
    aligned_inputs, aligned_meta = aligned_shard.fsdp_pre_all_gather(mesh)
    assert aligned_inputs[1].dtype is torch.float8_e4m3fn
    assert aligned_inputs[1].shape == aligned_shard.scale.shape
    assert torch._C._is_alias_of(aligned_inputs[1], aligned_shard.scale)
    if rank == 0:
        q2 = torch.cat([aligned_inputs[0], aligned_inputs[0]], dim=0)
        s2_bytes = aligned_inputs[1].view(torch.uint8)
        s2 = torch.cat([s2_bytes, s2_bytes], dim=0).view(torch.float8_e4m3fn)
        aligned_out, aligned_extra = aligned_shard.fsdp_post_all_gather(
            (q2, s2), aligned_meta, torch.bfloat16
        )
        assert aligned_extra == ()
        for bad_scale in (
            s2.view(torch.uint8),
            s2_bytes.t().contiguous().t().view(torch.float8_e4m3fn),
            s2[:1],
        ):
            with pytest.raises(ValueError, match="scale mismatch"):
                aligned_shard.fsdp_post_all_gather((q2, bad_scale), aligned_meta, torch.bfloat16)

        alt_aligned = _make_weight(
            PiperNVFP4Tensor, rows=128, features=128, swizzled=True, seed=776
        )
        alt_aligned.per_tensor_scale.fill_(9.0)
        alt_aligned.act_per_tensor_scale.fill_(7.0)
        _, alt_aligned_meta = alt_aligned.fsdp_pre_all_gather(mesh)
        snap_q = aligned_out.qdata.clone()
        snap_s = aligned_out.scale.view(torch.uint8).clone()
        snap_pts = aligned_out.per_tensor_scale.clone()
        snap_apts = aligned_out.act_per_tensor_scale.clone()
        with pytest.raises(RuntimeError, match=r"expected out\.scale to alias"):
            aligned_shard.fsdp_post_all_gather(
                (q2, s2.view(torch.uint8).clone().view(torch.float8_e4m3fn)),
                alt_aligned_meta,
                torch.bfloat16,
                out=aligned_out,
            )
        torch.testing.assert_close(aligned_out.qdata, snap_q, rtol=0, atol=0)
        torch.testing.assert_close(aligned_out.scale.view(torch.uint8), snap_s, rtol=0, atol=0)
        torch.testing.assert_close(aligned_out.per_tensor_scale, snap_pts, rtol=0, atol=0)
        torch.testing.assert_close(aligned_out.act_per_tensor_scale, snap_apts, rtol=0, atol=0)

    unaligned_shard = _make_weight(PiperNVFP4Tensor, rows=16, features=128, swizzled=True)
    unaligned_inputs, unaligned_meta = unaligned_shard.fsdp_pre_all_gather(mesh)
    assert unaligned_inputs[1].dtype is torch.uint8
    if rank == 0:
        q_full = torch.cat([unaligned_inputs[0], unaligned_inputs[0]], dim=0)
        s_full = torch.cat([unaligned_inputs[1], unaligned_inputs[1]], dim=0)
        bad_out, _ = unaligned_shard.fsdp_post_all_gather(
            (q_full, s_full), unaligned_meta, torch.bfloat16
        )
        bad_out.scale = bad_out.scale[:, :16].contiguous()
        alt_shard = _make_weight(PiperNVFP4Tensor, rows=16, features=128, swizzled=True, seed=777)
        alt_shard.per_tensor_scale.fill_(9.0)
        alt_shard.act_per_tensor_scale.fill_(7.0)
        _, alt_unaligned_meta = alt_shard.fsdp_pre_all_gather(mesh)
        snap_pts = bad_out.per_tensor_scale.clone()
        snap_apts = bad_out.act_per_tensor_scale.clone()
        with pytest.raises(ValueError, match="target scale mismatch"):
            unaligned_shard.fsdp_post_all_gather(
                (q_full, s_full), alt_unaligned_meta, torch.bfloat16, out=bad_out
            )
        torch.testing.assert_close(bad_out.per_tensor_scale, snap_pts, rtol=0, atol=0)
        torch.testing.assert_close(bad_out.act_per_tensor_scale, snap_apts, rtol=0, atol=0)


def _cpu_gloo_worker(rank: int, world: int, store_path: str) -> None:
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world,
        store=dist.FileStore(store_path, world),
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))

        # All 3 formats, direct fully_shard, swizzled aligned (zero-copy) & unaligned (reswizzled),
        # unswizzled, flat scales, high_first, with and without global scales, no_grad & inference.
        for cls, swizzled, high_first in _FORMATS:
            rows = 256 if (cls is not ConvRotInt8Tensor and swizzled) else 32
            _check_gloo_fsdp2_linear(
                mesh,
                rank,
                world,
                cls=cls,
                rows=rows,
                features=128,
                seed=400 + int(swizzled) * 10 + int(high_first),
                options={"swizzled": swizzled, "high_first": high_first},
            )

        # Swizzled NVFP4 reswizzle path (local_rows=16 % 128 != 0) across 3 steps.
        for cls in (PiperNVFP4Tensor, ConvRotNVFP4Tensor):
            _check_gloo_fsdp2_linear(
                mesh,
                rank,
                world,
                cls=cls,
                rows=32,
                features=128,
                seed=500,
                options={"swizzled": True, "flat_scales": True, "with_global_scales": False},
            )

        # ConvRotInt8 without global act_per_tensor_scale.
        _check_gloo_fsdp2_linear(
            mesh,
            rank,
            world,
            cls=ConvRotInt8Tensor,
            rows=32,
            features=128,
            seed=550,
            options={"with_global_scales": False, "reshard_after_forward": False},
        )

        # Meta init + load_state_dict(assign=True) + inference_mode across all 3 classes.
        for idx, cls in enumerate((ConvRotInt8Tensor, PiperNVFP4Tensor, ConvRotNVFP4Tensor)):
            _check_gloo_fsdp2_linear(
                mesh,
                rank,
                world,
                cls=cls,
                rows=32,
                features=128,
                seed=600 + idx,
                options={
                    "swizzled": cls is not ConvRotInt8Tensor,
                    "meta_init": True,
                    "inference_mode": True,
                },
            )

        # MixedPrecisionPolicy(param_dtype=float16) changes logical dtype without requantizing.
        for cls in (ConvRotInt8Tensor, ConvRotNVFP4Tensor):
            _check_gloo_fsdp2_linear(
                mesh,
                rank,
                world,
                cls=cls,
                rows=32,
                features=128,
                seed=700,
                options={
                    "swizzled": cls is ConvRotNVFP4Tensor,
                    "mp_param_dtype": torch.float16,
                },
            )

        # 2-D FSDP + TP wrapping DTensor parameter on ("dp", "tp") submeshes.
        mesh_2d = init_device_mesh("cpu", (world, 1), mesh_dim_names=("dp", "tp"))
        ref_2d, _ = _build_quantized_linear(
            ConvRotInt8Tensor, rows=32, features=128, bias=False, seed=800
        )
        tp_layer, _ = _build_quantized_linear(
            ConvRotInt8Tensor, rows=32, features=128, bias=False, seed=800
        )
        with torch.no_grad():
            parallelize_module(
                tp_layer,
                mesh_2d["tp"],
                ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
                src_data_rank=None,
            )
            fully_shard(tp_layer, mesh=mesh_2d["dp"], reshard_after_forward=True)
            for step in range(3):
                gen = torch.Generator(device="cpu").manual_seed(850 + step + rank)
                x = torch.randn(3, 128, generator=gen, dtype=torch.bfloat16)
                torch.testing.assert_close(tp_layer(x), ref_2d(x), rtol=0, atol=0)

        # Rejections: uneven dim-0 rows and Shard(1) placement.
        uneven, _ = _build_quantized_linear(
            ConvRotInt8Tensor, rows=31, features=64, bias=False, seed=900
        )
        with pytest.raises(NotImplementedError):
            fully_shard(uneven, mesh=mesh)

        col_bad, _ = _build_quantized_linear(
            ConvRotInt8Tensor, rows=32, features=64, bias=False, seed=901
        )
        with pytest.raises(NotImplementedError):
            fully_shard(col_bad, mesh=mesh, shard_placement_fn=lambda _p: Shard(1))

        _check_gloo_nvfp4_scale_paths(mesh, rank)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_fsdp2_cpu_gloo_end_to_end(tmp_path: Path) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires distributed Gloo support")
    mp.spawn(_cpu_gloo_worker, args=(2, str(tmp_path / "gloo_store")), nprocs=2, join=True)


# ---------------------------------------------------------------------------
# 2-GPU RX 9070 XT FSDP2 + torch.compile + hybrid TP+SP Attention tests
# ---------------------------------------------------------------------------


class _QuantizedSwiGLUFFN(nn.Module):
    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        *,
        act_per_tensor_scale: torch.Tensor | None = None,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.up = nn.Linear(_HIDDEN, _INTERMEDIATE, bias=False, dtype=torch.bfloat16)
        self.gate = nn.Linear(_HIDDEN, _INTERMEDIATE, bias=False, dtype=torch.bfloat16)
        self.down = nn.Linear(_INTERMEDIATE, _HIDDEN, bias=False, dtype=torch.bfloat16)
        for name in ("up", "gate", "down"):
            layer = getattr(self, name)
            qweight = ConvRotInt8Tensor.from_hp(
                weights[name], group_size=64, act_per_tensor_scale=act_per_tensor_scale
            ).to(device)
            layer.weight = nn.Parameter(qweight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _StackedFFN(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _ResidualFFN(nn.Module):
    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        gate_table: torch.Tensor,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.ff = _QuantizedSwiGLUFFN(weights, device=device)
        self.gate_table = nn.Parameter(gate_table.to(device), requires_grad=False)

    def forward(self, hidden: torch.Tensor, gate_indices: torch.Tensor) -> torch.Tensor:
        return hidden.copy_(hidden + self.gate_table[gate_indices] * self.ff(hidden))


class _RMSNorm(nn.Module):
    def __init__(self, values: torch.Tensor, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(values.to(device), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, 1e-5)


class _DenseAttn(nn.Module):
    def __init__(
        self,
        heads: int,
        weights: dict[str, torch.Tensor],
        norms: dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(_HIDDEN, _HIDDEN, bias=False, dtype=torch.bfloat16)
        self.to_k = nn.Linear(_HIDDEN, _HIDDEN, bias=False, dtype=torch.bfloat16)
        self.to_v = nn.Linear(_HIDDEN, _HIDDEN, bias=False, dtype=torch.bfloat16)
        self.norm_q = _RMSNorm(norms["norm_q"], device=device)
        self.norm_k = _RMSNorm(norms["norm_k"], device=device)
        self.to_out = nn.ModuleList([nn.Linear(_HIDDEN, _HIDDEN, bias=False, dtype=torch.bfloat16)])
        for name, mod in (
            ("to_q", self.to_q),
            ("to_k", self.to_k),
            ("to_v", self.to_v),
            ("to_out.0", self.to_out[0]),
        ):
            mod.weight = nn.Parameter(
                ConvRotInt8Tensor.from_hp(weights[name], group_size=64).to(device),
                requires_grad=False,
            )

    def _norm_rope(
        self, proj: torch.Tensor, norm: _RMSNorm, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, _ = proj.shape
        x = norm(proj.view(batch, seq, self.heads, _HEAD_DIM))
        rotary = x[..., :_ROTARY_DIM]
        half1, half2 = rotary.chunk(2, dim=-1)
        rot_cat = torch.cat((-half2, half1), dim=-1)
        rotary = (
            rotary * cos.to(proj.dtype)[None, :, None, :]
            + rot_cat * sin.to(proj.dtype)[None, :, None, :]
        )
        return torch.cat((rotary, x[..., _ROTARY_DIM:]), dim=-1).contiguous()

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        q = self._norm_rope(self.to_q(hidden), self.norm_q, cos[:seq], sin[:seq])
        k = self._norm_rope(self.to_k(hidden), self.norm_k, cos[:seq], sin[:seq])
        v = self.to_v(hidden).view(batch, seq, self.heads, _HEAD_DIM)
        attn = piper_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(
            1, 2
        )
        return self.to_out[0](attn.reshape(batch, seq, self.heads * _HEAD_DIM))


class _HybridTPSPFSDPBlock(nn.Module):
    def __init__(
        self,
        heads: int,
        attn_weights: dict[str, torch.Tensor],
        norms: dict[str, torch.Tensor],
        ffn_weights: dict[str, torch.Tensor],
        attn_gates: torch.Tensor,
        ffn_gates: torch.Tensor,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.attn = _DenseAttn(heads, attn_weights, norms, device=device)
        self.ff_res = _ResidualFFN(ffn_weights, ffn_gates, device=device)
        self.attn_gates = nn.Parameter(attn_gates.to(device), requires_grad=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        gate_indices: torch.Tensor,
    ) -> torch.Tensor:
        hidden.copy_(hidden + self.attn_gates[gate_indices] * self.attn(hidden, cos, sin))
        return self.ff_res(hidden, gate_indices)


def _shard_attn_linears(
    block: _HybridTPSPFSDPBlock,
    mesh: DeviceMesh,
    attn_weights: dict[str, torch.Tensor],
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    for mod, name in (
        (block.attn.to_q, "to_q"),
        (block.attn.to_k, "to_k"),
        (block.attn.to_v, "to_v"),
    ):
        full_q = ConvRotInt8Tensor.from_hp(attn_weights[name], group_size=64)
        length = full_q.shape[0] // world
        local_q = shard_quantized_weight(full_q, dim=0, start=rank * length, length=length).to(
            device
        )
        mod.weight = nn.Parameter(
            DTensor.from_local(
                local_q,
                mesh,
                [Shard(0)],
                run_check=False,
                shape=full_q.shape,
                stride=full_q.stride(),
            ),
            requires_grad=False,
        )
    out_full = ConvRotInt8Tensor.from_hp(attn_weights["to_out.0"], group_size=64)
    out_len = out_full.shape[1] // world
    out_local = shard_quantized_weight(out_full, dim=1, start=rank * out_len, length=out_len).to(
        device
    )
    block.attn.to_out[0].weight = nn.Parameter(
        DTensor.from_local(
            out_local,
            mesh,
            [Shard(1)],
            run_check=False,
            shape=out_full.shape,
            stride=out_full.stride(),
        ),
        requires_grad=False,
    )
    parallelize_module(
        block,
        mesh,
        {
            "attn": PrepareModuleInput(
                input_layouts=(Shard(1), None, None),
                desired_input_layouts=(Replicate(), None, None),
                use_local_output=True,
            ),
            "attn.to_q": ColwiseParallel(),
            "attn.to_k": ColwiseParallel(),
            "attn.to_v": ColwiseParallel(),
            "attn.to_out.0": RowwiseParallel(output_layouts=Shard(1), use_local_output=True),
        },
        src_data_rank=None,
    )


def _check_gpu_prefetch_and_meta(
    mesh: DeviceMesh,
    rank: int,
    world: int,
    device: torch.device,
    w0: dict[str, torch.Tensor],
    w1: dict[str, torch.Tensor],
) -> None:
    ref_stacked = _StackedFFN(
        [_QuantizedSwiGLUFFN(w0, device=device), _QuantizedSwiGLUFFN(w1, device=device)]
    )
    fsdp_stacked = _StackedFFN(
        [_QuantizedSwiGLUFFN(w0, device=device), _QuantizedSwiGLUFFN(w1, device=device)]
    )
    fully_shard(fsdp_stacked.layers[0], mesh=mesh, reshard_after_forward=True)
    fully_shard(fsdp_stacked.layers[1], mesh=mesh, reshard_after_forward=True)
    fully_shard(fsdp_stacked, mesh=mesh, reshard_after_forward=True)
    fsdp_stacked.layers[0].set_modules_to_forward_prefetch([fsdp_stacked.layers[1]])
    for _ in range(3):
        x = torch.randn(1, 64, _HIDDEN, device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(fsdp_stacked(x), ref_stacked(x), rtol=0, atol=0)

    meta_layer = nn.Linear(_HIDDEN, _INTERMEDIATE, bias=False, device="meta", dtype=torch.bfloat16)
    full_q = ConvRotInt8Tensor.from_hp(w0["up"], group_size=64)
    meta_layer.weight = nn.Parameter(_make_meta_like(full_q), requires_grad=False)
    fully_shard(meta_layer, mesh=mesh, reshard_after_forward=True)
    local_rows = _INTERMEDIATE // world
    local_q = shard_quantized_weight(full_q, dim=0, start=rank * local_rows, length=local_rows).to(
        device
    )
    meta_layer.load_state_dict(
        {
            "weight": DTensor.from_local(
                local_q,
                mesh,
                [Shard(0)],
                run_check=False,
                shape=full_q.shape,
                stride=full_q.stride(),
            )
        },
        assign=True,
    )
    gpu_full_w = full_q.to(device)
    for _ in range(3):
        x = torch.randn(8, _HIDDEN, device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(meta_layer(x), F.linear(x, gpu_full_w), rtol=0, atol=0)


def _check_gpu_cpu_offload(
    mesh: DeviceMesh,
    device: torch.device,
    w0: dict[str, torch.Tensor],
) -> None:
    static_scale = torch.tensor(0.05, dtype=torch.float32)
    ref_offload = _QuantizedSwiGLUFFN(w0, act_per_tensor_scale=static_scale, device=device)
    fsdp_offload = _QuantizedSwiGLUFFN(w0, act_per_tensor_scale=static_scale.clone(), device=device)
    fully_shard(
        fsdp_offload,
        mesh=mesh,
        reshard_after_forward=True,
        offload_policy=CPUOffloadPolicy(pin_memory=True),
    )
    sharded_up = fsdp_offload.up.weight.to_local()
    assert sharded_up.device.type == "cpu"
    assert sharded_up.is_pinned()
    for _ in range(3):
        x = torch.randn(1, 32, _HIDDEN, device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(fsdp_offload(x), ref_offload(x), rtol=0, atol=0)


def _check_gpu_compile(
    mesh: DeviceMesh,
    device: torch.device,
    w0: dict[str, torch.Tensor],
    gates: torch.Tensor,
) -> None:
    ref_res = _ResidualFFN(w0, gates, device=device)
    ref_compiled = torch.compile(
        ref_res,
        options=TargetCapturePass().wrap_options(compile_options(ffn=True, residual_updates=True)),
    )
    gi = torch.arange(128, device=device, dtype=torch.int64) % 4

    inner_mod = _ResidualFFN(w0, gates, device=device)
    fully_shard(inner_mod, mesh=mesh, reshard_after_forward=True)
    inner_hook_compiling: list[bool] = []
    inner_mod.register_forward_pre_hook(
        lambda _mod, _args: inner_hook_compiling.append(torch.compiler.is_compiling())
    )
    cap_inner = FunctionalTargetCapturePass()
    inner_mod.forward = torch.compile(
        inner_mod.forward,
        options=cap_inner.wrap_options(compile_options(ffn=True, residual_updates=True)),
    )

    outer_mod = _ResidualFFN(w0, gates, device=device)
    fully_shard(outer_mod, mesh=mesh, reshard_after_forward=True)
    cap_outer = FunctionalTargetCapturePass()
    outer_compiled = torch.compile(
        outer_mod,
        options=cap_outer.wrap_options(compile_options(ffn=True, residual_updates=True)),
    )

    for mod_fn in (inner_mod, outer_compiled):
        for _ in range(4):
            x0 = torch.randn(1, 128, _HIDDEN, device=device, dtype=torch.bfloat16)
            x_ref = x0.clone()
            x_dist = x0.clone()
            ref_compiled(x_ref, gi)
            mod_fn(x_dist, gi)
            assert _rel_l2(x_dist, x_ref) < 1e-4

    assert inner_hook_compiling == [False] * 4
    assert not any("c10d" in name or "all_gather" in name for name in cap_inner.target_names)
    for cap in (cap_inner, cap_outer):
        assert cap.calls == 1
        assert "piper_kernels.convrot_int8_ffn_update_.default" in cap.target_names


class _MultiGraphCapturePass(FunctionalTargetCapturePass):
    """Record target lists for every FX graph compiled across graph breaks."""

    def __init__(self) -> None:
        super().__init__()
        self.all_targets: list[list[str]] = []

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        super().__call__(graph, is_inference)
        self.all_targets.append(list(self.target_names))


def _check_gpu_hybrid_tpsp_fsdp(
    mesh: DeviceMesh,
    rank: int,
    world: int,
    device: torch.device,
    w0: dict[str, torch.Tensor],
    gates: torch.Tensor,
    gen: torch.Generator,
) -> None:
    pg = mesh.get_group("dp")
    attn_weights = {
        "to_q": torch.randn(_HIDDEN, _HIDDEN, generator=gen, dtype=torch.bfloat16) * 0.04,
        "to_k": torch.randn(_HIDDEN, _HIDDEN, generator=gen, dtype=torch.bfloat16) * 0.04,
        "to_v": torch.randn(_HIDDEN, _HIDDEN, generator=gen, dtype=torch.bfloat16) * 0.04,
        "to_out.0": torch.randn(_HIDDEN, _HIDDEN, generator=gen, dtype=torch.bfloat16) * 0.04,
    }
    norms = {
        "norm_q": torch.rand(_HEAD_DIM, generator=gen, dtype=torch.bfloat16).add_(0.5),
        "norm_k": torch.rand(_HEAD_DIM, generator=gen, dtype=torch.bfloat16).add_(0.5),
    }
    angles = torch.rand(256, _ROTARY_DIM, generator=gen, dtype=torch.float32) * (2 * torch.pi)
    cos_all = angles.cos().to(device)
    sin_all = angles.sin().to(device)

    ref_block = _HybridTPSPFSDPBlock(_HEADS, attn_weights, norms, w0, gates, gates, device=device)
    hybrid_block = _HybridTPSPFSDPBlock(
        _HEADS // world, attn_weights, norms, w0, gates, gates, device=device
    )
    _shard_attn_linears(hybrid_block, mesh, attn_weights, rank, world, device)
    fully_shard(hybrid_block.ff_res, mesh=mesh, reshard_after_forward=True)

    cap_hybrid = _MultiGraphCapturePass()
    hybrid_opts = compile_options(
        ffn=True, attention=True, attention_output=True, residual_updates=True
    )
    compiled_hybrid = torch.compile(
        hybrid_block,
        options=cap_hybrid.wrap_options(hybrid_opts),
    )
    compiled_ref_block = (
        torch.compile(ref_block, options=TargetCapturePass().wrap_options(hybrid_opts))
        if rank == 0
        else None
    )

    seq = 256
    local_seq = seq // world
    mine = slice(rank * local_seq, (rank + 1) * local_seq)
    for step in range(3):
        step_gen = torch.Generator(device="cpu").manual_seed(2000 + step)
        full_h = torch.randn(1, seq, _HIDDEN, generator=step_gen, dtype=torch.bfloat16)
        full_gi = torch.arange(seq, dtype=torch.int64) % 4
        local_h = full_h[:, mine].clone(memory_format=torch.contiguous_format).to(device)
        local_gi = full_gi[mine].contiguous().to(device)
        compiled_hybrid(local_h, cos_all, sin_all, local_gi)

        shards = [torch.empty_like(local_h) for _ in range(world)]
        dist.all_gather(shards, local_h.contiguous(), group=pg)
        gathered = torch.cat(shards, dim=1)
        if rank == 0:
            assert compiled_ref_block is not None
            ref_h = full_h.clone().to(device)
            compiled_ref_block(ref_h, cos_all, sin_all, full_gi.to(device))
            assert _rel_l2(gathered, ref_h) < 0.005

    assert cap_hybrid.calls == 2
    flat_targets = [target for graph_targets in cap_hybrid.all_targets for target in graph_targets]
    assert "_c10d_functional.all_gather_into_tensor_coalesced.default" in flat_targets
    assert (
        "piper_kernels."
        "convrot_int8_piper_projected_query_attention_reduce_scatter_residual_add_.default"
        in flat_targets
    )
    assert "piper_kernels.convrot_int8_ffn_update_.default" in flat_targets


def _gpu_fsdp2_worker(rank: int, world: int, store_path: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world,
        store=dist.FileStore(store_path, world),
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
        gen = torch.Generator(device="cpu").manual_seed(1234)
        w0 = {
            "up": torch.randn(_INTERMEDIATE, _HIDDEN, generator=gen, dtype=torch.bfloat16) * 0.04,
            "gate": torch.randn(_INTERMEDIATE, _HIDDEN, generator=gen, dtype=torch.bfloat16) * 0.04,
            "down": torch.randn(_HIDDEN, _INTERMEDIATE, generator=gen, dtype=torch.bfloat16) * 0.04,
        }
        w1 = {
            "up": torch.randn(_INTERMEDIATE, _HIDDEN, generator=gen, dtype=torch.bfloat16) * 0.04,
            "gate": torch.randn(_INTERMEDIATE, _HIDDEN, generator=gen, dtype=torch.bfloat16) * 0.04,
            "down": torch.randn(_HIDDEN, _INTERMEDIATE, generator=gen, dtype=torch.bfloat16) * 0.04,
        }
        gates = torch.full((4, _HIDDEN), 0.5, dtype=torch.bfloat16)

        with torch.inference_mode():
            _check_gpu_prefetch_and_meta(mesh, rank, world, device, w0, w1)
            _check_gpu_cpu_offload(mesh, device, w0)
            _check_gpu_compile(mesh, device, w0, gates)
            _check_gpu_hybrid_tpsp_fsdp(mesh, rank, world, device, w0, gates, gen)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
def test_fsdp2_2gpu_end_to_end(large_device_memory: None, tmp_path: Path) -> None:
    _ = large_device_memory
    if torch.cuda.device_count() < 2 or not dist.is_nccl_available():
        pytest.skip("requires at least 2 CUDA/ROCm GPUs with NCCL/RCCL support")
    mp.spawn(_gpu_fsdp2_worker, args=(2, str(tmp_path / "nccl_store")), nprocs=2, join=True)
