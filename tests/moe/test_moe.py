"""Contract, offline-compile, and GPU tests for quantized MoE decode and prefill."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
import triton
from _accelerators import (
    forbid_triton_driver,
    rocm_fences_only_lds,
    w4a16_nvfp4_available,
)
from _nvidia_resources import capture_cuda_spill_stores
from torch._dynamo.testing import CompileCounter
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F  # noqa: N812
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import LazyDict

from piper_kernels import compile_options
from piper_kernels.moe import (
    MoEWeights,
    PreparedMoEInput,
    moe_ffn,
    prepare_moe_input,
)
from piper_kernels.moe import reference as moe_reference
from piper_kernels.moe import triton as moe_triton
from piper_kernels.trellis import resolve_block_size, resolve_layout
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

MoEFormat = Literal["nvfp4_w4a16", "convrot_nvfp4_w4a16", "convrot_int8", "convrot_trellis"]
TrellisLayoutKind = Literal["uniform", "var2d", "var1d"]
ActivationKind = Literal["swiglu", "gelu_tanh"]

_INT8_MOE_GPU_AVAILABLE = w4a16_nvfp4_available() and (
    torch.version.hip is not None or torch.cuda.get_device_capability()[0] >= 8
)
_SM75_SHARED_MEMORY_LIMIT = 64 * 1024
_SM120_SHARED_MEMORY_LIMIT = 99 * 1024
_NVIDIA_TARGETS = [
    GPUTarget("cuda", 80, 32),
    GPUTarget("cuda", 86, 32),
    GPUTarget("cuda", 89, 32),
    GPUTarget("cuda", 120, 32),
]
_AMD_TARGETS = [
    ("gfx942", 64),
    ("gfx1100", 32),
    ("gfx1151", 32),
    ("gfx1201", 32),
]


def _shared_memory_limit(target: GPUTarget) -> int:
    return _SM75_SHARED_MEMORY_LIMIT if target.arch < 80 else _SM120_SHARED_MEMORY_LIMIT


def _compile_nvidia_and_assert_clean(
    target: GPUTarget,
    *,
    check_int8_mma: bool = False,
    max_spill_stores: dict[str, int] | None = None,
    **kwargs: Any,
) -> dict[str, triton.compiler.CompiledKernel]:
    spills_by_hash, compiler = capture_cuda_spill_stores()
    compiled = moe_triton.compile_moe_kernels(target, compiler=compiler, **kwargs)
    assert compiled
    smem_limit = _shared_memory_limit(target)
    for name, kernel in compiled.items():
        assert kernel.asm.get("cubin")
        ptx = kernel.asm.get("ptx", "")
        assert ptx
        limit = 0 if max_spill_stores is None else max_spill_stores.get(name, 0)
        assert spills_by_hash[kernel.hash] <= limit
        assert kernel.metadata.shared <= smem_limit
        if check_int8_mma and name.startswith(("int8_matmul", "int8_grouped")):
            assert "mma.sync" in ptx
    return compiled


def _make_dense_weights(
    num_slots: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    activation_fn: ActivationKind = "swiglu",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    seed: int = 101,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    up_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    up_gate_hp = (
        torch.randn((num_slots, up_out, hidden_size), generator=generator, dtype=dtype)
        * (hidden_size**-0.5)
    ).to(device)
    down_hp = (
        torch.randn((num_slots, hidden_size, intermediate_size), generator=generator, dtype=dtype)
        * (intermediate_size**-0.5)
    ).to(device)
    return up_gate_hp, down_hp


def _finite_e4m3(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    codes = torch.randint(0x18, 0x48, shape, dtype=torch.uint8, generator=generator)
    return codes.view(torch.float8_e4m3fn)


def _make_trellis_proj_slab(
    num_slots: int,
    out_features: int,
    in_features: int,
    *,
    block_size: int,
    layout_kind: TrellisLayoutKind,
    generator: torch.Generator,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    blocks_per_row = in_features // block_size
    block_scales = _finite_e4m3((num_slots, out_features, blocks_per_row), generator)
    scale = (torch.rand((num_slots, out_features), generator=generator) * 0.03 + 0.005) / max(
        1.0, (in_features / 256.0) ** 0.5
    )
    if layout_kind == "uniform":
        bpb = resolve_layout(block_size, 4.5).bytes_per_block
        qdata = torch.randint(
            0,
            256,
            (num_slots, out_features, blocks_per_row, bpb),
            dtype=torch.uint8,
            generator=generator,
        )
        return (
            qdata.to(device),
            block_scales.to(device),
            scale.to(device),
            None,
            None,
        )

    low_bytes = (2 * block_size + 7) // 8
    high_bytes = (6 * block_size) // 8
    # Vary widths across both rows and slots (especially making 1-D slot totals asymmetric).
    row_widths = torch.randint(
        low_bytes,
        high_bytes + 1,
        (num_slots, out_features),
        dtype=torch.uint8,
        generator=generator,
    )
    if num_slots > 1 and out_features > 0:
        row_widths[0, 0] = low_bytes
        row_widths[1, -1] = high_bytes
    slot_totals = [int(row_widths[slot].sum().item()) * blocks_per_row for slot in range(num_slots)]
    if layout_kind == "var2d":
        slot_bytes = max(slot_totals, default=0) + 17
        qdata = torch.zeros((num_slots, slot_bytes), dtype=torch.uint8)
        for slot, total in enumerate(slot_totals):
            qdata[slot, :total] = torch.randint(
                0, 256, (total,), dtype=torch.uint8, generator=generator
            )
        return (
            qdata.to(device),
            block_scales.to(device),
            scale.to(device),
            row_widths.to(device),
            None,
        )

    chunks: list[torch.Tensor] = []
    offsets: list[int] = []
    cursor = 0
    for total in slot_totals:
        offsets.append(cursor)
        chunks.append(torch.randint(0, 256, (total,), dtype=torch.uint8, generator=generator))
        # Non-zero gap ensures kernels index via slot_offsets rather than assuming tight packing.
        chunks.append(torch.full((13,), 0xA5, dtype=torch.uint8))
        cursor += total + 13
    qdata_1d = torch.cat(chunks) if chunks else torch.empty((0,), dtype=torch.uint8)
    slot_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
    return (
        qdata_1d.to(device),
        block_scales.to(device),
        scale.to(device),
        row_widths.to(device),
        slot_offsets,
    )


def _make_trellis_moe_weights(
    num_slots: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    group_size: int = 64,
    activation_fn: ActivationKind = "swiglu",
    gate_first: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    layouts: tuple[TrellisLayoutKind, TrellisLayoutKind] = ("uniform", "uniform"),
    seed: int = 101,
) -> MoEWeights:
    up_layout, down_layout = layouts
    generator = torch.Generator(device="cpu").manual_seed(seed)
    up_out = intermediate_size * (2 if activation_fn == "swiglu" else 1)
    resolved_up_bs = resolve_block_size(hidden_size)
    resolved_dn_bs = resolve_block_size(intermediate_size)
    up_qd, up_bs, up_sc, up_rw, up_so = _make_trellis_proj_slab(
        num_slots,
        up_out,
        hidden_size,
        block_size=resolved_up_bs,
        layout_kind=up_layout,
        generator=generator,
        device=device,
    )
    dn_qd, dn_bs, dn_sc, dn_rw, dn_so = _make_trellis_proj_slab(
        num_slots,
        hidden_size,
        intermediate_size,
        block_size=resolved_dn_bs,
        layout_kind=down_layout,
        generator=generator,
        device=device,
    )
    return MoEWeights.from_quantized(
        up_qd,
        up_sc,
        dn_qd,
        dn_sc,
        logical_dtype=dtype,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        up_gate_block_scales=up_bs,
        down_block_scales=dn_bs,
        up_gate_row_widths=up_rw,
        down_row_widths=dn_rw,
        up_gate_slot_offsets=up_so,
        down_slot_offsets=dn_so,
        up_gate_block_size=resolved_up_bs,
        down_block_size=resolved_dn_bs,
    )


def _decode_trellis_moe_weights_to_int8(weights: MoEWeights) -> MoEWeights:
    assert weights.up_gate_block_scales is not None
    assert weights.down_block_scales is not None
    num_slots = weights.num_slots
    up_int8: list[torch.Tensor] = []
    dn_int8: list[torch.Tensor] = []
    for slot in range(num_slots):
        up_stream = (
            weights.up_gate_qdata[slot]
            if weights.up_gate_slot_offsets is None
            else weights.up_gate_qdata[int(weights.up_gate_slot_offsets[slot].item()) :]
        )
        dn_stream = (
            weights.down_qdata[slot]
            if weights.down_slot_offsets is None
            else weights.down_qdata[int(weights.down_slot_offsets[slot].item()) :]
        )
        up_t = ConvRotInt8Tensor.from_trellis(
            up_stream,
            weights.up_gate_block_scales[slot],
            weights.up_gate_scale[slot],
            block_size=weights.up_gate_block_size,
            group_size=weights.group_size,
            logical_dtype=weights.logical_dtype,
            row_widths=None
            if weights.up_gate_row_widths is None
            else weights.up_gate_row_widths[slot],
        )
        dn_t = ConvRotInt8Tensor.from_trellis(
            dn_stream,
            weights.down_block_scales[slot],
            weights.down_scale[slot],
            block_size=weights.down_block_size,
            group_size=weights.group_size,
            logical_dtype=weights.logical_dtype,
            row_widths=None if weights.down_row_widths is None else weights.down_row_widths[slot],
        )
        up_int8.append(up_t.qdata)
        dn_int8.append(dn_t.qdata)
    return MoEWeights.from_quantized(
        torch.stack(up_int8).contiguous(),
        weights.up_gate_scale.contiguous(),
        torch.stack(dn_int8).contiguous(),
        weights.down_scale.contiguous(),
        logical_dtype=weights.logical_dtype,
        group_size=weights.group_size,
        activation_fn=weights.activation_fn,
        gate_first=weights.gate_first,
    )


def _make_moe_weights(
    num_slots: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    weight_format: MoEFormat,
    group_size: int = 0,
    activation_fn: ActivationKind = "swiglu",
    gate_first: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    seed: int = 101,
) -> MoEWeights:
    if weight_format == "convrot_trellis":
        return _make_trellis_moe_weights(
            num_slots,
            hidden_size,
            intermediate_size,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            dtype=dtype,
            device=device,
            seed=seed,
        )
    up_gate_hp, down_hp = _make_dense_weights(
        num_slots,
        hidden_size,
        intermediate_size,
        activation_fn=activation_fn,
        dtype=dtype,
        device=device,
        seed=seed,
    )
    return MoEWeights.from_hp(
        up_gate_hp,
        down_hp,
        weight_format=weight_format,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
    )


def _reference_for_weights(
    x: torch.Tensor,
    weights: MoEWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    expert_to_slot: torch.Tensor | None = None,
    prepared_input: PreparedMoEInput | None = None,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
) -> torch.Tensor:
    prepared_data = None if prepared_input is None else prepared_input.data.cpu()
    prepared_scale = (
        None
        if (prepared_input is None or prepared_input.scale is None)
        else prepared_input.scale.cpu()
    )
    return moe_reference.moe_ffn_reference(
        x.cpu(),
        topk_ids.cpu(),
        topk_weights.cpu(),
        weights.up_gate_qdata.cpu(),
        weights.up_gate_scale.cpu(),
        weights.down_qdata.cpu(),
        weights.down_scale.cpu(),
        None if weights.up_gate_global_scale is None else weights.up_gate_global_scale.cpu(),
        None if weights.down_global_scale is None else weights.down_global_scale.cpu(),
        None if expert_to_slot is None else expert_to_slot.cpu(),
        prepared_data,
        prepared_scale,
        None if out is None else out.detach().cpu().clone(),
        accumulate=accumulate,
        group_size=weights.group_size,
        activation_fn=weights.activation_fn,
        gate_first=weights.gate_first,
        high_first=weights.high_first,
        up_gate_block_scales=(
            None if weights.up_gate_block_scales is None else weights.up_gate_block_scales.cpu()
        ),
        down_block_scales=(
            None if weights.down_block_scales is None else weights.down_block_scales.cpu()
        ),
        up_gate_row_widths=(
            None if weights.up_gate_row_widths is None else weights.up_gate_row_widths.cpu()
        ),
        down_row_widths=(
            None if weights.down_row_widths is None else weights.down_row_widths.cpu()
        ),
        up_gate_slot_offsets=(
            None if weights.up_gate_slot_offsets is None else weights.up_gate_slot_offsets.cpu()
        ),
        down_slot_offsets=(
            None if weights.down_slot_offsets is None else weights.down_slot_offsets.cpu()
        ),
        up_gate_block_size=weights.up_gate_block_size,
        down_block_size=weights.down_block_size,
    )


def _oracle_nvfp4_tensor(
    weights: MoEWeights,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None,
) -> PiperNVFP4Tensor:
    if weights.group_size > 0:
        return ConvRotNVFP4Tensor(
            qdata,
            scale,
            16,
            weights.logical_dtype,
            weights.group_size,
            per_tensor_scale=global_scale,
            is_swizzled_scales=True,
            high_first=weights.high_first,
        )
    return PiperNVFP4Tensor(
        qdata,
        scale,
        16,
        weights.logical_dtype,
        per_tensor_scale=global_scale,
        is_swizzled_scales=True,
        high_first=weights.high_first,
    )


def _oracle_slot_output(
    x_row: torch.Tensor,
    weights: MoEWeights,
    slot: int,
) -> torch.Tensor:
    if weights.weight_format == "convrot_trellis":
        assert weights.up_gate_block_scales is not None
        assert weights.down_block_scales is not None
        up_stream = (
            weights.up_gate_qdata[slot].cpu()
            if weights.up_gate_slot_offsets is None
            else weights.up_gate_qdata[int(weights.up_gate_slot_offsets[slot].item()) :].cpu()
        )
        dn_stream = (
            weights.down_qdata[slot].cpu()
            if weights.down_slot_offsets is None
            else weights.down_qdata[int(weights.down_slot_offsets[slot].item()) :].cpu()
        )
        up_t = ConvRotTrellisTensor.from_quantized(
            up_stream,
            weights.up_gate_block_scales[slot].cpu(),
            weights.up_gate_scale[slot].cpu(),
            block_size=weights.up_gate_block_size,
            group_size=weights.group_size,
            logical_dtype=weights.logical_dtype,
            row_widths=(
                None
                if weights.up_gate_row_widths is None
                else weights.up_gate_row_widths[slot].cpu()
            ),
        )
        down_t = ConvRotTrellisTensor.from_quantized(
            dn_stream,
            weights.down_block_scales[slot].cpu(),
            weights.down_scale[slot].cpu(),
            block_size=weights.down_block_size,
            group_size=weights.group_size,
            logical_dtype=weights.logical_dtype,
            row_widths=(
                None if weights.down_row_widths is None else weights.down_row_widths[slot].cpu()
            ),
        )
        mid = F.linear(x_row.float(), up_t.dequantize(torch.float32))
    elif weights.weight_format == "convrot_int8":
        up_w = ConvRotInt8Tensor.from_quantized(
            weights.up_gate_qdata[slot].cpu(),
            weights.up_gate_scale[slot].cpu(),
            group_size=weights.group_size,
            logical_dtype=weights.logical_dtype,
        )
        down_w = ConvRotInt8Tensor.from_quantized(
            weights.down_qdata[slot].cpu(),
            weights.down_scale[slot].cpu(),
            group_size=weights.group_size,
            logical_dtype=torch.float32,
        )
        mid = F.linear(x_row, up_w).float()
    else:
        up_gs = (
            None
            if weights.up_gate_global_scale is None
            else weights.up_gate_global_scale[slot].cpu()
        )
        down_gs = (
            None if weights.down_global_scale is None else weights.down_global_scale[slot].cpu()
        )
        up_t = _oracle_nvfp4_tensor(
            weights, weights.up_gate_qdata[slot].cpu(), weights.up_gate_scale[slot].cpu(), up_gs
        )
        down_t = _oracle_nvfp4_tensor(
            weights, weights.down_qdata[slot].cpu(), weights.down_scale[slot].cpu(), down_gs
        )
        mid = F.linear(x_row.float(), up_t.dequantize(torch.float32))
    if weights.activation_fn == "swiglu":
        first, second = mid.chunk(2, dim=-1)
        activated = F.silu(first) * second if weights.gate_first else first * F.silu(second)
    else:
        activated = F.gelu(mid, approximate="tanh")
    if weights.weight_format == "convrot_int8":
        return F.linear(activated, down_w)[0]
    activated_q = activated.to(weights.logical_dtype)
    return F.linear(activated_q.float(), down_t.dequantize(torch.float32))[0]


def _independent_moe_oracle(
    x: torch.Tensor,
    weights: MoEWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    expert_to_slot: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """Compose public ConvRot/NVFP4 per-expert projections independently from moe.reference."""
    x_cpu = x.detach().cpu()
    ids_cpu = topk_ids.detach().cpu()
    wts_cpu = topk_weights.detach().cpu()
    slot_map = None if expert_to_slot is None else expert_to_slot.detach().cpu()
    rows = x_cpu.shape[:-1].numel()
    hidden_size = weights.hidden_size
    top_k = ids_cpu.shape[-1]
    result = torch.zeros_like(x_cpu, memory_format=torch.contiguous_format)
    if out is not None and accumulate:
        result.copy_(out.detach().cpu().clone())
    x_2d = x_cpu.reshape(rows, hidden_size)
    ids_2d = ids_cpu.reshape(rows, top_k)
    wts_2d = wts_cpu.reshape(rows, top_k)
    out_2d = result.view(rows, hidden_size)

    for row in range(rows):
        acc = torch.zeros((hidden_size,), dtype=torch.float32)
        had_hit = False
        for k_idx in range(top_k):
            expert = int(ids_2d[row, k_idx].item())
            if slot_map is None:
                slot = expert
            else:
                slot = int(slot_map[expert].item()) if 0 <= expert < slot_map.shape[0] else -1
            if not (0 <= slot < weights.num_slots):
                continue
            had_hit = True
            expert_out = _oracle_slot_output(x_2d[row : row + 1], weights, slot)
            acc.add_(expert_out, alpha=float(wts_2d[row, k_idx].float().item()))
        if accumulate:
            if had_hit:
                out_2d[row].copy_((out_2d[row].float() + acc).to(x_cpu.dtype))
        else:
            out_2d[row].copy_(acc.to(x_cpu.dtype))
    return result


def _assert_matches_reference(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    weight_format: MoEFormat,
    dtype: torch.dtype,
) -> None:
    expected_dev = expected.to(actual.device)
    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    if weight_format == "convrot_int8":
        denom = expected_dev.float().norm().clamp_min(1e-6)
        rel_l2 = (actual.float() - expected_dev.float()).norm() / denom
        assert float(rel_l2) < 0.015
        rtol = 1e-2 if dtype is torch.float32 else (2e-2 if dtype is torch.float16 else 4e-2)
        atol = 1.5e-2 if dtype is torch.float32 else (2e-2 if dtype is torch.float16 else 4e-2)
        torch.testing.assert_close(actual, expected_dev, rtol=rtol, atol=atol)
        return
    if dtype is torch.float32:
        torch.testing.assert_close(actual, expected_dev, rtol=1e-3, atol=1e-3)
    elif dtype is torch.float16:
        torch.testing.assert_close(actual, expected_dev, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(actual, expected_dev, rtol=3.5e-2, atol=2e-2)


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


def _assert_clean_amd_compile(
    compiled: triton.compiler.CompiledKernel,
    arch: str,
    *,
    check_vgpr_spills: bool = True,
    max_vgpr_spills: int = 0,
    max_private_scratch_bytes: int = 0,
    check_lds_fences: bool = False,
) -> None:
    assert compiled.asm.get("hsaco")
    assert compiled.metadata.global_scratch_size == 0
    amdgcn = compiled.asm.get("amdgcn", "")
    if check_vgpr_spills:
        vgpr_spills = re.search(r"vgpr_spill[^:]*:\s*(\d+)", amdgcn)
        priv_size = re.search(r"\.private_segment_fixed_size:\s*(\d+)", amdgcn)
        if vgpr_spills is not None:
            assert int(vgpr_spills.group(1)) <= max_vgpr_spills
        if priv_size is not None:
            assert int(priv_size.group(1)) <= max_private_scratch_bytes
    if check_lds_fences and arch == "gfx1201" and rocm_fences_only_lds():
        assert "global_inv" not in amdgcn


def test_from_hp_and_from_quantized_zero_copy_slab() -> None:
    for weight_format, group_size in (
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        ("convrot_int8", 64),
    ):
        weights = _make_moe_weights(
            4,
            128,
            192,
            weight_format=weight_format,  # type: ignore[arg-type]
            group_size=group_size,
            dtype=torch.bfloat16,
            seed=10 + group_size,
        )
        assert weights.weight_format == weight_format
        assert weights.num_slots == 4
        assert weights.hidden_size == 128
        assert weights.intermediate_size == 192

        up_scale = (
            weights.up_gate_scale.unsqueeze(-1)
            if weight_format == "convrot_int8"
            else weights.up_gate_scale
        )
        down_scale = (
            weights.down_scale.unsqueeze(-1)
            if weight_format == "convrot_int8"
            else weights.down_scale
        )
        up_gs = (
            None
            if weights.up_gate_global_scale is None
            else weights.up_gate_global_scale.unsqueeze(-1)
        )
        down_gs = (
            None if weights.down_global_scale is None else weights.down_global_scale.unsqueeze(-1)
        )
        wrapped = MoEWeights.from_quantized(
            weights.up_gate_qdata,
            up_scale,
            weights.down_qdata,
            down_scale,
            logical_dtype=weights.logical_dtype,
            up_gate_global_scale=up_gs,
            down_global_scale=down_gs,
            group_size=weights.group_size,
            activation_fn=weights.activation_fn,
            gate_first=weights.gate_first,
            high_first=weights.high_first,
        )
        assert wrapped.up_gate_qdata.data_ptr() == weights.up_gate_qdata.data_ptr()
        assert wrapped.up_gate_scale.data_ptr() == weights.up_gate_scale.data_ptr()
        assert wrapped.down_qdata.data_ptr() == weights.down_qdata.data_ptr()
        assert wrapped.down_scale.data_ptr() == weights.down_scale.data_ptr()
        if weights.up_gate_global_scale is not None:
            assert wrapped.up_gate_global_scale is not None
            assert (
                wrapped.up_gate_global_scale.data_ptr() == weights.up_gate_global_scale.data_ptr()
            )
            wrapped_3d = MoEWeights.from_quantized(
                weights.up_gate_qdata,
                up_scale,
                weights.down_qdata,
                down_scale,
                logical_dtype=weights.logical_dtype,
                up_gate_global_scale=up_gs.unsqueeze(-1),  # type: ignore[union-attr]
                down_global_scale=down_gs.unsqueeze(-1),  # type: ignore[union-attr]
                group_size=weights.group_size,
                activation_fn=weights.activation_fn,
                gate_first=weights.gate_first,
                high_first=weights.high_first,
            )
            assert wrapped_3d.up_gate_global_scale is not None
            assert (
                wrapped_3d.up_gate_global_scale.data_ptr()
                == weights.up_gate_global_scale.data_ptr()
            )


def test_validation_errors_leave_caller_out_untouched() -> None:
    up_hp, down_hp = _make_dense_weights(2, 128, 128, dtype=torch.bfloat16, seed=19)
    with pytest.raises(ValueError, match="group_size=0"):
        MoEWeights.from_hp(up_hp, down_hp, weight_format="nvfp4_w4a16", group_size=64)
    with pytest.raises(ValueError, match="group_size in"):
        MoEWeights.from_hp(up_hp, down_hp, weight_format="convrot_nvfp4_w4a16", group_size=0)

    weights = _make_moe_weights(
        3, 128, 128, weight_format="convrot_int8", group_size=64, dtype=torch.bfloat16, seed=21
    )
    nvfp4_weights = _make_moe_weights(
        3, 128, 128, weight_format="convrot_nvfp4_w4a16", group_size=64, dtype=torch.float16
    )
    x = torch.randn((1, 128), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.6, 0.4]], dtype=torch.bfloat16)
    sentinel = torch.full((1, 128), -0.0, dtype=torch.bfloat16)
    out = sentinel.clone()
    before_bits = out.view(torch.uint16).clone()

    with pytest.raises(ValueError, match="out tensor"):
        moe_ffn(x, weights, topk_ids, topk_weights, out=None, accumulate=True)
    with pytest.raises(ValueError, match="out tensor"):
        moe_ffn(x, weights, topk_ids, topk_weights, out=torch.empty((1, 128), dtype=torch.float32))
    with pytest.raises(ValueError, match="out tensor"):
        moe_ffn(
            x,
            weights,
            topk_ids,
            topk_weights,
            out=torch.empty((1, 256), dtype=torch.bfloat16)[:, ::2],
        )

    prep_int8 = prepare_moe_input(x, weights)
    bad_dtype_prep = PreparedMoEInput(
        data=prep_int8.data.to(torch.float32),
        scale=prep_int8.scale,
        logical_dtype=weights.logical_dtype,
        group_size=weights.group_size,
        is_int8=True,
    )
    bad_noncontig_prep = PreparedMoEInput(
        data=torch.zeros((1, 256), dtype=torch.int8)[:, ::2],
        scale=prep_int8.scale,
        logical_dtype=weights.logical_dtype,
        group_size=weights.group_size,
        is_int8=True,
    )
    bad_missing_scale = PreparedMoEInput(
        data=prep_int8.data,
        scale=None,
        logical_dtype=weights.logical_dtype,
        group_size=weights.group_size,
        is_int8=True,
    )
    for bad_prep in (bad_dtype_prep, bad_noncontig_prep, bad_missing_scale):
        with pytest.raises(ValueError, match="prepared_input"):
            moe_ffn(
                x,
                weights,
                topk_ids,
                topk_weights,
                out=out,
                accumulate=True,
                prepared_input=bad_prep,
            )
        assert torch.equal(out.view(torch.uint16), before_bits)

    x_fp16 = torch.randn((1, 128), dtype=torch.float16)
    bad_fp16_prep = PreparedMoEInput(
        data=torch.zeros((1, 128), dtype=torch.float16),
        scale=None,
        logical_dtype=torch.float16,
        group_size=64,
        is_int8=False,
    )
    with pytest.raises(ValueError, match="prepared_input"):
        moe_ffn(
            x_fp16,
            nvfp4_weights,
            topk_ids,
            topk_weights.to(torch.float16),
            prepared_input=bad_fp16_prep,
        )

    with torch.enable_grad():
        with pytest.raises(RuntimeError, match="gradients"):
            moe_ffn(
                x.clone().requires_grad_(True),
                weights,
                topk_ids,
                topk_weights,
                out=out,
                accumulate=True,
            )
        with pytest.raises(RuntimeError, match="gradients"):
            moe_ffn(
                x,
                weights,
                topk_ids,
                topk_weights.clone().requires_grad_(True),
                out=out,
                accumulate=True,
            )
        with pytest.raises(RuntimeError, match="autograd"):
            moe_ffn(
                x_fp16,
                nvfp4_weights,
                topk_ids,
                topk_weights.to(torch.float16),
                prepared_input=PreparedMoEInput(
                    data=torch.zeros((1, 128), dtype=torch.float32, requires_grad=True),
                    scale=None,
                    logical_dtype=torch.float16,
                    group_size=64,
                    is_int8=False,
                ),
            )
        with pytest.raises(RuntimeError, match="gradients"):
            moe_ffn(
                x,
                weights,
                topk_ids,
                topk_weights,
                out=torch.zeros((1, 128), dtype=torch.bfloat16, requires_grad=True),
                accumulate=True,
            )
    assert torch.equal(out.view(torch.uint16), before_bits)


def test_rejected_inplace_calls_preserve_out_version() -> None:
    torch.manual_seed(22)
    weights = _make_moe_weights(
        3, 128, 128, weight_format="convrot_int8", group_size=64, dtype=torch.bfloat16, seed=21
    )
    x = torch.randn((1, 128), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.6, 0.4]], dtype=torch.bfloat16)
    alias_out = weights.up_gate_scale[0, :64].view(torch.bfloat16).view(1, 128)

    # Rejected mutating calls never bump out._version or break saved autograd graphs.
    for ctx_factory in (torch.enable_grad, torch.no_grad, torch.inference_mode):
        tracked_out = torch.ones((1, 128), dtype=torch.bfloat16)
        grad_leaf = torch.ones((1, 128), dtype=torch.bfloat16, requires_grad=True)
        saved = (grad_leaf * tracked_out).sum()
        with ctx_factory(), pytest.raises(ValueError, match="topk_ids"):
            moe_ffn(x, weights, topk_ids.float(), topk_weights, out=tracked_out, accumulate=True)
        assert tracked_out._version == 0
        saved.backward()
        assert grad_leaf.grad is not None

        alias_before = alias_out.clone()
        alias_ver = alias_out._version
        with ctx_factory(), pytest.raises(ValueError, match="alias"):
            moe_ffn(x, weights, topk_ids, topk_weights, out=alias_out)
        assert alias_out._version == alias_ver
        assert torch.equal(alias_out, alias_before)

    normal_out = torch.zeros_like(x)
    v_before = normal_out._version
    with torch.inference_mode():
        assert moe_ffn(x, weights, topk_ids, topk_weights, out=normal_out) is normal_out
        assert normal_out._version > v_before
        inf_out = torch.zeros_like(x)
        assert moe_ffn(x, weights, topk_ids, topk_weights, out=inf_out) is inf_out


@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        ("convrot_int8", 64),
        ("convrot_trellis", 64),
    ],
)
@pytest.mark.parametrize(
    ("activation_fn", "gate_first"),
    [("swiglu", True), ("swiglu", False), ("gelu_tanh", True)],
)
def test_cpu_parity_formats_activations_and_gate_order(
    weight_format: MoEFormat,
    group_size: int,
    activation_fn: ActivationKind,
    gate_first: bool,
) -> None:
    torch.manual_seed(30 + group_size + int(gate_first))
    weights = _make_moe_weights(
        3,
        128,
        128,
        weight_format=weight_format,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        dtype=torch.bfloat16,
        seed=40 + group_size,
    )
    x = torch.randn((1, 128), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[2, 0]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.65, 0.35]], dtype=torch.bfloat16)

    actual = moe_ffn(x, weights, topk_ids, topk_weights)
    oracle = _independent_moe_oracle(x, weights, topk_ids, topk_weights)
    _assert_matches_reference(actual, oracle, weight_format=weight_format, dtype=torch.bfloat16)

    if weight_format == "convrot_int8" and activation_fn == "swiglu" and gate_first:
        w_scale_fp16 = MoEWeights.from_quantized(
            torch.full((1, 128, 64), 1, dtype=torch.int8),
            torch.full((1, 128), 0.125, dtype=torch.float32),
            torch.full((1, 64, 64), 1, dtype=torch.int8),
            torch.full((1, 64), 1e4, dtype=torch.float32),
            logical_dtype=torch.float16,
            group_size=64,
        )
        x_scale = torch.ones((1, 64), dtype=torch.float16)
        s_ids = torch.zeros((1, 1), dtype=torch.int64)
        s_wts = torch.full((1, 1), 1e-4, dtype=torch.float16)
        out_fp32_route = moe_ffn(x_scale, w_scale_fp16, s_ids, s_wts)
        oracle_fp32_route = _independent_moe_oracle(x_scale, w_scale_fp16, s_ids, s_wts)
        assert torch.isfinite(out_fp32_route).all()
        assert torch.isfinite(oracle_fp32_route).all()
        torch.testing.assert_close(out_fp32_route, oracle_fp32_route, rtol=1e-2, atol=1.0)


def test_cpu_gate_first_symmetry_and_high_first_parity() -> None:
    up_hp, down_hp = _make_dense_weights(2, 128, 128, dtype=torch.float32, seed=55)
    gate_half, up_half = up_hp.chunk(2, dim=1)
    swapped_hp = torch.cat([up_half, gate_half], dim=1).contiguous()
    w_gate_first = MoEWeights.from_hp(
        up_hp, down_hp, weight_format="convrot_int8", group_size=64, gate_first=True
    )
    w_up_first = MoEWeights.from_hp(
        swapped_hp, down_hp, weight_format="convrot_int8", group_size=64, gate_first=False
    )
    x = torch.randn((1, 128), dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    out_gate_first = moe_ffn(x, w_gate_first, topk_ids, topk_weights)
    out_up_first = moe_ffn(x, w_up_first, topk_ids, topk_weights)
    torch.testing.assert_close(out_gate_first, out_up_first, rtol=1e-5, atol=1e-5)

    hf_up_hp, hf_down_hp = _make_dense_weights(2, 128, 128, dtype=torch.float16, seed=56)
    w_high_first = MoEWeights.from_hp(
        hf_up_hp,
        hf_down_hp,
        weight_format="convrot_nvfp4_w4a16",
        group_size=64,
        high_first=True,
    )
    x16 = torch.randn((1, 128), dtype=torch.float16)
    actual_hf = moe_ffn(x16, w_high_first, topk_ids, topk_weights.to(torch.float16))
    expected_hf = _reference_for_weights(
        x16, w_high_first, topk_ids, topk_weights.to(torch.float16)
    )
    assert torch.equal(actual_hf, expected_hf)


def test_cpu_slot_indirection_misses_accumulate_and_strided_out() -> None:
    weights = _make_moe_weights(
        4, 128, 128, weight_format="nvfp4_w4a16", dtype=torch.float32, seed=71
    )
    x = torch.randn((1, 128), dtype=torch.float32)
    expert_to_slot = torch.tensor([2, -1, 0, -1, 3, -1], dtype=torch.int32)
    topk_ids = torch.tensor([[1, 2, 99, -2]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.2, 0.5, 0.2, 0.1]], dtype=torch.float32)

    backing = torch.full((1, 144), float("nan"), dtype=torch.float32)
    out_view = backing[:, :128]
    out_view.fill_(1.5)
    expected = _independent_moe_oracle(
        x,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=expert_to_slot,
        out=out_view,
        accumulate=True,
    )
    actual = moe_ffn(
        x,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=expert_to_slot,
        out=out_view,
        accumulate=True,
    )
    assert actual is out_view
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert torch.isnan(backing[:, 128:]).all()

    all_miss_ids = torch.tensor([[1, 3, -1, 42]], dtype=torch.int32)
    neg_zero = torch.full((1, 128), -0.0, dtype=torch.float32)
    moe_ffn(
        x,
        weights,
        all_miss_ids,
        topk_weights,
        expert_to_slot=expert_to_slot,
        out=neg_zero,
        accumulate=True,
    )
    assert torch.signbit(neg_zero).all()

    zero_backing = torch.full((1, 144), float("nan"), dtype=torch.float32)
    zero_view = zero_backing[:, :128]
    moe_ffn(
        x,
        weights,
        all_miss_ids,
        topk_weights,
        expert_to_slot=expert_to_slot,
        out=zero_view,
        accumulate=False,
    )
    assert torch.equal(zero_view, torch.zeros_like(zero_view))
    assert torch.isnan(zero_backing[:, 128:]).all()


@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        ("convrot_int8", 64),
        ("convrot_trellis", 64),
    ],
)
def test_cpu_two_wave_prepared_input_accumulation(
    weight_format: MoEFormat, group_size: int
) -> None:
    torch.manual_seed(92 + group_size)
    weights = _make_moe_weights(
        4,
        128,
        128,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.float16,
        seed=93 + group_size,
    )
    x = torch.randn((1, 128), dtype=torch.float16)
    topk_ids = torch.tensor([[0, 2, 3, 5]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float16)

    full_map = torch.tensor([0, -1, 1, 2, -1, 3], dtype=torch.int64)
    wave1_map = torch.tensor([0, -1, 1, -1, -1, -1], dtype=torch.int64)
    wave2_map = torch.tensor([-1, -1, -1, 2, -1, 3], dtype=torch.int64)

    prepared = prepare_moe_input(x, weights)
    expected = moe_ffn(
        x, weights, topk_ids, topk_weights, expert_to_slot=full_map, prepared_input=prepared
    )

    acc = torch.empty_like(x)
    moe_ffn(
        x,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=wave1_map,
        out=acc,
        accumulate=False,
        prepared_input=prepared,
    )
    moe_ffn(
        x,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=wave2_map,
        out=acc,
        accumulate=True,
        prepared_input=prepared,
    )
    rtol = 1e-3 if weight_format != "convrot_int8" else 1e-2
    atol = 1e-3 if weight_format != "convrot_int8" else 1e-2
    torch.testing.assert_close(acc, expected, rtol=rtol, atol=atol)

    # PreparedMoEInput owns an independent snapshot even when wave 1 writes out=x.
    x_inout = x.clone()
    prepared_inout = prepare_moe_input(x_inout, weights)
    moe_ffn(
        x_inout,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=wave1_map,
        out=x_inout,
        accumulate=False,
        prepared_input=prepared_inout,
    )
    moe_ffn(
        x_inout,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=wave2_map,
        out=x_inout,
        accumulate=True,
        prepared_input=prepared_inout,
    )
    torch.testing.assert_close(x_inout, expected, rtol=rtol, atol=atol)


def test_cpu_empty_meta_and_multi_row() -> None:
    weights = _make_moe_weights(
        2, 128, 128, weight_format="convrot_int8", group_size=64, dtype=torch.bfloat16, seed=105
    )

    x_empty = torch.empty((0, 128), dtype=torch.bfloat16)
    ids_empty = torch.empty((0, 2), dtype=torch.int64)
    wts_empty = torch.empty((0, 2), dtype=torch.bfloat16)
    out_empty = torch.empty((0, 128), dtype=torch.bfloat16)
    assert (
        moe_ffn(x_empty, weights, ids_empty, wts_empty, out=out_empty, accumulate=True) is out_empty
    )

    x_one = torch.randn((1, 128), dtype=torch.bfloat16)
    ids_k0 = torch.empty((1, 0), dtype=torch.int64)
    wts_k0 = torch.empty((1, 0), dtype=torch.bfloat16)
    neg_zero = torch.full((1, 128), -0.0, dtype=torch.bfloat16)
    moe_ffn(x_one, weights, ids_k0, wts_k0, out=neg_zero, accumulate=True)
    assert torch.signbit(neg_zero).all()

    up_e0, down_e0 = _make_dense_weights(0, 128, 128, dtype=torch.bfloat16, seed=106)
    w_s0 = MoEWeights.from_hp(up_e0, down_e0, weight_format="convrot_int8", group_size=64)
    empty_slot_out = torch.full((1, 128), 7.0, dtype=torch.bfloat16)
    moe_ffn(
        x_one,
        w_s0,
        torch.tensor([[0]], dtype=torch.int64),
        torch.tensor([[1.0]], dtype=torch.bfloat16),
        out=empty_slot_out,
        accumulate=False,
    )
    assert torch.equal(empty_slot_out, torch.zeros_like(empty_slot_out))

    # Multi-row 2-D and 3-D shapes match reference.
    x_2d = torch.randn((2, 128), dtype=torch.bfloat16)
    ids_2d = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    wts_2d = torch.tensor([[0.6, 0.4], [0.25, 0.75]], dtype=torch.bfloat16)
    actual_2d = moe_ffn(x_2d, weights, ids_2d, wts_2d)
    expected_2d = _reference_for_weights(x_2d, weights, ids_2d, wts_2d)
    assert torch.equal(actual_2d, expected_2d)

    x_3d = x_2d.unsqueeze(0)
    actual_3d = moe_ffn(x_3d, weights, ids_2d.unsqueeze(0), wts_2d.unsqueeze(0))
    assert actual_3d.shape == (1, 2, 128)
    assert torch.equal(actual_3d, expected_2d.unsqueeze(0))

    # device="meta" dispatch for functional and in-place custom ops.
    meta_w = MoEWeights.from_quantized(
        torch.empty((2, 256, 128), dtype=torch.int8, device="meta"),
        torch.empty((2, 256), dtype=torch.float32, device="meta"),
        torch.empty((2, 128, 128), dtype=torch.int8, device="meta"),
        torch.empty((2, 128), dtype=torch.float32, device="meta"),
        logical_dtype=torch.bfloat16,
        group_size=64,
    )
    meta_x = torch.empty((1, 128), dtype=torch.bfloat16, device="meta")
    meta_ids = torch.empty((1, 2), dtype=torch.int64, device="meta")
    meta_wts = torch.empty((1, 2), dtype=torch.bfloat16, device="meta")
    meta_out = moe_ffn(meta_x, meta_w, meta_ids, meta_wts)
    assert meta_out.device.type == "meta"
    assert meta_out.shape == (1, 128)
    assert meta_out.dtype is torch.bfloat16
    meta_inplace = torch.empty_like(meta_x)
    assert (
        moe_ffn(meta_x, meta_w, meta_ids, meta_wts, out=meta_inplace, accumulate=True)
        is meta_inplace
    )
    for ctx_factory in (torch.enable_grad, torch.no_grad, torch.inference_mode):
        v_meta = meta_inplace._version
        with ctx_factory(), pytest.raises(ValueError, match="topk_weights"):
            moe_ffn(meta_x, meta_w, meta_ids, meta_wts.to(torch.int32), out=meta_inplace)
        assert meta_inplace._version == v_meta


def test_cpu_noncontiguous_inputs_and_overlapping_out() -> None:
    for weight_format, group_size in (
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        ("convrot_int8", 64),
    ):
        torch.manual_seed(115 + group_size)
        weights = _make_moe_weights(
            3,
            64,
            64,
            weight_format=weight_format,  # type: ignore[arg-type]
            group_size=group_size,
            dtype=torch.bfloat16,
            seed=116 + group_size,
        )
        # 2-D transposed input with stride (1, 3) and 3-D transposed input.
        x_2d_t = torch.randn((64, 3), dtype=torch.bfloat16).t()
        ids_2d = torch.tensor([[0, 2], [1, 0], [2, 1]], dtype=torch.int64)
        wts_2d = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], dtype=torch.bfloat16)
        out_2d = moe_ffn(x_2d_t, weights, ids_2d, wts_2d)
        assert out_2d.is_contiguous()
        assert torch.equal(out_2d, _reference_for_weights(x_2d_t, weights, ids_2d, wts_2d))

        x_3d_t = torch.randn((2, 3, 64), dtype=torch.bfloat16).transpose(0, 1)
        ids_3d = torch.tensor([[[0, 1], [1, 2]], [[2, 0], [0, 2]], [[1, 0], [2, 1]]])
        wts_3d = torch.full((3, 2, 2), 0.5, dtype=torch.bfloat16)
        out_3d = moe_ffn(x_3d_t, weights, ids_3d, wts_3d)
        assert out_3d.shape == (3, 2, 64)
        assert out_3d.is_contiguous()
        assert torch.equal(out_3d, _reference_for_weights(x_3d_t, weights, ids_3d, wts_3d))

    # Reject overlapping output rows before mutation.
    w_nv = _make_moe_weights(2, 64, 64, weight_format="nvfp4_w4a16", dtype=torch.float32, seed=120)
    x3 = torch.randn((3, 64), dtype=torch.float32)
    ids3 = torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.int64)
    wts3 = torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]], dtype=torch.float32)

    expand_backing = torch.full((1, 64), 3.25, dtype=torch.float32)
    with pytest.raises(ValueError, match="overlap"):
        moe_ffn(x3, w_nv, ids3, wts3, out=expand_backing.expand(3, 64), accumulate=False)
    assert torch.equal(expand_backing, torch.full((1, 64), 3.25, dtype=torch.float32))

    half_stride_backing = torch.full((4, 64), 2.5, dtype=torch.float32)
    with pytest.raises(ValueError, match="overlap"):
        moe_ffn(
            x3,
            w_nv,
            ids3,
            wts3,
            out=torch.as_strided(half_stride_backing, (3, 64), (32, 1)),
            accumulate=True,
        )
    assert torch.equal(half_stride_backing, torch.full((4, 64), 2.5, dtype=torch.float32))


def test_cpu_aliasing_guards() -> None:
    w_nv = _make_moe_weights(2, 64, 64, weight_format="nvfp4_w4a16", dtype=torch.float32, seed=122)
    ids3 = torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.int64)
    wts3 = torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]], dtype=torch.float32)

    # Shifted input/out and prepared/out aliasing preserves correctness.
    buf = torch.randn((4, 64), dtype=torch.float32)
    x_shift = buf[:-1]
    out_shift = buf[1:]
    expected_shift = _reference_for_weights(x_shift, w_nv, ids3, wts3)
    moe_ffn(x_shift, w_nv, ids3, wts3, out=out_shift, accumulate=False)
    assert torch.equal(out_shift, expected_shift)

    buf_prep = torch.randn((4, 64), dtype=torch.float32)
    x_prep_shift = buf_prep[:-1]
    prep_shift = prepare_moe_input(x_prep_shift, w_nv)
    out_prep_shift = buf_prep[1:]
    expected_prep_shift = _reference_for_weights(
        x_prep_shift, w_nv, ids3, wts3, prepared_input=prep_shift
    )
    moe_ffn(x_prep_shift, w_nv, ids3, wts3, out=out_prep_shift, prepared_input=prep_shift)
    assert torch.equal(out_prep_shift, expected_prep_shift)

    # out aliasing routing weights preserves routing reads.
    wts_alias_buf = torch.tensor([[0.6, 0.4] + [0.0] * 62], dtype=torch.float32)
    x1 = torch.randn((1, 64), dtype=torch.float32)
    ids1 = torch.tensor([[0, 1]], dtype=torch.int64)
    expected_alias = _reference_for_weights(x1, w_nv, ids1, wts_alias_buf[:, :2].clone())
    moe_ffn(x1, w_nv, ids1, wts_alias_buf[:, :2], out=wts_alias_buf, accumulate=False)
    assert torch.equal(wts_alias_buf, expected_alias)

    # Exact M=1 in-place out is input stays valid.
    x_inplace = x1.clone()
    expected_m1_inplace = _reference_for_weights(x_inplace, w_nv, ids1, wts3[:1])
    assert moe_ffn(x_inplace, w_nv, ids1, wts3[:1], out=x_inplace) is x_inplace
    assert torch.equal(x_inplace, expected_m1_inplace)

    # out aliasing weight storage is rejected before mutation (including empty routing).
    aliased_scale_backing = torch.ones((2, 64), dtype=torch.float32)
    w_int8 = MoEWeights.from_quantized(
        torch.zeros((2, 128, 64), dtype=torch.int8),
        torch.ones((2, 128), dtype=torch.float32),
        torch.zeros((2, 64, 64), dtype=torch.int8),
        aliased_scale_backing,
        logical_dtype=torch.float32,
        group_size=64,
    )
    with pytest.raises(ValueError, match="alias weight storage"):
        moe_ffn(x1, w_int8, ids1[:, :0], wts3[:1, :0], out=aliased_scale_backing[:1])
    assert torch.equal(aliased_scale_backing, torch.ones((2, 64), dtype=torch.float32))

    # aot_eager functionalization must also reject weight-aliasing out before copying back.
    @torch.compile(fullgraph=True, backend="aot_eager")
    def compiled_weight_alias(inp: torch.Tensor) -> torch.Tensor:
        return moe_ffn(inp, w_int8, ids1, wts3[:1], out=aliased_scale_backing[1:])

    with pytest.raises(Exception, match="alias weight storage"):
        compiled_weight_alias(x1)
    assert torch.equal(aliased_scale_backing, torch.ones((2, 64), dtype=torch.float32))


def test_cpu_inference_mode_and_direct_custom_ops() -> None:
    weights = _make_moe_weights(
        2, 64, 64, weight_format="convrot_nvfp4_w4a16", group_size=16, dtype=torch.float16, seed=130
    )
    x = torch.randn((1, 64), dtype=torch.float16)
    ids = torch.tensor([[0, 1]], dtype=torch.int64)
    wts = torch.tensor([[0.6, 0.4]], dtype=torch.float16)

    with torch.inference_mode():
        inf_out = torch.full((2, 64), 4.0, dtype=torch.float16)
    x2 = torch.randn((2, 64), dtype=torch.float16)
    ids2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    wts2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float16)
    with pytest.raises(RuntimeError, match="InferenceMode"):
        moe_ffn(x2, weights, ids2, wts2, out=inf_out, accumulate=False)
    assert torch.equal(inf_out, torch.full((2, 64), 4.0, dtype=torch.float16))

    with pytest.raises(RuntimeError, match="InferenceMode"):
        moe_ffn(x2, weights, ids2[:, :0], wts2[:, :0], out=inf_out, accumulate=False)
    assert torch.equal(inf_out, torch.full((2, 64), 4.0, dtype=torch.float16))

    # Direct custom-op validation before mutation.
    bad_out = torch.full((1, 5), 9.0, dtype=torch.float16)
    with pytest.raises(ValueError, match="out tensor"):
        torch.ops.piper_kernels.moe_ffn_(
            x,
            ids[:, :0],
            wts[:, :0],
            weights.up_gate_qdata,
            weights.up_gate_scale,
            weights.down_qdata,
            weights.down_scale,
            bad_out,
            weights.up_gate_global_scale,
            weights.down_global_scale,
            group_size=weights.group_size,
            activation_fn=weights.activation_fn,
            gate_first=weights.gate_first,
            high_first=weights.high_first,
        )
    assert torch.equal(bad_out, torch.full((1, 5), 9.0, dtype=torch.float16))

    # Dangling prepared_scale and NVFP4 non-None scale rejection.
    with torch.enable_grad(), pytest.raises(ValueError, match="scale must be None"):
        moe_ffn(
            x,
            weights,
            ids,
            wts,
            prepared_input=PreparedMoEInput(
                data=torch.zeros((1, 64), dtype=torch.float32),
                scale=torch.ones((1,), dtype=torch.float32, requires_grad=True),
                logical_dtype=torch.float16,
                group_size=16,
                is_int8=False,
            ),
        )
    with pytest.raises(ValueError, match="scale must be None"):
        torch.ops.piper_kernels.moe_ffn(
            x,
            ids,
            wts,
            weights.up_gate_qdata,
            weights.up_gate_scale,
            weights.down_qdata,
            weights.down_scale,
            weights.up_gate_global_scale,
            weights.down_global_scale,
            prepared_scale=torch.ones((1,), dtype=torch.float32),
            group_size=16,
        )

    # Direct custom ops also enforce inference-only autograd guards.
    with torch.enable_grad():
        with pytest.raises(RuntimeError, match="gradients"):
            torch.ops.piper_kernels.moe_ffn(
                x.clone().requires_grad_(True),
                ids,
                wts,
                weights.up_gate_qdata,
                weights.up_gate_scale,
                weights.down_qdata,
                weights.down_scale,
                weights.up_gate_global_scale,
                weights.down_global_scale,
                group_size=weights.group_size,
            )
        grad_out = torch.ones_like(x, requires_grad=True)
        with pytest.raises(RuntimeError, match="gradients"):
            torch.ops.piper_kernels.moe_ffn_(
                x,
                ids,
                wts,
                weights.up_gate_qdata,
                weights.up_gate_scale,
                weights.down_qdata,
                weights.down_scale,
                grad_out,
                weights.up_gate_global_scale,
                weights.down_global_scale,
                group_size=weights.group_size,
            )
        assert torch.equal(grad_out, torch.ones_like(grad_out))
        assert grad_out._version == 0


def test_cpu_metadata_and_autocast_edge_cases() -> None:
    weights = _make_moe_weights(
        2, 64, 64, weight_format="convrot_nvfp4_w4a16", group_size=16, dtype=torch.float16, seed=132
    )
    x = torch.randn((1, 64), dtype=torch.float16)
    ids = torch.tensor([[0, 1]], dtype=torch.int64)
    wts = torch.tensor([[0.6, 0.4]], dtype=torch.float16)

    with pytest.raises(TypeError, match="bool"):
        moe_ffn(
            x,
            weights,
            ids,
            wts,
            out=torch.empty_like(x),
            accumulate=torch.tensor(True),  # type: ignore[arg-type]
        )

    neg_backing = torch.full((1, 64), 2.0, dtype=torch.float16)
    with pytest.raises(ValueError, match="negated or conjugated"):
        moe_ffn(x, weights, ids, wts, out=torch._neg_view(neg_backing), accumulate=False)
    assert torch.equal(neg_backing, torch.full((1, 64), 2.0, dtype=torch.float16))

    raw_x = torch.randn((1, 64), dtype=torch.float16)
    raw_wts = torch.tensor([[0.6, 0.4]], dtype=torch.float16)
    assert torch.equal(
        moe_ffn(torch._neg_view(raw_x), weights, ids, torch._neg_view(raw_wts)),
        moe_ffn(-raw_x, weights, ids, -raw_wts),
    )

    up_ok, down_ok = _make_dense_weights(1, 64, 64, dtype=torch.float16, seed=131)
    with pytest.raises(TypeError, match="booleans"):
        MoEWeights.from_hp(up_ok, down_ok, gate_first=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strided"):
        MoEWeights.from_hp(up_ok.to_sparse(), down_ok)

    # Post-rotation overflow in from_hp raises ValueError even with compute_per_tensor_scale=False.
    sign16 = torch.tensor(
        [1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1], dtype=torch.float32
    )
    up_overflow = sign16.repeat(1, 32, 1) * 1e38
    down_small = torch.ones((1, 16, 16), dtype=torch.float32) * 0.1
    for fmt in ("convrot_int8", "convrot_nvfp4_w4a16"):
        with pytest.raises(ValueError, match="non-finite"):
            MoEWeights.from_hp(
                up_overflow,
                down_small,
                weight_format=fmt,  # type: ignore[arg-type]
                group_size=16,
                compute_per_tensor_scale=False,
            )
    with pytest.raises(ValueError, match="non-finite"):
        ConvRotNVFP4Tensor.from_hp(up_overflow[0], group_size=16, compute_per_tensor_scale=False)
    with pytest.raises(ValueError, match="non-finite"):
        MoEWeights.from_hp(
            up_overflow,
            down_small,
            weight_format="convrot_nvfp4_w4a16",
            group_size=16,
            compute_per_tensor_scale=True,
        )

    # INT8 reduction width limit on meta device (K > 131_071 rejected).
    with pytest.raises(ValueError, match="131071"):
        MoEWeights.from_quantized(
            torch.empty((1, 32, 131_328), dtype=torch.int8, device="meta"),
            torch.empty((1, 32), dtype=torch.float32, device="meta"),
            torch.empty((1, 131_328, 16), dtype=torch.int8, device="meta"),
            torch.empty((1, 131_328), dtype=torch.float32, device="meta"),
            logical_dtype=torch.bfloat16,
            group_size=16,
        )


def test_cpu_autocast_and_bf16_rounding_parity() -> None:
    weights = _make_moe_weights(
        2, 64, 64, weight_format="convrot_nvfp4_w4a16", group_size=16, dtype=torch.float16, seed=132
    )
    x = torch.randn((1, 64), dtype=torch.float16)
    ids = torch.tensor([[0, 1]], dtype=torch.int64)
    wts = torch.tensor([[0.6, 0.4]], dtype=torch.float16)

    # CPU BF16 autocast preserves FP32 behavior across both ConvRot formats and prepared reuse.
    w_int8_ac = _make_moe_weights(
        1, 16, 16, weight_format="convrot_int8", group_size=16, dtype=torch.float32, seed=134
    )
    x_ac = torch.full((1, 16), 1.001, dtype=torch.float32)
    ids_ac = torch.zeros((1, 1), dtype=torch.int64)
    wts_ac = torch.ones((1, 1), dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prep_autocast = prepare_moe_input(x, weights)
        assert prep_autocast.data.dtype is torch.float32
        out_autocast = moe_ffn(x, weights, ids, wts, prepared_input=prep_autocast)
        assert out_autocast.dtype is torch.float16
        assert torch.equal(out_autocast, moe_ffn(x, weights, ids, wts))

        prep_int8_ac = prepare_moe_input(x_ac, w_int8_ac)
        out_int8_ac = moe_ffn(x_ac, w_int8_ac, ids_ac, wts_ac, prepared_input=prep_int8_ac)
        assert torch.equal(out_int8_ac, moe_ffn(x_ac, w_int8_ac, ids_ac, wts_ac))

    # BF16 ConvRot NVFP4 prepared vs unprepared parity on a rounding-sensitive group-16 vector.
    w_bf16 = _make_moe_weights(
        2,
        16,
        16,
        weight_format="convrot_nvfp4_w4a16",
        group_size=16,
        dtype=torch.bfloat16,
        seed=133,
    )
    x_round = torch.zeros((1, 16), dtype=torch.bfloat16)
    x_round[0, :4] = torch.tensor([-515.0, -517.0, 257.0, -259.0], dtype=torch.bfloat16)
    prep_bf16 = prepare_moe_input(x_round, w_bf16)
    assert prep_bf16.data.dtype is torch.float32
    assert torch.equal(
        moe_ffn(x_round, w_bf16, ids, wts.to(torch.bfloat16), prepared_input=prep_bf16),
        moe_ffn(x_round, w_bf16, ids, wts.to(torch.bfloat16)),
    )

    # FakeTensorMode covers prepare_moe_input and moe_ffn across all three formats.
    for fmt, gs in (("nvfp4_w4a16", 0), ("convrot_nvfp4_w4a16", 64), ("convrot_int8", 64)):
        w_fmt = _make_moe_weights(
            2,
            64,
            64,
            weight_format=fmt,  # type: ignore[arg-type]
            group_size=gs,
            dtype=torch.bfloat16,
            seed=140 + gs,
        )
        with FakeTensorMode(allow_non_fake_inputs=True):
            fx = torch.empty((1, 64), dtype=torch.bfloat16)
            fids = torch.empty((1, 2), dtype=torch.int64)
            fwts = torch.empty((1, 2), dtype=torch.bfloat16)
            fprep = prepare_moe_input(fx, w_fmt)
            fout = moe_ffn(fx, w_fmt, fids, fwts, prepared_input=fprep)
            assert fout.shape == (1, 64)
            assert fout.dtype is torch.bfloat16


def test_moe_import_without_optional_deps_and_cpu_runs_without_triton() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    blocker_prelude = """
import builtins
import importlib.util

def _install_module_blocker(blocked_roots):
    original_import = builtins.__import__
    original_find_spec = importlib.util.find_spec

    def _is_blocked(name):
        return name.partition(".")[0] in blocked_roots

    def guarded_import(name, *args, **kwargs):
        if _is_blocked(name):
            raise ModuleNotFoundError(f"Blocked optional dependency {name}", name=name)
        return original_import(name, *args, **kwargs)

    def guarded_find_spec(name, *args, **kwargs):
        if _is_blocked(name):
            return None
        return original_find_spec(name, *args, **kwargs)

    builtins.__import__ = guarded_import
    importlib.util.find_spec = guarded_find_spec
"""
    import_script = (
        blocker_prelude
        + """
_install_module_blocker(("triton", "torchao"))
import torch

from piper_kernels.moe import MoEWeights, PreparedMoEInput, moe_ffn, prepare_moe_input

assert PreparedMoEInput.__name__ == "PreparedMoEInput"
weights = MoEWeights.from_quantized(
    torch.zeros((2, 256, 128), dtype=torch.int8),
    torch.ones((2, 256), dtype=torch.float32),
    torch.zeros((2, 128, 128), dtype=torch.int8),
    torch.ones((2, 128), dtype=torch.float32),
    logical_dtype=torch.bfloat16,
    group_size=64,
)
assert weights.num_slots == 2
assert callable(prepare_moe_input)
assert callable(moe_ffn)
"""
    )
    import_result = subprocess.run(
        [sys.executable, "-c", import_script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_result.returncode == 0, import_result.stdout + import_result.stderr

    cpu_script = (
        blocker_prelude
        + """
_install_module_blocker(("triton",))
import torch

from piper_kernels.moe import MoEWeights, moe_ffn, prepare_moe_input

torch.manual_seed(7)
up_gate_hp = torch.randn((3, 256, 128), dtype=torch.bfloat16) * 0.1
down_hp = torch.randn((3, 128, 128), dtype=torch.bfloat16) * 0.1
int8_weights = MoEWeights.from_hp(
    up_gate_hp, down_hp, weight_format="convrot_int8", group_size=64
)
x = torch.randn((1, 128), dtype=torch.bfloat16)
topk_ids = torch.tensor([[0, 2, 1]], dtype=torch.int64)
topk_weights = torch.tensor([[0.5, 0.5, 0.25]], dtype=torch.bfloat16)
slot_map = torch.tensor([1, -1, 0], dtype=torch.int64)
prep = prepare_moe_input(x, int8_weights)
out = torch.full((1, 128), 0.25, dtype=torch.bfloat16)
res_int8 = moe_ffn(
    x,
    int8_weights,
    topk_ids,
    topk_weights,
    expert_to_slot=slot_map,
    out=out,
    accumulate=True,
    prepared_input=prep,
)
assert res_int8 is out
assert torch.isfinite(res_int8).all()

nvfp4_weights = MoEWeights.from_hp(up_gate_hp, down_hp, weight_format="nvfp4_w4a16")
res_nvfp4 = moe_ffn(x, nvfp4_weights, topk_ids, topk_weights, expert_to_slot=slot_map)
assert torch.isfinite(res_nvfp4).all()
"""
    )
    cpu_result = subprocess.run(
        [sys.executable, "-c", cpu_script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cpu_result.returncode == 0, cpu_result.stdout + cpu_result.stderr


@pytest.mark.parametrize(
    ("up_layout", "down_layout"),
    [
        ("uniform", "uniform"),
        ("var2d", "var2d"),
        ("var1d", "var1d"),
        ("var1d", "var2d"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_trellis_layouts_dtypes_block_sizes(
    up_layout: TrellisLayoutKind,
    down_layout: TrellisLayoutKind,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(145 + int(dtype is torch.float16))
    # Distinct up/down block sizes (256 vs 128) and mixed layouts across slots.
    weights = _make_trellis_moe_weights(
        3,
        256,
        128,
        group_size=64,
        dtype=dtype,
        layouts=(up_layout, down_layout),
        seed=150,
    )
    assert weights.weight_format == "convrot_trellis"
    assert weights.num_slots == 3
    assert weights.hidden_size == 256
    assert weights.intermediate_size == 128
    assert weights.up_gate_block_size == 256
    assert weights.down_block_size == 128

    x = torch.randn((2, 256), dtype=dtype)
    topk_ids = torch.tensor([[2, 0], [1, 1]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=dtype)
    actual = moe_ffn(x, weights, topk_ids, topk_weights)
    expected = _reference_for_weights(x, weights, topk_ids, topk_weights)
    oracle = _independent_moe_oracle(x, weights, topk_ids, topk_weights)
    assert torch.equal(actual, expected)
    _assert_matches_reference(actual, oracle, weight_format="convrot_trellis", dtype=dtype)

    if up_layout == "var1d" and down_layout == "var1d" and dtype is torch.bfloat16:
        # Non-power-of-two T = 224 on widths divisible by 224 and group_size=16.
        weights_224 = _make_trellis_moe_weights(
            2,
            448,
            224,
            group_size=16,
            activation_fn="gelu_tanh",
            dtype=dtype,
            layouts=("var1d", "var2d"),
            seed=151,
        )
        assert weights_224.up_gate_block_size == 224
        assert weights_224.down_block_size == 224
        x_224 = torch.randn((1, 448), dtype=dtype)
        ids_224 = torch.tensor([[1, 0]], dtype=torch.int64)
        wts_224 = torch.tensor([[0.55, 0.45]], dtype=dtype)
        out_224 = moe_ffn(x_224, weights_224, ids_224, wts_224)
        oracle_224 = _independent_moe_oracle(x_224, weights_224, ids_224, wts_224)
        assert torch.equal(out_224, _reference_for_weights(x_224, weights_224, ids_224, wts_224))
        _assert_matches_reference(out_224, oracle_224, weight_format="convrot_trellis", dtype=dtype)


def test_cpu_trellis_validation_and_empty_hp() -> None:
    # MoEWeights.from_hp validates trellis arguments and builds valid empty storage for num_slots=0.
    empty_up, empty_dn = _make_dense_weights(0, 256, 256, dtype=torch.bfloat16, seed=152)
    empty_w = MoEWeights.from_hp(
        empty_up,
        empty_dn,
        weight_format="convrot_trellis",
        group_size=64,
        bits_per_weight=4.5,
    )
    assert empty_w.weight_format == "convrot_trellis"
    assert empty_w.num_slots == 0
    assert empty_w.up_gate_qdata.ndim == 4
    assert empty_w.up_gate_block_size == 256
    assert empty_w.down_block_size == 256

    up_hp, dn_hp = _make_dense_weights(1, 256, 256, dtype=torch.bfloat16, seed=153)
    with pytest.raises(ValueError, match="bits_per_weight"):
        MoEWeights.from_hp(
            up_hp, dn_hp, weight_format="convrot_int8", group_size=64, bits_per_weight=4.5
        )
    with pytest.raises(ValueError, match="specify exactly one"):
        MoEWeights.from_hp(empty_up, empty_dn, weight_format="convrot_trellis", group_size=64)
    with pytest.raises(ValueError, match="specify exactly one"):
        MoEWeights.from_hp(
            empty_up,
            empty_dn,
            weight_format="convrot_trellis",
            group_size=64,
            bits_per_weight=4.5,
            bytes_per_block=143,
        )

    weights = _make_trellis_moe_weights(
        2,
        128,
        128,
        group_size=64,
        dtype=torch.bfloat16,
        layouts=("var1d", "var2d"),
        seed=154,
    )
    # Storage and metadata validation errors.
    with pytest.raises(ValueError, match="high_first"):
        MoEWeights.from_quantized(
            weights.up_gate_qdata,
            weights.up_gate_scale,
            weights.down_qdata,
            weights.down_scale,
            logical_dtype=torch.bfloat16,
            group_size=64,
            high_first=True,
            up_gate_block_scales=weights.up_gate_block_scales,
            down_block_scales=weights.down_block_scales,
            up_gate_row_widths=weights.up_gate_row_widths,
            down_row_widths=weights.down_row_widths,
            up_gate_slot_offsets=weights.up_gate_slot_offsets,
            down_slot_offsets=weights.down_slot_offsets,
            up_gate_block_size=128,
            down_block_size=128,
        )
    with pytest.raises(ValueError, match="global scales"):
        MoEWeights.from_quantized(
            weights.up_gate_qdata,
            weights.up_gate_scale,
            weights.down_qdata,
            weights.down_scale,
            logical_dtype=torch.bfloat16,
            group_size=64,
            up_gate_global_scale=torch.ones((2,), dtype=torch.float32),
            up_gate_block_scales=weights.up_gate_block_scales,
            down_block_scales=weights.down_block_scales,
            up_gate_row_widths=weights.up_gate_row_widths,
            down_row_widths=weights.down_row_widths,
            up_gate_slot_offsets=weights.up_gate_slot_offsets,
            down_slot_offsets=weights.down_slot_offsets,
            up_gate_block_size=128,
            down_block_size=128,
        )
    with pytest.raises(ValueError, match="slot_offsets requires"):
        MoEWeights.from_quantized(
            weights.up_gate_qdata,
            weights.up_gate_scale,
            weights.down_qdata,
            weights.down_scale,
            logical_dtype=torch.bfloat16,
            group_size=64,
            up_gate_block_scales=weights.up_gate_block_scales,
            down_block_scales=weights.down_block_scales,
            up_gate_row_widths=None,
            down_row_widths=weights.down_row_widths,
            up_gate_slot_offsets=weights.up_gate_slot_offsets,
            down_slot_offsets=weights.down_slot_offsets,
            up_gate_block_size=128,
            down_block_size=128,
        )
    with pytest.raises(ValueError, match="2-D variable-row"):
        MoEWeights.from_quantized(
            weights.up_gate_qdata,
            weights.up_gate_scale,
            weights.down_qdata.reshape(-1),
            weights.down_scale,
            logical_dtype=torch.bfloat16,
            group_size=64,
            up_gate_block_scales=weights.up_gate_block_scales,
            down_block_scales=weights.down_block_scales,
            up_gate_row_widths=weights.up_gate_row_widths,
            down_row_widths=weights.down_row_widths,
            up_gate_slot_offsets=weights.up_gate_slot_offsets,
            down_slot_offsets=None,
            up_gate_block_size=128,
            down_block_size=128,
        )
    with pytest.raises(ValueError, match="1-D variable-row"):
        MoEWeights.from_quantized(
            weights.up_gate_qdata.unsqueeze(0),
            weights.up_gate_scale,
            weights.down_qdata,
            weights.down_scale,
            logical_dtype=torch.bfloat16,
            group_size=64,
            up_gate_block_scales=weights.up_gate_block_scales,
            down_block_scales=weights.down_block_scales,
            up_gate_row_widths=weights.up_gate_row_widths,
            down_row_widths=weights.down_row_widths,
            up_gate_slot_offsets=weights.up_gate_slot_offsets,
            down_slot_offsets=weights.down_slot_offsets,
            up_gate_block_size=128,
            down_block_size=128,
        )
    with pytest.raises(ValueError, match="slot_offsets must be int64"):
        MoEWeights.from_quantized(
            weights.up_gate_qdata,
            weights.up_gate_scale,
            weights.down_qdata.reshape(-1),
            weights.down_scale,
            logical_dtype=torch.bfloat16,
            group_size=64,
            up_gate_block_scales=weights.up_gate_block_scales,
            down_block_scales=weights.down_block_scales,
            up_gate_row_widths=weights.up_gate_row_widths,
            down_row_widths=weights.down_row_widths,
            up_gate_slot_offsets=weights.up_gate_slot_offsets,
            down_slot_offsets=torch.zeros((2,), dtype=torch.int32),
            up_gate_block_size=128,
            down_block_size=128,
        )

    bad_offset_w = MoEWeights.from_quantized(
        weights.up_gate_qdata,
        weights.up_gate_scale,
        weights.down_qdata,
        weights.down_scale,
        logical_dtype=torch.bfloat16,
        group_size=64,
        up_gate_block_scales=weights.up_gate_block_scales,
        down_block_scales=weights.down_block_scales,
        up_gate_row_widths=weights.up_gate_row_widths,
        down_row_widths=weights.down_row_widths,
        up_gate_slot_offsets=torch.tensor([-1, 0], dtype=torch.int64),
        down_slot_offsets=weights.down_slot_offsets,
        up_gate_block_size=128,
        down_block_size=128,
    )
    # Negative slot_offsets violate caller preconditions (no host readback); fallback clamps safely.
    bad_offset_out = moe_ffn(
        torch.randn((1, 128), dtype=torch.bfloat16),
        bad_offset_w,
        torch.tensor([[0, 1]], dtype=torch.int64),
        torch.tensor([[0.5, 0.5]], dtype=torch.bfloat16),
    )
    assert bad_offset_out.shape == (1, 128)
    assert bad_offset_out.dtype == torch.bfloat16
    assert torch.isfinite(bad_offset_out).all()


def test_cpu_trellis_aliasing_and_autograd_guards() -> None:
    weights = _make_trellis_moe_weights(
        2,
        128,
        128,
        group_size=64,
        dtype=torch.bfloat16,
        layouts=("var1d", "var2d"),
        seed=154,
    )
    x = torch.randn((1, 128), dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1]], dtype=torch.int64)
    wts = torch.tensor([[0.6, 0.4]], dtype=torch.bfloat16)
    out = torch.full((1, 128), 3.0, dtype=torch.bfloat16)
    before = out.clone()

    # Prepared input validation & leaving caller out untouched.
    for bad_prep in (
        PreparedMoEInput(
            data=torch.zeros((1, 128), dtype=torch.int8),
            scale=torch.ones((1,), dtype=torch.float32),
            logical_dtype=torch.bfloat16,
            group_size=64,
            is_int8=True,
        ),
        PreparedMoEInput(
            data=torch.zeros((1, 128), dtype=torch.float32),
            scale=torch.ones((1,), dtype=torch.float32),
            logical_dtype=torch.bfloat16,
            group_size=64,
            is_int8=False,
        ),
    ):
        with pytest.raises(ValueError, match=r"(prepared_input|scale must be None)"):
            moe_ffn(x, weights, ids, wts, out=out, accumulate=True, prepared_input=bad_prep)
        assert torch.equal(out, before)

    # Autograd guard on FP8 block_scales.
    with torch.enable_grad():
        assert weights.up_gate_block_scales is not None
        with pytest.raises(RuntimeError, match="autograd"):
            torch.ops.piper_kernels.moe_ffn(
                x,
                ids,
                wts,
                weights.up_gate_qdata,
                weights.up_gate_scale,
                weights.down_qdata,
                weights.down_scale,
                group_size=64,
                up_gate_block_scales=weights.up_gate_block_scales.clone().requires_grad_(True),
                down_block_scales=weights.down_block_scales,
                up_gate_row_widths=weights.up_gate_row_widths,
                down_row_widths=weights.down_row_widths,
                up_gate_slot_offsets=weights.up_gate_slot_offsets,
                down_slot_offsets=weights.down_slot_offsets,
                up_gate_block_size=128,
                down_block_size=128,
            )

    # Out aliasing any trellis metadata tensor is rejected before mutation.
    buf = torch.ones((4096,), dtype=torch.uint8)
    out_alias = buf[:256].view(torch.bfloat16).view(1, 128)
    assert weights.up_gate_block_scales is not None
    assert weights.down_block_scales is not None
    assert weights.up_gate_row_widths is not None
    assert weights.down_row_widths is not None
    assert weights.up_gate_slot_offsets is not None
    alias_bs_up = buf[:512].view(torch.float8_e4m3fn).view(2, 256, 1)
    alias_bs_dn = buf[:256].view(torch.float8_e4m3fn).view(2, 128, 1)
    alias_rw_up = buf[:512].view(2, 256)
    alias_rw_dn = buf[:256].view(2, 128)
    alias_so = buf[:16].view(torch.int64).zero_()
    alias_cases: list[dict[str, object]] = [
        {"up_gate_block_scales": alias_bs_up},
        {"down_block_scales": alias_bs_dn},
        {"up_gate_row_widths": alias_rw_up},
        {"down_row_widths": alias_rw_dn},
        {"up_gate_slot_offsets": alias_so},
        {
            "down_row_widths": alias_rw_dn,
            "down_slot_offsets": alias_so,
            "down_qdata": torch.zeros((256,), dtype=torch.uint8),
        },
    ]
    for overrides in alias_cases:
        w_alias = MoEWeights.from_quantized(
            overrides.get("up_gate_qdata", weights.up_gate_qdata),  # type: ignore[arg-type]
            weights.up_gate_scale,
            overrides.get("down_qdata", weights.down_qdata),  # type: ignore[arg-type]
            weights.down_scale,
            logical_dtype=torch.bfloat16,
            group_size=64,
            up_gate_block_scales=overrides.get(
                "up_gate_block_scales",
                weights.up_gate_block_scales,  # type: ignore[arg-type]
            ),
            down_block_scales=overrides.get(
                "down_block_scales",
                weights.down_block_scales,  # type: ignore[arg-type]
            ),
            up_gate_row_widths=overrides.get(
                "up_gate_row_widths",
                weights.up_gate_row_widths,  # type: ignore[arg-type]
            ),
            down_row_widths=overrides.get(
                "down_row_widths",
                weights.down_row_widths,  # type: ignore[arg-type]
            ),
            up_gate_slot_offsets=overrides.get(
                "up_gate_slot_offsets",
                weights.up_gate_slot_offsets,  # type: ignore[arg-type]
            ),
            down_slot_offsets=overrides.get("down_slot_offsets", None),  # type: ignore[arg-type]
            up_gate_block_size=128,
            down_block_size=128,
        )
        snapshot = buf.clone()
        with pytest.raises(ValueError, match="alias weight storage"):
            moe_ffn(x, w_alias, ids[:, :0], wts[:, :0], out=out_alias)
        assert torch.equal(buf, snapshot)


def test_cpu_trellis_prepared_waves_misses_and_meta() -> None:
    torch.manual_seed(156)
    for layout in ("uniform", "var2d", "var1d"):
        w = _make_trellis_moe_weights(
            4,
            128,
            128,
            group_size=64,
            dtype=torch.bfloat16,
            layouts=(layout, layout),
            seed=157,
        )
        x = torch.randn((3, 128), dtype=torch.bfloat16)
        ids = torch.tensor([[0, 2], [1, 3], [-1, 5]], dtype=torch.int64)
        wts = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.2, 0.8]], dtype=torch.bfloat16)
        slot_map = torch.tensor([2, 0, -1, 3], dtype=torch.int64)
        prep = prepare_moe_input(x, w)
        assert prep.data.dtype is torch.float32
        assert prep.scale is None
        out = torch.full((3, 128), 1.0, dtype=torch.bfloat16)
        expected = _reference_for_weights(
            x, w, ids, wts, expert_to_slot=slot_map, out=out, accumulate=True
        )
        moe_ffn(
            x,
            w,
            ids,
            wts,
            expert_to_slot=slot_map,
            out=out,
            accumulate=True,
            prepared_input=prep,
        )
        assert torch.equal(out, expected)

    # Zero-slot / zero-row / FakeTensor / meta dispatch on 1-D trellis storage.
    w_1d = _make_trellis_moe_weights(
        2,
        128,
        128,
        group_size=64,
        dtype=torch.bfloat16,
        layouts=("var1d", "var1d"),
        seed=158,
    )
    with FakeTensorMode(allow_non_fake_inputs=True):
        fx = torch.empty((2, 128), dtype=torch.bfloat16)
        fids = torch.empty((2, 2), dtype=torch.int64)
        fwts = torch.empty((2, 2), dtype=torch.bfloat16)
        fprep = prepare_moe_input(fx, w_1d)
        fout = moe_ffn(fx, w_1d, fids, fwts, prepared_input=fprep)
        assert fout.shape == (2, 128)
        assert fout.dtype is torch.bfloat16

    meta_w = MoEWeights.from_quantized(
        torch.empty((1024,), dtype=torch.uint8, device="meta"),
        torch.empty((2, 256), dtype=torch.float32, device="meta"),
        torch.empty((512,), dtype=torch.uint8, device="meta"),
        torch.empty((2, 128), dtype=torch.float32, device="meta"),
        logical_dtype=torch.bfloat16,
        group_size=64,
        up_gate_block_scales=torch.empty((2, 256, 1), dtype=torch.float8_e4m3fn, device="meta"),
        down_block_scales=torch.empty((2, 128, 1), dtype=torch.float8_e4m3fn, device="meta"),
        up_gate_row_widths=torch.empty((2, 256), dtype=torch.uint8, device="meta"),
        down_row_widths=torch.empty((2, 128), dtype=torch.uint8, device="meta"),
        up_gate_slot_offsets=torch.empty((2,), dtype=torch.int64, device="meta"),
        down_slot_offsets=torch.empty((2,), dtype=torch.int64, device="meta"),
        up_gate_block_size=128,
        down_block_size=128,
    )
    meta_x = torch.empty((2, 128), dtype=torch.bfloat16, device="meta")
    meta_ids = torch.empty((2, 2), dtype=torch.int64, device="meta")
    meta_wts = torch.empty((2, 2), dtype=torch.bfloat16, device="meta")
    meta_out = moe_ffn(meta_x, meta_w, meta_ids, meta_wts)
    assert meta_out.device.type == "meta"
    assert meta_out.shape == (2, 128)


@pytest.mark.parametrize("target", _NVIDIA_TARGETS)
@pytest.mark.parametrize(
    ("weight_format", "group_size", "logical_dtype"),
    [
        ("nvfp4_w4a16", 0, torch.bfloat16),
        ("convrot_nvfp4_w4a16", 64, torch.float16),
        ("convrot_int8", 256, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(("split_k", "accumulate"), [(1, False), (2, True)])
def test_offline_nvidia_compile_zero_spills(
    target: GPUTarget,
    weight_format: MoEFormat,
    group_size: int,
    logical_dtype: torch.dtype,
    split_k: int,
    accumulate: bool,
) -> None:
    compiled = _compile_nvidia_and_assert_clean(
        target,
        check_int8_mma=True,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        hidden_size=5120,
        intermediate_size=2304,
        group_size=group_size,
        split_k=split_k,
        accumulate=accumulate,
    )
    expected_prefix = "int8_matmul_" if weight_format == "convrot_int8" else "nvfp4_gemv_"
    assert f"{expected_prefix}up" in compiled
    assert f"{expected_prefix}down" in compiled


@pytest.mark.parametrize("weight_format", ["nvfp4_w4a16", "convrot_nvfp4_w4a16"])
@pytest.mark.parametrize(("split_k", "accumulate"), [(1, False), (2, True)])
def test_offline_sm75_compile_matrix(
    weight_format: MoEFormat, split_k: int, accumulate: bool
) -> None:
    sm75 = GPUTarget("cuda", 75, 32)
    _compile_nvidia_and_assert_clean(
        sm75,
        weight_format=weight_format,
        logical_dtype=torch.float16,
        hidden_size=1024,
        intermediate_size=512,
        group_size=64 if weight_format == "convrot_nvfp4_w4a16" else 0,
        split_k=split_k,
        accumulate=accumulate,
    )

    with pytest.raises(ValueError, match="SM80"):
        moe_triton.compile_moe_kernels(
            sm75,
            weight_format="convrot_int8",
            logical_dtype=torch.float16,
            group_size=64,
        )
    sm80 = GPUTarget("cuda", 80, 32)
    with pytest.raises(ValueError, match="divisible"):
        moe_triton.compile_moe_kernels(
            sm80,
            weight_format="convrot_int8",
            hidden_size=80,
            intermediate_size=64,
            group_size=64,
        )
    with pytest.raises(ValueError, match="multiples of 16"):
        moe_triton.compile_moe_kernels(
            sm75,
            weight_format="nvfp4_w4a16",
            hidden_size=24,
            intermediate_size=64,
            group_size=0,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="ROCm HIP compiler is Linux-only")
@pytest.mark.parametrize(("arch", "warp_size"), _AMD_TARGETS)
@pytest.mark.parametrize(
    ("weight_format", "group_size", "logical_dtype"),
    [
        ("nvfp4_w4a16", 0, torch.bfloat16),
        ("convrot_nvfp4_w4a16", 64, torch.float16),
        ("convrot_int8", 256, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(("split_k", "accumulate"), [(1, False), (2, True)])
def test_offline_amd_compile_matrix_and_fences(
    arch: str,
    warp_size: int,
    weight_format: MoEFormat,
    group_size: int,
    logical_dtype: torch.dtype,
    split_k: int,
    accumulate: bool,
) -> None:
    target = GPUTarget("hip", arch, warp_size)
    compiled = moe_triton.compile_moe_kernels(
        target,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        hidden_size=5120,
        intermediate_size=2304,
        group_size=group_size,
        split_k=split_k,
        accumulate=accumulate,
    )
    assert compiled
    expected_prefix = "int8_matmul_" if weight_format == "convrot_int8" else "nvfp4_gemv_"
    assert f"{expected_prefix}up" in compiled
    assert f"{expected_prefix}down" in compiled
    expected_int8_isa = "v_mfma_i32" if arch == "gfx942" else "v_wmma_i32"
    for name, kernel in compiled.items():
        # gfx1201 short-batch INT8 intentionally uses wide block_k=256 for cold-DRAM
        # coalescing (matching linear/convrot/int8), staging tiles in VGPR scratch
        # while keeping global_scratch_size == 0.
        check_spills = not (arch == "gfx1201" and name.startswith("int8_matmul"))
        _assert_clean_amd_compile(
            kernel,
            arch,
            check_vgpr_spills=check_spills,
            check_lds_fences=name.startswith("int8_matmul"),
        )
        if name.startswith("int8_matmul"):
            assert expected_int8_isa in kernel.asm["amdgcn"]


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"), [("nvfp4_w4a16", 0), ("convrot_nvfp4_w4a16", 64)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("activation_fn", "gate_first"),
    [("swiglu", True), ("swiglu", False), ("gelu_tanh", True)],
)
def test_gpu_moe_matches_reference_nvfp4(
    weight_format: MoEFormat,
    group_size: int,
    dtype: torch.dtype,
    activation_fn: ActivationKind,
    gate_first: bool,
) -> None:
    torch.manual_seed(200 + group_size + int(gate_first))
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        dtype=dtype,
        device="cuda",
        seed=210 + group_size,
    )
    x = torch.randn((1, 256), device="cuda", dtype=dtype)
    topk_ids = torch.tensor([[3, 0, 2]], device="cuda", dtype=torch.int64)
    topk_weights = torch.tensor([[0.5, 0.3, 0.2]], device="cuda", dtype=dtype)

    actual = moe_ffn(x, weights, topk_ids, topk_weights)
    expected = _reference_for_weights(x, weights, topk_ids, topk_weights)
    _assert_matches_reference(actual, expected, weight_format=weight_format, dtype=dtype)


@pytest.mark.gpu
@pytest.mark.skipif(
    not _INT8_MOE_GPU_AVAILABLE,
    reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("activation_fn", "gate_first"),
    [("swiglu", True), ("swiglu", False), ("gelu_tanh", True)],
)
def test_gpu_moe_matches_reference_int8(
    dtype: torch.dtype,
    activation_fn: ActivationKind,
    gate_first: bool,
) -> None:
    torch.manual_seed(240 + int(gate_first))
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format="convrot_int8",
        group_size=64,
        activation_fn=activation_fn,
        gate_first=gate_first,
        dtype=dtype,
        device="cuda",
        seed=250 + int(gate_first),
    )
    x = torch.randn((1, 256), device="cuda", dtype=dtype)
    topk_ids = torch.tensor([[2, 1, 0]], device="cuda", dtype=torch.int64)
    topk_weights = torch.tensor([[0.45, 0.35, 0.2]], device="cuda", dtype=dtype)

    actual = moe_ffn(x, weights, topk_ids, topk_weights)
    expected = _reference_for_weights(x, weights, topk_ids, topk_weights)
    _assert_matches_reference(actual, expected, weight_format="convrot_int8", dtype=dtype)


@pytest.mark.gpu
@pytest.mark.skipif(
    not w4a16_nvfp4_available(),
    reason="requires NVIDIA SM75+ or supported ROCm",
)
def test_gpu_moe_tails_splitk_multirow_and_deepseek_anchor() -> None:
    # 1. Mid-tile swap_halves + unaligned block_k tails (I=80 g=16 and I=192 g=64)
    if _INT8_MOE_GPU_AVAILABLE:
        for hidden_size, intermediate_size, group_size in ((128, 80, 16), (256, 192, 64)):
            w_int8 = _make_moe_weights(
                2,
                hidden_size,
                intermediate_size,
                weight_format="convrot_int8",
                group_size=group_size,
                gate_first=True,
                dtype=torch.bfloat16,
                device="cuda",
                seed=300 + intermediate_size,
            )
            x = torch.randn((1, hidden_size), device="cuda", dtype=torch.bfloat16)
            ids = torch.tensor([[1, 0]], device="cuda", dtype=torch.int64)
            wts = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.bfloat16)
            _assert_matches_reference(
                moe_ffn(x, w_int8, ids, wts),
                _reference_for_weights(x, w_int8, ids, wts),
                weight_format="convrot_int8",
                dtype=torch.bfloat16,
            )

    # 2. high_first=True and compute_per_tensor_scale=False
    for high_first, two_level in ((True, True), (False, False)):
        hf_up, hf_down = _make_dense_weights(
            2, 256, 256, dtype=torch.float16, device="cuda", seed=310 + int(high_first)
        )
        w_nvfp4 = MoEWeights.from_hp(
            hf_up,
            hf_down,
            weight_format="convrot_nvfp4_w4a16",
            group_size=64,
            high_first=high_first,
            compute_per_tensor_scale=two_level,
        )
        x16 = torch.randn((1, 256), device="cuda", dtype=torch.float16)
        ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64)
        wts = torch.tensor([[0.55, 0.45]], device="cuda", dtype=torch.float16)
        _assert_matches_reference(
            moe_ffn(x16, w_nvfp4, ids, wts),
            _reference_for_weights(x16, w_nvfp4, ids, wts),
            weight_format="convrot_nvfp4_w4a16",
            dtype=torch.float16,
        )

    # 3. Narrow Split-K + M=2 multi-row across NVFP4 and INT8
    narrow_formats: list[tuple[MoEFormat, int]] = [("nvfp4_w4a16", 0)]
    if _INT8_MOE_GPU_AVAILABLE:
        narrow_formats.append(("convrot_int8", 256))
    for weight_format, group_size in narrow_formats:
        w_narrow = _make_moe_weights(
            2,
            4096,
            256,
            weight_format=weight_format,
            group_size=group_size,
            dtype=torch.bfloat16,
            device="cuda",
            seed=320 + group_size,
        )
        x2 = torch.randn((2, 4096), device="cuda", dtype=torch.bfloat16)
        ids2 = torch.tensor([[0, 1], [1, 0]], device="cuda", dtype=torch.int64)
        wts2 = torch.tensor([[0.7, 0.3], [0.2, 0.8]], device="cuda", dtype=torch.bfloat16)
        _assert_matches_reference(
            moe_ffn(x2, w_narrow, ids2, wts2),
            _reference_for_weights(x2, w_narrow, ids2, wts2),
            weight_format=weight_format,
            dtype=torch.bfloat16,
        )

    # 4. DeepSeek anchor decode shape H=5120, I=2304, top_k=6.
    torch.manual_seed(330)
    up_hp = torch.randn((6, 4608, 5120), device="cuda", dtype=torch.bfloat16) * (5120**-0.5)
    down_hp = torch.randn((6, 5120, 2304), device="cuda", dtype=torch.bfloat16) * (2304**-0.5)
    ds_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device="cuda", dtype=torch.int64)
    ds_wts = torch.tensor(
        [[0.25, 0.2, 0.18, 0.15, 0.12, 0.10]], device="cuda", dtype=torch.bfloat16
    )
    x_ds = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
    ds_formats: list[tuple[MoEFormat, int]] = [("nvfp4_w4a16", 0)]
    if _INT8_MOE_GPU_AVAILABLE:
        ds_formats.append(("convrot_int8", 256))
    for weight_format, group_size in ds_formats:
        w_ds = MoEWeights.from_hp(
            up_hp,
            down_hp,
            weight_format=weight_format,
            group_size=group_size,  # type: ignore[arg-type]
        )
        actual_ds = moe_ffn(x_ds, w_ds, ds_ids, ds_wts)
        expected_ds = _reference_for_weights(x_ds, w_ds, ds_ids, ds_wts)
        _assert_matches_reference(
            actual_ds,
            expected_ds,
            weight_format=weight_format,  # type: ignore[arg-type]
            dtype=torch.bfloat16,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_slot_indirection_misses_accumulate_and_strided_out(
    weight_format: MoEFormat, group_size: int
) -> None:
    torch.manual_seed(360 + group_size)
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=361 + group_size,
    )
    x = torch.randn((1, 256), device="cuda", dtype=torch.bfloat16)
    slot_map = torch.tensor([3, -1, 1, -1, 0, -1], device="cuda", dtype=torch.int32)
    ids = torch.tensor([[1, 2, 99, 0]], device="cuda", dtype=torch.int32)
    wts = torch.tensor([[0.25, 0.35, 0.2, 0.2]], device="cuda", dtype=torch.bfloat16)

    backing = torch.full((1, 272), float("nan"), device="cuda", dtype=torch.bfloat16)
    out_view = backing[:, :256]
    out_view.fill_(1.25)
    expected = _reference_for_weights(
        x, weights, ids, wts, expert_to_slot=slot_map, out=out_view, accumulate=True
    )
    actual = moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map, out=out_view, accumulate=True)
    assert actual is out_view
    _assert_matches_reference(actual, expected, weight_format=weight_format, dtype=torch.bfloat16)
    assert torch.isnan(backing[:, 256:]).all()

    neg_zero_backing = torch.full((1, 272), float("nan"), device="cuda", dtype=torch.bfloat16)
    neg_zero_view = neg_zero_backing[:, :256]
    neg_zero_view.fill_(-0.0)
    all_miss_ids = torch.tensor([[1, 3, -5, 100]], device="cuda", dtype=torch.int32)
    moe_ffn(
        x, weights, all_miss_ids, wts, expert_to_slot=slot_map, out=neg_zero_view, accumulate=True
    )
    assert torch.signbit(neg_zero_view).all()
    assert torch.isnan(neg_zero_backing[:, 256:]).all()


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_two_wave_prepared_input_accumulation(
    weight_format: MoEFormat, group_size: int
) -> None:
    torch.manual_seed(380 + group_size)
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.float16,
        device="cuda",
        seed=381 + group_size,
    )
    x = torch.randn((1, 256), device="cuda", dtype=torch.float16)
    ids = torch.tensor([[0, 2, 3, 5]], device="cuda", dtype=torch.int64)
    wts = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device="cuda", dtype=torch.float16)

    full_map = torch.tensor([0, -1, 1, 2, -1, 3], device="cuda", dtype=torch.int64)
    wave1_map = torch.tensor([0, -1, 1, -1, -1, -1], device="cuda", dtype=torch.int64)
    wave2_map = torch.tensor([-1, -1, -1, 2, -1, 3], device="cuda", dtype=torch.int64)

    prepared = prepare_moe_input(x, weights)
    expected = moe_ffn(x, weights, ids, wts, expert_to_slot=full_map, prepared_input=prepared)
    acc = torch.empty_like(x)
    moe_ffn(
        x,
        weights,
        ids,
        wts,
        expert_to_slot=wave1_map,
        out=acc,
        accumulate=False,
        prepared_input=prepared,
    )
    moe_ffn(
        x,
        weights,
        ids,
        wts,
        expert_to_slot=wave2_map,
        out=acc,
        accumulate=True,
        prepared_input=prepared,
    )
    rtol = 1e-3 if weight_format != "convrot_int8" else 1e-2
    atol = 1e-3 if weight_format != "convrot_int8" else 1e-2
    torch.testing.assert_close(acc, expected, rtol=rtol, atol=atol)

    # PreparedMoEInput owns an independent snapshot even when wave 1 writes out=x.
    x_inout = x.clone()
    prepared_inout = prepare_moe_input(x_inout, weights)
    moe_ffn(
        x_inout,
        weights,
        ids,
        wts,
        expert_to_slot=wave1_map,
        out=x_inout,
        accumulate=False,
        prepared_input=prepared_inout,
    )
    moe_ffn(
        x_inout,
        weights,
        ids,
        wts,
        expert_to_slot=wave2_map,
        out=x_inout,
        accumulate=True,
        prepared_input=prepared_inout,
    )
    torch.testing.assert_close(x_inout, expected, rtol=rtol, atol=atol)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_launch_canary_nvfp4_and_int8() -> None:
    torch.manual_seed(400)
    ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64)
    wts = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.bfloat16)

    # Up-projection Split-K canary (H=16384, I=512, active_slots_hint=1).
    x_up = torch.randn((1, 16384), device="cuda", dtype=torch.bfloat16)
    w_nvfp4_up = _make_moe_weights(
        2, 16384, 512, weight_format="nvfp4_w4a16", dtype=torch.bfloat16, device="cuda", seed=401
    )
    ref_nvfp4_up = _reference_for_weights(x_up, w_nvfp4_up, ids, wts)
    nvfp4_first, nvfp4_launches = _record_triton_launches(
        lambda: moe_ffn(x_up, w_nvfp4_up, ids, wts, active_slots_hint=1)
    )
    nvfp4_second = moe_ffn(x_up, w_nvfp4_up, ids, wts, active_slots_hint=1)
    assert torch.equal(nvfp4_first, nvfp4_second)
    _assert_matches_reference(
        nvfp4_first, ref_nvfp4_up, weight_format="nvfp4_w4a16", dtype=torch.bfloat16
    )
    assert "nvfp4_moe_m1_gemv_kernel" in nvfp4_launches
    assert "moe_splitk_reduce_kernel" in nvfp4_launches
    assert "moe_m1_activate_rotate_kernel" in nvfp4_launches
    assert "moe_m1_weighted_reduce_kernel" in nvfp4_launches

    # Split-K post_scale must run after Split-K reduction so canceling K-halves yield 0, not NaN.
    w_sk_cancel = MoEWeights.from_quantized(
        torch.full_like(w_nvfp4_up.up_gate_qdata[:1], 0x77),
        torch.ones_like(w_nvfp4_up.up_gate_scale[:1].float()).to(torch.float8_e4m3fn),
        torch.full_like(w_nvfp4_up.down_qdata[:1], 0x11),
        torch.ones_like(w_nvfp4_up.down_scale[:1].float()).to(torch.float8_e4m3fn),
        logical_dtype=torch.float32,
        up_gate_global_scale=torch.full((1,), 100.0, device="cuda", dtype=torch.float32),
        down_global_scale=torch.ones((1,), device="cuda", dtype=torch.float32),
    )
    # Use an exact power of two so odd Split-K partitions cancel to 0 without FP32 ULP drift
    # (which SwiGLU would otherwise square to inf) while pre-reduce post_scale still overflows.
    x_cancel = torch.empty((1, 16384), device="cuda", dtype=torch.float32)
    x_cancel[:, :8192] = 2.0**110
    x_cancel[:, 8192:] = -(2.0**110)
    ids8 = torch.tensor([[0] + [99] * 7], device="cuda", dtype=torch.int64)
    wts8 = torch.ones((1, 8), device="cuda", dtype=torch.float32)
    res_hint1 = moe_ffn(x_cancel, w_sk_cancel, ids8, wts8, active_slots_hint=1)
    res_hint8 = moe_ffn(x_cancel, w_sk_cancel, ids8, wts8, active_slots_hint=8)
    assert torch.equal(res_hint1, torch.zeros_like(x_cancel))
    assert torch.equal(res_hint8, torch.zeros_like(x_cancel))

    if _INT8_MOE_GPU_AVAILABLE:
        w_int8_up = _make_moe_weights(
            2,
            16384,
            512,
            weight_format="convrot_int8",
            group_size=256,
            dtype=torch.bfloat16,
            device="cuda",
            seed=402,
        )
        ref_int8_up = _reference_for_weights(x_up, w_int8_up, ids, wts)
        int8_first, int8_launches = _record_triton_launches(
            lambda: moe_ffn(x_up, w_int8_up, ids, wts, active_slots_hint=1)
        )
        int8_second = moe_ffn(x_up, w_int8_up, ids, wts, active_slots_hint=1)
        assert torch.equal(int8_first, int8_second)
        _assert_matches_reference(
            int8_first, ref_int8_up, weight_format="convrot_int8", dtype=torch.bfloat16
        )
        assert "int8_moe_m1_matmul_kernel" in int8_launches
        assert "moe_splitk_reduce_kernel" in int8_launches
        assert "moe_m1_weighted_reduce_kernel" in int8_launches

    # Down-projection Split-K parity and repeatability (H=512, I=16384, active_slots_hint=1).
    x_down = torch.randn((1, 512), device="cuda", dtype=torch.bfloat16)
    down_formats: list[tuple[MoEFormat, int]] = [("nvfp4_w4a16", 0)]
    if _INT8_MOE_GPU_AVAILABLE:
        down_formats.append(("convrot_int8", 256))
    for fmt, gs in down_formats:
        w_down = _make_moe_weights(
            2,
            512,
            16384,
            weight_format=fmt,
            group_size=gs,
            dtype=torch.bfloat16,
            device="cuda",
            seed=405 + gs,
        )
        res_1 = moe_ffn(x_down, w_down, ids, wts, active_slots_hint=1)
        res_2 = moe_ffn(x_down, w_down, ids, wts, active_slots_hint=1)
        assert torch.equal(res_1, res_2)
        _assert_matches_reference(
            res_1,
            _reference_for_weights(x_down, w_down, ids, wts),
            weight_format=fmt,
            dtype=torch.bfloat16,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        ("convrot_trellis", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_faketensor_prepare_and_ffn(weight_format: MoEFormat, group_size: int) -> None:
    weights = _make_moe_weights(
        2,
        128,
        128,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=415 + group_size,
    )
    with forbid_triton_driver() as guard, FakeTensorMode(allow_non_fake_inputs=True):
        fx = torch.empty((1, 128), device="cuda", dtype=torch.bfloat16)
        fids = torch.empty((1, 2), device="cuda", dtype=torch.int64)
        fwts = torch.empty((1, 2), device="cuda", dtype=torch.bfloat16)
        fprep = prepare_moe_input(fx, weights)
        fout = moe_ffn(fx, weights, fids, fwts, prepared_input=fprep)
        assert fout.shape == (1, 128)
        assert fout.dtype is torch.bfloat16
        assert fout.device.type == "cuda"
        assert guard.call_count == 0


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        ("convrot_trellis", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_torch_compile_fullgraph(weight_format: MoEFormat, group_size: int) -> None:
    torch._dynamo.reset()
    torch.manual_seed(420 + group_size)
    weights = _make_moe_weights(
        3,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=421 + group_size,
    )
    x = torch.randn((1, 256), device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2]], device="cuda", dtype=torch.int64)
    wts = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.bfloat16)
    slot_map = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int64)

    def run_functional(inp: torch.Tensor, r_ids: torch.Tensor, r_wts: torch.Tensor) -> torch.Tensor:
        return moe_ffn(inp, weights, r_ids, r_wts, expert_to_slot=slot_map)

    def run_prepared(inp: torch.Tensor, r_ids: torch.Tensor, r_wts: torch.Tensor) -> torch.Tensor:
        prep = prepare_moe_input(inp, weights)
        return moe_ffn(inp, weights, r_ids, r_wts, expert_to_slot=slot_map, prepared_input=prep)

    def run_inplace(
        inp: torch.Tensor, r_ids: torch.Tensor, r_wts: torch.Tensor, out_buf: torch.Tensor
    ) -> torch.Tensor:
        return moe_ffn(
            inp, weights, r_ids, r_wts, expert_to_slot=slot_map, out=out_buf, accumulate=True
        )

    with torch.inference_mode():
        eager_fn = run_functional(x, ids, wts)
        compiled_fn = torch.compile(run_functional, fullgraph=True, options=compile_options())(
            x, ids, wts
        )
        assert torch.equal(compiled_fn, eager_fn)

        eager_prep = run_prepared(x, ids, wts)
        compiled_prep = torch.compile(run_prepared, fullgraph=True, options=compile_options())(
            x, ids, wts
        )
        torch.testing.assert_close(compiled_prep, eager_prep, rtol=1e-4, atol=1e-4)

        eager_out = torch.ones_like(x)
        compiled_out = eager_out.clone()
        run_inplace(x, ids, wts, eager_out)
        torch.compile(run_inplace, fullgraph=True, options=compile_options())(
            x, ids, wts, compiled_out
        )
        assert torch.equal(compiled_out, eager_out)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_edge_layout_aliasing_and_inference_guards() -> None:
    torch.manual_seed(435)
    edge_formats: list[tuple[MoEFormat, int]] = [("nvfp4_w4a16", 0)]
    if _INT8_MOE_GPU_AVAILABLE:
        edge_formats.append(("convrot_int8", 64))
    for weight_format, group_size in edge_formats:
        weights = _make_moe_weights(
            3,
            64,
            64,
            weight_format=weight_format,
            group_size=group_size,
            dtype=torch.bfloat16,
            device="cuda",
            seed=436 + group_size,
        )
        x_2d_t = torch.randn((64, 3), device="cuda", dtype=torch.bfloat16).t()
        ids_2d = torch.tensor([[0, 2], [1, 0], [2, 1]], device="cuda", dtype=torch.int64)
        wts_2d = torch.tensor(
            [[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], device="cuda", dtype=torch.bfloat16
        )
        out_2d = moe_ffn(x_2d_t, weights, ids_2d, wts_2d)
        assert out_2d.is_contiguous()
        _assert_matches_reference(
            out_2d,
            _reference_for_weights(x_2d_t, weights, ids_2d, wts_2d),
            weight_format=weight_format,  # type: ignore[arg-type]
            dtype=torch.bfloat16,
        )

        x_3d_t = torch.randn((2, 3, 64), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
        ids_3d = torch.tensor(
            [[[0, 1], [1, 2]], [[2, 0], [0, 2]], [[1, 0], [2, 1]]],
            device="cuda",
            dtype=torch.int64,
        )
        wts_3d = torch.full((3, 2, 2), 0.5, device="cuda", dtype=torch.bfloat16)
        out_3d = moe_ffn(x_3d_t, weights, ids_3d, wts_3d)
        assert out_3d.shape == (3, 2, 64)
        assert out_3d.is_contiguous()
        _assert_matches_reference(
            out_3d,
            _reference_for_weights(x_3d_t, weights, ids_3d, wts_3d),
            weight_format=weight_format,  # type: ignore[arg-type]
            dtype=torch.bfloat16,
        )

        # Shifted buf[:-1] / buf[1:] aliasing.
        buf = torch.randn((4, 64), device="cuda", dtype=torch.bfloat16)
        x_shift = buf[:-1]
        out_shift = buf[1:]
        expected_shift = _reference_for_weights(x_shift, weights, ids_2d, wts_2d)
        moe_ffn(x_shift, weights, ids_2d, wts_2d, out=out_shift)
        _assert_matches_reference(
            out_shift,
            expected_shift,
            weight_format=weight_format,  # type: ignore[arg-type]
            dtype=torch.bfloat16,
        )

        # Inference-tensor out outside inference_mode fails before any GPU write.
        with torch.inference_mode():
            inf_out = torch.full((2, 64), 5.0, device="cuda", dtype=torch.bfloat16)
        for empty_k in (False, True):
            r_ids = ids_2d[:2, :0] if empty_k else ids_2d[:2]
            r_wts = wts_2d[:2, :0] if empty_k else wts_2d[:2]
            with pytest.raises(RuntimeError, match="InferenceMode"):
                moe_ffn(buf[:2], weights, r_ids, r_wts, out=inf_out, accumulate=False)
            assert torch.equal(
                inf_out, torch.full((2, 64), 5.0, device="cuda", dtype=torch.bfloat16)
            )

        # Direct custom-op validation before GPU mutation and _neg_view rejection on out.
        bad_out = torch.full((1, 5), 8.0, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="out tensor"):
            torch.ops.piper_kernels.moe_ffn_(
                buf[:1],
                ids_2d[:1, :0],
                wts_2d[:1, :0],
                weights.up_gate_qdata,
                weights.up_gate_scale,
                weights.down_qdata,
                weights.down_scale,
                bad_out,
                weights.up_gate_global_scale,
                weights.down_global_scale,
                group_size=weights.group_size,
            )
        assert torch.equal(bad_out, torch.full((1, 5), 8.0, device="cuda", dtype=torch.bfloat16))

        neg_backing = torch.full((1, 64), 3.0, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="negated or conjugated"):
            moe_ffn(buf[:1], weights, ids_2d[:1], wts_2d[:1], out=torch._neg_view(neg_backing))
        assert torch.equal(
            neg_backing, torch.full((1, 64), 3.0, device="cuda", dtype=torch.bfloat16)
        )

        # Compiled aot_eager rejects out aliasing weight storage before overwriting it.
        if weight_format == "convrot_int8":
            orig_scale = weights.down_scale.clone()
            f32_w = MoEWeights.from_quantized(
                weights.up_gate_qdata,
                weights.up_gate_scale,
                weights.down_qdata,
                weights.down_scale,
                logical_dtype=torch.float32,
                group_size=weights.group_size,
            )
            f32_x = torch.randn((2, 64), device="cuda", dtype=torch.float32)
            f32_w_wts = torch.tensor([[0.6, 0.4], [0.5, 0.5]], device="cuda", dtype=torch.float32)

            @torch.compile(fullgraph=True, backend="aot_eager")
            def gpu_alias_out(
                inp: torch.Tensor,
                alias_w: MoEWeights = f32_w,
                alias_ids: torch.Tensor = ids_2d[:2],
                alias_wts: torch.Tensor = f32_w_wts,
            ) -> torch.Tensor:
                return moe_ffn(inp, alias_w, alias_ids, alias_wts, out=alias_w.down_scale[1:])

            with pytest.raises(Exception, match="alias weight storage"):
                gpu_alias_out(f32_x)
            assert torch.equal(weights.down_scale, orig_scale)


@pytest.mark.gpu
@pytest.mark.skipif(
    not _INT8_MOE_GPU_AVAILABLE,
    reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
)
def test_gpu_numerical_overflow_zero_scale_and_bf16_parity() -> None:
    torch.manual_seed(439)
    # 1. Wide FP16 ConvRot INT8 unfused preparation (H=16400, ±24000 -> rotated 96000).
    sign16 = torch.tensor(
        [1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1],
        device="cuda",
        dtype=torch.float16,
    )
    x_wide = (sign16.repeat(16400 // 16) * 24000.0).unsqueeze(0)
    up_w_wide, down_w_wide = _make_dense_weights(
        1, 16400, 16, dtype=torch.float16, device="cuda", seed=440
    )
    up_w_wide = up_w_wide * 1e-3
    w_wide = MoEWeights.from_hp(up_w_wide, down_w_wide, weight_format="convrot_int8", group_size=16)
    wide_ids = torch.zeros((1, 1), device="cuda", dtype=torch.int64)
    wide_wts = torch.ones((1, 1), device="cuda", dtype=torch.float16)
    prep_wide = prepare_moe_input(x_wide, w_wide)
    assert prep_wide.scale is not None
    assert torch.isfinite(prep_wide.scale).all()
    actual_wide = moe_ffn(x_wide, w_wide, wide_ids, wide_wts, prepared_input=prep_wide)
    expected_wide = _reference_for_weights(x_wide, w_wide, wide_ids, wide_wts)
    assert torch.isfinite(actual_wide).all()
    assert float(actual_wide.abs().max()) > 0.0
    _assert_matches_reference(
        actual_wide, expected_wide, weight_format="convrot_int8", dtype=torch.float16
    )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_nvfp4_scale_extremes_and_bf16_parity() -> None:
    torch.manual_seed(441)
    # 1. NVFP4 zero block scale, tiny 1e-30, zero, and extreme 1e37 global scales.
    w_shape = _make_moe_weights(
        1, 16, 16, weight_format="nvfp4_w4a16", dtype=torch.float32, device="cuda", seed=441
    )
    q_max = torch.full_like(w_shape.up_gate_qdata, 0x77)
    dq_max = torch.full_like(w_shape.down_qdata, 0x77)
    ones_f8 = torch.ones_like(w_shape.up_gate_scale.float()).to(torch.float8_e4m3fn)
    d_ones_f8 = torch.ones_like(w_shape.down_scale.float()).to(torch.float8_e4m3fn)
    x_huge = torch.full((1, 16), 1e37, device="cuda", dtype=torch.float32)
    h_ids = torch.zeros((1, 1), device="cuda", dtype=torch.int64)
    h_wts = torch.ones((1, 1), device="cuda", dtype=torch.float32)

    w_zero_blk = MoEWeights.from_quantized(
        q_max,
        torch.zeros_like(w_shape.up_gate_scale),
        dq_max,
        d_ones_f8,
        logical_dtype=torch.float32,
        up_gate_global_scale=torch.ones((1,), device="cuda", dtype=torch.float32),
        down_global_scale=torch.ones((1,), device="cuda", dtype=torch.float32),
    )
    assert torch.equal(moe_ffn(x_huge, w_zero_blk, h_ids, h_wts), torch.zeros_like(x_huge))

    w_tiny_gs = MoEWeights.from_quantized(
        q_max,
        ones_f8,
        dq_max,
        d_ones_f8,
        logical_dtype=torch.float32,
        up_gate_global_scale=torch.full((1,), 1e-30, device="cuda", dtype=torch.float32),
        down_global_scale=torch.full((1,), 1e-30, device="cuda", dtype=torch.float32),
    )
    actual_tiny = moe_ffn(x_huge, w_tiny_gs, h_ids, h_wts)
    expected_tiny = _reference_for_weights(x_huge, w_tiny_gs, h_ids, h_wts)
    assert torch.isfinite(actual_tiny).all()
    torch.testing.assert_close(actual_tiny, expected_tiny.to("cuda"), rtol=1e-4, atol=1e-4)

    w_zero_gs = MoEWeights.from_quantized(
        q_max,
        ones_f8,
        dq_max,
        d_ones_f8,
        logical_dtype=torch.float32,
        up_gate_global_scale=torch.zeros((1,), device="cuda", dtype=torch.float32),
        down_global_scale=torch.ones((1,), device="cuda", dtype=torch.float32),
    )
    assert torch.equal(moe_ffn(x_huge, w_zero_gs, h_ids, h_wts), torch.zeros_like(x_huge))

    # Case 3 extreme: max E4M3 block scale (448) and huge global scale (1e37) with tiny input.
    max_f8 = torch.full_like(w_shape.up_gate_scale.float(), 448.0).to(torch.float8_e4m3fn)
    d_max_f8 = torch.full_like(w_shape.down_scale.float(), 1.0).to(torch.float8_e4m3fn)
    w_huge_gs = MoEWeights.from_quantized(
        torch.full_like(w_shape.up_gate_qdata, 0x22),
        max_f8,
        dq_max,
        d_max_f8,
        logical_dtype=torch.bfloat16,
        up_gate_global_scale=torch.full((1,), 1e37, device="cuda", dtype=torch.float32),
        down_global_scale=torch.ones((1,), device="cuda", dtype=torch.float32),
    )
    x_tiny = torch.full((1, 16), 1e-37, device="cuda", dtype=torch.bfloat16)
    actual_huge_gs = moe_ffn(x_tiny, w_huge_gs, h_ids, h_wts.to(torch.bfloat16))
    expected_huge_gs = _reference_for_weights(x_tiny, w_huge_gs, h_ids, h_wts.to(torch.bfloat16))
    assert torch.isfinite(actual_huge_gs).all()
    torch.testing.assert_close(actual_huge_gs, expected_huge_gs.to("cuda"), rtol=2e-2, atol=1e-3)

    # Case 4 extreme: representable tiny NVFP4 weight (block 2^-9 * global 2^-141 * code 6).
    s_rep_f8 = torch.full_like(w_shape.up_gate_scale.float(), 2.0**-9).to(torch.float8_e4m3fn)
    w_rep_tiny = MoEWeights.from_quantized(
        q_max,
        s_rep_f8,
        dq_max,
        d_ones_f8,
        logical_dtype=torch.float32,
        up_gate_global_scale=torch.full((1,), 2.0**-141, device="cuda", dtype=torch.float32),
        down_global_scale=torch.ones((1,), device="cuda", dtype=torch.float32),
    )
    x_rep = torch.full((1, 16), 3e38, device="cuda", dtype=torch.float32)
    actual_rep = moe_ffn(x_rep, w_rep_tiny, h_ids, h_wts)
    expected_rep = _reference_for_weights(x_rep, w_rep_tiny, h_ids, h_wts).to("cuda")
    assert float(actual_rep.abs().min()) > 0.0
    torch.testing.assert_close(actual_rep, expected_rep, rtol=1e-4, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_convrot_nvfp4_rotation_range_and_bf16_parity() -> None:
    torch.manual_seed(442)
    h_ids = torch.zeros((1, 1), device="cuda", dtype=torch.int64)
    h_wts = torch.ones((1, 1), device="cuda", dtype=torch.float32)

    # ConvRot NVFP4 1e38 rotation stays finite across prepare_moe_input and activate_rotate.
    w_rot = _make_moe_weights(
        1,
        16,
        16,
        weight_format="convrot_nvfp4_w4a16",
        group_size=16,
        dtype=torch.float32,
        device="cuda",
        seed=442,
    )
    x_1e38 = torch.full((1, 16), 1e38, device="cuda", dtype=torch.float32)
    prep_1e38 = prepare_moe_input(x_1e38, w_rot)
    assert torch.isfinite(prep_1e38.data).all()
    torch.testing.assert_close(prep_1e38.data, x_1e38, rtol=1e-5, atol=1e2)

    gelu_shape = _make_moe_weights(
        1,
        16,
        16,
        weight_format="convrot_nvfp4_w4a16",
        group_size=16,
        activation_fn="gelu_tanh",
        dtype=torch.float32,
        device="cuda",
        seed=444,
    )
    w_act_1e38 = MoEWeights.from_quantized(
        torch.full_like(gelu_shape.up_gate_qdata, 0x11),
        torch.ones_like(gelu_shape.up_gate_scale.float()).to(torch.float8_e4m3fn),
        torch.full_like(gelu_shape.down_qdata, 0x11),
        torch.ones_like(gelu_shape.down_scale.float()).to(torch.float8_e4m3fn),
        logical_dtype=torch.float32,
        up_gate_global_scale=torch.ones((1,), device="cuda", dtype=torch.float32),
        down_global_scale=torch.full((1,), 1e-10, device="cuda", dtype=torch.float32),
        group_size=16,
        activation_fn="gelu_tanh",
    )
    x_act_1e38 = torch.full((1, 16), 1.25e37, device="cuda", dtype=torch.float32)
    out_rot = moe_ffn(x_act_1e38, w_act_1e38, h_ids, h_wts)
    assert torch.isfinite(out_rot).all()
    torch.testing.assert_close(
        out_rot,
        _reference_for_weights(x_act_1e38, w_act_1e38, h_ids, h_wts).to("cuda"),
        rtol=1e-4,
        atol=1e-4,
    )

    w_bf16 = _make_moe_weights(
        2,
        16,
        16,
        weight_format="convrot_nvfp4_w4a16",
        group_size=16,
        dtype=torch.bfloat16,
        device="cuda",
        seed=443,
    )
    x_round = torch.zeros((1, 16), device="cuda", dtype=torch.bfloat16)
    x_round[0, :4] = torch.tensor(
        [-515.0, -517.0, 257.0, -259.0], device="cuda", dtype=torch.bfloat16
    )
    prep_bf16 = prepare_moe_input(x_round, w_bf16)
    assert prep_bf16.data.dtype is torch.float32
    b_ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64)
    b_wts = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.bfloat16)
    assert torch.equal(
        moe_ffn(x_round, w_bf16, b_ids, b_wts, prepared_input=prep_bf16),
        moe_ffn(x_round, w_bf16, b_ids, b_wts),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
def test_gpu_sm75_int8_falls_back_cleanly_in_cuda_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(446)
    weights = _make_moe_weights(
        3, 128, 128, weight_format="convrot_int8", group_size=64, dtype=torch.float16, device="cuda"
    )
    x = torch.randn((1, 128), device="cuda", dtype=torch.float16)
    ids = torch.tensor([[0, 2]], device="cuda", dtype=torch.int64)
    wts = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.float16)
    slot_map = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int64)
    expected = _reference_for_weights(x, weights, ids, wts, expert_to_slot=slot_map).to("cuda")

    # Initialize Triton driver before stubbing torch.version.hip so isolated ROCm runs keep HIP.
    assert triton.runtime.driver.active is not None
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (7, 5))
    actual, launches = _record_triton_launches(
        lambda: moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map)
    )
    assert "int8_moe_m1_matmul_kernel" not in launches
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    out = torch.empty_like(x)
    moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map, out=out, accumulate=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map, out=out, accumulate=True)
    out.fill_(1.0)
    graph.replay()
    torch.testing.assert_close(out, expected + 1.0, rtol=1e-3, atol=1e-3)
    slot_map.fill_(-1)
    sentinel = torch.full_like(out, -0.0)
    out.copy_(sentinel)
    graph.replay()
    assert torch.signbit(out).all()


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_cuda_graph_replay_slot_updates(weight_format: MoEFormat, group_size: int) -> None:
    torch.manual_seed(450 + group_size)
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=451 + group_size,
    )
    x = torch.randn((1, 256), device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2]], device="cuda", dtype=torch.int64)
    wts = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.bfloat16)
    slot_map = torch.tensor([1, -1, 3, -1], device="cuda", dtype=torch.int64)
    out = torch.empty_like(x)

    # Warm up Triton compilation before graph capture.
    moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map, out=out, accumulate=True)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map, out=out, accumulate=True)

    # Replay 1: hit case
    init_1 = torch.full((1, 256), 0.5, device="cuda", dtype=torch.bfloat16)
    out.copy_(init_1)
    graph.replay()
    exp_1 = moe_ffn(
        x, weights, ids, wts, expert_to_slot=slot_map, out=init_1.clone(), accumulate=True
    )
    assert torch.equal(out, exp_1)

    # Replay 2: all-miss in-place update -> preserves out unchanged
    slot_map.fill_(-1)
    sentinel = torch.full((1, 256), -0.0, device="cuda", dtype=torch.bfloat16)
    out.copy_(sentinel)
    graph.replay()
    assert torch.signbit(out).all()

    # Replay 3: different hit routing & weights mutated in place
    slot_map.copy_(torch.tensor([2, 0, -1, 1], device="cuda", dtype=torch.int64))
    ids.copy_(torch.tensor([[3, 1]], device="cuda", dtype=torch.int64))
    wts.copy_(torch.tensor([[0.3, 0.7]], device="cuda", dtype=torch.bfloat16))
    init_3 = torch.full((1, 256), -1.0, device="cuda", dtype=torch.bfloat16)
    out.copy_(init_3)
    graph.replay()
    exp_3 = moe_ffn(
        x, weights, ids, wts, expert_to_slot=slot_map, out=init_3.clone(), accumulate=True
    )
    assert torch.equal(out, exp_3)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two GPUs")
def test_gpu_noncurrent_device_preserves_caller_device() -> None:
    torch.manual_seed(470)
    with torch.cuda.device(0):
        target_device = torch.device("cuda", 1)
        x = torch.randn((1, 256), device=target_device, dtype=torch.bfloat16)
        ids = torch.tensor([[1, 0]], device=target_device, dtype=torch.int64)
        wts = torch.tensor([[0.55, 0.45]], device=target_device, dtype=torch.bfloat16)
        for weight_format, group_size in (
            ("nvfp4_w4a16", 0),
            ("convrot_nvfp4_w4a16", 64),
            ("convrot_int8", 64),
        ):
            weights = _make_moe_weights(
                2,
                256,
                256,
                weight_format=weight_format,  # type: ignore[arg-type]
                group_size=group_size,
                dtype=torch.bfloat16,
                device=target_device,
                seed=471 + group_size,
            )
            prepared = prepare_moe_input(x, weights)
            assert torch.cuda.current_device() == 0
            assert prepared.data.device == target_device
            actual = moe_ffn(x, weights, ids, wts, prepared_input=prepared)
            assert torch.cuda.current_device() == 0
            assert actual.device == target_device
            expected = _reference_for_weights(x, weights, ids, wts)
            _assert_matches_reference(
                actual,
                expected,
                weight_format=weight_format,  # type: ignore[arg-type]
                dtype=torch.bfloat16,
            )


@pytest.mark.parametrize("target", _NVIDIA_TARGETS)
@pytest.mark.parametrize(
    ("weight_format", "group_size", "logical_dtype"),
    [
        ("nvfp4_w4a16", 0, torch.bfloat16),
        ("nvfp4_w4a16", 0, torch.float16),
        ("convrot_nvfp4_w4a16", 64, torch.bfloat16),
        ("convrot_nvfp4_w4a16", 64, torch.float16),
        ("convrot_int8", 256, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(("rows", "split_k", "accumulate"), [(4, 1, False), (8, 2, True)])
def test_offline_nvidia_small_batch_compile_zero_spills(
    target: GPUTarget,
    weight_format: MoEFormat,
    group_size: int,
    logical_dtype: torch.dtype,
    rows: int,
    split_k: int,
    accumulate: bool,
) -> None:
    compiled = _compile_nvidia_and_assert_clean(
        target,
        check_int8_mma=True,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        rows=rows,
        top_k=6,
        hidden_size=5120,
        intermediate_size=2304,
        group_size=group_size,
        split_k=split_k,
        accumulate=accumulate,
    )
    assert "bucket" in compiled
    expected_prefix = "int8_matmul_" if weight_format == "convrot_int8" else "nvfp4_gemv_"
    assert f"{expected_prefix}up" in compiled
    assert f"{expected_prefix}down" in compiled


@pytest.mark.parametrize("weight_format", ["nvfp4_w4a16", "convrot_nvfp4_w4a16"])
@pytest.mark.parametrize("logical_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("rows", "split_k", "accumulate"), [(4, 1, False), (8, 2, True)])
def test_offline_sm75_small_batch_compile_zero_spills(
    weight_format: MoEFormat,
    logical_dtype: torch.dtype,
    rows: int,
    split_k: int,
    accumulate: bool,
) -> None:
    sm75 = GPUTarget("cuda", 75, 32)
    compiled = _compile_nvidia_and_assert_clean(
        sm75,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        rows=rows,
        top_k=6,
        hidden_size=1024,
        intermediate_size=512,
        group_size=64 if weight_format == "convrot_nvfp4_w4a16" else 0,
        split_k=split_k,
        accumulate=accumulate,
    )
    assert "bucket" in compiled
    assert "nvfp4_gemv_up" in compiled
    assert "nvfp4_gemv_down" in compiled

    if rows == 4 and logical_dtype is torch.float16 and weight_format == "nvfp4_w4a16":
        for prefill_rows, prefill_top_k in ((17, 6), (16, 65)):
            routed = _compile_nvidia_and_assert_clean(sm75, rows=prefill_rows, top_k=prefill_top_k)
            assert "nvfp4_grouped" in routed
            assert "bucket_count" in routed
        with pytest.raises(ValueError, match="must be positive"):
            moe_triton.compile_moe_kernels(sm75, rows=0)
        with pytest.raises(ValueError, match="SM80"):
            moe_triton.compile_moe_kernels(
                sm75,
                rows=8,
                top_k=6,
                weight_format="convrot_int8",
                logical_dtype=torch.float16,
                group_size=64,
            )


@pytest.mark.skipif(sys.platform != "linux", reason="ROCm HIP compiler is Linux-only")
@pytest.mark.parametrize(("arch", "warp_size"), _AMD_TARGETS)
@pytest.mark.parametrize(
    ("weight_format", "group_size", "logical_dtype"),
    [
        ("nvfp4_w4a16", 0, torch.bfloat16),
        ("convrot_nvfp4_w4a16", 64, torch.float16),
        ("convrot_nvfp4_w4a16", 64, torch.bfloat16),
        ("convrot_int8", 256, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(("rows", "split_k", "accumulate"), [(4, 1, False), (8, 2, True)])
def test_offline_amd_small_batch_compile_matrix_and_fences(
    arch: str,
    warp_size: int,
    weight_format: MoEFormat,
    group_size: int,
    logical_dtype: torch.dtype,
    rows: int,
    split_k: int,
    accumulate: bool,
) -> None:
    target = GPUTarget("hip", arch, warp_size)
    compiled = moe_triton.compile_moe_kernels(
        target,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        rows=rows,
        top_k=6,
        hidden_size=5120,
        intermediate_size=2304,
        group_size=group_size,
        split_k=split_k,
        accumulate=accumulate,
    )
    assert "bucket" in compiled
    expected_prefix = "int8_matmul_" if weight_format == "convrot_int8" else "nvfp4_gemv_"
    assert f"{expected_prefix}up" in compiled
    assert f"{expected_prefix}down" in compiled
    expected_int8_isa = "v_mfma_i32" if arch == "gfx942" else "v_wmma_i32"
    for name, kernel in compiled.items():
        check_spills = not (arch == "gfx1201" and name.startswith("int8_matmul"))
        _assert_clean_amd_compile(
            kernel,
            arch,
            check_vgpr_spills=check_spills,
            check_lds_fences=name.startswith("int8_matmul"),
        )
        if name.startswith("int8_matmul"):
            assert expected_int8_isa in kernel.asm["amdgcn"]
    if arch == "gfx1201" and weight_format == "convrot_int8" and rows == 8 and split_k == 2:
        narrow = moe_triton.compile_moe_kernels(
            target,
            weight_format="convrot_int8",
            logical_dtype=torch.bfloat16,
            rows=8,
            top_k=6,
            hidden_size=5120,
            intermediate_size=1024,
            group_size=256,
            split_k=2,
        )
        for n_name, kernel in narrow.items():
            _assert_clean_amd_compile(
                kernel,
                arch,
                check_vgpr_spills=not n_name.startswith("int8_matmul"),
                check_lds_fences=n_name.startswith("int8_matmul"),
            )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_small_batch_matches_reference(weight_format: MoEFormat, group_size: int) -> None:
    torch.manual_seed(500 + group_size)
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=501 + group_size,
    )
    slot_map = torch.tensor([3, -1, 1, -1, 0, 2], device="cuda", dtype=torch.int64)
    for rows in (2, 4, 8, 16, 20):
        x = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, 6, (rows, 3), device="cuda", dtype=torch.int64)
        wts = torch.rand((rows, 3), device="cuda", dtype=torch.bfloat16) + 0.1
        actual = moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map)
        expected = _reference_for_weights(x, weights, ids, wts, expert_to_slot=slot_map)
        _assert_matches_reference(
            actual, expected, weight_format=weight_format, dtype=torch.bfloat16
        )

    # Multi-wave prepared input reuse + out=x aliasing across a small batch.
    x_wave = torch.randn((6, 256), device="cuda", dtype=torch.bfloat16)
    ids_wave = torch.tensor(
        [[0, 2, 4, 5], [1, 3, 2, 0], [4, 5, 1, 2], [0, 3, 4, 1], [5, 2, 0, 3], [2, 4, 5, 0]],
        device="cuda",
        dtype=torch.int64,
    )
    wts_wave = torch.full((6, 4), 0.25, device="cuda", dtype=torch.bfloat16)
    wave1 = torch.tensor([3, -1, 1, -1, -1, -1], device="cuda", dtype=torch.int64)
    wave2 = torch.tensor([-1, -1, -1, -1, 0, 2], device="cuda", dtype=torch.int64)
    prep = prepare_moe_input(x_wave, weights)
    expected_full = moe_ffn(
        x_wave, weights, ids_wave, wts_wave, expert_to_slot=slot_map, prepared_input=prep
    )
    out_wave = x_wave.clone()
    moe_ffn(
        out_wave,
        weights,
        ids_wave,
        wts_wave,
        expert_to_slot=wave1,
        out=out_wave,
        accumulate=False,
        prepared_input=prep,
    )
    moe_ffn(
        out_wave,
        weights,
        ids_wave,
        wts_wave,
        expert_to_slot=wave2,
        out=out_wave,
        accumulate=True,
        prepared_input=prep,
    )
    rtol = 1e-2 if weight_format != "convrot_int8" else 2e-2
    atol = 1e-2 if weight_format != "convrot_int8" else 2e-2
    torch.testing.assert_close(out_wave, expected_full, rtol=rtol, atol=atol)

    # FP16 and FP32 parity at M=4 (nodot) and M=8 (dot/nodot), plus activations & packing variants.
    for dtype in (torch.float16, torch.float32):
        w_dt = _make_moe_weights(
            4,
            256,
            256,
            weight_format=weight_format,
            group_size=group_size,
            dtype=dtype,
            device="cuda",
            seed=520 + group_size,
        )
        for rows in (4, 8):
            x_dt = torch.randn((rows, 256), device="cuda", dtype=dtype)
            ids_dt = torch.randint(0, 4, (rows, 3), device="cuda", dtype=torch.int64)
            wts_dt = torch.rand((rows, 3), device="cuda", dtype=dtype) + 0.1
            _assert_matches_reference(
                moe_ffn(x_dt, w_dt, ids_dt, wts_dt),
                _reference_for_weights(x_dt, w_dt, ids_dt, wts_dt),
                weight_format=weight_format,
                dtype=dtype,
            )

    for activation_fn, gate_first in (("swiglu", False), ("gelu_tanh", True)):
        w_act = _make_moe_weights(
            4,
            256,
            256,
            weight_format=weight_format,
            group_size=group_size,
            activation_fn=activation_fn,
            gate_first=gate_first,
            dtype=torch.bfloat16,
            device="cuda",
            seed=540 + int(gate_first),
        )
        x_act = torch.randn((8, 256), device="cuda", dtype=torch.bfloat16)
        ids_act = torch.randint(0, 4, (8, 3), device="cuda", dtype=torch.int64)
        wts_act = torch.rand((8, 3), device="cuda", dtype=torch.bfloat16) + 0.1
        _assert_matches_reference(
            moe_ffn(x_act, w_act, ids_act, wts_act),
            _reference_for_weights(x_act, w_act, ids_act, wts_act),
            weight_format=weight_format,
            dtype=torch.bfloat16,
        )

    if weight_format != "convrot_int8":
        hf_up, hf_down = _make_dense_weights(
            4, 256, 256, dtype=torch.bfloat16, device="cuda", seed=560
        )
        w_hf = MoEWeights.from_hp(
            hf_up,
            hf_down,
            weight_format=weight_format,
            group_size=group_size,
            high_first=True,
        )
        x_hf = torch.randn((8, 256), device="cuda", dtype=torch.bfloat16)
        ids_hf = torch.randint(0, 4, (8, 3), device="cuda", dtype=torch.int64)
        wts_hf = torch.rand((8, 3), device="cuda", dtype=torch.bfloat16) + 0.1
        _assert_matches_reference(
            moe_ffn(x_hf, w_hf, ids_hf, wts_hf),
            _reference_for_weights(x_hf, w_hf, ids_hf, wts_hf),
            weight_format=weight_format,
            dtype=torch.bfloat16,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_small_batch_edge_cases_and_launch_counts(
    weight_format: MoEFormat, group_size: int
) -> None:
    torch.manual_seed(600 + group_size)
    w = _make_moe_weights(
        3,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=610 + group_size,
    )
    # 1. Duplicate topk_ids + slot overflow (> 16 hits) + trailing 17th row.
    x_dup = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    ids_dup = torch.tensor(
        [[1, 1, 0, 1], [2, 2, 2, 2], [0, 1, 0, 2], [1, 0, 1, 0]],
        device="cuda",
        dtype=torch.int64,
    )
    wts_dup = torch.tensor(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.2, 0.2, 0.1],
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    _assert_matches_reference(
        moe_ffn(x_dup, w, ids_dup, wts_dup),
        _reference_for_weights(x_dup, w, ids_dup, wts_dup),
        weight_format=weight_format,
        dtype=torch.bfloat16,
    )

    x_ovf = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16)
    ids_ovf = torch.zeros((16, 2), device="cuda", dtype=torch.int64)
    wts_ovf = torch.full((16, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    _assert_matches_reference(
        moe_ffn(x_ovf, w, ids_ovf, wts_ovf),
        _reference_for_weights(x_ovf, w, ids_ovf, wts_ovf),
        weight_format=weight_format,
        dtype=torch.bfloat16,
    )

    x_17 = torch.randn((17, 256), device="cuda", dtype=torch.bfloat16)
    ids_17 = torch.randint(0, 3, (17, 2), device="cuda", dtype=torch.int64)
    wts_17 = torch.rand((17, 2), device="cuda", dtype=torch.bfloat16) + 0.1
    _assert_matches_reference(
        moe_ffn(x_17, w, ids_17, wts_17),
        _reference_for_weights(x_17, w, ids_17, wts_17),
        weight_format=weight_format,
        dtype=torch.bfloat16,
    )

    # 2. Mixed hit & all-miss rows preserving -0.0 and leaving padded out guard intact.
    backing = torch.full((8, 272), float("nan"), device="cuda", dtype=torch.bfloat16)
    out_view = backing[:, :256]
    out_view.fill_(-0.0)
    slot_map = torch.tensor([2, -1, 0, -1], device="cuda", dtype=torch.int64)
    ids_mix = torch.tensor(
        [
            [0, 2, 1],
            [1, 3, 99],
            [2, 1, 0],
            [-1, 1, 3],
            [3, 3, 1],
            [0, 1, 2],
            [10, -2, 1],
            [2, 2, 0],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    wts_mix = torch.rand((8, 3), device="cuda", dtype=torch.bfloat16) + 0.1
    x_mix = torch.randn((8, 256), device="cuda", dtype=torch.bfloat16)
    expected_mix = _reference_for_weights(
        x_mix, w, ids_mix, wts_mix, expert_to_slot=slot_map, out=out_view, accumulate=True
    )
    actual_mix = moe_ffn(
        x_mix, w, ids_mix, wts_mix, expert_to_slot=slot_map, out=out_view, accumulate=True
    )
    assert actual_mix is out_view
    assert torch.signbit(actual_mix[[1, 3, 4, 6]]).all()
    _assert_matches_reference(
        actual_mix[[0, 2, 5, 7]],
        expected_mix[[0, 2, 5, 7]],
        weight_format=weight_format,
        dtype=torch.bfloat16,
    )
    assert torch.isnan(backing[:, 256:]).all()

    # 3. Single-pass batched launch counts at M=4 and M=8.
    ids_cnt = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]], device="cuda", dtype=torch.int64)
    wts_cnt = torch.full((4, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    x_cnt = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
    for rows_cnt in (4, 8):
        ids_r = ids_cnt.repeat(rows_cnt // 4, 1)
        wts_r = wts_cnt.repeat(rows_cnt // 4, 1)
        x_r = x_cnt.repeat(rows_cnt // 4, 1)
        _, launches = _record_triton_launches(
            lambda x_r=x_r, ids_r=ids_r, wts_r=wts_r, w=w: moe_ffn(x_r, w, ids_r, wts_r)
        )
        assert launches.count("moe_bucket_tokens_kernel") == 1
        assert launches.count("moe_m1_weighted_reduce_kernel") == 1
        if weight_format == "convrot_int8":
            assert launches.count("int8_moe_batched_matmul_kernel") == 2
            assert "int8_moe_m1_matmul_kernel" not in launches
        else:
            assert launches.count("nvfp4_moe_batched_gemv_kernel") == 2
            assert "nvfp4_moe_m1_gemv_kernel" not in launches


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_small_batch_big_k_fallback_and_global_scales() -> None:
    torch.manual_seed(640)
    # rows=2, top_k=513 (> _BUCKET_MAX_PAIRS) routes cleanly through prefill bucketing.
    w_tiny = _make_moe_weights(
        2, 16, 16, weight_format="nvfp4_w4a16", dtype=torch.bfloat16, device="cuda", seed=640
    )
    x_bigk = torch.randn((2, 16), device="cuda", dtype=torch.bfloat16)
    ids_bigk = (
        torch.arange(513, device="cuda", dtype=torch.int64).remainder(2).unsqueeze(0).repeat(2, 1)
    )
    wts_bigk = torch.full((2, 513), 1.0 / 513.0, device="cuda", dtype=torch.bfloat16)
    _assert_matches_reference(
        moe_ffn(x_bigk, w_tiny, ids_bigk, wts_bigk),
        _reference_for_weights(x_bigk, w_tiny, ids_bigk, wts_bigk),
        weight_format="nvfp4_w4a16",
        dtype=torch.bfloat16,
    )

    # Extreme NVFP4 global scales (0.0, 1e-30, and FP16 1e-8) at M=4 and M=8.
    w_gs_base = _make_moe_weights(
        2, 64, 64, weight_format="nvfp4_w4a16", dtype=torch.float32, device="cuda", seed=650
    )
    q_max = torch.full_like(w_gs_base.up_gate_qdata, 0x77)
    dq_max = torch.full_like(w_gs_base.down_qdata, 0x77)
    ones_f8 = torch.ones_like(w_gs_base.up_gate_scale.float()).to(torch.float8_e4m3fn)
    d_ones_f8 = torch.ones_like(w_gs_base.down_scale.float()).to(torch.float8_e4m3fn)
    w_zero_tiny = MoEWeights.from_quantized(
        q_max,
        ones_f8,
        dq_max,
        d_ones_f8,
        logical_dtype=torch.float32,
        up_gate_global_scale=torch.tensor([0.0, 1e-30], device="cuda", dtype=torch.float32),
        down_global_scale=torch.tensor([1.0, 1e-30], device="cuda", dtype=torch.float32),
    )
    w_fp16_base = _make_moe_weights(
        2,
        64,
        64,
        weight_format="nvfp4_w4a16",
        activation_fn="gelu_tanh",
        dtype=torch.float16,
        device="cuda",
        seed=651,
    )
    w_fp16_tiny = MoEWeights.from_quantized(
        torch.full_like(w_fp16_base.up_gate_qdata, 0x77),
        torch.full_like(w_fp16_base.up_gate_scale.float(), 448.0).to(torch.float8_e4m3fn),
        torch.full_like(w_fp16_base.down_qdata, 0x77),
        torch.full_like(w_fp16_base.down_scale.float(), 448.0).to(torch.float8_e4m3fn),
        logical_dtype=torch.float16,
        up_gate_global_scale=torch.tensor([0.0, 1e-8], device="cuda", dtype=torch.float32),
        down_global_scale=torch.tensor([1.0, 1e4], device="cuda", dtype=torch.float32),
        activation_fn="gelu_tanh",
    )
    for rows_gs in (4, 8):
        ids_gs = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64).repeat(rows_gs, 1)
        wts_gs = torch.ones((rows_gs, 2), device="cuda", dtype=torch.float32)
        x_huge = torch.full((rows_gs, 64), 1e37, device="cuda", dtype=torch.float32)
        actual_32 = moe_ffn(x_huge, w_zero_tiny, ids_gs, wts_gs)
        expected_32 = _reference_for_weights(x_huge, w_zero_tiny, ids_gs, wts_gs).to("cuda")
        assert torch.isfinite(actual_32).all()
        assert float(actual_32.abs().min()) > 0.0
        torch.testing.assert_close(actual_32, expected_32, rtol=1e-4, atol=1e-4)

        x_fp16_sub = torch.full((rows_gs, 64), 2e-4, device="cuda", dtype=torch.float16)
        wts_16 = torch.ones((rows_gs, 2), device="cuda", dtype=torch.float16)
        actual_16 = moe_ffn(x_fp16_sub, w_fp16_tiny, ids_gs, wts_16)
        expected_16 = _reference_for_weights(x_fp16_sub, w_fp16_tiny, ids_gs, wts_16).to("cuda")
        assert torch.isfinite(actual_16).all()
        assert float(actual_16.abs().min()) > 0.0
        torch.testing.assert_close(actual_16, expected_16, rtol=2e-2, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
@pytest.mark.parametrize("rows", [4, 8])
def test_gpu_small_batch_cuda_graph_replay(
    weight_format: MoEFormat, group_size: int, rows: int
) -> None:
    torch.manual_seed(700 + group_size + rows)
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=710 + group_size,
    )
    x = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2]], device="cuda", dtype=torch.int64).repeat(rows, 1)
    wts = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.bfloat16).repeat(rows, 1)
    slot_map = torch.tensor([1, -1, 3, -1], device="cuda", dtype=torch.int64)
    out = torch.empty_like(x)
    moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map, out=out, accumulate=True)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map, out=out, accumulate=True)

    init_1 = torch.full((rows, 256), 0.5, device="cuda", dtype=torch.bfloat16)
    out.copy_(init_1)
    graph.replay()
    exp_1 = moe_ffn(
        x, weights, ids, wts, expert_to_slot=slot_map, out=init_1.clone(), accumulate=True
    )
    assert torch.equal(out, exp_1)

    slot_map.fill_(-1)
    out.fill_(-0.0)
    graph.replay()
    assert torch.signbit(out).all()

    slot_map.copy_(torch.tensor([2, 0, -1, 1], device="cuda", dtype=torch.int64))
    ids.copy_(torch.tensor([[3, 1]], device="cuda", dtype=torch.int64).repeat(rows, 1))
    wts.copy_(torch.tensor([[0.3, 0.7]], device="cuda", dtype=torch.bfloat16).repeat(rows, 1))
    init_3 = torch.full((rows, 256), -1.0, device="cuda", dtype=torch.bfloat16)
    out.copy_(init_3)
    graph.replay()
    exp_3 = moe_ffn(
        x, weights, ids, wts, expert_to_slot=slot_map, out=init_3.clone(), accumulate=True
    )
    assert torch.equal(out, exp_3)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size", "dtype"),
    [
        ("nvfp4_w4a16", 0, torch.bfloat16),
        ("nvfp4_w4a16", 0, torch.float16),
        ("convrot_nvfp4_w4a16", 256, torch.bfloat16),
        pytest.param(
            "convrot_int8",
            256,
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_small_batch_splitk_with_misses(
    weight_format: MoEFormat,
    group_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(770 + group_size)
    weights = _make_moe_weights(
        2,
        16384,
        512,
        weight_format=weight_format,
        group_size=group_size,
        dtype=dtype,
        device="cuda",
        seed=771 + group_size,
    )
    slot_map = torch.tensor([1, -1, 0], device="cuda", dtype=torch.int64)
    for rows in (4, 8):
        x = torch.randn((rows, 16384), device="cuda", dtype=dtype)
        ids = torch.tensor(
            [[0, 1], [1, 2], [2, 0], [1, 1]], device="cuda", dtype=torch.int64
        ).repeat(rows // 4, 1)
        wts = torch.tensor(
            [[0.6, 0.4], [0.3, 0.7], [0.5, 0.5], [0.5, 0.5]], device="cuda", dtype=dtype
        ).repeat(rows // 4, 1)
        actual, launches = _record_triton_launches(
            lambda x=x, ids=ids, wts=wts: moe_ffn(
                x, weights, ids, wts, expert_to_slot=slot_map, active_slots_hint=1
            )
        )
        assert "moe_splitk_reduce_kernel" in launches
        expected = _reference_for_weights(x, weights, ids, wts, expert_to_slot=slot_map)
        _assert_matches_reference(actual, expected, weight_format=weight_format, dtype=dtype)


@pytest.mark.gpu
@pytest.mark.skipif(
    not _INT8_MOE_GPU_AVAILABLE,
    reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
)
@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size", "group_size"), [(128, 80, 16), (256, 192, 64)]
)
@pytest.mark.parametrize(
    ("activation_fn", "gate_first"), [("swiglu", True), ("swiglu", False), ("gelu_tanh", True)]
)
def test_gpu_small_batch_int8_unaligned_tails(
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
    activation_fn: ActivationKind,
    gate_first: bool,
) -> None:
    torch.manual_seed(780 + intermediate_size + int(gate_first))
    weights = _make_moe_weights(
        3,
        hidden_size,
        intermediate_size,
        weight_format="convrot_int8",
        group_size=group_size,
        activation_fn=activation_fn,
        gate_first=gate_first,
        dtype=torch.bfloat16,
        device="cuda",
        seed=781 + intermediate_size,
    )
    for rows in (4, 8):
        x = torch.randn((rows, hidden_size), device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, 3, (rows, 2), device="cuda", dtype=torch.int64)
        wts = torch.rand((rows, 2), device="cuda", dtype=torch.bfloat16) + 0.1
        _assert_matches_reference(
            moe_ffn(x, weights, ids, wts),
            _reference_for_weights(x, weights, ids, wts),
            weight_format="convrot_int8",
            dtype=torch.bfloat16,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_small_batch_compile_dynamic(weight_format: MoEFormat, group_size: int) -> None:
    torch.manual_seed(720 + group_size)
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=730 + group_size,
    )
    slot_map = torch.tensor([2, -1, 0, 3], device="cuda", dtype=torch.int64)
    rtol = 1e-3 if weight_format != "convrot_int8" else 1e-2
    atol = 1e-3 if weight_format != "convrot_int8" else 1e-2

    def run_closure(inp: torch.Tensor, r_ids: torch.Tensor, r_wts: torch.Tensor) -> torch.Tensor:
        return moe_ffn(inp, weights, r_ids, r_wts, expert_to_slot=slot_map)

    def run_with_weights(
        inp: torch.Tensor,
        w: MoEWeights,
        r_ids: torch.Tensor,
        r_wts: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        return moe_ffn(inp, w, r_ids, r_wts, expert_to_slot=s)

    def run_extras(
        inp: torch.Tensor,
        r_ids: torch.Tensor,
        r_wts: torch.Tensor,
        out_buf: torch.Tensor,
        hint: int,
    ) -> torch.Tensor:
        prep = prepare_moe_input(inp, weights)
        return moe_ffn(
            inp,
            weights,
            r_ids,
            r_wts,
            expert_to_slot=slot_map,
            out=out_buf,
            accumulate=True,
            prepared_input=prep,
            active_slots_hint=hint,
        )

    torch._dynamo.reset()
    counter_closure = CompileCounter()
    compiled_count_closure = torch.compile(
        run_closure, fullgraph=True, dynamic=True, backend=counter_closure
    )
    counter_arg = CompileCounter()
    compiled_count_arg = torch.compile(
        run_with_weights, fullgraph=True, dynamic=True, backend=counter_arg
    )
    with torch.inference_mode():
        # Use M not equal to num_slots (4) so rows==num_slots cannot guard-specialize.
        for rows in (3, 7, 13, 16):
            x_cnt = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
            ids_cnt = torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int64)
            wts_cnt = torch.rand((rows, 2), device="cuda", dtype=torch.bfloat16) + 0.1
            compiled_count_closure(x_cnt, ids_cnt, wts_cnt)
            compiled_count_arg(x_cnt, weights, ids_cnt, wts_cnt, slot_map)
    assert counter_closure.frame_count == 1
    assert counter_arg.frame_count == 1

    torch._dynamo.reset()
    compiled_closure = torch.compile(
        run_closure, fullgraph=True, dynamic=True, options=compile_options()
    )
    compiled_arg = torch.compile(
        run_with_weights, fullgraph=True, dynamic=True, options=compile_options()
    )
    compiled_extras = torch.compile(
        run_extras, fullgraph=True, dynamic=True, options=compile_options()
    )
    with torch.inference_mode():
        for rows in (3, 7, 16, 1):
            x_dyn = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
            ids_dyn = torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int64)
            wts_dyn = torch.rand((rows, 2), device="cuda", dtype=torch.bfloat16) + 0.1
            eager = run_closure(x_dyn, ids_dyn, wts_dyn)
            torch.testing.assert_close(
                compiled_closure(x_dyn, ids_dyn, wts_dyn), eager, rtol=rtol, atol=atol
            )
            torch.testing.assert_close(
                compiled_arg(x_dyn, weights, ids_dyn, wts_dyn, slot_map),
                eager,
                rtol=rtol,
                atol=atol,
            )
            out_eager = torch.ones_like(x_dyn)
            out_comp = out_eager.clone()
            run_extras(x_dyn, ids_dyn, wts_dyn, out_eager, 2)
            compiled_extras(x_dyn, ids_dyn, wts_dyn, out_comp, 2)
            torch.testing.assert_close(out_comp, out_eager, rtol=rtol, atol=atol)

        x_bad = torch.randn((3, 256), device="cuda", dtype=torch.bfloat16)
        ids_bad = torch.randint(0, 4, (3, 2), device="cuda", dtype=torch.int64)
        wts_bad = torch.rand((3, 2), device="cuda", dtype=torch.bfloat16) + 0.1
        out_bad = torch.ones_like(x_bad)
        for bad_hint in (0, -1, True):
            with pytest.raises((TypeError, ValueError, torch._dynamo.exc.TorchDynamoException)):
                compiled_extras(x_bad, ids_bad, wts_bad, out_bad, bad_hint)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two GPUs")
def test_gpu_small_batch_noncurrent_device() -> None:
    torch.manual_seed(760)
    formats: list[tuple[MoEFormat, int]] = [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
    ]
    if _INT8_MOE_GPU_AVAILABLE:
        formats.append(("convrot_int8", 64))

    with torch.cuda.device(0):
        target_device = torch.device("cuda", 1)
        x_d1 = torch.randn((8, 256), device=target_device, dtype=torch.bfloat16)
        ids_d1 = torch.randint(0, 2, (8, 2), device=target_device, dtype=torch.int64)
        wts_d1 = torch.rand((8, 2), device=target_device, dtype=torch.bfloat16) + 0.1
        for weight_format, group_size in formats:
            w_d1 = _make_moe_weights(
                2,
                256,
                256,
                weight_format=weight_format,
                group_size=group_size,
                dtype=torch.bfloat16,
                device=target_device,
                seed=760 + group_size,
            )
            prepared = prepare_moe_input(x_d1, w_d1)
            assert torch.cuda.current_device() == 0
            assert prepared.data.device == target_device
            actual = moe_ffn(x_d1, w_d1, ids_d1, wts_d1, prepared_input=prepared)
            assert torch.cuda.current_device() == 0
            assert actual.device == target_device
            expected = _reference_for_weights(x_d1, w_d1, ids_d1, wts_d1)
            _assert_matches_reference(
                actual, expected, weight_format=weight_format, dtype=torch.bfloat16
            )


@pytest.mark.parametrize("target", _NVIDIA_TARGETS)
@pytest.mark.parametrize(
    ("weight_format", "group_size", "logical_dtype"),
    [
        ("nvfp4_w4a16", 0, torch.bfloat16),
        ("nvfp4_w4a16", 0, torch.float16),
        ("nvfp4_w4a16", 0, torch.float32),
        ("convrot_nvfp4_w4a16", 64, torch.bfloat16),
        ("convrot_nvfp4_w4a16", 64, torch.float16),
        ("convrot_int8", 256, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(("rows", "accumulate"), [(128, False), (256, True)])
def test_offline_nvidia_prefill_compile_zero_spills(
    target: GPUTarget,
    weight_format: MoEFormat,
    group_size: int,
    logical_dtype: torch.dtype,
    rows: int,
    accumulate: bool,
) -> None:
    compiled = _compile_nvidia_and_assert_clean(
        target,
        check_int8_mma=True,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        rows=rows,
        top_k=6,
        num_slots=64,
        hidden_size=5120,
        intermediate_size=2304,
        group_size=group_size,
        accumulate=accumulate,
    )
    for key in ("bucket_count", "bucket_scatter", "gather", "weighted_reduce"):
        assert key in compiled
    if weight_format == "convrot_int8":
        assert "int8_grouped_up" in compiled
        assert "int8_grouped_down" in compiled
    else:
        assert "nvfp4_grouped" in compiled
        assert "activate_rotate" in compiled


@pytest.mark.parametrize("weight_format", ["nvfp4_w4a16", "convrot_nvfp4_w4a16"])
@pytest.mark.parametrize("logical_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("rows", "accumulate"), [(128, False), (256, True)])
def test_offline_sm75_prefill_compile_zero_spills(
    weight_format: MoEFormat,
    logical_dtype: torch.dtype,
    rows: int,
    accumulate: bool,
) -> None:
    sm75 = GPUTarget("cuda", 75, 32)
    compiled = _compile_nvidia_and_assert_clean(
        sm75,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        rows=rows,
        top_k=6,
        num_slots=64,
        hidden_size=1024,
        intermediate_size=512,
        group_size=64 if weight_format == "convrot_nvfp4_w4a16" else 0,
        accumulate=accumulate,
    )
    assert "nvfp4_grouped" in compiled

    if rows == 128 and logical_dtype is torch.float16 and weight_format == "nvfp4_w4a16":
        wide_sm75 = _compile_nvidia_and_assert_clean(sm75, rows=20000, top_k=6)
        assert "nvfp4_grouped" in wide_sm75

        sm80 = _NVIDIA_TARGETS[0]
        wide_sm80 = moe_triton.compile_moe_kernels(
            sm80,
            weight_format="convrot_nvfp4_w4a16",
            logical_dtype=torch.bfloat16,
            rows=2048,
            top_k=6,
            num_slots=256,
            hidden_size=8192,
            intermediate_size=4096,
            group_size=64,
        )
        chunk_sm80 = moe_triton.compile_moe_kernels(
            sm80,
            weight_format="convrot_nvfp4_w4a16",
            logical_dtype=torch.bfloat16,
            rows=390,
            top_k=6,
            num_slots=256,
            hidden_size=8192,
            intermediate_size=4096,
            group_size=64,
        )
        assert wide_sm80["nvfp4_grouped"].hash == chunk_sm80["nvfp4_grouped"].hash
        assert wide_sm80["bucket_scatter"].hash == chunk_sm80["bucket_scatter"].hash

        with pytest.raises(ValueError, match="SM80"):
            moe_triton.compile_moe_kernels(
                sm75,
                rows=256,
                top_k=6,
                weight_format="convrot_int8",
                logical_dtype=torch.float16,
                group_size=64,
            )


@pytest.mark.skipif(sys.platform != "linux", reason="ROCm HIP compiler is Linux-only")
@pytest.mark.parametrize(("arch", "warp_size"), _AMD_TARGETS)
@pytest.mark.parametrize(
    ("weight_format", "group_size", "logical_dtype"),
    [
        ("nvfp4_w4a16", 0, torch.bfloat16),
        ("convrot_nvfp4_w4a16", 64, torch.float16),
        ("convrot_nvfp4_w4a16", 64, torch.bfloat16),
        ("convrot_int8", 256, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(("rows", "accumulate"), [(128, False), (256, True)])
def test_offline_amd_prefill_compile_matrix_and_fences(
    arch: str,
    warp_size: int,
    weight_format: MoEFormat,
    group_size: int,
    logical_dtype: torch.dtype,
    rows: int,
    accumulate: bool,
) -> None:
    target = GPUTarget("hip", arch, warp_size)
    compiled = moe_triton.compile_moe_kernels(
        target,
        weight_format=weight_format,
        logical_dtype=logical_dtype,
        rows=rows,
        top_k=6,
        num_slots=64,
        hidden_size=5120,
        intermediate_size=2304,
        group_size=group_size,
        accumulate=accumulate,
    )
    if weight_format == "convrot_int8":
        assert "int8_grouped_up" in compiled
        assert "int8_grouped_down" in compiled
    else:
        assert "nvfp4_grouped" in compiled
    expected_int8_isa = "v_mfma_i32" if arch == "gfx942" else "v_wmma_i32"
    for name, kernel in compiled.items():
        # Sparse gfx1201 INT8 (block_m=16, block_k=256) trades L1-cached private scratch for
        # ~10-12% faster weight-bandwidth-bound DRAM reads; dense prefill (block_m=64, block_k=64)
        # has strictly zero VGPR spills.
        sparse_gfx1201_int8 = arch == "gfx1201" and rows == 128 and name.startswith("int8_grouped")
        _assert_clean_amd_compile(
            kernel,
            arch,
            check_vgpr_spills=True,
            max_vgpr_spills=640 if sparse_gfx1201_int8 else 0,
            max_private_scratch_bytes=1536 if sparse_gfx1201_int8 else 0,
            check_lds_fences=name.startswith("int8_grouped"),
        )
        if name.startswith("int8_grouped"):
            assert expected_int8_isa in kernel.asm["amdgcn"]
    if arch == "gfx1201" and weight_format == "convrot_int8" and rows == 256:
        unaligned = moe_triton.compile_moe_kernels(
            target,
            weight_format="convrot_int8",
            logical_dtype=torch.bfloat16,
            rows=256,
            top_k=6,
            num_slots=64,
            hidden_size=144,
            intermediate_size=112,
            group_size=16,
        )
        for u_name, kernel in unaligned.items():
            _assert_clean_amd_compile(
                kernel,
                arch,
                check_vgpr_spills=True,
                check_lds_fences=u_name.startswith("int8_grouped"),
            )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_prefill_matches_reference(weight_format: MoEFormat, group_size: int) -> None:
    torch.manual_seed(800 + group_size)
    # 1. Dense regime (avg_m >= 20 > 16) across dtypes and activations.
    for activation_fn, gate_first in (("swiglu", True), ("swiglu", False), ("gelu_tanh", True)):
        for dtype in (torch.float16, torch.bfloat16):
            w_dense = _make_moe_weights(
                2,
                256,
                256,
                weight_format=weight_format,
                group_size=group_size,
                activation_fn=activation_fn,
                gate_first=gate_first,
                dtype=dtype,
                device="cuda",
                seed=810 + group_size + int(gate_first),
            )
            x = torch.randn((24, 256), device="cuda", dtype=dtype)
            ids = torch.randint(0, 2, (24, 2), device="cuda", dtype=torch.int64)
            wts = torch.rand((24, 2), device="cuda", dtype=dtype) + 0.1
            _assert_matches_reference(
                moe_ffn(x, w_dense, ids, wts),
                _reference_for_weights(x, w_dense, ids, wts),
                weight_format=weight_format,
                dtype=dtype,
            )

    # 2. Sparse regime (avg_m = 9 <= 16) across 8 slots.
    w_sparse = _make_moe_weights(
        8,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=820 + group_size,
    )
    x_sp = torch.randn((36, 256), device="cuda", dtype=torch.bfloat16)
    ids_sp = torch.randint(0, 8, (36, 2), device="cuda", dtype=torch.int64)
    wts_sp = torch.rand((36, 2), device="cuda", dtype=torch.bfloat16) + 0.1
    _assert_matches_reference(
        moe_ffn(x_sp, w_sparse, ids_sp, wts_sp),
        _reference_for_weights(x_sp, w_sparse, ids_sp, wts_sp),
        weight_format=weight_format,
        dtype=torch.bfloat16,
    )

    # 3-D [B, T, H] input + int32 routing + active_slots_hint switching sparse (9) -> dense (18).
    x_3d = x_sp.view(3, 12, 256)
    ids_32 = torch.randint(0, 10, (3, 12, 2), device="cuda", dtype=torch.int32)
    wts_3d = wts_sp.view(3, 12, 2)
    slot_map_32 = torch.tensor([3, -1, 0, 5, -1, 2, 7, 1, 4, 6], device="cuda", dtype=torch.int32)
    res_3d = moe_ffn(
        x_3d, w_sparse, ids_32, wts_3d, expert_to_slot=slot_map_32, active_slots_hint=4
    )
    assert res_3d.shape == (3, 12, 256)
    _assert_matches_reference(
        res_3d,
        _reference_for_weights(x_3d, w_sparse, ids_32, wts_3d, expert_to_slot=slot_map_32),
        weight_format=weight_format,
        dtype=torch.bfloat16,
    )

    # 3. FP32 logical dtype + high_first=True (NVFP4).
    w_f32 = _make_moe_weights(
        2,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.float32,
        device="cuda",
        seed=830 + group_size,
    )
    x_f32 = torch.randn((20, 256), device="cuda", dtype=torch.float32)
    ids_f32 = torch.randint(0, 2, (20, 2), device="cuda", dtype=torch.int64)
    wts_f32 = torch.rand((20, 2), device="cuda", dtype=torch.float32) + 0.1
    _assert_matches_reference(
        moe_ffn(x_f32, w_f32, ids_f32, wts_f32),
        _reference_for_weights(x_f32, w_f32, ids_f32, wts_f32),
        weight_format=weight_format,
        dtype=torch.float32,
    )
    if weight_format == "convrot_nvfp4_w4a16":
        hf_up, hf_dn = _make_dense_weights(
            2, 256, 256, dtype=torch.float16, device="cuda", seed=835
        )
        w_hf = MoEWeights.from_hp(
            hf_up, hf_dn, weight_format="convrot_nvfp4_w4a16", group_size=64, high_first=True
        )
        x_hf = torch.randn((24, 256), device="cuda", dtype=torch.float16)
        _assert_matches_reference(
            moe_ffn(x_hf, w_hf, ids_sp[:24] % 2, wts_sp[:24].half()),
            _reference_for_weights(x_hf, w_hf, ids_sp[:24] % 2, wts_sp[:24].half()),
            weight_format=weight_format,
            dtype=torch.float16,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_prefill_skew_tails_waves_and_launches(
    weight_format: MoEFormat, group_size: int
) -> None:
    torch.manual_seed(850 + group_size)
    w = _make_moe_weights(
        3,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=855 + group_size,
    )
    # 1. Single-slot heavy skew (> block_m hits on one slot) + duplicate topk_ids + launches.
    x_skew = torch.randn((72, 256), device="cuda", dtype=torch.bfloat16)
    ids_skew = torch.ones((72, 2), device="cuda", dtype=torch.int64)
    wts_skew = torch.rand((72, 2), device="cuda", dtype=torch.bfloat16) + 0.1
    actual_skew, launches = _record_triton_launches(
        lambda: moe_ffn(
            x_skew, w, ids_skew, wts_skew, expert_to_slot=torch.tensor([1, 2, 0], device="cuda")
        )
    )
    for k_name in (
        "moe_prefill_bucket_count_kernel",
        "moe_prefill_bucket_scatter_kernel",
        "moe_prefill_gather_kernel",
        "moe_m1_weighted_reduce_kernel",
    ):
        assert launches.count(k_name) == 1
    if weight_format == "convrot_int8":
        assert launches.count("int8_moe_grouped_gemm_kernel") == 2
        assert "int8_moe_batched_matmul_kernel" not in launches
    else:
        assert launches.count("nvfp4_moe_grouped_gemm_kernel") == 2
        assert "nvfp4_moe_batched_gemv_kernel" not in launches
    _assert_matches_reference(
        actual_skew,
        _reference_for_weights(
            x_skew, w, ids_skew, wts_skew, expert_to_slot=torch.tensor([1, 2, 0], device="cuda")
        ),
        weight_format=weight_format,
        dtype=torch.bfloat16,
    )

    # 2. Mixed hit / all-miss rows preserving -0.0 and padded NaN guard columns + two-wave out=x.
    backing = torch.full((24, 272), float("nan"), device="cuda", dtype=torch.bfloat16)
    out_view = backing[:, :256]
    out_view.fill_(-0.0)
    slot_map = torch.tensor([2, -1, 0, -1], device="cuda", dtype=torch.int64)
    ids_mix = torch.randint(0, 4, (24, 3), device="cuda", dtype=torch.int64)
    ids_mix[1::3] = torch.tensor([1, 3, 99], device="cuda", dtype=torch.int64)
    ids_mix[2::6] = torch.tensor([-1, 1, -2], device="cuda", dtype=torch.int64)
    wts_mix = torch.rand((24, 3), device="cuda", dtype=torch.bfloat16) + 0.1
    x_mix = torch.randn((24, 256), device="cuda", dtype=torch.bfloat16)
    expected_mix = _reference_for_weights(
        x_mix, w, ids_mix, wts_mix, expert_to_slot=slot_map, out=out_view, accumulate=True
    )
    actual_mix = moe_ffn(
        x_mix, w, ids_mix, wts_mix, expert_to_slot=slot_map, out=out_view, accumulate=True
    )
    assert actual_mix is out_view
    assert torch.isnan(backing[:, 256:]).all()
    all_miss_rows = [r for r in range(24) if r % 3 == 1 or r % 6 == 2]
    hit_rows = [r for r in range(24) if r not in all_miss_rows]
    assert torch.signbit(actual_mix[all_miss_rows]).all()
    _assert_matches_reference(
        actual_mix[hit_rows],
        expected_mix[hit_rows],
        weight_format=weight_format,
        dtype=torch.bfloat16,
    )

    x_wave = x_mix.clone()
    prep = prepare_moe_input(x_wave, w)
    slot_w0 = torch.tensor([0, -1, 1, -1], device="cuda", dtype=torch.int64)
    slot_w1 = torch.tensor([-1, 0, -1, 2], device="cuda", dtype=torch.int64)
    exp_w0 = _reference_for_weights(
        x_wave, w, ids_mix, wts_mix, expert_to_slot=slot_w0, out=x_wave, accumulate=True
    )
    exp_w1 = _reference_for_weights(
        x_wave,
        w,
        ids_mix,
        wts_mix,
        expert_to_slot=slot_w1,
        prepared_input=prep,
        out=exp_w0,
        accumulate=True,
    )
    moe_ffn(
        x_wave,
        w,
        ids_mix,
        wts_mix,
        expert_to_slot=slot_w0,
        out=x_wave,
        accumulate=True,
        prepared_input=prep,
    )
    assert (
        moe_ffn(
            x_wave,
            w,
            ids_mix,
            wts_mix,
            expert_to_slot=slot_w1,
            out=x_wave,
            accumulate=True,
            prepared_input=prep,
        )
        is x_wave
    )
    _assert_matches_reference(x_wave, exp_w1, weight_format=weight_format, dtype=torch.bfloat16)

    x_alias = x_mix.clone()
    exp_alias = _reference_for_weights(
        x_alias, w, ids_mix, wts_mix, expert_to_slot=slot_map, out=x_alias, accumulate=True
    )
    assert (
        moe_ffn(x_alias, w, ids_mix, wts_mix, expert_to_slot=slot_map, out=x_alias, accumulate=True)
        is x_alias
    )
    _assert_matches_reference(x_alias, exp_alias, weight_format=weight_format, dtype=torch.bfloat16)

    # 3. Unaligned N/K tails at M=32 (group_size=16).
    for h_tail, i_tail in ((80, 80), (144, 112)):
        w_tail = _make_moe_weights(
            2,
            h_tail,
            i_tail,
            weight_format=weight_format,
            group_size=16 if group_size > 0 else 0,
            gate_first=True,
            dtype=torch.bfloat16,
            device="cuda",
            seed=870 + h_tail,
        )
        x_t = torch.randn((32, h_tail), device="cuda", dtype=torch.bfloat16)
        ids_t = torch.randint(0, 2, (32, 2), device="cuda", dtype=torch.int64)
        wts_t = torch.rand((32, 2), device="cuda", dtype=torch.bfloat16) + 0.1
        _assert_matches_reference(
            moe_ffn(x_t, w_tail, ids_t, wts_t),
            _reference_for_weights(x_t, w_tail, ids_t, wts_t),
            weight_format=weight_format,
            dtype=torch.bfloat16,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_prefill_global_scales_and_multichunk() -> None:
    torch.manual_seed(880)
    # 1. Extreme NVFP4 global scales (0.0, 1e-30, and FP16 1e-8) in dense M=36 prefill.
    w_base = _make_moe_weights(
        2, 64, 64, weight_format="nvfp4_w4a16", dtype=torch.float32, device="cuda", seed=881
    )
    q_max = torch.full_like(w_base.up_gate_qdata, 0x77)
    dq_max = torch.full_like(w_base.down_qdata, 0x77)
    ones_f8 = torch.ones_like(w_base.up_gate_scale.float()).to(torch.float8_e4m3fn)
    d_ones_f8 = torch.ones_like(w_base.down_scale.float()).to(torch.float8_e4m3fn)
    w_zero_tiny = MoEWeights.from_quantized(
        q_max,
        ones_f8,
        dq_max,
        d_ones_f8,
        logical_dtype=torch.float32,
        up_gate_global_scale=torch.tensor([0.0, 1e-30], device="cuda", dtype=torch.float32),
        down_global_scale=torch.tensor([1.0, 1e-30], device="cuda", dtype=torch.float32),
    )
    w_fp16_base = _make_moe_weights(
        2,
        64,
        64,
        weight_format="nvfp4_w4a16",
        activation_fn="gelu_tanh",
        dtype=torch.float16,
        device="cuda",
        seed=882,
    )
    w_fp16_tiny = MoEWeights.from_quantized(
        torch.full_like(w_fp16_base.up_gate_qdata, 0x77),
        torch.full_like(w_fp16_base.up_gate_scale.float(), 448.0).to(torch.float8_e4m3fn),
        torch.full_like(w_fp16_base.down_qdata, 0x77),
        torch.full_like(w_fp16_base.down_scale.float(), 448.0).to(torch.float8_e4m3fn),
        logical_dtype=torch.float16,
        up_gate_global_scale=torch.tensor([0.0, 1e-8], device="cuda", dtype=torch.float32),
        down_global_scale=torch.tensor([1.0, 1e4], device="cuda", dtype=torch.float32),
        activation_fn="gelu_tanh",
    )
    ids_gs = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64).repeat(36, 1)
    wts_gs = torch.ones((36, 2), device="cuda", dtype=torch.float32)
    x_huge = torch.full((36, 64), 1e37, device="cuda", dtype=torch.float32)
    actual_32 = moe_ffn(x_huge, w_zero_tiny, ids_gs, wts_gs)
    expected_32 = _reference_for_weights(x_huge, w_zero_tiny, ids_gs, wts_gs).to("cuda")
    assert torch.isfinite(actual_32).all()
    assert float(actual_32.abs().min()) > 0.0
    torch.testing.assert_close(actual_32, expected_32, rtol=1e-4, atol=1e-4)

    x_fp16_sub = torch.full((36, 64), 2e-4, device="cuda", dtype=torch.float16)
    wts_16 = torch.ones((36, 2), device="cuda", dtype=torch.float16)
    actual_16 = moe_ffn(x_fp16_sub, w_fp16_tiny, ids_gs, wts_16)
    expected_16 = _reference_for_weights(x_fp16_sub, w_fp16_tiny, ids_gs, wts_16).to("cuda")
    assert torch.isfinite(actual_16).all()
    assert float(actual_16.abs().min()) > 0.0
    torch.testing.assert_close(actual_16, expected_16, rtol=2e-2, atol=1e-4)

    # 2. Natural multi-chunk prefill (M=300, top_k=64, H=2048, I=1024 exceeds 256 MiB workspace).
    multi_formats: list[tuple[MoEFormat, int]] = [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
    ]
    if _INT8_MOE_GPU_AVAILABLE:
        multi_formats.append(("convrot_int8", 256))
    for fmt, gs in multi_formats:
        w_mc = _make_moe_weights(
            2,
            2048,
            1024,
            weight_format=fmt,
            group_size=gs,
            dtype=torch.bfloat16,
            device="cuda",
            seed=890 + gs,
        )
        x_mc = torch.randn((300, 2048), device="cuda", dtype=torch.bfloat16)
        ids_mc = torch.randint(0, 2, (300, 64), device="cuda", dtype=torch.int64)
        ids_mc[186] = -1
        wts_mc = torch.full((300, 64), 1.0 / 64.0, device="cuda", dtype=torch.bfloat16)
        backing = torch.full((300, 2064), float("nan"), device="cuda", dtype=torch.bfloat16)
        out_mc = backing[:, :2048]
        out_mc.fill_(0.25)
        res_mc, launches = _record_triton_launches(
            lambda x_mc=x_mc, w_mc=w_mc, ids_mc=ids_mc, wts_mc=wts_mc, out_mc=out_mc: moe_ffn(
                x_mc, w_mc, ids_mc, wts_mc, out=out_mc, accumulate=True
            )
        )
        assert res_mc is out_mc
        assert torch.isnan(backing[:, 2048:]).all()
        assert torch.all(res_mc[186] == 0.25)
        assert launches.count("moe_prefill_bucket_count_kernel") >= 2
        assert launches.count("moe_m1_weighted_reduce_kernel") >= 2

        sample_rows = [0, 150, 185, 273, 299]
        sample_out = torch.full((len(sample_rows), 2048), 0.25, dtype=torch.bfloat16)
        expected_sample = _reference_for_weights(
            x_mc[sample_rows],
            w_mc,
            ids_mc[sample_rows],
            wts_mc[sample_rows],
            out=sample_out,
            accumulate=True,
        )
        _assert_matches_reference(
            res_mc[sample_rows], expected_sample, weight_format=fmt, dtype=torch.bfloat16
        )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_prefill_cuda_graph_replay(weight_format: MoEFormat, group_size: int) -> None:
    torch.manual_seed(900 + group_size)
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=905 + group_size,
    )
    x_buf = torch.randn((64, 256), device="cuda", dtype=torch.bfloat16)
    ids_buf = torch.randint(0, 4, (64, 2), device="cuda", dtype=torch.int64)
    wts_buf = torch.rand((64, 2), device="cuda", dtype=torch.bfloat16) + 0.1
    slot_map_buf = torch.tensor([1, 2, -1, 0], device="cuda", dtype=torch.int64)
    out_buf = torch.full((64, 256), 0.25, device="cuda", dtype=torch.bfloat16)

    for _ in range(2):
        moe_ffn(
            x_buf,
            weights,
            ids_buf,
            wts_buf,
            expert_to_slot=slot_map_buf,
            out=out_buf,
            accumulate=True,
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_ffn(
            x_buf,
            weights,
            ids_buf,
            wts_buf,
            expert_to_slot=slot_map_buf,
            out=out_buf,
            accumulate=True,
        )

    patterns = [
        (
            torch.randint(0, 4, (64, 2), device="cuda", dtype=torch.int64),
            torch.tensor([3, 0, -1, 1], device="cuda", dtype=torch.int64),
        ),
        (
            torch.zeros((64, 2), device="cuda", dtype=torch.int64),
            torch.tensor([2, -1, -1, -1], device="cuda", dtype=torch.int64),
        ),
        (
            torch.randint(0, 4, (64, 2), device="cuda", dtype=torch.int64),
            torch.full((4,), -1, device="cuda", dtype=torch.int64),
        ),
    ]
    for new_ids, new_slots in patterns:
        x_buf.normal_()
        ids_buf.copy_(new_ids)
        slot_map_buf.copy_(new_slots)
        out_buf.fill_(0.5)
        expected = _reference_for_weights(
            x_buf,
            weights,
            ids_buf,
            wts_buf,
            expert_to_slot=slot_map_buf,
            out=out_buf,
            accumulate=True,
        )
        graph.replay()
        _assert_matches_reference(
            out_buf, expected, weight_format=weight_format, dtype=torch.bfloat16
        )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
        pytest.param(
            "convrot_int8",
            64,
            marks=pytest.mark.skipif(
                not _INT8_MOE_GPU_AVAILABLE,
                reason="requires supported ConvRot INT8 GPU target (ROCm or NVIDIA SM80+)",
            ),
        ),
    ],
)
def test_gpu_prefill_compile_dynamic_all_regimes(weight_format: MoEFormat, group_size: int) -> None:
    torch.manual_seed(920 + group_size)
    weights = _make_moe_weights(
        4,
        256,
        256,
        weight_format=weight_format,
        group_size=group_size,
        dtype=torch.bfloat16,
        device="cuda",
        seed=925 + group_size,
    )
    slot_map = torch.tensor([2, -1, 0, 3], device="cuda", dtype=torch.int64)
    rtol = 1e-3 if weight_format != "convrot_int8" else 1e-2
    atol = 1e-3 if weight_format != "convrot_int8" else 1e-2

    def run_fn(
        inp: torch.Tensor,
        r_ids: torch.Tensor,
        r_wts: torch.Tensor,
        out_buf: torch.Tensor,
        hint: int,
    ) -> torch.Tensor:
        prep = prepare_moe_input(inp, weights)
        return moe_ffn(
            inp,
            weights,
            r_ids,
            r_wts,
            expert_to_slot=slot_map,
            out=out_buf,
            accumulate=True,
            prepared_input=prep,
            active_slots_hint=hint,
        )

    torch._dynamo.reset()
    counter = CompileCounter()
    counted = torch.compile(run_fn, fullgraph=True, dynamic=True, backend=counter)
    with torch.inference_mode():
        # Exclude rows=1 from frame_count == 1 (PyTorch Dynamo's 0/1 static specialization).
        for rows in (3, 8, 32, 128):
            x_c = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
            ids_c = torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int64)
            wts_c = torch.rand((rows, 2), device="cuda", dtype=torch.bfloat16) + 0.1
            out_c = torch.ones_like(x_c)
            counted(x_c, ids_c, wts_c, out_c, 2)
    assert counter.frame_count == 1

    torch._dynamo.reset()
    compiled = torch.compile(run_fn, fullgraph=True, dynamic=True, options=compile_options())
    with torch.inference_mode():
        for rows in (3, 8, 32, 128, 1):
            x_d = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
            ids_d = torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int64)
            wts_d = torch.rand((rows, 2), device="cuda", dtype=torch.bfloat16) + 0.1
            out_eager = torch.ones_like(x_d)
            out_comp = out_eager.clone()
            run_fn(x_d, ids_d, wts_d, out_eager, 2)
            compiled(x_d, ids_d, wts_d, out_comp, 2)
            torch.testing.assert_close(out_comp, out_eager, rtol=rtol, atol=atol)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two GPUs")
def test_gpu_prefill_noncurrent_device() -> None:
    torch.manual_seed(950)
    formats: list[tuple[MoEFormat, int]] = [
        ("nvfp4_w4a16", 0),
        ("convrot_nvfp4_w4a16", 64),
    ]
    if _INT8_MOE_GPU_AVAILABLE:
        formats.append(("convrot_int8", 64))

    with torch.cuda.device(0):
        target_device = torch.device("cuda", 1)
        x_d1 = torch.randn((32, 256), device=target_device, dtype=torch.bfloat16)
        ids_d1 = torch.randint(0, 2, (32, 2), device=target_device, dtype=torch.int64)
        wts_d1 = torch.rand((32, 2), device=target_device, dtype=torch.bfloat16) + 0.1
        for weight_format, group_size in formats:
            w_d1 = _make_moe_weights(
                2,
                256,
                256,
                weight_format=weight_format,
                group_size=group_size,
                dtype=torch.bfloat16,
                device=target_device,
                seed=950 + group_size,
            )
            prepared = prepare_moe_input(x_d1, w_d1)
            assert torch.cuda.current_device() == 0
            assert prepared.data.device == target_device
            actual = moe_ffn(x_d1, w_d1, ids_d1, wts_d1, prepared_input=prepared)
            assert torch.cuda.current_device() == 0
            assert actual.device == target_device
            expected = _reference_for_weights(x_d1, w_d1, ids_d1, wts_d1)
            _assert_matches_reference(
                actual, expected, weight_format=weight_format, dtype=torch.bfloat16
            )


_NVIDIA_TRELLIS_TARGETS = [GPUTarget("cuda", 75, 32), *_NVIDIA_TARGETS]
_TRELLIS_COMPILE_LAYOUTS = [
    pytest.param((False, False, 256, 512, 512), id="uniform-256"),
    pytest.param((True, False, 256, 512, 512), id="var2d-256"),
    pytest.param((True, True, 224, 896, 896), id="var1d-224"),
]
_TRELLIS_COMPILE_REGIMES = [
    pytest.param(("m1", 1, 2, 1, False), id="m1"),
    pytest.param(("batched", 4, 2, 2, True), id="batched"),
    pytest.param(("prefill", 256, 6, 1, True), id="prefill"),
]


@pytest.mark.parametrize("target", _NVIDIA_TRELLIS_TARGETS)
@pytest.mark.parametrize("layout_spec", _TRELLIS_COMPILE_LAYOUTS)
@pytest.mark.parametrize("regime_spec", _TRELLIS_COMPILE_REGIMES)
def test_offline_nvidia_trellis_compile_zero_spills(
    target: GPUTarget,
    layout_spec: tuple[bool, bool, int, int, int],
    regime_spec: tuple[str, int, int, int, bool],
) -> None:
    variable_rows, has_slot_offsets, block_size, hidden_size, inter_size = layout_spec
    regime, rows, top_k, split_k, accumulate = regime_spec
    compiled = _compile_nvidia_and_assert_clean(
        target,
        check_int8_mma=True,
        weight_format="convrot_trellis",
        logical_dtype=torch.bfloat16,
        rows=rows,
        top_k=top_k,
        num_slots=64,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        group_size=64,
        split_k=split_k,
        accumulate=accumulate,
        block_size=block_size,
        variable_rows=variable_rows,
        has_slot_offsets=has_slot_offsets,
    )
    if regime == "prefill" and target.arch >= 80:
        assert {"trellis_decode", "int8_grouped_up", "int8_grouped_down"}.issubset(compiled)
    else:
        assert {"trellis_gemv_up", "trellis_gemv_down"}.issubset(compiled)
        if regime == "m1":
            assert "trellis_reduce" not in compiled
        elif regime == "batched":
            assert {"bucket", "trellis_reduce"}.issubset(compiled)
        else:
            assert "bucket" in compiled


@pytest.mark.skipif(sys.platform != "linux", reason="ROCm HIP compiler is Linux-only")
@pytest.mark.parametrize(("arch", "warp_size"), _AMD_TARGETS)
@pytest.mark.parametrize("layout_spec", _TRELLIS_COMPILE_LAYOUTS)
@pytest.mark.parametrize("regime_spec", _TRELLIS_COMPILE_REGIMES)
def test_offline_amd_trellis_compile_matrix_and_fences(
    arch: str,
    warp_size: int,
    layout_spec: tuple[bool, bool, int, int, int],
    regime_spec: tuple[str, int, int, int, bool],
) -> None:
    variable_rows, has_slot_offsets, block_size, hidden_size, inter_size = layout_spec
    regime, rows, top_k, split_k, accumulate = regime_spec
    target = GPUTarget("hip", arch, warp_size)
    compiled = moe_triton.compile_moe_kernels(
        target,
        weight_format="convrot_trellis",
        logical_dtype=torch.bfloat16,
        rows=rows,
        top_k=top_k,
        num_slots=64,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        group_size=64,
        split_k=split_k,
        accumulate=accumulate,
        block_size=block_size,
        variable_rows=variable_rows,
        has_slot_offsets=has_slot_offsets,
    )
    if regime == "prefill":
        assert {"trellis_decode", "int8_grouped_up", "int8_grouped_down"}.issubset(compiled)
    else:
        assert {"trellis_gemv_up", "trellis_gemv_down"}.issubset(compiled)
        if regime == "m1":
            assert "trellis_reduce" not in compiled
        else:
            assert {"bucket", "trellis_reduce"}.issubset(compiled)
    expected_int8_isa = "v_mfma_i32" if arch == "gfx942" else "v_wmma_i32"
    for name, kernel in compiled.items():
        _assert_clean_amd_compile(
            kernel,
            arch,
            check_vgpr_spills=True,
            check_lds_fences=name.startswith("int8_grouped"),
        )
        if name.startswith("int8_grouped"):
            assert expected_int8_isa in kernel.asm["amdgcn"]


def test_offline_trellis_compile_validation_errors() -> None:
    sm80 = GPUTarget("cuda", 80, 32)
    with pytest.raises(ValueError, match="high_first"):
        moe_triton.compile_moe_kernels(
            sm80, weight_format="convrot_trellis", group_size=64, high_first=True
        )
    with pytest.raises(ValueError, match="variable_rows=True"):
        moe_triton.compile_moe_kernels(
            sm80,
            weight_format="convrot_trellis",
            group_size=64,
            variable_rows=False,
            has_slot_offsets=True,
        )
    with pytest.raises(ValueError, match="block_size"):
        moe_triton.compile_moe_kernels(
            sm80,
            weight_format="convrot_trellis",
            group_size=64,
            hidden_size=256,
            intermediate_size=256,
            block_size=24,
        )
    with pytest.raises(ValueError, match="divisible"):
        moe_triton.compile_moe_kernels(
            sm80,
            weight_format="convrot_trellis",
            group_size=64,
            hidden_size=320,
            intermediate_size=256,
            block_size=256,
        )
    with pytest.raises(ValueError, match="bytes_per_block"):
        moe_triton.compile_moe_kernels(
            sm80,
            weight_format="convrot_trellis",
            group_size=64,
            hidden_size=256,
            intermediate_size=256,
            block_size=256,
            variable_rows=True,
            bytes_per_block=143,
        )


def _assert_trellis_prefill_matches_oracle(
    actual: torch.Tensor,
    x: torch.Tensor,
    weights: MoEWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    expert_to_slot: torch.Tensor | None = None,
    out_before: torch.Tensor | None = None,
    accumulate: bool = False,
) -> None:
    assert actual.dtype is weights.logical_dtype
    assert torch.isfinite(actual).all()
    if _INT8_MOE_GPU_AVAILABLE:
        int8_oracle_weights = _decode_trellis_moe_weights_to_int8(weights)
        expected_w8a8 = _reference_for_weights(
            x,
            int8_oracle_weights,
            topk_ids,
            topk_weights,
            expert_to_slot=expert_to_slot,
            out=out_before,
            accumulate=accumulate,
        )
        _assert_matches_reference(
            actual, expected_w8a8, weight_format="convrot_int8", dtype=weights.logical_dtype
        )
        return
    expected_a16 = _reference_for_weights(
        x,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=expert_to_slot,
        out=out_before,
        accumulate=accumulate,
    )
    _assert_matches_reference(
        actual, expected_a16, weight_format="convrot_trellis", dtype=weights.logical_dtype
    )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize(
    ("up_layout", "down_layout"),
    [
        ("uniform", "uniform"),
        ("var2d", "var2d"),
        ("var1d", "var1d"),
        ("var1d", "var2d"),
    ],
)
def test_gpu_trellis_m1_and_batched_parity(
    rows: int,
    up_layout: TrellisLayoutKind,
    down_layout: TrellisLayoutKind,
) -> None:
    torch.manual_seed(960 + rows)
    # Cover asymmetric block sizes (256 vs 128), misses, duplicate experts, and out accumulation.
    weights = _make_trellis_moe_weights(
        4,
        256,
        128,
        group_size=64,
        activation_fn="swiglu",
        gate_first=(rows != 4),
        dtype=torch.bfloat16,
        device="cuda",
        layouts=(up_layout, down_layout),
        seed=961 + rows,
    )
    x = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, 6, (rows, 3), device="cuda", dtype=torch.int64)
    topk_ids[0, 0] = topk_ids[0, 1] = 2  # duplicate slot hit within row
    topk_weights = torch.rand((rows, 3), device="cuda", dtype=torch.bfloat16) + 0.1
    slot_map = torch.tensor([1, -1, 0, 3, -1, 2], device="cuda", dtype=torch.int64)

    out, launches = _record_triton_launches(
        lambda: moe_ffn(x, weights, topk_ids, topk_weights, expert_to_slot=slot_map)
    )
    assert any("trellis_moe_gemv_kernel" in name for name in launches)
    _assert_matches_reference(
        out,
        _reference_for_weights(x, weights, topk_ids, topk_weights, expert_to_slot=slot_map),
        weight_format="convrot_trellis",
        dtype=torch.bfloat16,
    )

    # Preserve caller-owned strided out on all-miss accumulate and verify two-wave prepared reuse.
    if up_layout == down_layout:
        all_miss = torch.full((rows, 3), -1, device="cuda", dtype=torch.int64)
        backing = torch.full((rows, 288), float("nan"), device="cuda", dtype=torch.bfloat16)
        out_view = backing[:, :256]
        out_view.fill_(2.5)
        moe_ffn(x, weights, all_miss, topk_weights, out=out_view, accumulate=True)
        assert torch.equal(out_view, torch.full_like(out_view, 2.5))
        assert torch.isnan(backing[:, 256:]).all()

        prep = prepare_moe_input(x, weights)
        wave1 = torch.tensor([1, -1, 0, -1, -1, -1], device="cuda", dtype=torch.int64)
        wave2 = torch.tensor([-1, -1, -1, 3, -1, 2], device="cuda", dtype=torch.int64)
        acc = torch.full_like(x, 0.5)
        expected_acc = _reference_for_weights(
            x, weights, topk_ids, topk_weights, expert_to_slot=slot_map, out=acc, accumulate=True
        )
        moe_ffn(
            x,
            weights,
            topk_ids,
            topk_weights,
            expert_to_slot=wave1,
            out=acc,
            accumulate=True,
            prepared_input=prep,
        )
        moe_ffn(
            x,
            weights,
            topk_ids,
            topk_weights,
            expert_to_slot=wave2,
            out=acc,
            accumulate=True,
            prepared_input=prep,
        )
        _assert_matches_reference(
            acc, expected_acc, weight_format="convrot_trellis", dtype=torch.bfloat16
        )

    # Extra coverage: non-pow2 T=224, GeLU, FP16/FP32, and wide Split-K > 1.
    if rows == 1 and up_layout == "var1d" and down_layout == "var1d":
        for dtype, act in ((torch.float16, "gelu_tanh"), (torch.float32, "swiglu")):
            w_224 = _make_trellis_moe_weights(
                3,
                448,
                224,
                group_size=16,
                activation_fn=act,
                dtype=dtype,
                device="cuda",
                layouts=("var1d", "var2d"),
                seed=965,
            )
            x_224 = torch.randn((1, 448), device="cuda", dtype=dtype)
            ids_224 = torch.tensor([[0, 2]], device="cuda", dtype=torch.int64)
            wts_224 = torch.tensor([[0.6, 0.4]], device="cuda", dtype=dtype)
            _assert_matches_reference(
                moe_ffn(x_224, w_224, ids_224, wts_224),
                _reference_for_weights(x_224, w_224, ids_224, wts_224),
                weight_format="convrot_trellis",
                dtype=dtype,
            )
    if rows == 4 and up_layout == "uniform" and down_layout == "uniform":
        # Narrow out_features + deep K triggers split_k_up > 1 and moe_splitk_reduce_kernel.
        w_splitk = _make_trellis_moe_weights(
            2,
            36864,
            64,
            group_size=64,
            activation_fn="gelu_tanh",
            dtype=torch.bfloat16,
            device="cuda",
            layouts=("var1d", "uniform"),
            seed=966,
        )
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        for splitk_rows in (1, 2):
            x_sk = torch.randn((splitk_rows, 36864), device="cuda", dtype=torch.bfloat16)
            ids_sk = torch.zeros((splitk_rows, 1), device="cuda", dtype=torch.int64)
            wts_sk = torch.full((splitk_rows, 1), 0.75, device="cuda", dtype=torch.bfloat16)
            out_sk, launches_sk = _record_triton_launches(
                lambda x_sk=x_sk, ids_sk=ids_sk, wts_sk=wts_sk: moe_ffn(
                    x_sk, w_splitk, ids_sk, wts_sk, active_slots_hint=1
                )
            )
            assert any("trellis_moe_gemv_kernel" in name for name in launches_sk)
            if sm_count >= 32:
                assert any("moe_splitk_reduce_kernel" in name for name in launches_sk)
            _assert_matches_reference(
                out_sk,
                _reference_for_weights(x_sk, w_splitk, ids_sk, wts_sk),
                weight_format="convrot_trellis",
                dtype=torch.bfloat16,
            )


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    ("up_layout", "down_layout"),
    [
        ("uniform", "uniform"),
        ("var2d", "var2d"),
        ("var1d", "var1d"),
        ("var1d", "var2d"),
    ],
)
def test_gpu_trellis_prefill_parity_and_multichunk(
    up_layout: TrellisLayoutKind,
    down_layout: TrellisLayoutKind,
) -> None:
    torch.manual_seed(970 + int(up_layout == "var1d"))
    rows = 32 if up_layout == down_layout else 64
    weights = _make_trellis_moe_weights(
        6,
        256,
        128,
        group_size=64,
        activation_fn="swiglu" if up_layout != "var2d" else "gelu_tanh",
        gate_first=(down_layout == "uniform"),
        dtype=torch.float16 if up_layout == "var2d" else torch.bfloat16,
        device="cuda",
        layouts=(up_layout, down_layout),
        seed=971,
    )
    dtype = weights.logical_dtype
    x = torch.randn((rows, 256), device="cuda", dtype=dtype)
    topk_ids = torch.randint(0, 8, (rows, 2), device="cuda", dtype=torch.int64)
    topk_weights = torch.rand((rows, 2), device="cuda", dtype=dtype) + 0.1
    # Map subsets of experts with unmapped slots (slots 4 and 5 stay inactive and get skipped).
    slot_map = torch.tensor([1, -1, 0, 3, -1, 2, -1, -1], device="cuda", dtype=torch.int64)

    actual = moe_ffn(x, weights, topk_ids, topk_weights, expert_to_slot=slot_map)
    _assert_trellis_prefill_matches_oracle(
        actual, x, weights, topk_ids, topk_weights, expert_to_slot=slot_map
    )

    # Prepared input reuse across prefill waves with accumulate=True.
    prep = prepare_moe_input(x, weights)
    wave1 = torch.tensor([1, -1, 0, -1, -1, -1, -1, -1], device="cuda", dtype=torch.int64)
    wave2 = torch.tensor([-1, -1, -1, 3, -1, 2, -1, -1], device="cuda", dtype=torch.int64)
    out_acc = torch.full_like(x, 0.75)
    oracle_acc = out_acc.clone()
    moe_ffn(
        x,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=wave1,
        out=out_acc,
        accumulate=True,
        prepared_input=prep,
    )
    moe_ffn(
        x,
        weights,
        topk_ids,
        topk_weights,
        expert_to_slot=wave2,
        out=out_acc,
        accumulate=True,
        prepared_input=prep,
    )
    if _INT8_MOE_GPU_AVAILABLE:
        int8_oracle = _decode_trellis_moe_weights_to_int8(weights)
        prep_i8 = prepare_moe_input(x, int8_oracle)
        moe_ffn(
            x,
            int8_oracle,
            topk_ids,
            topk_weights,
            expert_to_slot=wave1,
            out=oracle_acc,
            accumulate=True,
            prepared_input=prep_i8,
        )
        moe_ffn(
            x,
            int8_oracle,
            topk_ids,
            topk_weights,
            expert_to_slot=wave2,
            out=oracle_acc,
            accumulate=True,
            prepared_input=prep_i8,
        )
        torch.testing.assert_close(out_acc, oracle_acc, rtol=1e-4, atol=1e-4)
    else:
        _assert_matches_reference(
            out_acc,
            _reference_for_weights(
                x,
                weights,
                topk_ids,
                topk_weights,
                expert_to_slot=slot_map,
                out=oracle_acc,
                accumulate=True,
            ),
            weight_format="convrot_trellis",
            dtype=dtype,
        )

    if up_layout == "var1d" and down_layout == "var2d":
        # Non-pow2 T=224 in prefill regime.
        w_224 = _make_trellis_moe_weights(
            4,
            448,
            224,
            group_size=16,
            dtype=torch.bfloat16,
            device="cuda",
            layouts=("var1d", "var2d"),
            seed=974,
        )
        x_224 = torch.randn((32, 448), device="cuda", dtype=torch.bfloat16)
        ids_224 = torch.randint(0, 4, (32, 2), device="cuda", dtype=torch.int64)
        wts_224 = torch.rand((32, 2), device="cuda", dtype=torch.bfloat16) + 0.1
        _assert_trellis_prefill_matches_oracle(
            moe_ffn(x_224, w_224, ids_224, wts_224), x_224, w_224, ids_224, wts_224
        )

    if up_layout == "var1d" and down_layout == "var1d" and _INT8_MOE_GPU_AVAILABLE:
        # 3 slots x 5632 x 2048 > 32 MiB decoded scratch forces multiple decode + GEMM windows.
        w_multi = _make_trellis_moe_weights(
            3,
            2048,
            2816,
            group_size=64,
            dtype=torch.bfloat16,
            device="cuda",
            layouts=("var1d", "uniform"),
            seed=975,
        )
        x_m = torch.randn((24, 2048), device="cuda", dtype=torch.bfloat16)
        ids_m = torch.randint(0, 3, (24, 2), device="cuda", dtype=torch.int64)
        wts_m = torch.rand((24, 2), device="cuda", dtype=torch.bfloat16) + 0.1
        out_m, launches_m = _record_triton_launches(lambda: moe_ffn(x_m, w_multi, ids_m, wts_m))
        decode_launches = [name for name in launches_m if "decode_slots_int8_kernel" in name]
        grouped_launches = [name for name in launches_m if "int8_moe_grouped_gemm_kernel" in name]
        assert len(decode_launches) >= 3  # >=2 up windows + >=1 down window
        assert len(grouped_launches) >= 3
        _assert_trellis_prefill_matches_oracle(out_m, x_m, w_multi, ids_m, wts_m)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_trellis_prefill_edge_cases() -> None:
    torch.manual_seed(973)
    weights = _make_trellis_moe_weights(
        3,
        256,
        128,
        group_size=64,
        dtype=torch.float32,
        device="cuda",
        layouts=("var1d", "var2d"),
        seed=973,
    )
    x = torch.randn((24, 256), device="cuda", dtype=torch.float32)
    ids_i32 = torch.randint(0, 4, (24, 2), device="cuda", dtype=torch.int32)
    wts = torch.rand((24, 2), device="cuda", dtype=torch.float32) + 0.1
    map_i32 = torch.tensor([1, -1, 0, 2], device="cuda", dtype=torch.int32)

    direct = moe_ffn(x, weights, ids_i32, wts, expert_to_slot=map_i32)
    _assert_trellis_prefill_matches_oracle(direct, x, weights, ids_i32, wts, expert_to_slot=map_i32)
    prep = prepare_moe_input(x, weights)
    alias_out = x.clone()
    moe_ffn(
        alias_out,
        weights,
        ids_i32,
        wts,
        expert_to_slot=map_i32,
        out=alias_out,
        prepared_input=prep,
    )
    torch.testing.assert_close(alias_out, direct, rtol=1e-5, atol=1e-5)

    all_miss = torch.full((24, 2), -1, device="cuda", dtype=torch.int32)
    backing = torch.full((24, 288), float("nan"), device="cuda", dtype=torch.float32)
    out_view = backing[:, :256]
    out_view.fill_(1.75)
    moe_ffn(x, weights, all_miss, wts, out=out_view, accumulate=True)
    assert torch.equal(out_view, torch.full_like(out_view, 1.75))
    assert torch.isnan(backing[:, 256:]).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    not _INT8_MOE_GPU_AVAILABLE, reason="requires SM80+ CUDA or supported ROCm INT8 MoE"
)
@pytest.mark.parametrize(
    ("up_layout", "down_layout"),
    [("uniform", "uniform"), ("var1d", "var2d")],
)
def test_gpu_trellis_prefill_intra_expert_row_windowing(
    up_layout: TrellisLayoutKind,
    down_layout: TrellisLayoutKind,
) -> None:
    torch.manual_seed(976 + int(up_layout == "var1d"))
    # Single expert where up_gate=72 MiB and down=36 MiB both exceed the 32 MiB scratch cap.
    weights = _make_trellis_moe_weights(
        1,
        4608,
        8192,
        group_size=64,
        activation_fn="swiglu",
        gate_first=True,
        dtype=torch.bfloat16,
        device="cuda",
        layouts=(up_layout, down_layout),
        seed=977,
    )
    x = torch.randn((24, 4608), device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.zeros((24, 1), device="cuda", dtype=torch.int64)
    topk_weights = torch.full((24, 1), 0.75, device="cuda", dtype=torch.bfloat16)

    # Warm JIT/caches first, then measure peak workspace during windowed execution.
    _ = moe_ffn(x, weights, topk_ids, topk_weights)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual, launches = _record_triton_launches(lambda: moe_ffn(x, weights, topk_ids, topk_weights))
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline

    decode_launches = [name for name in launches if "decode_slots_int8_kernel" in name]
    grouped_launches = [name for name in launches if "int8_moe_grouped_gemm_kernel" in name]
    assert len(decode_launches) > 2
    assert len(grouped_launches) == len(decode_launches)
    assert peak_delta < 56 * 1024 * 1024
    _assert_trellis_prefill_matches_oracle(actual, x, weights, topk_ids, topk_weights)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("layout_kind", ["uniform", "var1d"])
def test_gpu_trellis_compile_dynamic_all_regimes(layout_kind: TrellisLayoutKind) -> None:
    torch.manual_seed(980 + int(layout_kind == "var1d"))
    weights = _make_trellis_moe_weights(
        4,
        256,
        256,
        group_size=64,
        dtype=torch.bfloat16,
        device="cuda",
        layouts=(layout_kind, layout_kind),
        seed=981,
    )
    slot_map = torch.tensor([2, -1, 0, 3], device="cuda", dtype=torch.int64)

    def run_fn(
        inp: torch.Tensor,
        r_ids: torch.Tensor,
        r_wts: torch.Tensor,
        out_buf: torch.Tensor,
    ) -> torch.Tensor:
        prep = prepare_moe_input(inp, weights)
        return moe_ffn(
            inp,
            weights,
            r_ids,
            r_wts,
            expert_to_slot=slot_map,
            out=out_buf,
            accumulate=True,
            prepared_input=prep,
        )

    torch._dynamo.reset()
    counter = CompileCounter()
    counted = torch.compile(run_fn, fullgraph=True, dynamic=True, backend=counter)
    with torch.inference_mode():
        # Start above top_k=2 so Dynamo duck-sizing does not specialize rows == top_k.
        for rows in (3, 8, 12, 32):
            x_c = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
            ids_c = torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int64)
            wts_c = torch.rand((rows, 2), device="cuda", dtype=torch.bfloat16) + 0.1
            out_c = torch.ones_like(x_c)
            counted(x_c, ids_c, wts_c, out_c)
    assert counter.frame_count == 1

    torch._dynamo.reset()
    compiled = torch.compile(run_fn, fullgraph=True, dynamic=True, options=compile_options())
    with torch.inference_mode():
        for rows in (3, 8, 12, 32, 1):
            x_d = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
            ids_d = torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int64)
            wts_d = torch.rand((rows, 2), device="cuda", dtype=torch.bfloat16) + 0.1
            out_eager = torch.ones_like(x_d)
            out_comp = out_eager.clone()
            run_fn(x_d, ids_d, wts_d, out_eager)
            compiled(x_d, ids_d, wts_d, out_comp)
            torch.testing.assert_close(out_comp, out_eager, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(1, id="m1"),
        pytest.param(4, id="batched"),
        pytest.param(32, id="prefill"),
    ],
)
@pytest.mark.parametrize("layout_kind", ["uniform", "var2d", "var1d"])
def test_gpu_trellis_cuda_graph_replay(
    rows: int,
    layout_kind: TrellisLayoutKind,
) -> None:
    torch.manual_seed(990 + rows + int(layout_kind == "var1d"))
    weights = _make_trellis_moe_weights(
        4,
        256,
        256,
        group_size=64,
        dtype=torch.bfloat16,
        device="cuda",
        layouts=(layout_kind, layout_kind),
        seed=991 + rows,
    )
    x_buf = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16)
    ids_buf = torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int64)
    wts_buf = torch.rand((rows, 2), device="cuda", dtype=torch.bfloat16) + 0.1
    slot_map_buf = torch.tensor([1, 2, -1, 0], device="cuda", dtype=torch.int64)
    out_buf = torch.full((rows, 256), 0.25, device="cuda", dtype=torch.bfloat16)

    for _ in range(2):
        moe_ffn(
            x_buf,
            weights,
            ids_buf,
            wts_buf,
            expert_to_slot=slot_map_buf,
            out=out_buf,
            accumulate=True,
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_ffn(
            x_buf,
            weights,
            ids_buf,
            wts_buf,
            expert_to_slot=slot_map_buf,
            out=out_buf,
            accumulate=True,
        )

    patterns = [
        (
            torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int64),
            torch.tensor([3, 0, -1, 1], device="cuda", dtype=torch.int64),
        ),
        (
            torch.zeros((rows, 2), device="cuda", dtype=torch.int64),
            torch.tensor([2, -1, -1, -1], device="cuda", dtype=torch.int64),
        ),
        (
            torch.randint(0, 4, (rows, 2), device="cuda", dtype=torch.int64),
            torch.full((4,), -1, device="cuda", dtype=torch.int64),
        ),
    ]
    for new_ids, new_slots in patterns:
        x_buf.normal_()
        ids_buf.copy_(new_ids)
        slot_map_buf.copy_(new_slots)
        out_buf.fill_(0.5)
        expected_eager = moe_ffn(
            x_buf,
            weights,
            ids_buf,
            wts_buf,
            expert_to_slot=slot_map_buf,
            out=out_buf.clone(),
            accumulate=True,
        )
        graph.replay()
        torch.testing.assert_close(out_buf, expected_eager, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
@pytest.mark.parametrize("layout_kind", ["uniform", "var2d", "var1d"])
def test_gpu_trellis_reference_cuda_graph_replay(layout_kind: TrellisLayoutKind) -> None:
    torch.manual_seed(995 + int(layout_kind == "var1d"))
    weights = _make_trellis_moe_weights(
        4,
        256,
        256,
        group_size=64,
        dtype=torch.bfloat16,
        device="cuda",
        layouts=(layout_kind, layout_kind),
        seed=996,
    )
    x_buf = torch.randn((2, 256), device="cuda", dtype=torch.bfloat16)
    ids_buf = torch.randint(0, 4, (2, 2), device="cuda", dtype=torch.int64)
    wts_buf = torch.rand((2, 2), device="cuda", dtype=torch.bfloat16) + 0.1
    slot_map_buf = torch.tensor([1, 2, -1, 0], device="cuda", dtype=torch.int64)
    out_buf = torch.full((2, 256), 0.25, device="cuda", dtype=torch.bfloat16)

    def _run_ref(out_target: torch.Tensor) -> torch.Tensor:
        return moe_reference.moe_ffn_reference(
            x_buf,
            ids_buf,
            wts_buf,
            weights.up_gate_qdata,
            weights.up_gate_scale,
            weights.down_qdata,
            weights.down_scale,
            expert_to_slot=slot_map_buf,
            out=out_target,
            accumulate=True,
            group_size=weights.group_size,
            activation_fn=weights.activation_fn,
            gate_first=weights.gate_first,
            up_gate_block_scales=weights.up_gate_block_scales,
            down_block_scales=weights.down_block_scales,
            up_gate_row_widths=weights.up_gate_row_widths,
            down_row_widths=weights.down_row_widths,
            up_gate_slot_offsets=weights.up_gate_slot_offsets,
            down_slot_offsets=weights.down_slot_offsets,
            up_gate_block_size=weights.up_gate_block_size,
            down_block_size=weights.down_block_size,
        )

    for _ in range(2):
        _run_ref(out_buf)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_ref(out_buf)

    patterns = [
        (
            torch.randint(0, 4, (2, 2), device="cuda", dtype=torch.int64),
            torch.tensor([3, 0, -1, 1], device="cuda", dtype=torch.int64),
        ),
        (
            torch.zeros((2, 2), device="cuda", dtype=torch.int64),
            torch.tensor([2, -1, -1, -1], device="cuda", dtype=torch.int64),
        ),
        (
            torch.randint(0, 4, (2, 2), device="cuda", dtype=torch.int64),
            torch.full((4,), -1, device="cuda", dtype=torch.int64),
        ),
    ]
    for new_ids, new_slots in patterns:
        x_buf.normal_()
        ids_buf.copy_(new_ids)
        slot_map_buf.copy_(new_slots)
        out_buf.fill_(0.5)
        expected_eager = _run_ref(out_buf.clone())
        graph.replay()
        torch.testing.assert_close(out_buf, expected_eager, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("target", "weight_format", "group_size"),
    [
        (target, weight_format, group_size)
        for target in _NVIDIA_TRELLIS_TARGETS
        for weight_format, group_size in (
            ("nvfp4_w4a16", 0),
            ("convrot_nvfp4_w4a16", 64),
            ("convrot_trellis", 64),
            ("convrot_int8", 64),
        )
        if not (target.arch < 80 and weight_format == "convrot_int8")
    ],
)
@pytest.mark.parametrize(("rows", "split_k"), [(1, 2), (256, 1)])
def test_offline_nvidia_gelu_tanh_compile_zero_spills(
    target: GPUTarget,
    weight_format: MoEFormat,
    group_size: int,
    rows: int,
    split_k: int,
) -> None:
    _compile_nvidia_and_assert_clean(
        target,
        check_int8_mma=True,
        weight_format=weight_format,
        logical_dtype=torch.bfloat16,
        rows=rows,
        top_k=4,
        hidden_size=512,
        intermediate_size=512,
        group_size=group_size,
        activation_fn="gelu_tanh",
        split_k=split_k,
    )


@pytest.mark.parametrize("target", _NVIDIA_TARGETS)
@pytest.mark.parametrize(("rows", "split_k"), [(1, 2), (4, 1), (256, 1)])
def test_offline_nvidia_int8_unaligned_nk_compile_zero_spills(
    target: GPUTarget,
    rows: int,
    split_k: int,
) -> None:
    _compile_nvidia_and_assert_clean(
        target,
        check_int8_mma=True,
        weight_format="convrot_int8",
        logical_dtype=torch.bfloat16,
        rows=rows,
        top_k=4,
        num_slots=64,
        hidden_size=144,
        intermediate_size=112,
        group_size=16,
        split_k=split_k,
    )


@pytest.mark.parametrize(
    ("target", "weight_format", "group_size"),
    [
        (target, weight_format, group_size)
        for target in _NVIDIA_TRELLIS_TARGETS
        for weight_format, group_size in (
            ("nvfp4_w4a16", 0),
            ("convrot_trellis", 64),
            ("convrot_int8", 64),
        )
        if not (target.arch < 80 and weight_format == "convrot_int8")
    ],
)
def test_offline_nvidia_batched_block_p_1024_compile_zero_spills(
    target: GPUTarget,
    weight_format: MoEFormat,
    group_size: int,
) -> None:
    compiled = _compile_nvidia_and_assert_clean(
        target,
        check_int8_mma=True,
        weight_format=weight_format,
        logical_dtype=torch.bfloat16,
        rows=16,
        top_k=64,
        num_slots=64,
        hidden_size=256,
        intermediate_size=256,
        group_size=group_size,
    )
    assert "bucket" in compiled


@pytest.mark.parametrize(
    ("weight_format", "group_size"),
    [("nvfp4_w4a16", 0), ("convrot_trellis", 64), ("convrot_int8", 64)],
)
@pytest.mark.parametrize(("rows", "split_k"), [(1, 2), (4, 1), (256, 1)])
def test_offline_optional_null_pointers_compile_zero_spills(
    weight_format: MoEFormat,
    group_size: int,
    rows: int,
    split_k: int,
) -> None:
    for target in _NVIDIA_TRELLIS_TARGETS:
        if target.arch < 80 and weight_format == "convrot_int8":
            continue
        # On 48-warp/SM SM86/SM89, ptxas trades 8 stack bytes (REG:72, STACK:8) for
        # higher CTA occupancy when use_dot=False and has_global_scale=False.
        spill_limits = (
            {"nvfp4_gemv_up": 8}
            if (weight_format == "nvfp4_w4a16" and rows == 4 and target.arch in (86, 89))
            else None
        )
        _compile_nvidia_and_assert_clean(
            target,
            check_int8_mma=True,
            max_spill_stores=spill_limits,
            weight_format=weight_format,
            logical_dtype=torch.bfloat16,
            rows=rows,
            top_k=4,
            hidden_size=512,
            intermediate_size=512,
            group_size=group_size,
            has_slot_map=False,
            has_global_scale=False,
            split_k=split_k,
        )
    if sys.platform == "linux":
        amd_target = GPUTarget("hip", "gfx1201", 32)
        compiled = moe_triton.compile_moe_kernels(
            amd_target,
            weight_format=weight_format,
            logical_dtype=torch.bfloat16,
            rows=rows,
            top_k=4,
            hidden_size=512,
            intermediate_size=512,
            group_size=group_size,
            has_slot_map=False,
            has_global_scale=False,
            split_k=split_k,
        )
        for name, kernel in compiled.items():
            check_spills = not name.startswith(("int8_matmul", "int8_grouped"))
            _assert_clean_amd_compile(kernel, "gfx1201", check_vgpr_spills=check_spills)


@pytest.mark.gpu
@pytest.mark.skipif(not w4a16_nvfp4_available(), reason="requires NVIDIA SM75+ or supported ROCm")
def test_gpu_m1_top_k_exceeds_cuda_grid_y_limit() -> None:
    torch.manual_seed(1001)
    formats: list[tuple[MoEFormat, int, int, str]] = [
        ("nvfp4_w4a16", 0, 64, "nvfp4_moe_m1_gemv_kernel"),
        ("convrot_trellis", 64, 256, "trellis_moe_gemv_kernel"),
    ]
    if _INT8_MOE_GPU_AVAILABLE:
        formats.append(("convrot_int8", 64, 64, "int8_moe_m1_matmul_kernel"))

    top_k = 70_000
    hit_cols = torch.tensor([0, 35_000, 69_999], device="cuda", dtype=torch.int64)
    ids = torch.full((1, top_k), -1, device="cuda", dtype=torch.int64)
    ids[0, hit_cols] = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int64)
    wts = torch.zeros((1, top_k), device="cuda", dtype=torch.bfloat16)
    wts[0, hit_cols] = torch.tensor([0.3, 0.5, 0.2], device="cuda", dtype=torch.bfloat16)

    for weight_format, group_size, width, expected_kernel in formats:
        weights = _make_moe_weights(
            2,
            width,
            width,
            weight_format=weight_format,
            group_size=group_size,
            dtype=torch.bfloat16,
            device="cuda",
            seed=1002 + width + group_size,
        )
        x = torch.randn((1, width), device="cuda", dtype=torch.bfloat16)
        actual, launches = _record_triton_launches(
            lambda x=x, weights=weights: moe_ffn(x, weights, ids, wts, active_slots_hint=2)
        )
        assert expected_kernel in launches
        assert "moe_m1_weighted_reduce_kernel" in launches
        if weight_format != "convrot_int8":
            assert "moe_m1_activate_rotate_kernel" in launches
        expected = _independent_moe_oracle(x, weights, ids, wts)
        _assert_matches_reference(
            actual, expected, weight_format=weight_format, dtype=torch.bfloat16
        )

    # rows=2, top_k=70_000 triggers _select_moe_execution_regime prefill-to-M1 fallback.
    w_nvfp4 = _make_moe_weights(
        2, 64, 64, weight_format="nvfp4_w4a16", dtype=torch.bfloat16, device="cuda", seed=1005
    )
    x_two = torch.randn((2, 64), device="cuda", dtype=torch.bfloat16)
    ids_two = ids.repeat(2, 1)
    wts_two = wts.repeat(2, 1)
    actual_two, launches_two = _record_triton_launches(
        lambda: moe_ffn(x_two, w_nvfp4, ids_two, wts_two, active_slots_hint=2)
    )
    assert "nvfp4_moe_m1_gemv_kernel" in launches_two
    assert "moe_m1_weighted_reduce_kernel" in launches_two
    _assert_matches_reference(
        actual_two,
        _independent_moe_oracle(x_two, w_nvfp4, ids_two, wts_two),
        weight_format="nvfp4_w4a16",
        dtype=torch.bfloat16,
    )

    # Wide H triggers Split-K on up_gate and exercises flattened moe_splitk_reduce_kernel.
    if torch.cuda.get_device_properties(0).multi_processor_count >= 32:
        w_splitk = _make_moe_weights(
            2,
            4096,
            512,
            weight_format="nvfp4_w4a16",
            dtype=torch.bfloat16,
            device="cuda",
            seed=1006,
        )
        x_splitk = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
        actual_sk, launches_sk = _record_triton_launches(
            lambda: moe_ffn(x_splitk, w_splitk, ids, wts, active_slots_hint=2)
        )
        assert "nvfp4_moe_m1_gemv_kernel" in launches_sk
        assert "moe_splitk_reduce_kernel" in launches_sk
        assert "moe_m1_activate_rotate_kernel" in launches_sk
        _assert_matches_reference(
            actual_sk,
            _independent_moe_oracle(x_splitk, w_splitk, ids, wts),
            weight_format="nvfp4_w4a16",
            dtype=torch.bfloat16,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
def test_gpu_sm70_trellis_gelu_tanh_falls_back_to_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1009)
    weights = _make_trellis_moe_weights(
        3,
        256,
        256,
        group_size=64,
        activation_fn="gelu_tanh",
        dtype=torch.float16,
        device="cuda",
        seed=1010,
    )
    x = torch.randn((2, 256), device="cuda", dtype=torch.float16)
    ids = torch.tensor([[0, 2], [1, 0]], device="cuda", dtype=torch.int64)
    wts = torch.tensor([[0.6, 0.4], [0.25, 0.75]], device="cuda", dtype=torch.float16)
    slot_map = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int64)
    expected = _reference_for_weights(x, weights, ids, wts, expert_to_slot=slot_map).to("cuda")

    # Initialize Triton driver before stubbing torch.version.hip so isolated ROCm runs keep HIP.
    assert triton.runtime.driver.active is not None
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (7, 0))
    actual, launches = _record_triton_launches(
        lambda: moe_ffn(x, weights, ids, wts, expert_to_slot=slot_map)
    )
    assert "trellis_moe_gemv_kernel" not in launches
    assert "moe_m1_activate_rotate_kernel" not in launches
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_compile_dynamic_active_slots_hint_single_frame() -> None:
    torch.manual_seed(1015)
    weights = _make_moe_weights(
        4, 64, 64, weight_format="nvfp4_w4a16", dtype=torch.float32, seed=1016
    )

    def _fn(inp: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        return moe_ffn(inp, weights, ids, wts, active_slots_hint=inp.shape[0])

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(_fn, backend=counter, fullgraph=True, dynamic=True)
    for rows in (3, 5, 7):
        x = torch.randn((rows, 64), dtype=torch.float32)
        ids = torch.randint(0, 4, (rows, 2), dtype=torch.int64)
        wts = torch.full((rows, 2), 0.5, dtype=torch.float32)
        actual = compiled(x, ids, wts)
        expected = _reference_for_weights(x, weights, ids, wts)
        _assert_matches_reference(
            actual, expected, weight_format="nvfp4_w4a16", dtype=torch.float32
        )
    assert counter.frame_count == 1
    torch._dynamo.reset()
