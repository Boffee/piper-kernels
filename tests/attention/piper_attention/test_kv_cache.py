"""Tests for global chunk-major KV cache storage and freeze_kv_chunks_."""

from __future__ import annotations

import math
import os
import shutil
import sys
from collections.abc import Callable
from typing import Any

import pytest
import torch
from _accelerators import (
    forbid_device_probes,
    forbid_triton_driver,
    mock_cuda_target,
    mock_rocm_target,
    rdna4_available,
)
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.nn import functional as fn
from torch.utils._python_dispatch import TorchDispatchMode

from piper_kernels import (
    PiperKVChunkPool,
    freeze_kv_chunks_,
    kv_chunk_storage_specs,
    piper_attention,
)
from piper_kernels.attention.piper_attention.kv_cache import KV_CHUNK_TOKENS
from piper_kernels.attention.piper_attention.reference import freeze_kv_chunks_reference
from piper_kernels.moe import MoEWeights, moe_ffn
from piper_kernels.trellis import resolve_block_size

_SIGNED_HADAMARD_MASK = (0xF62B39F3, 0x802576E6, 0x910D4CFD, 0x4F877B5A)
_OPCHECK_UTILS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


class _NoTensorOperations(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"metadata validation performed {func}")


class _FreezeFakeOpOnly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.piper_kernels.piper_kv_freeze_.default:
            return func(*args, **(kwargs or {}))
        raise AssertionError(f"fake freeze performed unexpected tensor op {func}")


def _record_triton_launches[T](fn_call: Callable[[], T]) -> tuple[T, list[str]]:
    import triton  # noqa: PLC0415 - GPU-only launch hook keeps CPU collection Triton-free

    launched: list[str] = []

    def _record_launch(metadata: Any) -> None:
        data = metadata.get()
        if isinstance(data, dict) and "name" in data:
            launched.append(str(data["name"]))

    triton.knobs.runtime.launch_enter_hook.add(_record_launch)
    try:
        output = fn_call()
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(_record_launch)
    return output, launched


def _signed_hadamard_inverse(values: torch.Tensor) -> torch.Tensor:
    head_dim = values.shape[-1]
    out = values.float()
    distance = 1
    while distance < head_dim:
        grouped = out.reshape(*out.shape[:-1], -1, 2, distance)
        low, high = grouped.unbind(dim=-2)
        out = torch.stack((low + high, low - high), dim=-2).flatten(-3)
        distance *= 2
    offsets = torch.arange(head_dim, device=values.device)
    word_index = offsets // 32
    selected = torch.full_like(word_index, _SIGNED_HADAMARD_MASK[3], dtype=torch.int64)
    for index in (2, 1, 0):
        selected = torch.where(word_index == index, _SIGNED_HADAMARD_MASK[index], selected)
    signs = 2 * ((selected >> (offsets % 32)) & 1) - 1
    return (out * (1.0 / math.sqrt(head_dim))) * signs


def _wmma_token_perm(tile_tokens: int, device: torch.device) -> torch.Tensor:
    tokens = torch.arange(tile_tokens, device=device, dtype=torch.int64)
    return (tokens & ~24) | ((tokens & 8) << 1) | ((tokens & 16) >> 1)


def _dequantize_slot(
    pool: PiperKVChunkPool,
    slot: int,
    *,
    pack_wmma: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, kv_heads, tiles, head_dim, tile_tokens = pool.value_int8.shape
    key_q = pool.key_int8[slot : slot + 1].view(torch.int8).float()
    if pool.key_scales == tiles:
        key_scaled = (
            key_q.reshape(1, kv_heads, tiles, tile_tokens, head_dim)
            * pool.key_scale[slot : slot + 1].unsqueeze(-1).unsqueeze(-1)
        ).reshape(1, kv_heads, KV_CHUNK_TOKENS, head_dim)
    else:
        key_scaled = key_q * pool.key_scale[slot : slot + 1].unsqueeze(-1)
    key_rot = key_scaled + pool.key_center[slot : slot + 1].unsqueeze(2)
    key_rec = _signed_hadamard_inverse(key_rot)

    val_q = pool.value_int8[slot : slot + 1].view(torch.int8)
    if pack_wmma:
        val_q = val_q.index_select(-1, _wmma_token_perm(tile_tokens, val_q.device))
    val_tok = val_q.transpose(-2, -1).reshape(1, kv_heads, KV_CHUNK_TOKENS, head_dim)
    val_rec = val_tok.float() * pool.value_scale[slot : slot + 1].unsqueeze(-1) + pool.value_mean[
        slot : slot + 1
    ].unsqueeze(2)
    return key_rec, val_rec


def _make_interleaved_pool(
    num_slots: int,
    kv_heads: int,
    head_dim: int,
    *,
    device: torch.device | str = "cpu",
    kv_dtype: torch.dtype = torch.uint8,
    key_scales: int = 16,
    meta_stride_pad_floats: int = 0,
) -> tuple[torch.Tensor, PiperKVChunkPool]:
    tiles = key_scales if key_scales == 16 else 16
    tile_tokens = KV_CHUNK_TOKENS // tiles
    kv = torch.empty(
        (num_slots, 2, kv_heads, KV_CHUNK_TOKENS, head_dim),
        device=device,
        dtype=kv_dtype,
    )
    key_int8 = kv[:, 0]
    value_int8 = kv[:, 1].view(num_slots, kv_heads, tiles, head_dim, tile_tokens)
    scale_stride = kv_heads * key_scales + meta_stride_pad_floats
    key_scale = torch.empty((num_slots, scale_stride), device=device, dtype=torch.float32)[
        :, : kv_heads * key_scales
    ].view(num_slots, kv_heads, key_scales)
    key_center = torch.empty((num_slots, kv_heads, head_dim), device=device, dtype=torch.float32)
    value_scale = torch.empty(
        (num_slots, kv_heads, KV_CHUNK_TOKENS), device=device, dtype=torch.float32
    )
    value_mean = torch.empty((num_slots, kv_heads, head_dim), device=device, dtype=torch.float32)
    pool = PiperKVChunkPool.from_tensors(
        key_int8,
        key_scale,
        key_center,
        value_int8,
        value_scale,
        value_mean,
    )
    return kv, pool


def _clone_pool(pool: PiperKVChunkPool) -> PiperKVChunkPool:
    return PiperKVChunkPool.from_tensors(
        pool.key_int8.clone(),
        pool.key_scale.clone(),
        pool.key_center.clone(),
        pool.value_int8.clone(),
        pool.value_scale.clone(),
        pool.value_mean.clone(),
    )


def _fill_pool_sentinels(pool: PiperKVChunkPool) -> None:
    for idx, tensor in enumerate(
        (
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
        )
    ):
        if tensor.dtype in (torch.int8, torch.uint8):
            tensor.fill_(0x5A if tensor.dtype is torch.uint8 else -42)
        else:
            tensor.fill_(100.0 + idx)


def _pool_versions(pool: PiperKVChunkPool) -> tuple[int, ...]:
    return tuple(
        tensor._version
        for tensor in (
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
        )
    )


def _assert_slot_equal(actual: PiperKVChunkPool, expected: PiperKVChunkPool, slot: int) -> None:
    torch.testing.assert_close(
        actual.key_int8[slot].view(torch.int8),
        expected.key_int8[slot].view(torch.int8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        actual.value_int8[slot].view(torch.int8),
        expected.value_int8[slot].view(torch.int8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(actual.key_scale[slot], expected.key_scale[slot], atol=0, rtol=0)
    torch.testing.assert_close(actual.key_center[slot], expected.key_center[slot], atol=0, rtol=0)
    torch.testing.assert_close(actual.value_scale[slot], expected.value_scale[slot], atol=0, rtol=0)
    torch.testing.assert_close(actual.value_mean[slot], expected.value_mean[slot], atol=0, rtol=0)


def _assert_pools_exact(actual: PiperKVChunkPool, expected: PiperKVChunkPool) -> None:
    for slot in range(int(actual.num_slots)):
        _assert_slot_equal(actual, expected, slot)


def _assert_pools_close_triton(actual: PiperKVChunkPool, expected: PiperKVChunkPool) -> None:
    # Separate FP32 reduction trees can flip rare half-integer boundaries by +-1 code.
    for a_q, e_q in (
        (actual.key_int8.view(torch.int8), expected.key_int8.view(torch.int8)),
        (actual.value_int8.view(torch.int8), expected.value_int8.view(torch.int8)),
    ):
        diff = (a_q.int() - e_q.int()).abs()
        assert int(diff.max().item()) <= 1
        assert float((diff > 0).float().mean().item()) < 1e-4
    torch.testing.assert_close(actual.key_scale, expected.key_scale, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.key_center, expected.key_center, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual.value_scale, expected.value_scale, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.value_mean, expected.value_mean, atol=2e-5, rtol=2e-5)


def _assert_relative_l2(actual: torch.Tensor, expected: torch.Tensor, *, limit: float) -> None:
    num = (actual.float() - expected.float()).norm()
    den = expected.float().norm().clamp_min(1e-6)
    assert float((num / den).item()) < limit


def _build_shared_float32_meta(
    *,
    device: torch.device,
    num_kv_slots: int,
    kv_heads: int,
    head_dim: int,
    key_scales: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Align every FP32 slice offset to a multiple of 4 floats (16 bytes).
    ks_per_slot = ((kv_heads * key_scales + 3) // 4) * 4
    kc_per_slot = ((kv_heads * head_dim + 3) // 4) * 4
    vs_per_slot = ((kv_heads * KV_CHUNK_TOKENS + 3) // 4) * 4
    vm_per_slot = ((kv_heads * head_dim + 3) // 4) * 4
    meta_stride = ks_per_slot + kc_per_slot + vs_per_slot + vm_per_slot
    meta_arena = torch.zeros((num_kv_slots, meta_stride), dtype=torch.float32, device=device)
    c0 = 0
    c1 = c0 + ks_per_slot
    c2 = c1 + kc_per_slot
    c3 = c2 + vs_per_slot
    return (
        meta_arena[:, c0 : c0 + kv_heads * key_scales].view(num_kv_slots, kv_heads, key_scales),
        meta_arena[:, c1 : c1 + kv_heads * head_dim].view(num_kv_slots, kv_heads, head_dim),
        meta_arena[:, c2 : c2 + kv_heads * KV_CHUNK_TOKENS].view(
            num_kv_slots, kv_heads, KV_CHUNK_TOKENS
        ),
        meta_arena[:, c3 : c3 + kv_heads * head_dim].view(num_kv_slots, kv_heads, head_dim),
    )


def _build_shared_arena_moe_and_kv(
    *,
    device: torch.device,
    num_kv_slots: int,
    kv_heads: int,
    head_dim: int,
    hidden_size: int = 256,
    intermediate_size: int = 256,
    num_experts: int = 2,
    key_scales: int = 16,
) -> tuple[torch.Tensor, int, MoEWeights, PiperKVChunkPool]:
    gen = torch.Generator(device="cpu").manual_seed(77)
    up_out = intermediate_size * 2
    up_bs = resolve_block_size(hidden_size)
    dn_bs = resolve_block_size(intermediate_size)

    def _pack_1d_proj(out_dim: int, in_features: int, block_size: int) -> tuple[Any, ...]:
        in_blocks = in_features // block_size
        low_b = (2 * block_size + 7) // 8
        high_b = (6 * block_size) // 8
        rw = torch.randint(
            low_b, high_b + 1, (num_experts, out_dim), dtype=torch.uint8, generator=gen
        )
        totals = [int(rw[slot].sum().item()) * in_blocks for slot in range(num_experts)]
        offsets: list[int] = []
        chunks: list[torch.Tensor] = []
        cursor = 0
        for total in totals:
            offsets.append(cursor)
            chunks.append(torch.randint(0, 256, (total,), dtype=torch.uint8, generator=gen))
            chunks.append(torch.full((13,), 0xA5, dtype=torch.uint8))
            cursor += total + 13
        bs = (
            (torch.rand((num_experts, out_dim, in_blocks), generator=gen) * 0.25 + 0.05)
            .to(torch.float8_e4m3fn)
            .to(device)
        )
        scale = (torch.rand((num_experts, out_dim), generator=gen) * 0.02 + 0.005).to(device)
        return (
            torch.cat(chunks),
            bs,
            scale,
            rw.to(device),
            torch.tensor(offsets, dtype=torch.int64, device=device),
        )

    up_qd_cpu, up_bs_t, up_sc, up_rw, up_so = _pack_1d_proj(up_out, hidden_size, up_bs)
    dn_qd_cpu, dn_bs_t, dn_sc, dn_rw, dn_so = _pack_1d_proj(hidden_size, intermediate_size, dn_bs)

    kv_slot_pair_bytes = 2 * kv_heads * KV_CHUNK_TOKENS * head_dim
    up_bytes = up_qd_cpu.numel()
    dn_start = ((up_bytes + 15) // 16) * 16
    dn_end = dn_start + dn_qd_cpu.numel()
    kv_start = ((dn_end + 15) // 16) * 16
    kv_bytes = num_kv_slots * kv_slot_pair_bytes
    arena = torch.empty(kv_start + kv_bytes, dtype=torch.uint8, device=device)

    up_qd = arena[:up_bytes]
    dn_qd = arena[dn_start:dn_end]
    up_qd.copy_(up_qd_cpu.to(device))
    dn_qd.copy_(dn_qd_cpu.to(device))

    kv_view = arena[kv_start:].view(num_kv_slots, 2, kv_heads, KV_CHUNK_TOKENS, head_dim)
    _, _, tiles, _, tile_tokens = kv_chunk_storage_specs(
        num_kv_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8
    )["value_int8"][0]
    key_u8 = kv_view[:, 0]
    val_u8 = kv_view[:, 1].view(num_kv_slots, kv_heads, tiles, head_dim, tile_tokens)
    key_scale, key_center, value_scale, value_mean = _build_shared_float32_meta(
        device=device,
        num_kv_slots=num_kv_slots,
        kv_heads=kv_heads,
        head_dim=head_dim,
        key_scales=key_scales,
    )

    moe_weights = MoEWeights.from_quantized(
        up_qd,
        up_sc,
        dn_qd,
        dn_sc,
        logical_dtype=torch.bfloat16,
        group_size=64,
        up_gate_block_scales=up_bs_t,
        down_block_scales=dn_bs_t,
        up_gate_row_widths=up_rw,
        down_row_widths=dn_rw,
        up_gate_slot_offsets=up_so,
        down_slot_offsets=dn_so,
        up_gate_block_size=up_bs,
        down_block_size=dn_bs,
    )
    pool = PiperKVChunkPool.from_tensors(
        key_u8,
        key_scale,
        key_center,
        val_u8,
        value_scale,
        value_mean,
    )
    return arena, kv_start, moe_weights, pool


@pytest.mark.parametrize(
    ("target", "expected_scales"),
    [
        ("sm75", 16),
        ("sm80", 1024),
        ("sm_80", 1024),
        ("sm89", 1024),
        ("sm_89", 1024),
        ("sm120", 16),
        ("sm_120", 16),
        ("gfx942", 16),
        ("gfx1100", 16),
        ("gfx1151", 16),
        ("gfx1200", 16),
        ("gfx1201", 16),
        ("cpu", 16),
        ("meta", 16),
    ],
)
def test_kv_chunk_storage_specs_across_targets(target: str, expected_scales: int) -> None:
    specs = kv_chunk_storage_specs(5, 3, 128, device=target, kv_dtype=torch.uint8)
    assert specs["key_int8"] == ((5, 3, 1024, 128), torch.uint8)
    assert specs["key_scale"] == ((5, 3, expected_scales), torch.float32)
    assert specs["key_center"] == ((5, 3, 128), torch.float32)
    assert specs["value_int8"] == ((5, 3, 16, 128, 64), torch.uint8)
    assert specs["value_scale"] == ((5, 3, 1024), torch.float32)
    assert specs["value_mean"] == ((5, 3, 128), torch.float32)


def test_kv_chunk_storage_specs_mocked_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cuda_target(monkeypatch, (8, 9))
    assert kv_chunk_storage_specs(2, 4, 64, device="cuda:0")["key_scale"][0] == (2, 4, 1024)

    mock_cuda_target(monkeypatch, (12, 0))
    assert kv_chunk_storage_specs(2, 4, 64, device="cuda:0")["key_scale"][0] == (2, 4, 16)

    mock_rocm_target(monkeypatch, "gfx1201")
    assert kv_chunk_storage_specs(2, 4, 64, device=torch.device("cuda:0"))["key_scale"][0] == (
        2,
        4,
        16,
    )


def test_kv_chunk_storage_specs_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="num_slots must be >= 0"):
        kv_chunk_storage_specs(-1, 2, 64)
    with pytest.raises(ValueError, match="kv_heads must be >= 1"):
        kv_chunk_storage_specs(2, 0, 64)
    with pytest.raises(ValueError, match="head_dim must be one of 64, 128"):
        kv_chunk_storage_specs(2, 2, 96)
    with pytest.raises(ValueError, match=r"kv_dtype must be torch\.int8 or torch\.uint8"):
        kv_chunk_storage_specs(2, 2, 64, kv_dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="num_slots must be an integer"):
        kv_chunk_storage_specs(True, 2, 64)
    with pytest.raises(TypeError, match=r"device must be torch\.device, str, or None"):
        kv_chunk_storage_specs(2, 2, 64, device=123)  # type: ignore[arg-type]
    for bad_target in ("sm", "smfoo", "sm1", "sm120a", "gfx", "gfxfoo"):
        with pytest.raises(ValueError, match="target specifier"):
            kv_chunk_storage_specs(2, 2, 64, device=bad_target)


@pytest.mark.parametrize("kv_dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("num_slots", [1, 3])
@pytest.mark.parametrize("kv_heads", [1, 3])
def test_pool_from_tensors_interleaved_and_zero_probe_validation(
    monkeypatch: pytest.MonkeyPatch,
    kv_dtype: torch.dtype,
    num_slots: int,
    kv_heads: int,
) -> None:
    forbid_device_probes(monkeypatch, "pool construction should not probe hardware")
    kv, pool = _make_interleaved_pool(
        num_slots, kv_heads, 64, device="cpu", kv_dtype=kv_dtype, key_scales=16
    )
    assert pool.num_slots == num_slots
    assert pool.kv_heads == kv_heads
    assert pool.head_dim == 64
    assert pool.key_scales == 16
    assert pool.kv_dtype is kv_dtype
    assert pool.key_int8.untyped_storage().data_ptr() == kv.untyped_storage().data_ptr()

    with FakeTensorMode():
        fake_specs = kv_chunk_storage_specs(num_slots, kv_heads, 128, kv_dtype=kv_dtype)
        fake_kv = torch.empty(
            (num_slots, 2, kv_heads, KV_CHUNK_TOKENS, 128),
            device="cuda:0",
            dtype=kv_dtype,
        )
        fake_k_view = fake_kv[:, 0]
        fake_v_view = fake_kv[:, 1].view(fake_specs["value_int8"][0])
        fake_key_scale = torch.empty((num_slots, kv_heads, 1024), device="cuda:0")
        fake_key_center = torch.empty((num_slots, kv_heads, 128), device="cuda:0")
        fake_value_scale = torch.empty((num_slots, kv_heads, KV_CHUNK_TOKENS), device="cuda:0")
        fake_value_mean = torch.empty((num_slots, kv_heads, 128), device="cuda:0")
        fake_k = torch.empty((2, kv_heads, 1024, 128), device="cuda:0", dtype=torch.bfloat16)
        fake_v = torch.empty_like(fake_k)
        fake_slots = torch.arange(2, device="cuda:0", dtype=torch.int32)
        with _NoTensorOperations():
            fake_pool = PiperKVChunkPool.from_tensors(
                fake_k_view,
                fake_key_scale,
                fake_key_center,
                fake_v_view,
                fake_value_scale,
                fake_value_mean,
            )
        with _FreezeFakeOpOnly():
            freeze_kv_chunks_(fake_k, fake_v, fake_pool, fake_slots)

    # Standalone FakeTensors outside an active FakeTensorMode context must not probe CUDA.
    with FakeTensorMode() as fm:
        ext_kv = fm.from_tensor(torch.empty((2, 2, 1, KV_CHUNK_TOKENS, 64), dtype=kv_dtype))
        ext_ks = fm.from_tensor(torch.empty((2, 1, 16), dtype=torch.float32))
        ext_kc = fm.from_tensor(torch.empty((2, 1, 64), dtype=torch.float32))
        ext_vs = fm.from_tensor(torch.empty((2, 1, KV_CHUNK_TOKENS), dtype=torch.float32))
        ext_vm = fm.from_tensor(torch.empty((2, 1, 64), dtype=torch.float32))
        ext_cuda_pool = PiperKVChunkPool.from_tensors(
            ext_kv[:, 0].to("cuda"),
            ext_ks.to("cuda"),
            ext_kc.to("cuda"),
            ext_kv[:, 1].view(2, 1, 16, 64, 64).to("cuda"),
            ext_vs.to("cuda"),
            ext_vm.to("cuda"),
        )
        ext_k = fm.from_tensor(torch.empty((1, 1, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16)).to(
            "cuda"
        )
        ext_v = fm.from_tensor(torch.empty((1, 1, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16)).to(
            "cuda"
        )
        ext_s = fm.from_tensor(torch.empty((1,), dtype=torch.int32)).to("cuda")
    freeze_kv_chunks_(ext_k, ext_v, ext_cuda_pool, ext_s)


def test_pool_properties_preserve_symint_under_symbolic_tracing() -> None:
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    n_sym = mode.from_tensor(torch.empty((3,)), static_shapes=False).shape[0]
    with mode:
        kv = torch.empty((n_sym, 2, 2, KV_CHUNK_TOKENS, 64), dtype=torch.uint8)
        ks = torch.empty((n_sym, 2, 16), dtype=torch.float32)
        kc = torch.empty((n_sym, 2, 64), dtype=torch.float32)
        vs = torch.empty((n_sym, 2, KV_CHUNK_TOKENS), dtype=torch.float32)
        vm = torch.empty((n_sym, 2, 64), dtype=torch.float32)
        pool = PiperKVChunkPool.from_tensors(
            kv[:, 0], ks, kc, kv[:, 1].view(n_sym, 2, 16, 64, 64), vs, vm
        )
        assert isinstance(pool.num_slots, torch.SymInt)


def test_pool_validation_rejects_malformed_layouts_and_overlaps() -> None:
    base = PiperKVChunkPool.empty(2, 2, 64, device="cpu", kv_dtype=torch.uint8)

    with pytest.raises(ValueError, match="must not overlap in storage"):
        PiperKVChunkPool.from_tensors(
            base.key_int8,
            base.key_scale,
            base.key_center,
            base.key_int8.reshape(2, 2, 16, 64, 64),
            base.value_scale,
            base.value_mean,
        )

    bad_stride0 = base.key_int8[:1].expand(2, 2, 1024, 64)
    with pytest.raises(ValueError, match=r"stride\(0\)"):
        PiperKVChunkPool.from_tensors(
            bad_stride0,
            base.key_scale,
            base.key_center,
            base.value_int8,
            base.value_scale,
            base.value_mean,
        )

    raw_f = torch.empty(2 * 2 * 16 + 4, dtype=torch.float32)
    unaligned_scale = raw_f[1 : 1 + 2 * 2 * 16].view(2, 2, 16)
    with pytest.raises(ValueError, match="16-byte aligned"):
        PiperKVChunkPool.from_tensors(
            base.key_int8,
            unaligned_scale,
            base.key_center,
            base.value_int8,
            base.value_scale,
            base.value_mean,
        )

    raw_padded = torch.empty((2, 2 * 16 + 1), dtype=torch.float32)
    unaligned_step_scale = raw_padded[:, : 2 * 16].view(2, 2, 16)
    with pytest.raises(ValueError, match="16-byte aligned"):
        PiperKVChunkPool.from_tensors(
            base.key_int8,
            unaligned_step_scale,
            base.key_center,
            base.value_int8,
            base.value_scale,
            base.value_mean,
        )

    non_contig_k = torch.empty((2, 2, 64, 1024), dtype=torch.uint8).transpose(-2, -1)
    with pytest.raises(ValueError, match="dense row-major"):
        PiperKVChunkPool.from_tensors(
            non_contig_k,
            base.key_scale,
            base.key_center,
            base.value_int8,
            base.value_scale,
            base.value_mean,
        )

    with pytest.raises(ValueError, match="negated or conjugated"):
        PiperKVChunkPool.from_tensors(
            base.key_int8,
            torch._neg_view(base.key_scale),
            base.key_center,
            base.value_int8,
            base.value_scale,
            base.value_mean,
        )

    with pytest.raises(ValueError, match="match key_int8 dtype"):
        PiperKVChunkPool.from_tensors(
            base.key_int8,
            base.key_scale,
            base.key_center,
            base.value_int8.to(torch.int8),
            base.value_scale,
            base.value_mean,
        )

    with pytest.raises(ValueError, match="must use float32"):
        PiperKVChunkPool.from_tensors(
            base.key_int8,
            base.key_scale.to(torch.bfloat16),
            base.key_center,
            base.value_int8,
            base.value_scale,
            base.value_mean,
        )

    with pytest.raises(ValueError, match="share device"):
        PiperKVChunkPool.from_tensors(
            base.key_int8,
            base.key_scale.to("meta"),
            base.key_center,
            base.value_int8,
            base.value_scale,
            base.value_mean,
        )

    # Unequal-width periodic overlaps in a shared float32 buffer.
    shared_fwd = torch.empty((2, 160), dtype=torch.float32)
    fwd_scale = shared_fwd[:, :32].view(2, 2, 16)
    fwd_center = shared_fwd[:, 28 : 28 + 128].view(2, 2, 64)
    with pytest.raises(ValueError, match="must not overlap in storage"):
        PiperKVChunkPool.from_tensors(
            base.key_int8,
            fwd_scale,
            fwd_center,
            base.value_int8,
            base.value_scale,
            base.value_mean,
        )

    shared_wrap = torch.empty(316, dtype=torch.float32)
    wrap_scale = shared_wrap.as_strided((2, 2, 16), (156, 16, 1), storage_offset=0)
    wrap_center = shared_wrap.as_strided((2, 2, 64), (156, 64, 1), storage_offset=32)
    with pytest.raises(ValueError, match="must not overlap in storage"):
        PiperKVChunkPool.from_tensors(
            base.key_int8,
            wrap_scale,
            wrap_center,
            base.value_int8,
            base.value_scale,
            base.value_mean,
        )

    # Distinct UntypedStorage handles aliasing overlapping memory via frombuffer.
    raw_mv = memoryview(bytearray(2 * 2 * 1024 * 64 + 64))
    base_ptr = torch.frombuffer(raw_mv, dtype=torch.uint8, count=1).data_ptr()
    align_off = (-base_ptr) % 16
    aligned_buf = raw_mv[align_off:]
    k_alias = torch.frombuffer(aligned_buf, dtype=torch.uint8, count=2 * 2 * 1024 * 64).view(
        2, 2, 1024, 64
    )
    v_alias = torch.frombuffer(aligned_buf[16:], dtype=torch.uint8, count=2 * 2 * 1024 * 64).view(
        2, 2, 16, 64, 64
    )
    with pytest.raises(ValueError, match="must not overlap in storage"):
        PiperKVChunkPool.from_tensors(
            k_alias,
            base.key_scale,
            base.key_center,
            v_alias,
            base.value_scale,
            base.value_mean,
        )


def _finite_intervals_overlap(
    a_start: int,
    a_step: int,
    a_len: int,
    a_slots: int,
    b_start: int,
    b_step: int,
    b_len: int,
    b_slots: int,
) -> bool:
    for i in range(a_slots):
        a0 = a_start + i * a_step
        a1 = a0 + a_len
        for j in range(b_slots):
            b0 = b_start + j * b_step
            if a0 < b0 + b_len and b0 < a1:
                return True
    return False


@pytest.mark.parametrize("swap_offsets", [False, True])
def test_finite_slot_different_stride_arena_disjointness_and_collision(
    swap_offsets: bool,
) -> None:
    # With strides (256, 320) and offsets (0, 128), slots stay disjoint for num_slots <= 3
    # and collide at num_slots == 4; with swapped offsets (128, 0), num_slots == 2 is disjoint
    # and num_slots == 3 collides at key_center[2] == value_mean[2] == [640, 704).
    disjoint_counts = (2,) if swap_offsets else (2, 3)
    collide_count = 3 if swap_offsets else 4
    off_k, off_v = (128, 0) if swap_offsets else (0, 128)

    for num_slots in disjoint_counts:
        base = PiperKVChunkPool.empty(num_slots, 1, 64, device="cpu", kv_dtype=torch.uint8)
        arena = torch.full((1280,), -999.0, dtype=torch.float32)
        kc = arena.as_strided((num_slots, 1, 64), (256, 64, 1), storage_offset=off_k)
        vm = arena.as_strided((num_slots, 1, 64), (320, 64, 1), storage_offset=off_v)
        kc.fill_(101.0)
        vm.fill_(105.0)
        pool = PiperKVChunkPool.from_tensors(
            base.key_int8, base.key_scale, kc, base.value_int8, base.value_scale, vm
        )
        _fill_pool_sentinels(pool)
        ref = _clone_pool(pool)
        sentinel = _clone_pool(pool)
        # Mask out written elements so untouched arena padding stays unchanged at -999.0.
        pad_mask = torch.ones_like(arena, dtype=torch.bool)
        for s in range(num_slots):
            pad_mask[off_k + s * 256 : off_k + s * 256 + 64] = False
            pad_mask[off_v + s * 320 : off_v + s * 320 + 64] = False
        pad_snap = arena[pad_mask].clone()

        k_in = torch.randn((num_slots - 1, 1, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16)
        v_in = torch.randn_like(k_in)
        slot_ids = torch.arange(num_slots - 1, dtype=torch.int32)
        with torch.no_grad():
            freeze_kv_chunks_(k_in, v_in, pool, slot_ids)
            freeze_kv_chunks_reference(
                k_in,
                v_in,
                ref.key_int8,
                ref.key_scale,
                ref.key_center,
                ref.value_int8,
                ref.value_scale,
                ref.value_mean,
                slot_ids,
            )
        _assert_pools_exact(pool, ref)
        _assert_slot_equal(pool, sentinel, num_slots - 1)
        torch.testing.assert_close(arena[pad_mask], pad_snap, atol=0, rtol=0)

    base_col = PiperKVChunkPool.empty(collide_count, 1, 64, device="cpu", kv_dtype=torch.uint8)
    arena_col = torch.empty(1400, dtype=torch.float32)
    kc_col = arena_col.as_strided((collide_count, 1, 64), (256, 64, 1), storage_offset=off_k)
    vm_col = arena_col.as_strided((collide_count, 1, 64), (320, 64, 1), storage_offset=off_v)
    with pytest.raises(ValueError, match="must not overlap in storage"):
        PiperKVChunkPool.from_tensors(
            base_col.key_int8,
            base_col.key_scale,
            kc_col,
            base_col.value_int8,
            base_col.value_scale,
            vm_col,
        )


def test_finite_slot_arena_overlap_matches_brute_force_oracle() -> None:
    arena = torch.empty(4096, dtype=torch.float32)
    for num_slots in (2, 3, 4):
        base = PiperKVChunkPool.empty(num_slots, 1, 64, device="cpu", kv_dtype=torch.uint8)
        for a_step in (64, 96, 128, 160, 192):
            for b_step in (64, 96, 128, 160, 192):
                for a_off in (0, 64, 96, 128):
                    for b_off in (0, 64, 96, 128):
                        kc = arena.as_strided(
                            (num_slots, 1, 64), (a_step, 64, 1), storage_offset=a_off
                        )
                        vm = arena.as_strided(
                            (num_slots, 1, 64), (b_step, 64, 1), storage_offset=b_off
                        )
                        should_overlap = _finite_intervals_overlap(
                            a_off, a_step, 64, num_slots, b_off, b_step, 64, num_slots
                        )
                        if should_overlap:
                            with pytest.raises(ValueError, match="must not overlap in storage"):
                                PiperKVChunkPool.from_tensors(
                                    base.key_int8,
                                    base.key_scale,
                                    kc,
                                    base.value_int8,
                                    base.value_scale,
                                    vm,
                                )
                        else:
                            PiperKVChunkPool.from_tensors(
                                base.key_int8,
                                base.key_scale,
                                kc,
                                base.value_int8,
                                base.value_scale,
                                vm,
                            )


def test_meta_and_zero_slot_pools() -> None:
    meta_pool = PiperKVChunkPool.empty(2, 2, 64, device="meta", kv_dtype=torch.uint8)
    meta_k = torch.empty((1, 2, KV_CHUNK_TOKENS, 64), device="meta", dtype=torch.bfloat16)
    meta_v = torch.empty_like(meta_k)
    meta_s = torch.empty((1,), device="meta", dtype=torch.int32)
    freeze_kv_chunks_(meta_k, meta_v, meta_pool, meta_s)

    with pytest.raises(ValueError, match="must not overlap in storage"):
        PiperKVChunkPool.from_tensors(
            meta_pool.key_int8,
            meta_pool.key_scale,
            meta_pool.key_center,
            meta_pool.key_int8.view(2, 2, 16, 64, 64),
            meta_pool.value_scale,
            meta_pool.value_mean,
        )

    zero_specs = kv_chunk_storage_specs(0, 2, 64, device="cpu", kv_dtype=torch.uint8)
    assert zero_specs["key_int8"][0][0] == 0
    zero_pool = PiperKVChunkPool.empty(0, 2, 64, device="cpu", kv_dtype=torch.uint8)
    k0 = torch.randn((2, 2, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    s0 = torch.tensor([0, 1], dtype=torch.int32)
    with forbid_triton_driver():
        freeze_kv_chunks_(k0, v0, zero_pool, s0)


@pytest.mark.parametrize("use_inference_mode", [False, True])
def test_freeze_rejects_bad_inputs_and_preserves_versions_and_bytes(
    use_inference_mode: bool,
) -> None:
    pool = PiperKVChunkPool.empty(3, 2, 64, device="cpu", kv_dtype=torch.uint8)
    _fill_pool_sentinels(pool)
    snapshot = _clone_pool(pool)
    versions_before = _pool_versions(pool)

    k = torch.randn((2, 2, 1024, 64), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = torch.tensor([0, 2], dtype=torch.int32)

    ctx = torch.inference_mode() if use_inference_mode else torch.no_grad()
    with ctx:
        with pytest.raises(ValueError, match="must share a dtype"):
            freeze_kv_chunks_(k, v.to(torch.float16), pool, slots)
        with pytest.raises(ValueError, match="int32"):
            freeze_kv_chunks_(k, v, pool, slots.to(torch.int64))
        with pytest.raises(ValueError, match="slot_ids must be contiguous"):
            freeze_kv_chunks_(k, v, pool, torch.arange(4, dtype=torch.int32)[::2])
        with pytest.raises(ValueError, match="shape"):
            freeze_kv_chunks_(torch.randn((2, 2, 1023, 64), dtype=torch.bfloat16), v, pool, slots)
        with pytest.raises(ValueError, match="head dimension must be contiguous"):
            freeze_kv_chunks_(k.transpose(-2, -1)[..., :64], v[..., :64], pool, slots)
        with pytest.raises(ValueError, match="must not use negated or conjugated views"):
            freeze_kv_chunks_(torch._neg_view(k), v, pool, slots)
        with pytest.raises(ValueError, match="must not use negated or conjugated views"):
            freeze_kv_chunks_(k, v, pool, torch._neg_view(slots))
        with pytest.raises(ValueError, match="key must have shape"):
            freeze_kv_chunks_(torch.tensor(1.0, dtype=torch.bfloat16), v, pool, slots)
        with pytest.raises(TypeError, match=r"key must be a torch\.Tensor"):
            freeze_kv_chunks_("not_a_tensor", v, pool, slots)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="pool must be a PiperKVChunkPool"):
            freeze_kv_chunks_(k, v, "bad_pool", slots)  # type: ignore[arg-type]

    assert _pool_versions(pool) == versions_before
    _assert_pools_exact(pool, snapshot)

    # Default grad mode rejects requires_grad and preserves saved autograd graph versions.
    w = torch.ones((3, 2, 64), requires_grad=True)
    saved = (pool.key_center * w).sum()
    with pytest.raises(ValueError, match="must share a dtype"):
        freeze_kv_chunks_(k, v.to(torch.float16), pool, slots)
    with pytest.raises(RuntimeError, match="inference-only"):
        freeze_kv_chunks_(k.clone().requires_grad_(True), v, pool, slots)
    with pytest.raises(RuntimeError, match="inference-only"):
        freeze_kv_chunks_(k, v.clone().requires_grad_(True), pool, slots)
    assert _pool_versions(pool) == versions_before
    saved.backward()


def test_freeze_rejects_overlapping_inputs_and_accepts_disjoint_arena() -> None:
    base = PiperKVChunkPool.empty(3, 2, 64, device="cpu", kv_dtype=torch.uint8)
    arena = torch.empty(3 * 2 * KV_CHUNK_TOKENS * 64 + 256 * 1024, dtype=torch.uint8)
    k_arena_slice = arena[3 * 2 * KV_CHUNK_TOKENS * 64 :].view(torch.bfloat16)[: 1 * 2 * 1024 * 64]
    k_disjoint = k_arena_slice.view(1, 2, 1024, 64).zero_()
    pool_arena = PiperKVChunkPool.from_tensors(
        arena[: 3 * 2 * KV_CHUNK_TOKENS * 64].view(3, 2, KV_CHUNK_TOKENS, 64),
        base.key_scale,
        base.key_center,
        base.value_int8,
        base.value_scale,
        base.value_mean,
    )
    _fill_pool_sentinels(pool_arena)
    slots = torch.tensor([1], dtype=torch.int32)
    freeze_kv_chunks_(k_disjoint, torch.zeros_like(k_disjoint), pool_arena, slots)

    snap = _clone_pool(pool_arena)
    versions_before = _pool_versions(pool_arena)
    overlap_k = arena[: 1 * 2 * 1024 * 64 * 2].view(torch.bfloat16).view(1, 2, 1024, 64)
    overlap_slots = pool_arena.key_scale[0, 0, :4].view(torch.int32)[:1]
    with pytest.raises(ValueError, match="must not overlap key_int8"):
        freeze_kv_chunks_(overlap_k, torch.zeros_like(overlap_k), pool_arena, slots)
    with pytest.raises(ValueError, match="must not overlap key_scale"):
        freeze_kv_chunks_(k_disjoint, torch.zeros_like(k_disjoint), pool_arena, overlap_slots)
    assert _pool_versions(pool_arena) == versions_before
    _assert_pools_exact(pool_arena, snap)


def test_freeze_interleaved_strided_inputs_in_shared_arena() -> None:
    torch.manual_seed(913)
    num_slots = 2
    kv_heads = 1
    head_dim = 64
    chunk_u8_bytes = kv_heads * KV_CHUNK_TOKENS * head_dim
    chunk_bf16_bytes = chunk_u8_bytes * 2
    stride_bytes = chunk_bf16_bytes + chunk_u8_bytes

    base = PiperKVChunkPool.empty(num_slots, kv_heads, head_dim, device="cpu", kv_dtype=torch.uint8)
    arena_k = torch.empty(num_slots * stride_bytes, dtype=torch.uint8)
    arena_v = torch.empty(num_slots * stride_bytes, dtype=torch.uint8)
    # Place 2-chunk BF16 inputs at [0, 131072), [196608, 327680) interleaved with
    # uint8 pool slots at [131072, 196608), [327680, 393216) in the same arena.
    k_in = arena_k.view(torch.bfloat16).as_strided(
        (num_slots, kv_heads, KV_CHUNK_TOKENS, head_dim),
        (stride_bytes // 2, chunk_u8_bytes, head_dim, 1),
        storage_offset=0,
    )
    v_in = arena_v.view(torch.bfloat16).as_strided(
        (num_slots, kv_heads, KV_CHUNK_TOKENS, head_dim),
        (stride_bytes // 2, chunk_u8_bytes, head_dim, 1),
        storage_offset=0,
    )
    k_in.copy_(torch.randn_like(k_in))
    v_in.copy_(torch.randn_like(v_in))
    k_snap = k_in.clone()
    v_snap = v_in.clone()

    key_pool_view = arena_k.as_strided(
        (num_slots, kv_heads, KV_CHUNK_TOKENS, head_dim),
        (stride_bytes, chunk_u8_bytes, head_dim, 1),
        storage_offset=chunk_bf16_bytes,
    )
    val_pool_view = arena_v.as_strided(
        (num_slots, kv_heads, 16, head_dim, 64),
        (stride_bytes, chunk_u8_bytes, head_dim * 64, 64, 1),
        storage_offset=chunk_bf16_bytes,
    )
    pool = PiperKVChunkPool.from_tensors(
        key_pool_view,
        base.key_scale,
        base.key_center,
        val_pool_view,
        base.value_scale,
        base.value_mean,
    )
    _fill_pool_sentinels(pool)
    ref = _clone_pool(pool)
    slots = torch.tensor([0, 1], dtype=torch.int32)
    with torch.no_grad():
        freeze_kv_chunks_(k_in, v_in, pool, slots)
        freeze_kv_chunks_reference(
            k_in,
            v_in,
            ref.key_int8,
            ref.key_scale,
            ref.key_center,
            ref.value_int8,
            ref.value_scale,
            ref.value_mean,
            slots,
        )
    _assert_pools_exact(pool, ref)
    torch.testing.assert_close(k_in, k_snap, atol=0, rtol=0)
    torch.testing.assert_close(v_in, v_snap, atol=0, rtol=0)

    # Shifting key input so chunk 1 overlaps pool.key_int8[0] must still be rejected.
    bad_k = arena_k.view(torch.bfloat16).as_strided(
        (2, 1, KV_CHUNK_TOKENS, head_dim),
        ((chunk_bf16_bytes + 4096) // 2, chunk_u8_bytes, head_dim, 1),
        storage_offset=0,
    )
    snap = _clone_pool(pool)
    versions_before = _pool_versions(pool)
    with pytest.raises(ValueError, match="must not overlap key_int8"):
        freeze_kv_chunks_(bad_k, v_in, pool, slots)
    assert _pool_versions(pool) == versions_before
    _assert_pools_exact(pool, snap)


def test_freeze_multi_dim_strided_input_in_shared_arena() -> None:
    torch.manual_seed(914)
    head_dim = 64
    head_bytes_bf16 = KV_CHUNK_TOKENS * head_dim * 2
    slot_pool_bytes = 2 * KV_CHUNK_TOKENS * head_dim
    head_stride_bytes = head_bytes_bf16 + slot_pool_bytes
    slot_stride_2 = head_stride_bytes + head_bytes_bf16

    # Pool slot [131072, 262144) sits inside k_multi's stride(1) gap between head 0 and head 1.
    base2 = PiperKVChunkPool.empty(2, 2, head_dim, device="cpu", kv_dtype=torch.uint8)
    arena2 = torch.empty(2 * slot_stride_2, dtype=torch.uint8)
    k_multi = arena2.view(torch.bfloat16).as_strided(
        (2, 2, KV_CHUNK_TOKENS, head_dim),
        (slot_stride_2 // 2, head_stride_bytes // 2, head_dim, 1),
        storage_offset=0,
    )
    k_multi.copy_(torch.randn_like(k_multi))
    key_pool_2 = arena2.as_strided(
        (2, 2, KV_CHUNK_TOKENS, head_dim),
        (slot_stride_2, KV_CHUNK_TOKENS * head_dim, head_dim, 1),
        storage_offset=head_bytes_bf16,
    )
    pool2 = PiperKVChunkPool.from_tensors(
        key_pool_2,
        base2.key_scale,
        base2.key_center,
        base2.value_int8,
        base2.value_scale,
        base2.value_mean,
    )
    _fill_pool_sentinels(pool2)
    v_multi = torch.randn((2, 2, KV_CHUNK_TOKENS, head_dim), dtype=torch.bfloat16)
    slots = torch.tensor([0, 1], dtype=torch.int32)
    ref2 = _clone_pool(pool2)
    with torch.no_grad():
        freeze_kv_chunks_(k_multi, v_multi, pool2, slots)
        freeze_kv_chunks_reference(
            k_multi,
            v_multi,
            ref2.key_int8,
            ref2.key_scale,
            ref2.key_center,
            ref2.value_int8,
            ref2.value_scale,
            ref2.value_mean,
            slots,
        )
    _assert_pools_exact(pool2, ref2)


def test_freeze_rejects_inference_pool_outside_inference_mode() -> None:
    normal_pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu", kv_dtype=torch.uint8)
    _fill_pool_sentinels(normal_pool)
    with torch.inference_mode():
        inf_pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu", kv_dtype=torch.uint8)
        _fill_pool_sentinels(inf_pool)
        mixed_inf_value_mean = inf_pool.value_mean
        freeze_kv_chunks_(
            torch.randn((1, 2, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16),
            torch.randn((1, 2, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16),
            inf_pool,
            torch.tensor([0], dtype=torch.int32),
        )

    mixed_pool = PiperKVChunkPool.from_tensors(
        normal_pool.key_int8,
        normal_pool.key_scale,
        normal_pool.key_center,
        normal_pool.value_int8,
        normal_pool.value_scale,
        mixed_inf_value_mean,
    )
    snap = _clone_pool(normal_pool)
    inf_snap = mixed_inf_value_mean.clone()
    versions_before = _pool_versions(normal_pool)
    k = torch.randn((1, 2, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    s = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="outside InferenceMode"):
        freeze_kv_chunks_(k, v, mixed_pool, s)
    assert _pool_versions(normal_pool) == versions_before
    _assert_pools_exact(normal_pool, snap)
    torch.testing.assert_close(mixed_inf_value_mean, inf_snap, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kv_dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("key_scales", [16, 1024])
def test_freeze_cpu_reference_parity_and_miss_skipping(
    dtype: torch.dtype,
    head_dim: int,
    kv_dtype: torch.dtype,
    key_scales: int,
) -> None:
    torch.manual_seed(123 + head_dim + key_scales)
    num_slots = 5
    kv_heads = 2
    kv_raw, pool = _make_interleaved_pool(
        num_slots, kv_heads, head_dim, device="cpu", kv_dtype=kv_dtype, key_scales=key_scales
    )
    _fill_pool_sentinels(pool)
    sentinel_pool = _clone_pool(pool)

    # Shifted non-zero per-channel means exercise key_center rotation and value_mean directly.
    shift_k = torch.linspace(-5.0, 4.0, head_dim, dtype=dtype).view(1, 1, 1, head_dim)
    shift_v = torch.linspace(3.0, -3.5, head_dim, dtype=dtype).view(1, 1, 1, head_dim)
    key_proj = (
        torch.randn((4, KV_CHUNK_TOKENS, kv_heads, head_dim), dtype=dtype).transpose(1, 2) + shift_k
    )
    val_proj = (
        torch.randn((4, KV_CHUNK_TOKENS, kv_heads, head_dim), dtype=dtype).transpose(1, 2) + shift_v
    )
    # Chunk 0 uses broadcast zero-stride constant V without copying; chunk 1 is unused (-1).
    const_row = torch.randn((1, kv_heads, 1, head_dim), dtype=dtype)
    val_expand = const_row.expand(1, kv_heads, KV_CHUNK_TOKENS, head_dim)
    val_proj[0].copy_(val_expand[0])
    slot_ids = torch.tensor([0, -1, 3, num_slots], dtype=torch.int32)

    with torch.amp.autocast(device_type="cpu", dtype=dtype):
        freeze_kv_chunks_(key_proj, val_proj, pool, slot_ids)

    for untouched in (1, 2, 4):
        _assert_slot_equal(pool, sentinel_pool, untouched)
    assert kv_raw[:, 0].data_ptr() == pool.key_int8.data_ptr()
    assert kv_raw[:, 1].data_ptr() == pool.value_int8.data_ptr()

    k0_rec, v0_rec = _dequantize_slot(pool, 0, pack_wmma=False)
    torch.testing.assert_close(v0_rec[0], val_proj[0].float(), atol=1e-4, rtol=1e-4)
    _assert_relative_l2(k0_rec, key_proj[0], limit=0.03)

    k3_rec, v3_rec = _dequantize_slot(pool, 3, pack_wmma=False)
    _assert_relative_l2(k3_rec, key_proj[2], limit=0.03)
    _assert_relative_l2(v3_rec, val_proj[2], limit=0.03)

    # Reconstructed attention parity for both grouped (16) and per-thread (1024) scales.
    q = torch.randn((1, 4, KV_CHUNK_TOKENS, head_dim), dtype=dtype)
    k_gqa = k3_rec.to(dtype).repeat_interleave(2, dim=1)
    v_gqa = v3_rec.to(dtype).repeat_interleave(2, dim=1)
    reconstructed_attn = fn.scaled_dot_product_attention(q, k_gqa, v_gqa, is_causal=True)
    if key_scales == 16:
        actual_attn = piper_attention(q, key_proj[2:3], val_proj[2:3], is_causal=True)
        torch.testing.assert_close(actual_attn, reconstructed_attn, atol=8e-2, rtol=5e-2)
    else:
        exact_attn = fn.scaled_dot_product_attention(
            q,
            key_proj[2:3].repeat_interleave(2, dim=1),
            val_proj[2:3].repeat_interleave(2, dim=1),
            is_causal=True,
        )
        torch.testing.assert_close(exact_attn, reconstructed_attn, atol=8e-2, rtol=5e-2)

    # Direct broadcast zero-stride inputs, all-zeros, and 2**100 constant V scale-floor test.
    if key_scales == 16:
        freeze_kv_chunks_(
            val_expand,
            torch.full(
                (1, kv_heads, 1, head_dim),
                2.0**100 if dtype is torch.bfloat16 else 0.0,
                dtype=dtype,
            ).expand(1, kv_heads, KV_CHUNK_TOKENS, head_dim),
            pool,
            torch.tensor([2], dtype=torch.int32),
        )
        assert torch.isfinite(pool.value_scale[2]).all()
        assert torch.isfinite(pool.key_scale[2]).all()

        # pack_wmma=True permutes value tokens inside each 64-token tile identically.
        _, wmma_pool = _make_interleaved_pool(
            2, kv_heads, head_dim, device="cpu", kv_dtype=kv_dtype, key_scales=key_scales
        )
        freeze_kv_chunks_reference(
            key_proj[2:3],
            val_proj[2:3],
            wmma_pool.key_int8,
            wmma_pool.key_scale,
            wmma_pool.key_center,
            wmma_pool.value_int8,
            wmma_pool.value_scale,
            wmma_pool.value_mean,
            torch.tensor([1], dtype=torch.int32),
            pack_wmma=True,
        )
        _, v_wmma_rec = _dequantize_slot(wmma_pool, 1, pack_wmma=True)
        torch.testing.assert_close(v_wmma_rec, v3_rec, atol=0.0, rtol=0.0)

    # num_new == 0 is a clean no-op.
    snap = _clone_pool(pool)
    empty_k = torch.empty((0, kv_heads, KV_CHUNK_TOKENS, head_dim), dtype=dtype)
    empty_v = torch.empty_like(empty_k)
    empty_slots = torch.empty((0,), dtype=torch.int32)
    with forbid_triton_driver():
        freeze_kv_chunks_(empty_k, empty_v, pool, empty_slots)
    _assert_pools_exact(pool, snap)


def test_cpu_opcheck_and_standalone_dynamic_compile() -> None:
    pool = PiperKVChunkPool.empty(4, 2, 64, device="cpu", kv_dtype=torch.uint8)
    k = torch.randn((2, 2, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = torch.tensor([2, -1], dtype=torch.int32)
    res = torch.library.opcheck(
        torch.ops.piper_kernels.piper_kv_freeze_.default,
        (
            k,
            v,
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
            slots,
        ),
        test_utils=_OPCHECK_UTILS,
    )
    assert set(res.values()) == {"SUCCESS"}

    def _fn(
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        ki8: torch.Tensor,
        ks: torch.Tensor,
        kc: torch.Tensor,
        vi8: torch.Tensor,
        vs: torch.Tensor,
        vm: torch.Tensor,
        s_in: torch.Tensor,
    ) -> torch.Tensor:
        dyn_pool = PiperKVChunkPool.from_tensors(ki8, ks, kc, vi8, vs, vm)
        freeze_kv_chunks_(k_in, v_in, dyn_pool, s_in)
        return ks.sum()

    has_cpu_inductor = (
        sys.platform != "win32" or shutil.which(os.environ.get("CXX", "cl")) is not None
    )
    backends = ("aot_eager", "inductor") if has_cpu_inductor else ("aot_eager",)

    for backend in backends:
        torch._dynamo.reset()
        counter = CompileCounterWithBackend(backend)
        compiled = torch.compile(_fn, backend=counter, fullgraph=True, dynamic=True)
        with torch.no_grad():
            for num_slots, slot_list in ((5, [0, 3, 1]), (8, [5, -1, 2, 0])):
                p = PiperKVChunkPool.empty(num_slots, 2, 64, device="cpu", kv_dtype=torch.uint8)
                _fill_pool_sentinels(p)
                p_ref = _clone_pool(p)
                kk = torch.randn((len(slot_list), 2, KV_CHUNK_TOKENS, 64), dtype=torch.bfloat16)
                vv = torch.randn_like(kk)
                ss = torch.tensor(slot_list, dtype=torch.int32)
                compiled(
                    kk,
                    vv,
                    p.key_int8,
                    p.key_scale,
                    p.key_center,
                    p.value_int8,
                    p.value_scale,
                    p.value_mean,
                    ss,
                )
                freeze_kv_chunks_reference(
                    kk,
                    vv,
                    p_ref.key_int8,
                    p_ref.key_scale,
                    p_ref.key_center,
                    p_ref.value_int8,
                    p_ref.value_scale,
                    p_ref.value_mean,
                    ss,
                )
                _assert_pools_exact(p, p_ref)
        assert counter.frame_count == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kv_dtype", [torch.int8, torch.uint8])
def test_gpu_triton_freeze_parity_with_reference_and_dense_attention(
    dtype: torch.dtype,
    head_dim: int,
    kv_dtype: torch.dtype,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(300 + head_dim)
    is_rdna4 = rdna4_available()
    num_slots = 5
    kv_heads = 2
    key_scales = kv_chunk_storage_specs(num_slots, kv_heads, head_dim, device=device)["key_scale"][
        0
    ][2]

    # Exercise 4-float (16-byte) aligned float32 metadata stride (not div16 in elements).
    _, gpu_pool = _make_interleaved_pool(
        num_slots,
        kv_heads,
        head_dim,
        device=device,
        kv_dtype=kv_dtype,
        key_scales=key_scales,
        meta_stride_pad_floats=4,
    )
    _fill_pool_sentinels(gpu_pool)
    ref_pool = _clone_pool(gpu_pool)
    sentinel_pool = _clone_pool(gpu_pool)

    shift_k = torch.linspace(-4.0, 5.0, head_dim, device=device, dtype=dtype).view(
        1, 1, 1, head_dim
    )
    shift_v = torch.linspace(3.5, -2.5, head_dim, device=device, dtype=dtype).view(
        1, 1, 1, head_dim
    )
    key_proj = (
        torch.randn((4, KV_CHUNK_TOKENS, kv_heads, head_dim), device=device, dtype=dtype).transpose(
            1, 2
        )
        + shift_k
    )
    val_proj = (
        torch.randn((4, KV_CHUNK_TOKENS, kv_heads, head_dim), device=device, dtype=dtype).transpose(
            1, 2
        )
        + shift_v
    )
    if dtype is torch.bfloat16:
        val_proj[0].fill_(2.0**100)
    else:
        val_proj[0].zero_()
        key_proj[0].zero_()
    slot_ids = torch.tensor([0, -1, 3, num_slots], device=device, dtype=torch.int32)

    _, launches = _record_triton_launches(
        lambda: freeze_kv_chunks_(key_proj, val_proj, gpu_pool, slot_ids)
    )
    assert "_quantize_key_k1024_kernel" in launches
    assert "_quantize_value_k1024_kernel" in launches

    freeze_kv_chunks_reference(
        key_proj,
        val_proj,
        ref_pool.key_int8,
        ref_pool.key_scale,
        ref_pool.key_center,
        ref_pool.value_int8,
        ref_pool.value_scale,
        ref_pool.value_mean,
        slot_ids,
        pack_wmma=is_rdna4,
    )
    _assert_pools_close_triton(gpu_pool, ref_pool)
    for untouched in (1, 2, 4):
        _assert_slot_equal(gpu_pool, sentinel_pool, untouched)

    k3_rec, v3_rec = _dequantize_slot(gpu_pool, 3, pack_wmma=is_rdna4)
    _assert_relative_l2(k3_rec, key_proj[2], limit=0.03)
    _assert_relative_l2(v3_rec, val_proj[2], limit=0.03)
    assert torch.isfinite(gpu_pool.value_scale[0]).all()
    assert torch.isfinite(gpu_pool.key_scale[0]).all()

    q = torch.randn((1, 4, KV_CHUNK_TOKENS, head_dim), device=device, dtype=dtype)
    actual_attn = piper_attention(q, key_proj[2:3], val_proj[2:3], is_causal=True)
    reconstructed_attn = fn.scaled_dot_product_attention(
        q,
        k3_rec.to(dtype).repeat_interleave(2, dim=1),
        v3_rec.to(dtype).repeat_interleave(2, dim=1),
        is_causal=True,
    )
    torch.testing.assert_close(actual_attn, reconstructed_attn, atol=8e-2, rtol=5e-2)

    # Size-1 pool + kv_heads == 1 with non-div16 dummy strides launches Triton cleanly.
    raw_k = torch.empty((2, 1, KV_CHUNK_TOKENS, head_dim), device=device, dtype=kv_dtype)
    raw_ks = torch.empty((2, 1, key_scales), device=device, dtype=torch.float32)
    raw_kc = torch.empty((2, 1, head_dim), device=device, dtype=torch.float32)
    raw_v = torch.empty((2, 1, 16, head_dim, 64), device=device, dtype=kv_dtype)
    raw_vs = torch.empty((2, 1, KV_CHUNK_TOKENS), device=device, dtype=torch.float32)
    raw_vm = torch.empty((2, 1, head_dim), device=device, dtype=torch.float32)
    dummy_s1_pool = PiperKVChunkPool.from_tensors(
        raw_k[:1].as_strided((1, 1, KV_CHUNK_TOKENS, head_dim), (1, 1, head_dim, 1)),
        raw_ks[:1].as_strided((1, 1, key_scales), (1, 1, 1)),
        raw_kc[:1].as_strided((1, 1, head_dim), (1, 1, 1)),
        raw_v[:1].as_strided((1, 1, 16, head_dim, 64), (1, 1, head_dim * 64, 64, 1)),
        raw_vs[:1].as_strided((1, 1, KV_CHUNK_TOKENS), (1, 1, 1)),
        raw_vm[:1].as_strided((1, 1, head_dim), (1, 1, 1)),
    )
    _, s1_launches = _record_triton_launches(
        lambda: freeze_kv_chunks_(
            key_proj[:1, :1],
            val_proj[:1, :1],
            dummy_s1_pool,
            torch.tensor([0], device=device, dtype=torch.int32),
        )
    )
    assert "_quantize_key_k1024_kernel" in s1_launches
    assert "_quantize_value_k1024_kernel" in s1_launches


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
@pytest.mark.parametrize("ctx_factory", [torch.enable_grad, torch.no_grad, torch.inference_mode])
def test_gpu_rejects_target_scale_mismatch_and_inference_pool_without_mutating(
    ctx_factory: Callable[[], Any],
) -> None:
    device = torch.device("cuda")
    gpu_scales = kv_chunk_storage_specs(2, 2, 64, device=device)["key_scale"][0][2]
    wrong_scales = 1024 if gpu_scales == 16 else 16
    _, mismatch_pool = _make_interleaved_pool(
        2, 2, 64, device=device, kv_dtype=torch.uint8, key_scales=wrong_scales
    )
    _fill_pool_sentinels(mismatch_pool)
    snap = _clone_pool(mismatch_pool)
    versions_before = _pool_versions(mismatch_pool)
    k = torch.randn((1, 2, KV_CHUNK_TOKENS, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    s = torch.tensor([0], device=device, dtype=torch.int32)

    with ctx_factory(), forbid_triton_driver(), pytest.raises(ValueError, match="key_scales"):
        freeze_kv_chunks_(k, v, mismatch_pool, s)
    assert _pool_versions(mismatch_pool) == versions_before
    _assert_pools_exact(mismatch_pool, snap)

    with torch.inference_mode():
        inf_pool = PiperKVChunkPool.empty(2, 2, 64, device=device, kv_dtype=torch.uint8)
        _fill_pool_sentinels(inf_pool)
    inf_snap = _clone_pool(inf_pool)
    with pytest.raises(RuntimeError, match="outside InferenceMode"):
        freeze_kv_chunks_(k, v, inf_pool, s)
    _assert_pools_exact(inf_pool, inf_snap)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
def test_gpu_cudagraph_replay_and_empty_noop_triton_and_reference() -> None:
    device = torch.device("cuda")
    torch.manual_seed(404)
    is_rdna4 = rdna4_available()
    num_slots = 4
    kv_heads = 2
    head_dim = 64
    key_scales = kv_chunk_storage_specs(num_slots, kv_heads, head_dim, device=device)["key_scale"][
        0
    ][2]

    _, pool_triton = _make_interleaved_pool(
        num_slots, kv_heads, head_dim, device=device, kv_dtype=torch.uint8, key_scales=key_scales
    )
    _fill_pool_sentinels(pool_triton)
    pool_ref = _clone_pool(pool_triton)
    sentinel_pool = _clone_pool(pool_triton)

    k_buf = torch.randn(
        (2, kv_heads, KV_CHUNK_TOKENS, head_dim), device=device, dtype=torch.bfloat16
    )
    v_buf = torch.randn_like(k_buf)
    slots_buf = torch.tensor([1, -1], device=device, dtype=torch.int32)

    # num_new == 0 and num_slots == 0 must trigger zero Triton launches.
    zero_pool = PiperKVChunkPool.empty(0, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    _, zero_launches = _record_triton_launches(
        lambda: (
            freeze_kv_chunks_(k_buf[:0], v_buf[:0], pool_triton, slots_buf[:0]),
            freeze_kv_chunks_(k_buf, v_buf, zero_pool, slots_buf),
        )
    )
    assert zero_launches == []

    for _ in range(2):
        _, cap_launches = _record_triton_launches(
            lambda: freeze_kv_chunks_(k_buf, v_buf, pool_triton, slots_buf)
        )
        assert "_quantize_key_k1024_kernel" in cap_launches
        assert "_quantize_value_k1024_kernel" in cap_launches
        freeze_kv_chunks_reference(
            k_buf,
            v_buf,
            pool_ref.key_int8,
            pool_ref.key_scale,
            pool_ref.key_center,
            pool_ref.value_int8,
            pool_ref.value_scale,
            pool_ref.value_mean,
            slots_buf,
            pack_wmma=is_rdna4,
        )
    torch.cuda.synchronize()

    graph_triton = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_triton):
        freeze_kv_chunks_(k_buf, v_buf, pool_triton, slots_buf)

    graph_ref = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_ref):
        freeze_kv_chunks_reference(
            k_buf,
            v_buf,
            pool_ref.key_int8,
            pool_ref.key_scale,
            pool_ref.key_center,
            pool_ref.value_int8,
            pool_ref.value_scale,
            pool_ref.value_mean,
            slots_buf,
            pack_wmma=is_rdna4,
        )

    # Replay [0, -1] (slot 0 vs invalid -1 collision) while slots 2 & 3 remain sentinels.
    k_buf.copy_(torch.randn_like(k_buf))
    v_buf.copy_(torch.randn_like(v_buf))
    slots_buf.copy_(torch.tensor([0, -1], device=device, dtype=torch.int32))
    graph_triton.replay()
    graph_ref.replay()
    torch.cuda.synchronize()

    _assert_pools_close_triton(pool_triton, pool_ref)
    for untouched in (2, 3):
        _assert_slot_equal(pool_triton, sentinel_pool, untouched)
        _assert_slot_equal(pool_ref, sentinel_pool, untouched)

    # All-miss replay ([-1, num_slots]) leaves every slot completely unchanged.
    snap_triton = _clone_pool(pool_triton)
    snap_ref = _clone_pool(pool_ref)
    k_buf.copy_(torch.randn_like(k_buf))
    v_buf.copy_(torch.randn_like(v_buf))
    slots_buf.copy_(torch.tensor([-1, num_slots], device=device, dtype=torch.int32))
    graph_triton.replay()
    graph_ref.replay()
    torch.cuda.synchronize()
    _assert_pools_exact(pool_triton, snap_triton)
    _assert_pools_exact(pool_ref, snap_ref)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
def test_gpu_opcheck_and_standalone_dynamic_compile() -> None:
    device = torch.device("cuda")
    pool = PiperKVChunkPool.empty(4, 2, 64, device=device, kv_dtype=torch.uint8)
    k = torch.randn((2, 2, KV_CHUNK_TOKENS, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = torch.tensor([1, -1], device=device, dtype=torch.int32)
    res = torch.library.opcheck(
        torch.ops.piper_kernels.piper_kv_freeze_.default,
        (
            k,
            v,
            pool.key_int8,
            pool.key_scale,
            pool.key_center,
            pool.value_int8,
            pool.value_scale,
            pool.value_mean,
            slots,
        ),
        test_utils=_OPCHECK_UTILS,
    )
    assert set(res.values()) == {"SUCCESS"}

    is_rdna4 = rdna4_available()
    counter = CompileCounterWithBackend("inductor")

    def _fn(
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        ki8: torch.Tensor,
        ks: torch.Tensor,
        kc: torch.Tensor,
        vi8: torch.Tensor,
        vs: torch.Tensor,
        vm: torch.Tensor,
        s_in: torch.Tensor,
    ) -> torch.Tensor:
        dyn_pool = PiperKVChunkPool.from_tensors(ki8, ks, kc, vi8, vs, vm)
        freeze_kv_chunks_(k_in, v_in, dyn_pool, s_in)
        return ks.sum()

    compiled = torch.compile(_fn, backend=counter, fullgraph=True, dynamic=True)
    with torch.inference_mode():
        for num_slots, slot_list in ((5, [0, 3, 1]), (8, [5, -1, 2, 0])):
            p = PiperKVChunkPool.empty(num_slots, 2, 64, device=device, kv_dtype=torch.uint8)
            _fill_pool_sentinels(p)
            p_ref = _clone_pool(p)
            kk = torch.randn(
                (len(slot_list), 2, KV_CHUNK_TOKENS, 64), device=device, dtype=torch.bfloat16
            )
            vv = torch.randn_like(kk)
            ss = torch.tensor(slot_list, device=device, dtype=torch.int32)
            _, launches = _record_triton_launches(
                lambda kk=kk, vv=vv, p=p, ss=ss: compiled(
                    kk,
                    vv,
                    p.key_int8,
                    p.key_scale,
                    p.key_center,
                    p.value_int8,
                    p.value_scale,
                    p.value_mean,
                    ss,
                )
            )
            assert "_quantize_key_k1024_kernel" in launches
            assert "_quantize_value_k1024_kernel" in launches
            freeze_kv_chunks_reference(
                kk,
                vv,
                p_ref.key_int8,
                p_ref.key_scale,
                p_ref.key_center,
                p_ref.value_int8,
                p_ref.value_scale,
                p_ref.value_mean,
                ss,
                pack_wmma=is_rdna4,
            )
            _assert_pools_close_triton(p, p_ref)
    assert counter.frame_count == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm GPU")
def test_gpu_torch_compile_shared_uint8_moe_and_kv_arena_fullgraph_dynamic() -> None:
    device = torch.device("cuda")
    torch.manual_seed(505)
    is_rdna4 = rdna4_available()
    kv_heads = 2
    head_dim = 64
    hidden_size = 256
    num_kv_slots = 7
    key_scales = kv_chunk_storage_specs(num_kv_slots, kv_heads, head_dim, device=device)[
        "key_scale"
    ][0][2]

    arena, kv_start, moe_weights, kv_pool = _build_shared_arena_moe_and_kv(
        device=device,
        num_kv_slots=num_kv_slots,
        kv_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=256,
        num_experts=2,
        key_scales=key_scales,
    )
    _fill_pool_sentinels(kv_pool)
    moe_bytes_snapshot = arena[:kv_start].clone()

    counter = CompileCounterWithBackend("inductor")

    def _step(
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = moe_ffn(x, moe_weights, topk_ids, topk_weights)
        freeze_kv_chunks_(k, v, kv_pool, slot_ids)
        return out

    compiled_step = torch.compile(_step, backend=counter, fullgraph=True, dynamic=True)

    with torch.inference_mode():
        # Keep dynamic dims >= 2 and distinct from fixed 2-expert / 2-head sizes (duck sizing).
        for num_tokens, slot_list in ((3, [0, 3, 1, 5, -1]), (4, [2, -1, 1, 6, 0, 4])):
            x = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
            topk_ids = torch.zeros((num_tokens, 1), device=device, dtype=torch.int32)
            topk_weights = torch.ones((num_tokens, 1), device=device, dtype=torch.float32)
            k = torch.randn(
                (len(slot_list), kv_heads, KV_CHUNK_TOKENS, head_dim),
                device=device,
                dtype=torch.bfloat16,
            )
            v = torch.randn_like(k)
            slots = torch.tensor(slot_list, device=device, dtype=torch.int32)

            expected_moe = moe_ffn(x, moe_weights, topk_ids, topk_weights)
            expected_kv = _clone_pool(kv_pool)
            freeze_kv_chunks_reference(
                k,
                v,
                expected_kv.key_int8,
                expected_kv.key_scale,
                expected_kv.key_center,
                expected_kv.value_int8,
                expected_kv.value_scale,
                expected_kv.value_mean,
                slots,
                pack_wmma=is_rdna4,
            )

            actual_moe, launches = _record_triton_launches(
                lambda x=x, topk_ids=topk_ids, topk_weights=topk_weights, k=k, v=v, slots=slots: (
                    compiled_step(x, topk_ids, topk_weights, k, v, slots)
                )
            )
            assert "_quantize_key_k1024_kernel" in launches
            assert "_quantize_value_k1024_kernel" in launches
            torch.testing.assert_close(actual_moe, expected_moe, atol=1e-2, rtol=1e-2)
            _assert_pools_close_triton(kv_pool, expected_kv)
            torch.testing.assert_close(arena[:kv_start], moe_bytes_snapshot, atol=0, rtol=0)
    assert counter.frame_count == 1
