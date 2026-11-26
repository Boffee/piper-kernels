"""Package-boundary smoke tests."""

import importlib
import subprocess
import sys

import pytest

import piper_kernels
import piper_kernels.attention
import piper_kernels.attention.piper_attention
import piper_kernels.attention.sage_attention_2pp
import piper_kernels.attention.sparse_piper_attention
import piper_kernels.conv3d.convrot.int8
import piper_kernels.fusions
import piper_kernels.fusions.convrot_int8_piper
import piper_kernels.fusions.convrot_int8_sparse_piper
import piper_kernels.fusions.ffn
import piper_kernels.fusions.nvfp4_sparse_piper
import piper_kernels.fusions.residual_updates
import piper_kernels.linear
import piper_kernels.linear.convrot
import piper_kernels.linear.convrot.int8
import piper_kernels.linear.convrot.nvfp4
import piper_kernels.linear.nvfp4
from piper_kernels import (
    PiperKVChunkPool,
    SparsePiperAttention,
    apply_coarse_attention_residual,
    coarse_attention,
    coarse_attention_residual,
    compile_options,
    freeze_kv_chunks_,
    kv_chunk_storage_specs,
    mean_pool_block_values,
    merge_attention_lse,
    piper_attention,
    piper_attention_decode,
    sage_attention_2pp,
    sparse_piper_coarse_residual,
)
from piper_kernels.linear.convrot import convrot_int8_linear
from piper_kernels.linear.convrot.nvfp4 import convrot_nvfp4_linear
from piper_kernels.weights import QuantizedWeight, sharding
from piper_kernels.weights.convrot import SUPPORTED_GROUP_SIZES
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor


def test_removed_convrot_root_package_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module("piper_kernels.convrot")

    assert error.value.name == "piper_kernels.convrot"


def test_convrot_int8_shared_ops_preserve_cpu_imports_without_triton() -> None:
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
orig_cap = torch.cuda.get_device_capability
orig_prop = torch.cuda.get_device_properties
torch.cuda.get_device_capability = lambda *_a: (_ for _ in ()).throw(AssertionError("probe"))
torch.cuda.get_device_properties = lambda *_a: (_ for _ in ()).throw(AssertionError("probe"))
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.linear.convrot import convrot_int8_linear
from piper_kernels.linear.convrot.int8 import _backend, _generic, _ops
import piper_kernels.fusions.convrot_int8_sparse_piper._backend as sparse_proj_backend
from piper_kernels._residual import ResidualAdd, validate_residual_add
from piper_kernels._distributed import _fx as distributed_fx
assert hasattr(distributed_fx, "has_c10d_collectives")
residual_input = torch.ones(2, 32)
validate_residual_add(
    residual_input,
    ResidualAdd(residual_input, torch.ones(1, 32), torch.zeros(2, dtype=torch.int64), False),
    32,
)

weight = ConvRotInt8Tensor.from_quantized(
    torch.ones(7, 32, dtype=torch.int8), torch.ones(7, 1),
    group_size=16, logical_dtype=torch.float32,
)
result = convrot_int8_linear(torch.ones(2, 32), weight)
assert result.shape == (2, 7)
assert _backend.select_linear_backend(torch.ones(1)) is None
assert _backend.select_linear_backend_for_target(AcceleratorTarget("cuda", "sm120")) is None
assert _backend.select_linear_backend_for_target(AcceleratorTarget("hip", "gfx1201")) is None
assert sparse_proj_backend.select_projection_backend(torch.ones(1)) is None
assert (
    sparse_proj_backend.select_projection_backend(
        torch.ones(1), target=AcceleratorTarget("cuda", "sm120")
    )
    is None
)
assert "triton" not in sys.modules
assert hasattr(torch.ops.piper_kernels, "convrot_int8_prepare_input")
assert _generic._triton_backend is None
from piper_kernels.fusions.convrot_int8_projection._validation import validate_qk_projection_inputs
assert validate_qk_projection_inputs(
    torch.ones(1, 64, 32, dtype=torch.int8),
    torch.ones(1, 64),
    torch.ones(128, 32, dtype=torch.int8),
    torch.ones(128, 1),
    torch.ones(128, dtype=torch.bfloat16),
    torch.ones(64, 96),
    torch.zeros(64, 96),
    norm_epsilon=1e-6,
    name="Q",
) == (1, 64, 1)
assert "piper_kernels.fusions.convrot_int8_projection.triton" not in sys.modules
assert "piper_kernels.fusions.attention.triton" not in sys.modules
assert "piper_kernels.fusions.attention._projected_qk_compile" not in sys.modules
assert "piper_kernels.linear._compile_fx" not in sys.modules
assert "triton" not in sys.modules
from piper_kernels.attention.piper_attention._nvidia import policy as n_pol
from piper_kernels.attention.sage_attention_2pp import _policy as s_pol
sm120 = AcceleratorTarget("cuda", "sm120")
n_plan = n_pol.select_execution_plan(sm120, head_dim=128, is_causal=False, query_length=128)
s_plan = s_pol.select_execution_plan(sm120, candidate_block_m=128, head_dim=128, is_causal=False)
assert isinstance(n_plan, n_pol.PiperAttentionExecutionPlan)
assert isinstance(s_plan, s_pol.SageAttention2ppExecutionPlan)
assert "triton" not in sys.modules
prepared, scales = _ops.prepare_input(torch.ones(2, 32), 16)
assert prepared.shape == (2, 32) and scales.shape == (2,)
weight.add_(torch.ones(7, 32))
weight.addmm_(torch.ones(7, 2), torch.ones(2, 32))
assert torch.isfinite(weight.dequantize()).all()
assert "triton" not in sys.modules
torch.cuda.get_device_capability, torch.cuda.get_device_properties = orig_cap, orig_prop
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor
nv = PiperNVFP4Tensor.from_hp(torch.ones(4, 32, dtype=torch.bfloat16))
nv.add_(torch.ones(4, 32, dtype=torch.bfloat16))
nv.addmm_(torch.ones(4, 2, dtype=torch.bfloat16), torch.ones(2, 32, dtype=torch.bfloat16))
assert torch.isfinite(nv.dequantize()).all() and "triton" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_compile_options_is_lazy_and_triton_torchao_free() -> None:
    script = """
import builtins
import importlib.util
import sys

original_import = builtins.__import__
original_find_spec = importlib.util.find_spec
blocked = ("triton", "torchao")

def guarded_import(name, *args, **kwargs):
    if any(name == pkg or name.startswith(pkg + ".") for pkg in blocked):
        raise ModuleNotFoundError(f"{name} intentionally unavailable", name=name)
    return original_import(name, *args, **kwargs)

def guarded_find_spec(name, *args, **kwargs):
    if any(name == pkg or name.startswith(pkg + ".") for pkg in blocked):
        return None
    return original_find_spec(name, *args, **kwargs)

builtins.__import__ = guarded_import
importlib.util.find_spec = guarded_find_spec

import piper_kernels
assert "piper_kernels._inductor_options" not in sys.modules

options = piper_kernels.compile_options(
    linear=True,
    ffn=True,
    attention=True,
    attention_output=True,
    residual_updates=True,
    conv3d=True,
)
assert options["allow_buffer_reuse"] is False
post = options["post_grad_custom_pre_pass"]
pre = options["pre_grad_custom_pass"]
assert isinstance(post, tuple) and len(post) == 1
assert isinstance(pre, tuple) and len(pre) == 1
assert post[0].uuid() != pre[0].uuid()

for forbidden_prefix in (
    "triton",
    "torchao",
    "piper_kernels.linear",
    "piper_kernels.conv3d",
    "piper_kernels.fusions",
    "piper_kernels.attention.sparse_piper_attention._quantized_dispatch",
):
    assert not any(
        mod == forbidden_prefix or mod.startswith(forbidden_prefix + ".")
        for mod in sys.modules
    ), f"unexpected eager import of {forbidden_prefix}"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_kv_cache_cpu_fallback_remains_usable_without_triton() -> None:
    script = """
import builtins
import importlib.util
import sys
import torch

original_import = builtins.__import__
original_find_spec = importlib.util.find_spec

def guarded_import(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        raise ModuleNotFoundError(f"{name} intentionally unavailable", name=name)
    return original_import(name, *args, **kwargs)

def guarded_find_spec(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        return None
    return original_find_spec(name, *args, **kwargs)

builtins.__import__ = guarded_import
importlib.util.find_spec = guarded_find_spec

from piper_kernels import (
    PiperKVChunkPool,
    freeze_kv_chunks_,
    kv_chunk_storage_specs,
    merge_attention_lse,
    piper_attention_decode,
)

specs = kv_chunk_storage_specs(2, 2, 64, device="cpu", kv_dtype=torch.uint8)
assert specs["key_scale"] == ((2, 2, 16), torch.float32)
pool = PiperKVChunkPool.empty(2, 2, 64, device="cpu", kv_dtype=torch.uint8)
key = torch.randn(1, 2, 1024, 64, dtype=torch.bfloat16)
value = torch.randn_like(key)
slots = torch.tensor([1], dtype=torch.int32)
freeze_kv_chunks_(key, value, pool, slots)
q = torch.randn(1, 4, 2, 64, dtype=torch.bfloat16)
c2s = torch.tensor([[1]], dtype=torch.int32)
nc = torch.tensor([1], dtype=torch.int32)
tk = torch.randn(1, 2, 4, 64, dtype=torch.bfloat16)
tv = torch.randn_like(tk)
tl = torch.tensor([4], dtype=torch.int32)
out, lse = piper_attention_decode(
    q, pool, c2s, nc, tail_key=tk, tail_value=tv, tail_len=tl, return_lse=True
)
merged_out, merged_lse = merge_attention_lse(
    [out, torch.full_like(out, float("nan"))],
    [lse, torch.full_like(lse, -float("inf"))],
)
assert merged_out.shape == q.shape
assert merged_lse.shape == lse.shape
assert torch.isfinite(merged_out).all()
assert torch.isfinite(merged_lse).all()
assert "triton" not in sys.modules
assert not any(mod == "triton" or mod.startswith("triton.") for mod in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("family", ["sparse_piper", "projection"])
def test_convrot_int8_fusion_packages_have_explicit_names(family: str) -> None:
    current = f"piper_kernels.fusions.convrot_int8_{family}"
    assert importlib.import_module(current).__name__ == current
    previous = f"piper_kernels.fusions.convrot_{family}"
    with pytest.raises(ModuleNotFoundError) as error:
        importlib.import_module(previous)
    assert error.value.name == previous


def test_subpackages_keep_compiler_modules_lazy_on_import() -> None:
    script = """
import sys
import piper_kernels.linear.convrot
import piper_kernels.linear.nvfp4
import piper_kernels.linear.convrot.nvfp4
import piper_kernels.fusions.ffn
import piper_kernels.fusions.residual_updates
import piper_kernels.fusions.convrot_int8_piper
import piper_kernels.fusions.convrot_int8_sparse_piper
import piper_kernels.fusions.nvfp4_sparse_piper

for forbidden in (
    "piper_kernels._inductor_options",
    "piper_kernels.linear._preparation_sharing",
    "piper_kernels.linear.convrot.int8._compile",
    "piper_kernels.linear.convrot.int8._distributed_compile",
    "piper_kernels.linear.nvfp4._compile",
    "piper_kernels.linear.convrot.nvfp4._compile",
    "piper_kernels.fusions.ffn._compile",
    "piper_kernels.fusions.residual_updates._compile",
    "piper_kernels.fusions.convrot_int8_piper._compile",
    "piper_kernels.fusions.convrot_int8_piper._output_compile",
    "piper_kernels.fusions.convrot_int8_sparse_piper._compile",
    "piper_kernels.fusions.convrot_int8_sparse_piper._output_compile",
    "piper_kernels.fusions.convrot_int8_sparse_piper._distributed_compile",
    "piper_kernels.fusions.nvfp4_sparse_piper._compile",
    "piper_kernels.fusions.nvfp4_sparse_piper._output_compile",
    "piper_kernels._distributed._fx",
    "piper_kernels._distributed._peer_exchange",
):
    assert forbidden not in sys.modules, f"unexpected eager import: {forbidden}"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_sparse_attention_does_not_load_convrot() -> None:
    script = """
import sys
import piper_kernels.attention.sparse_piper_attention

assert "piper_kernels.linear.convrot" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_convrot_int8_projection_does_not_load_sparse_piper_fusion() -> None:
    script = """
import sys
import piper_kernels.fusions.convrot_int8_projection.triton

assert "piper_kernels.fusions.convrot_int8_sparse_piper._compile" not in sys.modules
assert "piper_kernels.fusions.convrot_int8_sparse_piper.query" not in sys.modules
assert "piper_kernels.fusions.convrot_int8_sparse_piper.key" not in sys.modules
assert "piper_kernels.attention.sparse_piper_attention._quantized_dispatch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_packages_import() -> None:
    assert piper_kernels.__name__ == "piper_kernels"
    assert piper_kernels.PiperKVChunkPool is PiperKVChunkPool
    assert piper_kernels.compile_options is compile_options
    assert piper_kernels.freeze_kv_chunks_ is freeze_kv_chunks_
    assert piper_kernels.kv_chunk_storage_specs is kv_chunk_storage_specs
    assert piper_kernels.merge_attention_lse is merge_attention_lse
    assert piper_kernels.piper_attention is piper_attention
    assert piper_kernels.piper_attention_decode is piper_attention_decode
    assert piper_kernels.sage_attention_2pp is sage_attention_2pp
    assert piper_kernels.SparsePiperAttention is SparsePiperAttention
    assert piper_kernels.apply_coarse_attention_residual is apply_coarse_attention_residual
    assert piper_kernels.coarse_attention is coarse_attention
    assert piper_kernels.coarse_attention_residual is coarse_attention_residual
    assert piper_kernels.mean_pool_block_values is mean_pool_block_values
    assert piper_kernels.sparse_piper_coarse_residual is sparse_piper_coarse_residual
    assert piper_kernels.__all__ == [
        "PiperKVChunkPool",
        "SparsePiperAttention",
        "apply_coarse_attention_residual",
        "coarse_attention",
        "coarse_attention_residual",
        "compile_options",
        "freeze_kv_chunks_",
        "kv_chunk_storage_specs",
        "mean_pool_block_values",
        "merge_attention_lse",
        "piper_attention",
        "piper_attention_decode",
        "sage_attention_2pp",
        "sparse_piper_coarse_residual",
    ]
    assert piper_kernels.attention.__name__ == "piper_kernels.attention"
    assert piper_kernels.attention.piper_attention.__name__ == (
        "piper_kernels.attention.piper_attention"
    )
    assert piper_kernels.attention.sage_attention_2pp.__name__ == (
        "piper_kernels.attention.sage_attention_2pp"
    )
    assert piper_kernels.attention.sparse_piper_attention.__name__ == (
        "piper_kernels.attention.sparse_piper_attention"
    )
    assert piper_kernels.fusions.__name__ == "piper_kernels.fusions"
    assert piper_kernels.fusions.nvfp4_sparse_piper.__name__ == (
        "piper_kernels.fusions.nvfp4_sparse_piper"
    )
    assert piper_kernels.fusions.convrot_int8_sparse_piper.__name__ == (
        "piper_kernels.fusions.convrot_int8_sparse_piper"
    )
    assert piper_kernels.linear.__name__ == "piper_kernels.linear"
    assert piper_kernels.linear.convrot.__name__ == "piper_kernels.linear.convrot"
    assert piper_kernels.linear.convrot.int8.__name__ == "piper_kernels.linear.convrot.int8"
    assert piper_kernels.linear.convrot.nvfp4.__name__ == "piper_kernels.linear.convrot.nvfp4"
    assert piper_kernels.weights.convrot.SUPPORTED_GROUP_SIZES is SUPPORTED_GROUP_SIZES
    assert piper_kernels.weights.convrot.int8.ConvRotInt8Tensor is ConvRotInt8Tensor
    assert piper_kernels.weights.convrot.nvfp4.ConvRotNVFP4Tensor is ConvRotNVFP4Tensor
    assert piper_kernels.linear.convrot.convrot_int8_linear is convrot_int8_linear
    assert piper_kernels.linear.convrot.nvfp4.convrot_nvfp4_linear is convrot_nvfp4_linear
    assert convrot_int8_linear.__module__ == "piper_kernels.linear.convrot.int8._functional"
    assert not hasattr(piper_kernels.linear.convrot, "convrot_linear")
    assert not hasattr(piper_kernels.linear.convrot, "convrot_compile_options")
    assert not hasattr(piper_kernels.linear.convrot, "convrot_int8_compile_options")
    assert not hasattr(piper_kernels.linear.nvfp4, "nvfp4_compile_options")
    assert not hasattr(piper_kernels.linear.convrot.nvfp4, "convrot_nvfp4_compile_options")
    assert not hasattr(piper_kernels.conv3d.convrot.int8, "convrot_int8_conv3d_compile_options")
    assert not hasattr(piper_kernels.fusions.ffn, "ffn_compile_options")
    assert not hasattr(piper_kernels.fusions.residual_updates, "residual_updates_compile_options")
    assert not hasattr(
        piper_kernels.fusions.convrot_int8_piper, "convrot_int8_piper_compile_options"
    )
    assert not hasattr(
        piper_kernels.fusions.convrot_int8_sparse_piper, "convrot_int8_sparse_piper_compile_options"
    )
    assert not hasattr(
        piper_kernels.fusions.nvfp4_sparse_piper, "nvfp4_sparse_piper_compile_options"
    )
    assert not hasattr(piper_kernels.linear.convrot.int8, "convrot_int8_linear")
    assert not hasattr(piper_kernels.linear.convrot, "linear_input_act")
    assert not hasattr(piper_kernels.linear.convrot.int8, "linear_input_act")
    assert piper_kernels.linear.convrot.__all__ == [
        "convrot_int8_linear",
    ]
    assert not hasattr(piper_kernels.linear.convrot.int8, "ConvRotInt8Tensor")
    assert piper_kernels.linear.convrot.nvfp4.__all__ == [
        "convrot_nvfp4_linear",
        "dynamic_scale",
        "prepare_dynamic",
        "prepare_static",
        "prepare_static_out",
    ]


def test_weights_public_exports() -> None:
    assert piper_kernels.weights.__all__ == ["QuantizedWeight"]
    assert piper_kernels.weights.QuantizedWeight is QuantizedWeight
    assert sharding.__all__ == [
        "ByteRange",
        "shard_quantized_weight",
        "trellis_row_shard_byte_range",
    ]
    assert hasattr(sharding, "ByteRange")
    assert hasattr(sharding, "shard_quantized_weight")
    assert hasattr(sharding, "trellis_row_shard_byte_range")
