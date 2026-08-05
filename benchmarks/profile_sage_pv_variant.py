"""Measure one fixed-schedule Sage PV variant and inspect its generated code."""

# Profiling launchers intentionally expose each controlled schedule dimension.
# Triton JIT pointer arguments intentionally have no Python runtime types.
# ruff: noqa: ANN001, ANN202, PLR0912, PLR0913

import argparse
import json
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Literal

import torch
import triton
import triton.language as tl
import triton.testing
from triton.compiler.compiler import CompiledKernel
from triton.runtime.jit import JITFunction
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.experiments.int8_pv import (
    _int8_pv_attention_kernel,
    _launch_int8_pv_attention,
    _launch_uint8_grouped_output_pv_attention,
    _launch_uint8_k32_feature_pv_attention,
    _launch_uint8_run_scaled_output_pv_attention,
    _prepare_block_int8_pv_inputs,
    _prepare_int8_pv_inputs,
    _prepare_uint8_grouped_output_pv_inputs,
    _prepare_uint8_k32_feature_pv_inputs,
    _uint8_grouped_output_pv_attention_kernel,
    _uint8_k32_feature_pv_attention_kernel,
    _uint8_run_scaled_output_pv_attention_kernel,
)
from piper_kernels.attention._sage2pp.experiments.uint8_pv_feature_convrot import (
    _launch_uint8_pv_feature_convrot_attention,
    _prepare_uint8_pv_feature_convrot_inputs,
    _uint8_pv_feature_convrot_attention_kernel,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        choices=[
            "fp8-fp16",
            "fp8-fp16-transposed",
            "fp8-fp16-descriptor",
            "fp8-fp32",
            "fp8-fp32-transposed",
            "fp8-fp32-descriptor",
            "int8-fixed",
            "int8-fixed-transposed",
            "int8-fixed-descriptor",
            "int8-fixed-int32-descriptor",
            "int8-fixed-int32-raw-descriptor",
            "int8-fixed-fp32-raw-descriptor",
            "int8-fixed-magic-pv-descriptor",
            "int8-fixed-magic-all-descriptor",
            "int8-fixed-fp16-convert-descriptor",
            "int8-fixed-bf16-convert-descriptor",
            "int8-fixed-unmasked-descriptor",
            "int8-fixed-unmasked-split-descriptor",
            "int8-fixed-uint8-unmasked-split-descriptor",
            "int8-fixed-pair-descriptor",
            "int8-fixed-pair-split-descriptor",
            "int8-fixed-pair-prequantized-split-descriptor",
            "int8-fixed-split-transposed",
            "int8-fixed-split-descriptor",
            "int8-block",
            "int8-block-transposed",
            "int8-block-descriptor",
            "int8-block-unmasked-descriptor",
            "int8-block-unmasked-split-descriptor",
            "int8-block-uint8-unmasked-split-descriptor",
            "int8-block-global-p-unmasked-descriptor",
            "int8-block-uint8-global-p-unmasked-descriptor",
            "int8-block-global-p-unmasked-split-descriptor",
            "int8-block-uint8-global-p-unmasked-split-descriptor",
            "int8-output-k32-feature-native-descriptor",
            "int8-output-group1-native-descriptor",
            "int8-output-group1-run256-global-p-unmasked-native-descriptor",
            "int8-output-group1-run512-global-p-unmasked-native-descriptor",
            "int8-output-group1-run512-dominant-unmasked-native-descriptor",
            "int8-output-group1-run512-scaled-fp16-numerator-unmasked-native-descriptor",
            "int8-output-group1-run1024-global-p-unmasked-native-descriptor",
            "int8-output-group4-native-descriptor",
            "int8-output-group4-unmasked-native-descriptor",
            "int8-output-group4-run256-native-descriptor",
            "int8-output-group4-run256-global-p-unmasked-native-descriptor",
            "int8-output-group4-run512-native-descriptor",
            "int8-output-group4-run512-unmasked-native-descriptor",
            "int8-output-group4-run512-scaled-fp16-numerator-unmasked-native-descriptor",
            "int8-output-group4-run512-global-p-unmasked-native-descriptor",
            "int8-output-group4-run512-dominant-unmasked-native-descriptor",
            "int8-output-group4-run1024-native-descriptor",
            "int8-output-group4-run1024-global-p-unmasked-native-descriptor",
            "int8-output-group4-int32-native-descriptor",
            "int8-output-group4-int32-commonexp-native-descriptor",
            "int8-output-group8-native-descriptor",
            "int8-output-group8-run512-native-descriptor",
            "int8-output-group8-int32-native-descriptor",
            "int8-output-group8-int32-commonexp-native-descriptor",
            "int8-output-group16-native-descriptor",
            "int8-output-group16-run512-native-descriptor",
            "int8-output-group32-native-descriptor",
            "int8-output-group32-run512-native-descriptor",
            "int8-output-group64-native-descriptor",
            "int8-output-group64-run512-native-descriptor",
            "int8-output-scalar-native-descriptor",
            "int8-log",
            "int8-log-transposed",
            "int8-log-descriptor",
            "int8-log-unweighted-transposed",
            "int8-log-signed-transposed",
            "int8-log-signed-descriptor",
            "int8-log-native-descriptor",
            "int8-log-scale-forward-descriptor",
            "int8-log-scale-forward-native-descriptor",
            "int8-log-scale-forward-precomputed-pv-scale-fp32-metadata-scaled-fp16-numerator-unmasked-native-descriptor",
            "int8-log-split-descriptor",
            "int8-log-split-unweighted-descriptor",
            "int8-log-split-unshifted-descriptor",
            "int8-log-split-running-max-descriptor",
            "int8-log-split-scale-forward-descriptor",
            "int8-log-split-narrow-denom-descriptor",
            "int8-log-split-tile-common-descriptor",
            "int8-log-split-native-descriptor",
            "int8-log-split-unweighted-native-descriptor",
            "int8-log-split-unshifted-native-descriptor",
            "int8-log-split-unshifted-paired-native-descriptor",
            "int8-log-split-running-max-native-descriptor",
            "int8-log-split-scale-forward-native-descriptor",
            "int8-log-split-scale-forward-fp16-pv-scale-native-descriptor",
            "int8-log-split-scale-forward-factored-pv-scale-native-descriptor",
            "int8-log-split-scale-forward-factored-pv-scale-fp32-metadata-native-descriptor",
            "int8-log-split-scale-forward-precomputed-pv-scale-native-descriptor",
            "int8-log-split-scale-forward-precomputed-pv-scale-scaled-fp16-numerator-unmasked-native-descriptor",
            "int8-log-split-scale-forward-precomputed-pv-scale-fp32-metadata-native-descriptor",
            "int8-log-split-scale-forward-precomputed-pv-scale-fp32-metadata-unmasked-native-descriptor",
            "int8-log-split-scale-forward-precomputed-pv-scale-fp32-metadata-scaled-fp16-numerator-unmasked-native-descriptor",
            "int8-log-split-scale-forward-precomputed-pv-scale-fp32-metadata-scaled-fp16-numerator-scaled-fp16-denominator-unmasked-native-descriptor",
            "int8-log-split-scale-forward-precomputed-pv-scale-fp32-metadata-k32-immediate-native-descriptor",
            "int8-log-split-scale-forward-precomputed-pv-scale-fp32-metadata-scale-descriptor-native-descriptor",
            "int8-log-split-scale-forward-fp32-metadata-native-descriptor",
            "int8-log-split-scale-forward-no-pv-scale-native-descriptor",
            "int8-log-split-scale-forward-paired-native-descriptor",
            "int8-log-split-scale-forward-pair-p-native-descriptor",
            "int8-log-split-scale-forward-pair-p-precomputed-pv-scale-native-descriptor",
            "int8-log-split-scale-forward-pair-p-precomputed-pv-scale-fp32-metadata-native-descriptor",
            "int8-log-split-scale-forward-pair-p-precomputed-pv-scale-scaled-fp16-numerator-native-descriptor",
            "int8-log-split-scale-forward-sampled-pair-p-native-descriptor",
            "int8-log-split-narrow-denom-native-descriptor",
            "int8-log-split-tile-common-native-descriptor",
            "int8-log-paired-native-descriptor",
            "int8-log-split-paired-native-descriptor",
            "int8-log-native-fp16p-descriptor",
            "int8-log-split-native-fp16p-descriptor",
            "int8-log-native-fp16norm-descriptor",
            "int8-log-split-native-fp16norm-descriptor",
            "int8-log-split-int32-lazy-h0-native-descriptor",
            "int8-log-split-int32-lazy-h1-native-descriptor",
            "int8-log-split-int32-lazy-h2-native-descriptor",
            "int8-log-int32-affine-transposed",
            "int8-log-int32-signed-descriptor",
            "int8-log-int32-native-descriptor",
            "int8-log-int32-tile-native-descriptor",
            "int8-log-int32-tile-single-shift-native-descriptor",
            "int8-log-int32-tile-scale-forward-single-shift-precomputed-pv-scale-fp32-metadata-native-descriptor",
            "int8-log-split-int32-tile-scale-forward-single-shift-precomputed-pv-scale-fp32-metadata-native-descriptor",
            "int8-log-split-int32-tile-scale-forward-predot-nearest-precomputed-pv-scale-fp32-metadata-native-descriptor",
            "int8-log-split-int32-tile-scale-forward-predot-dithered-precomputed-pv-scale-fp32-metadata-native-descriptor",
            "int8-affine-block-transposed",
            "int8-affine-block-native-descriptor",
            "int8-per-key-dynamic-transposed",
            "int8-per-key-tile-transposed",
        ],
    )
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--block-m", type=int, choices=[32, 64, 128, 256], default=64)
    parser.add_argument("--block-n", type=int, choices=[64, 128, 256], default=64)
    parser.add_argument("--num-stages", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--num-warps", type=int, choices=[4, 8], default=4)
    parser.add_argument("--maxnreg", type=int, default=None)
    parser.add_argument(
        "--sampled-headroom-log2",
        type=float,
        default=0.0,
        help="Log2 headroom above the sampled K64 maximum for sampled K128 normalization.",
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--repeat-ms", type=int, default=1000)
    parser.add_argument(
        "--compiler-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report registers, shared memory, and static SASS opcode counts.",
    )
    return parser.parse_args(argv)


def _compiled_kernel(jit_kernel: JITFunction) -> CompiledKernel:
    """Return the sole specialization compiled by this fresh benchmark process."""
    device_cache = next(iter(jit_kernel.device_caches.values()))
    specialization_cache = device_cache[0]
    kernels = list(specialization_cache.values())
    if len(kernels) != 1:
        raise RuntimeError(f"expected one compiled specialization, found {len(kernels)}")
    return kernels[0]


def _sass_counts(cubin: bytes) -> tuple[int, dict[str, int], dict[str, int]]:
    """Disassemble a cubin and summarize static instruction opcodes."""
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(cubin)
        cubin_file.flush()
        result = subprocess.run(
            ["/usr/local/cuda/bin/nvdisasm", "--print-code", cubin_file.name],
            check=True,
            capture_output=True,
            text=True,
        )

    instruction_pattern = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)")
    full = Counter(instruction_pattern.findall(result.stdout))
    families: Counter[str] = Counter()
    for opcode, count in full.items():
        families[opcode.split(".", maxsplit=1)[0]] += count
    selected_families = {
        family: families[family]
        for family in (
            "IMMA",
            "QMMA",
            "I2FP",
            "IADD3",
            "IMAD",
            "IMNMX",
            "SHF",
            "LOP3",
            "HADD2",
            "HMUL2",
            "HFMA2",
            "F2FP",
            "FFMA",
            "FADD",
            "FMUL",
            "MUFU",
            "PRMT",
            "LDSM",
            "LDS",
            "STS",
            "LDL",
            "STL",
            "LDG",
            "STG",
            "BAR",
            "DEPBAR",
        )
        if families[family]
    }
    selected_mma = {
        opcode: count
        for opcode, count in sorted(full.items())
        if opcode.startswith(("IMMA", "QMMA"))
    }
    return sum(full.values()), selected_families, selected_mma


def _compiler_report(jit_kernel: JITFunction, latency_ms: float) -> dict[str, object]:
    compiled = _compiled_kernel(jit_kernel)
    metadata = compiled.metadata
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    threads_per_cta = metadata.num_warps * properties.warp_size
    register_blocks = properties.regs_per_multiprocessor // (compiled.n_regs * threads_per_cta)
    shared_blocks = (
        properties.shared_memory_per_multiprocessor // metadata.shared
        if metadata.shared
        else properties.max_threads_per_multi_processor // threads_per_cta
    )
    thread_blocks = properties.max_threads_per_multi_processor // threads_per_cta
    resident_blocks_ceiling = min(register_blocks, shared_blocks, thread_blocks)
    static_instructions, instruction_families, mma_opcodes = _sass_counts(compiled.asm["cubin"])
    return {
        "latency_ms": latency_ms,
        "registers_per_thread": compiled.n_regs,
        "spills": compiled.n_spills,
        "shared_bytes_per_cta": metadata.shared,
        "warps_per_cta": metadata.num_warps,
        "resident_ctas_per_sm_ceiling": resident_blocks_ceiling,
        "resident_warps_per_sm_ceiling": resident_blocks_ceiling * metadata.num_warps,
        "static_sass_instructions": static_instructions,
        "sass_families": instruction_families,
        "mma_opcodes": mma_opcodes,
    }


def _fp8_launcher(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    *,
    accumulator_fp32: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    num_warps: int,
    value_transposed: bool,
    use_tensor_descriptors: bool,
) -> Callable[[], torch.Tensor]:
    batch, heads, sequence, head_dim = query.shape
    prepared = _prepare_int8_pv_inputs(query, key, value, scale, grouped_qk=True)
    query_int8, key_int8, _, query_scale, key_scale, folded_scale = prepared
    value_fp8_shape = (batch, heads, head_dim, sequence) if value_transposed else value.shape
    value_fp8 = torch.empty(value_fp8_shape, device=value.device, dtype=torch.float8_e4m3fn)
    _sage_backend._quantize_value_kernel[(triton.cdiv(sequence, block_n), heads, batch)](
        value,
        folded_scale,
        value_fp8,
        sequence,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value_fp8.stride(0),
        value_fp8.stride(1),
        value_fp8.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=block_n,
        output_transposed=value_transposed,
        num_warps=4,
    )

    key_argument: torch.Tensor | TensorDescriptor = key_int8
    value_argument: torch.Tensor | TensorDescriptor = value_fp8
    if use_tensor_descriptors:
        key_argument = TensorDescriptor(
            base=key_int8,
            shape=[batch * heads, sequence, head_dim],
            strides=[sequence * head_dim, head_dim, 1],
            block_shape=[1, block_n, head_dim],
        )
        value_argument = TensorDescriptor(
            base=value_fp8,
            shape=[batch * heads, head_dim, sequence],
            strides=[head_dim * sequence, sequence, 1],
            block_shape=[1, head_dim, block_n],
        )

    def launch() -> torch.Tensor:
        _sage_backend._sage_attention_kernel[(triton.cdiv(sequence, block_m), heads, batch)](
            query_int8,
            key_argument,
            value_argument,
            query_scale,
            key_scale,
            folded_scale,
            output,
            sequence,
            sequence,
            is_causal=False,
            grouped_qk=True,
            pv_accumulator_fp32=accumulator_fp32,
            heads=heads,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            value_transposed=value_transposed,
            use_tensor_descriptors=use_tensor_descriptors,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output

    return launch


@triton.jit
def _pair_probability_attention_kernel(  # noqa: PLR0915, PLR0917
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_ptr,
    value_log_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    sampled_normalization: tl.constexpr,
    sampled_headroom_log2: tl.constexpr,
    precomputed_pv_multiplier: tl.constexpr,
    scaled_fp16_numerator: tl.constexpr,
):
    """Pair two K64 tiles before PV so both MMAs share one UINT8 scale."""
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    half_head_dim: tl.constexpr = head_dim // 2
    offsets_vd = tl.arange(0, half_head_dim)
    valid_queries = offsets_m < query_length
    query = tl.load(
        query_ptr
        + (batch_head * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    query_scale = tl.load(
        query_scale_ptr
        + batch_head * tl.cdiv(query_length, 32)
        + offsets_m // 32,
        mask=valid_queries,
        other=0.0,
    )
    if scaled_fp16_numerator:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
    else:
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)

    for pair_start in tl.range(0, key_length, 2 * block_n, disable_licm=True):
        current_n0 = pair_start + offsets_n
        key0 = _sage_backend._load_attention_key_tile(
            key_ptr,
            batch_head,
            pair_start,
            current_n0,
            offsets_d,
            key_length,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        integer_scores0 = tl.dot(query, key0, out_dtype=tl.int32)
        key_scale0 = tl.load(
            key_scale_ptr
            + batch_head * tl.cdiv(key_length, block_n)
            + pair_start // block_n
        )
        scores0 = integer_scores0.to(tl.float32) * (query_scale * key_scale0)[:, None]
        scores0 = tl.where(valid_queries[:, None], scores0, -float("inf"))
        value_log_scale0 = tl.load(
            value_log_scale_ptr + batch_head * key_length + current_n0
        )
        block_max0 = tl.max(scores0 + value_log_scale0[None, :], axis=1)
        safe_block_max0 = tl.where(valid_queries, block_max0, 0.0)
        probabilities0 = tl.where(
            valid_queries[:, None],
            tl.exp2(scores0 - safe_block_max0[:, None]),
            0.0,
        )
        denominator0 = tl.sum(probabilities0, axis=1)
        value_scale0 = tl.load(value_scale_ptr + batch_head * key_length + current_n0)
        if precomputed_pv_multiplier:
            probability_multiplier0 = value_scale0
        else:
            probability_multiplier0 = value_scale0 * 255.0
        if sampled_normalization:
            # The first physically interleaved K64 sample defines the pair's
            # probability coordinate. Quantizing it immediately tests whether
            # holding packed UINT8 codes is cheaper than retaining FP16 P.
            probability_codes0 = tl.minimum(
                255.0,
                probabilities0
                * tl.exp2(-sampled_headroom_log2)
                * probability_multiplier0[None, :]
                + 0.5,
            ).to(tl.uint8)
        else:
            # Exact pair normalization needs the second maximum before P0 can
            # be quantized, so retain the first probability tile in FP16.
            pending_probabilities0 = probabilities0.to(tl.float16)

        start_n1 = pair_start + block_n
        current_n1 = start_n1 + offsets_n
        key1 = _sage_backend._load_attention_key_tile(
            key_ptr,
            batch_head,
            start_n1,
            current_n1,
            offsets_d,
            key_length,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        integer_scores1 = tl.dot(query, key1, out_dtype=tl.int32)
        key_scale1 = tl.load(
            key_scale_ptr
            + batch_head * tl.cdiv(key_length, block_n)
            + start_n1 // block_n
        )
        scores1 = integer_scores1.to(tl.float32) * (query_scale * key_scale1)[:, None]
        scores1 = tl.where(valid_queries[:, None], scores1, -float("inf"))
        value_log_scale1 = tl.load(
            value_log_scale_ptr + batch_head * key_length + current_n1
        )
        block_max1 = tl.max(scores1 + value_log_scale1[None, :], axis=1)
        safe_block_max1 = tl.where(valid_queries, block_max1, 0.0)
        probabilities1 = tl.where(
            valid_queries[:, None],
            tl.exp2(scores1 - safe_block_max1[:, None]),
            0.0,
        )
        denominator1 = tl.sum(probabilities1, axis=1)

        pair_max = tl.maximum(block_max0, block_max1)
        next_max = tl.maximum(running_max, pair_max)
        old_weight = tl.where(valid_queries, tl.exp2(running_max - next_max), 0.0)
        pair_weight = tl.where(valid_queries, tl.exp2(pair_max - next_max), 0.0)
        pair_scale0 = tl.exp2(block_max0 - pair_max)
        pair_scale1 = tl.exp2(block_max1 - pair_max)
        denominator_pair = denominator0 * pair_scale0 + denominator1 * pair_scale1
        denominator = denominator * old_weight + denominator_pair * pair_weight

        value_scale1 = tl.load(value_scale_ptr + batch_head * key_length + current_n1)
        if precomputed_pv_multiplier:
            probability_multiplier1 = value_scale1
        else:
            probability_multiplier1 = value_scale1 * 255.0
        if sampled_normalization:
            sampled_pair_max = block_max0 + sampled_headroom_log2
            sampled_scale1 = tl.exp2(block_max1 - sampled_pair_max)
            probability_codes1 = tl.minimum(
                255.0,
                probabilities1
                * sampled_scale1[:, None]
                * probability_multiplier1[None, :]
                + 0.5,
            ).to(tl.uint8)
            partial_weight = tl.where(
                valid_queries,
                tl.exp2(sampled_pair_max - next_max),
                0.0,
            )
        else:
            probability_codes0 = tl.minimum(
                255.0,
                pending_probabilities0.to(tl.float32)
                * pair_scale0[:, None]
                * probability_multiplier0[None, :]
                + 0.5,
            ).to(tl.uint8)
            probability_codes1 = tl.minimum(
                255.0,
                probabilities1
                * pair_scale1[:, None]
                * probability_multiplier1[None, :]
                + 0.5,
            ).to(tl.uint8)
            partial_weight = pair_weight

        value0_low = _sage_backend._load_attention_value_subtile(
            value_ptr,
            batch,
            head,
            batch_head,
            pair_start,
            current_n0,
            offsets_vd,
            key_length,
            feature_start=0,
            feature_block=half_head_dim,
            value_transposed=True,
            use_tensor_descriptors=use_tensor_descriptors,
            heads=heads,
            head_dim=head_dim,
            block_n=block_n,
        )
        partial_low = tl.dot(probability_codes0, value0_low, out_dtype=tl.int32)
        value1_low = _sage_backend._load_attention_value_subtile(
            value_ptr,
            batch,
            head,
            batch_head,
            start_n1,
            current_n1,
            offsets_vd,
            key_length,
            feature_start=0,
            feature_block=half_head_dim,
            value_transposed=True,
            use_tensor_descriptors=use_tensor_descriptors,
            heads=heads,
            head_dim=head_dim,
            block_n=block_n,
        )
        partial_low = tl.dot(
            probability_codes1,
            value1_low,
            partial_low,
            out_dtype=tl.int32,
        )
        if scaled_fp16_numerator:
            partial_low_scaled = (partial_low.to(tl.float32) * (1.0 / 65536.0)).to(
                tl.float16
            )
            accumulator_low = (
                accumulator_low * old_weight[:, None].to(tl.float16)
                + partial_low_scaled * partial_weight[:, None].to(tl.float16)
            )
        else:
            accumulator_low = (
                accumulator_low * old_weight[:, None]
                + partial_low.to(tl.float32) * (1.0 / 255.0) * partial_weight[:, None]
            )

        value0_high = _sage_backend._load_attention_value_subtile(
            value_ptr,
            batch,
            head,
            batch_head,
            pair_start,
            current_n0,
            offsets_vd,
            key_length,
            feature_start=half_head_dim,
            feature_block=half_head_dim,
            value_transposed=True,
            use_tensor_descriptors=use_tensor_descriptors,
            heads=heads,
            head_dim=head_dim,
            block_n=block_n,
        )
        partial_high = tl.dot(probability_codes0, value0_high, out_dtype=tl.int32)
        value1_high = _sage_backend._load_attention_value_subtile(
            value_ptr,
            batch,
            head,
            batch_head,
            start_n1,
            current_n1,
            offsets_vd,
            key_length,
            feature_start=half_head_dim,
            feature_block=half_head_dim,
            value_transposed=True,
            use_tensor_descriptors=use_tensor_descriptors,
            heads=heads,
            head_dim=head_dim,
            block_n=block_n,
        )
        partial_high = tl.dot(
            probability_codes1,
            value1_high,
            partial_high,
            out_dtype=tl.int32,
        )
        if scaled_fp16_numerator:
            partial_high_scaled = (partial_high.to(tl.float32) * (1.0 / 65536.0)).to(
                tl.float16
            )
            accumulator_high = (
                accumulator_high * old_weight[:, None].to(tl.float16)
                + partial_high_scaled * partial_weight[:, None].to(tl.float16)
            )
        else:
            accumulator_high = (
                accumulator_high * old_weight[:, None]
                + partial_high.to(tl.float32) * (1.0 / 255.0) * partial_weight[:, None]
            )
        running_max = next_max

    denominator_safe = tl.maximum(denominator, 1e-30)
    if scaled_fp16_numerator:
        denominator_safe *= 255.0 / 65536.0
    denominator_safe = denominator_safe[:, None]
    output_base = output_ptr + (batch_head * query_length + offsets_m[:, None]) * head_dim
    tl.store(
        output_base + offsets_vd[None, :],
        accumulator_low / denominator_safe,
        mask=valid_queries[:, None],
    )
    tl.store(
        output_base + half_head_dim + offsets_vd[None, :],
        accumulator_high / denominator_safe,
        mask=valid_queries[:, None],
    )


@triton.jit
def _fixed_pair_probability_attention_kernel(  # noqa: PLR0915, PLR0917
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    folded_fp8_scale_ptr,
    output_ptr,
    query_length,
    key_length,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    prequantize_first_probability: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
):
    """Pair K64 PV MMAs so one INT32-to-FP32 conversion covers K128."""
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    half_head_dim: tl.constexpr = head_dim // 2
    if split_pv_head_dim:
        offsets_vd = tl.arange(0, half_head_dim)
        accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
        accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    else:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    valid_queries = offsets_m < query_length
    query = tl.load(
        query_ptr
        + (batch_head * query_length + offsets_m[:, None]) * head_dim
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    query_scale = tl.load(
        query_scale_ptr
        + batch_head * tl.cdiv(query_length, 32)
        + offsets_m // 32,
        mask=valid_queries,
        other=0.0,
    )
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)

    for pair_start in tl.range(0, key_length, 2 * block_n, disable_licm=True):
        current_n0 = pair_start + offsets_n
        key0 = _sage_backend._load_attention_key_tile(
            key_ptr,
            batch_head,
            pair_start,
            current_n0,
            offsets_d,
            key_length,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        integer_scores0 = tl.dot(query, key0, out_dtype=tl.int32)
        key_scale0 = tl.load(
            key_scale_ptr
            + batch_head * tl.cdiv(key_length, block_n)
            + pair_start // block_n
        )
        scores0 = integer_scores0.to(tl.float32) * (query_scale * key_scale0)[:, None]
        valid_keys0 = current_n0[None, :] < key_length
        scores0 = tl.where(
            valid_queries[:, None] & valid_keys0,
            scores0,
            -float("inf"),
        )
        block_max0 = tl.max(scores0, axis=1)
        safe_block_max0 = tl.where(valid_queries, block_max0, 0.0)
        probabilities0 = tl.where(
            valid_queries[:, None] & valid_keys0,
            tl.exp2(scores0 - safe_block_max0[:, None]),
            0.0,
        )
        denominator0 = tl.sum(probabilities0, axis=1)
        if prequantize_first_probability:
            # Retaining signed-INT8 P0 instead of FP16 P0 tests whether one
            # extra rounding step is a profitable way to halve PV conversion
            # frequency without extending the large floating-point live range.
            pending_probability_codes0 = (
                probabilities0 * 127.0 + 0.5
            ).to(tl.int8)
        else:
            pending_probabilities0 = probabilities0.to(tl.float16)

        start_n1 = pair_start + block_n
        current_n1 = start_n1 + offsets_n
        key1 = _sage_backend._load_attention_key_tile(
            key_ptr,
            batch_head,
            start_n1,
            current_n1,
            offsets_d,
            key_length,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        integer_scores1 = tl.dot(query, key1, out_dtype=tl.int32)
        key_scale1 = tl.load(
            key_scale_ptr
            + batch_head * tl.cdiv(key_length, block_n)
            + start_n1 // block_n
        )
        scores1 = integer_scores1.to(tl.float32) * (query_scale * key_scale1)[:, None]
        valid_keys1 = current_n1[None, :] < key_length
        scores1 = tl.where(
            valid_queries[:, None] & valid_keys1,
            scores1,
            -float("inf"),
        )
        block_max1 = tl.max(scores1, axis=1)
        safe_block_max1 = tl.where(valid_queries, block_max1, 0.0)
        probabilities1 = tl.where(
            valid_queries[:, None] & valid_keys1,
            tl.exp2(scores1 - safe_block_max1[:, None]),
            0.0,
        )
        denominator1 = tl.sum(probabilities1, axis=1)

        pair_max = tl.maximum(block_max0, block_max1)
        next_max = tl.maximum(running_max, pair_max)
        old_weight = tl.where(valid_queries, tl.exp2(running_max - next_max), 0.0)
        pair_weight = tl.where(valid_queries, tl.exp2(pair_max - next_max), 0.0)
        pair_scale0 = tl.exp2(block_max0 - pair_max)
        pair_scale1 = tl.exp2(block_max1 - pair_max)
        denominator_pair = denominator0 * pair_scale0 + denominator1 * pair_scale1
        denominator = denominator * old_weight + denominator_pair * pair_weight
        if prequantize_first_probability:
            probability_codes0 = (
                pending_probability_codes0.to(tl.float32)
                * pair_scale0[:, None]
                + 0.5
            ).to(tl.int8)
        else:
            probability_codes0 = (
                pending_probabilities0.to(tl.float32)
                * pair_scale0[:, None]
                * 127.0
                + 0.5
            ).to(tl.int8)
        probability_codes1 = (
            probabilities1 * pair_scale1[:, None] * 127.0 + 0.5
        ).to(tl.int8)

        if split_pv_head_dim:
            value0_low = _sage_backend._load_attention_value_subtile(
                value_ptr,
                batch,
                head,
                batch_head,
                pair_start,
                current_n0,
                offsets_vd,
                key_length,
                feature_start=0,
                feature_block=half_head_dim,
                value_transposed=True,
                use_tensor_descriptors=use_tensor_descriptors,
                heads=heads,
                head_dim=head_dim,
                block_n=block_n,
            )
            partial_low = tl.dot(probability_codes0, value0_low, out_dtype=tl.int32)
            value1_low = _sage_backend._load_attention_value_subtile(
                value_ptr,
                batch,
                head,
                batch_head,
                start_n1,
                current_n1,
                offsets_vd,
                key_length,
                feature_start=0,
                feature_block=half_head_dim,
                value_transposed=True,
                use_tensor_descriptors=use_tensor_descriptors,
                heads=heads,
                head_dim=head_dim,
                block_n=block_n,
            )
            partial_low = tl.dot(
                probability_codes1,
                value1_low,
                partial_low,
                out_dtype=tl.int32,
            )
            accumulator_low = (
                accumulator_low * old_weight[:, None]
                + partial_low.to(tl.float32) * pair_weight[:, None]
            )

            value0_high = _sage_backend._load_attention_value_subtile(
                value_ptr,
                batch,
                head,
                batch_head,
                pair_start,
                current_n0,
                offsets_vd,
                key_length,
                feature_start=half_head_dim,
                feature_block=half_head_dim,
                value_transposed=True,
                use_tensor_descriptors=use_tensor_descriptors,
                heads=heads,
                head_dim=head_dim,
                block_n=block_n,
            )
            partial_high = tl.dot(probability_codes0, value0_high, out_dtype=tl.int32)
            value1_high = _sage_backend._load_attention_value_subtile(
                value_ptr,
                batch,
                head,
                batch_head,
                start_n1,
                current_n1,
                offsets_vd,
                key_length,
                feature_start=half_head_dim,
                feature_block=half_head_dim,
                value_transposed=True,
                use_tensor_descriptors=use_tensor_descriptors,
                heads=heads,
                head_dim=head_dim,
                block_n=block_n,
            )
            partial_high = tl.dot(
                probability_codes1,
                value1_high,
                partial_high,
                out_dtype=tl.int32,
            )
            accumulator_high = (
                accumulator_high * old_weight[:, None]
                + partial_high.to(tl.float32) * pair_weight[:, None]
            )
        else:
            value0 = _sage_backend._load_attention_value_tile(
                value_ptr,
                batch,
                head,
                batch_head,
                pair_start,
                current_n0,
                offsets_d,
                key_length,
                True,
                use_tensor_descriptors,
                heads,
                head_dim,
                block_n,
            )
            partial = tl.dot(probability_codes0, value0, out_dtype=tl.int32)
            value1 = _sage_backend._load_attention_value_tile(
                value_ptr,
                batch,
                head,
                batch_head,
                start_n1,
                current_n1,
                offsets_d,
                key_length,
                True,
                use_tensor_descriptors,
                heads,
                head_dim,
                block_n,
            )
            partial = tl.dot(probability_codes1, value1, partial, out_dtype=tl.int32)
            accumulator = (
                accumulator * old_weight[:, None]
                + partial.to(tl.float32) * pair_weight[:, None]
            )
        running_max = next_max

    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    output_base = output_ptr + (batch_head * query_length + offsets_m[:, None]) * head_dim
    output_scale: tl.constexpr = 1008.0 / 16129.0
    if split_pv_head_dim:
        folded_scale_low = tl.load(
            folded_fp8_scale_ptr + batch_head * head_dim + offsets_vd
        )
        folded_scale_high = tl.load(
            folded_fp8_scale_ptr + batch_head * head_dim + half_head_dim + offsets_vd
        )
        tl.store(
            output_base + offsets_vd[None, :],
            accumulator_low
            / denominator_safe
            * folded_scale_low[None, :]
            * output_scale,
            mask=valid_queries[:, None],
        )
        tl.store(
            output_base + half_head_dim + offsets_vd[None, :],
            accumulator_high
            / denominator_safe
            * folded_scale_high[None, :]
            * output_scale,
            mask=valid_queries[:, None],
        )
    else:
        folded_scale = tl.load(
            folded_fp8_scale_ptr + batch_head * head_dim + offsets_d
        )
        tl.store(
            output_base + offsets_d[None, :],
            accumulator / denominator_safe * folded_scale[None, :] * output_scale,
            mask=valid_queries[:, None],
        )


def _fixed_pair_probability_launcher(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    *,
    block_m: int,
    num_stages: int,
    num_warps: int,
    split_pv_head_dim: bool,
    prequantize_first_probability: bool,
    maxnreg: int | None,
) -> Callable[[], torch.Tensor]:
    """Prepare and launch the fixed-scale signed-INT8 K64-pair kernel."""
    batch, heads, sequence, head_dim = query.shape
    if head_dim != 128 or sequence % 128:
        raise ValueError("fixed pair-P profiling requires D128 and sequence % 128 == 0")
    prepared = _prepare_int8_pv_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=True,
        value_transposed=True,
    )
    query_int8, key_int8, value_int8, query_scale, key_scale, folded_fp8_scale = prepared
    key_argument, value_argument = _sage_backend._make_attention_arguments(
        key_int8,
        value_int8,
        batch,
        heads,
        sequence,
        head_dim,
        True,
        True,
        head_dim // 2 if split_pv_head_dim else None,
        64,
    )
    launch_options = {"num_warps": num_warps, "num_stages": num_stages}
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg

    def launch() -> torch.Tensor:
        _fixed_pair_probability_attention_kernel[
            (triton.cdiv(sequence, block_m), heads, batch)
        ](
            query_int8,
            key_argument,
            value_argument,
            query_scale,
            key_scale,
            folded_fp8_scale,
            output,
            sequence,
            sequence,
            heads=heads,
            head_dim=head_dim,
            block_m=block_m,
            block_n=64,
            split_pv_head_dim=split_pv_head_dim,
            prequantize_first_probability=prequantize_first_probability,
            use_tensor_descriptors=True,
            **launch_options,
        )
        return output

    return launch


def _pair_probability_launcher(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    *,
    block_m: int,
    num_stages: int,
    num_warps: int,
    use_tensor_descriptors: bool,
    maxnreg: int | None,
    sampled_normalization: bool,
    sampled_headroom_log2: float,
    precomputed_pv_multiplier: bool,
    fp32_pv_metadata: bool,
    scaled_fp16_numerator: bool,
) -> Callable[[], torch.Tensor]:
    """Prepare and launch the perf-only K64-pair probability-side alignment kernel."""
    batch, heads, sequence, head_dim = query.shape
    if head_dim != 128 or sequence % 128:
        raise ValueError("pair-P profiling requires D128 and a sequence divisible by 128")
    if sampled_headroom_log2 < 0:
        raise ValueError("sampled normalization headroom must be nonnegative")
    if sampled_normalization:
        pair_order = torch.cat(
            (
                torch.arange(0, 128, 2, device=key.device),
                torch.arange(1, 128, 2, device=key.device),
            )
        )
        key = (
            key.reshape(batch, heads, sequence // 128, 128, head_dim)
            .index_select(3, pair_order)
            .reshape_as(key)
            .contiguous()
        )
        value = (
            value.reshape(batch, heads, sequence // 128, 128, head_dim)
            .index_select(3, pair_order)
            .reshape_as(value)
            .contiguous()
        )
    prepared = _prepare_uint8_pv_feature_convrot_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=True,
        rotation_group=0,
        value_scale_axis="key",
        value_scale_floor=0.0,
        probability_scale_mode="log",
        value_transposed=True,
        affine_probability=True,
        native_uint8_mma=True,
        scale_forward_log_recurrence=True,
        fp32_scale_forward_metadata=fp32_pv_metadata,
        precompute_pv_multiplier=precomputed_pv_multiplier,
    )
    query_int8, key_int8, value_int8, query_scale, key_scale, value_scale, value_log_scale, *_ = (
        prepared
    )
    key_argument, value_argument = _sage_backend._make_attention_arguments(
        key_int8,
        value_int8,
        batch,
        heads,
        sequence,
        head_dim,
        True,
        use_tensor_descriptors,
        head_dim // 2,
    )
    launch_options = {"num_warps": num_warps, "num_stages": num_stages}
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg

    def launch() -> torch.Tensor:
        _pair_probability_attention_kernel[
            (triton.cdiv(sequence, block_m), heads, batch)
        ](
            query_int8,
            key_argument,
            value_argument,
            query_scale,
            key_scale,
            value_scale,
            value_log_scale,
            output,
            sequence,
            sequence,
            heads=heads,
            head_dim=head_dim,
            block_m=block_m,
            block_n=64,
            use_tensor_descriptors=use_tensor_descriptors,
            sampled_normalization=sampled_normalization,
            sampled_headroom_log2=sampled_headroom_log2,
            precomputed_pv_multiplier=precomputed_pv_multiplier,
            scaled_fp16_numerator=scaled_fp16_numerator,
            **launch_options,
        )
        return output

    return launch


def _int8_launcher(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    *,
    block_scaled: bool,
    block_global_probability: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    num_warps: int,
    value_transposed: bool,
    use_tensor_descriptors: bool,
    split_pv_head_dim: bool,
    native_unsigned_probability: bool,
    integer_pv_recurrence: bool,
    raw_integer_pv_recurrence: bool,
    raw_fp32_pv_recurrence: bool,
    magic_score_conversion: bool,
    magic_pv_conversion: bool,
    fp16_pv_conversion: bool,
    bf16_pv_conversion: bool,
    unmasked_self_attention: bool,
    maxnreg: int | None,
) -> Callable[[], torch.Tensor]:
    sequence = query.shape[2]
    if unmasked_self_attention and (
        sequence % block_m or sequence % block_n
    ):
        raise ValueError("unmasked self-attention requires complete M and K tiles")
    prepared: tuple[torch.Tensor, ...] = (
        _prepare_block_int8_pv_inputs(
            query,
            key,
            value,
            scale,
            grouped_qk=True,
            value_transposed=value_transposed,
        )
        if block_scaled
        else _prepare_int8_pv_inputs(
            query,
            key,
            value,
            scale,
            grouped_qk=True,
            value_transposed=value_transposed,
        )
    )

    def launch() -> torch.Tensor:
        return _launch_int8_pv_attention(
            prepared,
            output,
            sequence,
            sequence,
            False,
            grouped_qk=True,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
            block_scaled_pv=block_scaled,
            block_global_probability=block_global_probability,
            split_pv_head_dim=split_pv_head_dim,
            native_unsigned_probability=native_unsigned_probability,
            integer_pv_recurrence=integer_pv_recurrence,
            raw_integer_pv_recurrence=raw_integer_pv_recurrence,
            raw_fp32_pv_recurrence=raw_fp32_pv_recurrence,
            magic_score_conversion=magic_score_conversion,
            magic_pv_conversion=magic_pv_conversion,
            fp16_pv_conversion=fp16_pv_conversion,
            bf16_pv_conversion=bf16_pv_conversion,
            unmasked_self_attention=unmasked_self_attention,
            maxnreg=maxnreg,
            value_transposed=value_transposed,
            use_tensor_descriptors=use_tensor_descriptors,
        )

    return launch


def _uint8_k32_feature_launcher(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    *,
    block_m: int,
    num_stages: int,
    num_warps: int,
    maxnreg: int | None,
) -> Callable[[], torch.Tensor]:
    sequence = query.shape[2]
    prepared = _prepare_uint8_k32_feature_pv_inputs(
        query,
        key,
        value,
        scale,
    )

    def launch() -> torch.Tensor:
        return _launch_uint8_k32_feature_pv_attention(
            prepared,
            output,
            sequence,
            sequence,
            block_m=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )

    return launch


def _log_int8_launcher(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    rotated_output: torch.Tensor,
    scale: float,
    *,
    block_m: int,
    num_stages: int,
    num_warps: int,
    value_transposed: bool,
    shift_log_scores: bool,
    weighted_log_denominator: bool,
    value_scale_axis: Literal["feature", "key"],
    probability_scale_mode: Literal["dynamic", "tile", "log"],
    affine_probability: bool,
    native_uint8_mma: bool,
    integer_output_recurrence: bool,
    integer_tile_exponent_recurrence: bool,
    single_shift_tile_exponent_recurrence: bool,
    predot_exponent_alignment: bool,
    dithered_predot_alignment: bool,
    immediate_k32_pv_conversion: bool,
    lazy_int32_exponent_recurrence: bool,
    integer_exponent_headroom: int,
    paired_int32_tiles: bool,
    probability_fp16: bool,
    fp16_pv_scaling: bool,
    factored_pv_scaling: bool,
    precomputed_pv_multiplier: bool,
    use_pv_scale_descriptor: bool,
    omit_pv_scaling: bool,
    fp32_scale_forward_metadata: bool,
    normalized_fp16_recurrence: bool,
    scaled_fp16_numerator: bool,
    scaled_fp16_denominator: bool,
    split_pv_head_dim: bool,
    tile_common_log_denominator: bool,
    narrow_int8_log_denominator: bool,
    running_max_probability_recurrence: bool,
    scale_forward_log_recurrence: bool,
    use_tensor_descriptors: bool,
    unmasked_self_attention: bool,
    maxnreg: int | None,
) -> Callable[[], torch.Tensor]:
    sequence = query.shape[2]
    prepared = _prepare_uint8_pv_feature_convrot_inputs(
        query,
        key,
        value,
        scale,
        grouped_qk=True,
        rotation_group=0,
        value_scale_axis=value_scale_axis,
        value_scale_floor=0.0,
        probability_scale_mode=probability_scale_mode,
        value_transposed=value_transposed,
        affine_probability=affine_probability,
        native_uint8_mma=native_uint8_mma,
        tile_common_log_denominator=tile_common_log_denominator,
        narrow_int8_log_denominator=narrow_int8_log_denominator,
        scale_forward_log_recurrence=scale_forward_log_recurrence,
        fp32_scale_forward_metadata=fp32_scale_forward_metadata,
        precompute_pv_multiplier=precomputed_pv_multiplier,
    )

    def launch() -> torch.Tensor:
        return _launch_uint8_pv_feature_convrot_attention(
            prepared,
            rotated_output,
            output,
            sequence,
            sequence,
            False,
            grouped_qk=True,
            rotation_group=0,
            value_scale_axis=value_scale_axis,
            probability_scale_mode=probability_scale_mode,
            fuse_output_rotation=True,
            block_m=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
            value_transposed=value_transposed,
            shift_log_scores=shift_log_scores,
            weighted_log_denominator=weighted_log_denominator,
            affine_probability=affine_probability,
            native_uint8_mma=native_uint8_mma,
            integer_output_recurrence=integer_output_recurrence,
            integer_tile_exponent_recurrence=integer_tile_exponent_recurrence,
            single_shift_tile_exponent_recurrence=single_shift_tile_exponent_recurrence,
            predot_exponent_alignment=predot_exponent_alignment,
            dithered_predot_alignment=dithered_predot_alignment,
            immediate_k32_pv_conversion=immediate_k32_pv_conversion,
            lazy_int32_exponent_recurrence=lazy_int32_exponent_recurrence,
            integer_exponent_headroom=integer_exponent_headroom,
            paired_int32_tiles=paired_int32_tiles,
            probability_fp16=probability_fp16,
            fp16_pv_scaling=fp16_pv_scaling,
            factored_pv_scaling=factored_pv_scaling,
            precomputed_pv_multiplier=precomputed_pv_multiplier,
            use_pv_scale_descriptor=use_pv_scale_descriptor,
            omit_pv_scaling=omit_pv_scaling,
            normalized_fp16_recurrence=normalized_fp16_recurrence,
            scaled_fp16_numerator=scaled_fp16_numerator,
            scaled_fp16_denominator=scaled_fp16_denominator,
            split_pv_head_dim=split_pv_head_dim,
            tile_common_log_denominator=tile_common_log_denominator,
            narrow_int8_log_denominator=narrow_int8_log_denominator,
            running_max_probability_recurrence=running_max_probability_recurrence,
            scale_forward_log_recurrence=scale_forward_log_recurrence,
            use_tensor_descriptors=use_tensor_descriptors,
            unmasked_self_attention=unmasked_self_attention,
            maxnreg=maxnreg,
        )

    return launch


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:  # noqa: PLR0915
    """Prepare once, then repeatedly launch one fixed-schedule attention kernel."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 12:
        raise SystemExit("This profiling configuration currently targets consumer SM12x")
    if args.variant == "int8-log" and args.block_n != 64:
        raise SystemExit("int8-log currently requires --block-n 64")
    shape = (args.batch, args.heads, args.sequence, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    output = torch.empty_like(query)
    rotated_output = torch.empty(shape, device="cuda", dtype=torch.float32)
    scale = args.head_dim**-0.5

    if args.variant.startswith("fp8"):
        launch = _fp8_launcher(
            query,
            key,
            value,
            output,
            scale,
            accumulator_fp32=args.variant.startswith("fp8-fp32"),
            block_m=args.block_m,
            block_n=args.block_n,
            num_stages=args.num_stages,
            num_warps=args.num_warps,
            value_transposed=args.variant.endswith(("transposed", "descriptor")),
            use_tensor_descriptors=args.variant.endswith("descriptor"),
        )
        jit_kernel = _sage_backend._sage_attention_kernel
    elif args.variant.startswith("int8-fixed-pair"):
        launch = _fixed_pair_probability_launcher(
            query,
            key,
            value,
            output,
            scale,
            block_m=args.block_m,
            num_stages=args.num_stages,
            num_warps=args.num_warps,
            split_pv_head_dim="split" in args.variant,
            prequantize_first_probability="prequantized" in args.variant,
            maxnreg=args.maxnreg,
        )
        jit_kernel = _fixed_pair_probability_attention_kernel
    elif args.variant in (
        "int8-log-split-scale-forward-pair-p-native-descriptor",
        "int8-log-split-scale-forward-pair-p-precomputed-pv-scale-native-descriptor",
        "int8-log-split-scale-forward-pair-p-precomputed-pv-scale-fp32-metadata-native-descriptor",
        "int8-log-split-scale-forward-pair-p-precomputed-pv-scale-scaled-fp16-numerator-native-descriptor",
        "int8-log-split-scale-forward-sampled-pair-p-native-descriptor",
    ):
        sampled_normalization = "sampled-pair" in args.variant
        launch = _pair_probability_launcher(
            query,
            key,
            value,
            output,
            scale,
            block_m=args.block_m,
            num_stages=args.num_stages,
            num_warps=args.num_warps,
            use_tensor_descriptors=True,
            maxnreg=args.maxnreg,
            sampled_normalization=sampled_normalization,
            sampled_headroom_log2=args.sampled_headroom_log2,
            precomputed_pv_multiplier="precomputed-pv-scale" in args.variant,
            fp32_pv_metadata="fp32-metadata" in args.variant,
            scaled_fp16_numerator="scaled-fp16-numerator" in args.variant,
        )
        jit_kernel = _pair_probability_attention_kernel
    elif args.variant.startswith("int8-output-group") or args.variant.startswith(
        "int8-output-scalar"
    ):
        feature_group = (
            128
            if args.variant.startswith("int8-output-scalar")
            else int(args.variant.removeprefix("int8-output-group").split("-", 1)[0])
        )
        integer_output_recurrence = "-int32-" in args.variant
        scale_run_n = next(
            (
                candidate
                for candidate in (256, 512, 1024)
                if f"-run{candidate}-" in args.variant
            ),
            None,
        )
        prepared = _prepare_uint8_grouped_output_pv_inputs(
            query,
            key,
            value,
            scale,
            feature_group=feature_group,
            block_n=args.block_n,
            scale_run_n=scale_run_n,
            integer_output_recurrence=integer_output_recurrence,
        )

        if scale_run_n is not None:

            def launch() -> torch.Tensor:
                return _launch_uint8_run_scaled_output_pv_attention(
                    prepared,
                    output,
                    args.sequence,
                    args.sequence,
                    feature_group=feature_group,
                    block_n=args.block_n,
                    scale_run_n=scale_run_n,
                    block_m=args.block_m,
                    num_stages=args.num_stages,
                    num_warps=args.num_warps,
                    global_probability_codes="global-p" in args.variant,
                    dominant_weight_merge="dominant" in args.variant,
                    scaled_fp16_numerator="scaled-fp16-numerator" in args.variant,
                    unmasked_self_attention="unmasked" in args.variant,
                    maxnreg=args.maxnreg,
                )

            jit_kernel = _uint8_run_scaled_output_pv_attention_kernel
        else:

            def launch() -> torch.Tensor:
                return _launch_uint8_grouped_output_pv_attention(
                    prepared,
                    output,
                    args.sequence,
                    args.sequence,
                    feature_group=feature_group,
                    block_n=args.block_n,
                    block_m=args.block_m,
                    num_stages=args.num_stages,
                    num_warps=args.num_warps,
                    integer_output_recurrence=integer_output_recurrence,
                    common_feature_exponent="-commonexp-" in args.variant,
                    unmasked_self_attention="unmasked" in args.variant,
                    maxnreg=args.maxnreg,
                )

            jit_kernel = _uint8_grouped_output_pv_attention_kernel
    elif args.variant == "int8-output-k32-feature-native-descriptor":
        launch = _uint8_k32_feature_launcher(
            query,
            key,
            value,
            output,
            scale,
            block_m=args.block_m,
            num_stages=args.num_stages,
            num_warps=args.num_warps,
            maxnreg=args.maxnreg,
        )
        jit_kernel = _uint8_k32_feature_pv_attention_kernel
    elif args.variant.startswith(("int8-log", "int8-affine", "int8-per-key")):
        value_scale_axis: Literal["feature", "key"] = (
            "feature" if args.variant.startswith("int8-affine") else "key"
        )
        if args.variant.startswith("int8-per-key-dynamic"):
            probability_scale_mode: Literal["dynamic", "tile", "log"] = "dynamic"
        elif args.variant.startswith("int8-per-key-tile"):
            probability_scale_mode = "tile"
        else:
            probability_scale_mode = "log" if value_scale_axis == "key" else "dynamic"
        launch = _log_int8_launcher(
            query,
            key,
            value,
            output,
            rotated_output,
            scale,
            block_m=args.block_m,
            num_stages=args.num_stages,
            num_warps=args.num_warps,
            value_transposed=args.variant.endswith(("transposed", "descriptor")),
            shift_log_scores="unshifted" not in args.variant,
            weighted_log_denominator=not any(
                control in args.variant for control in ("unweighted", "unshifted")
            ),
            value_scale_axis=value_scale_axis,
            probability_scale_mode=probability_scale_mode,
            affine_probability="signed" not in args.variant,
            native_uint8_mma="native" in args.variant,
            integer_output_recurrence=(
                "int32" in args.variant
                and "int32-tile" not in args.variant
                and "int32-lazy" not in args.variant
            ),
            integer_tile_exponent_recurrence="int32-tile" in args.variant,
            single_shift_tile_exponent_recurrence="single-shift" in args.variant,
            predot_exponent_alignment="predot" in args.variant,
            dithered_predot_alignment="predot-dithered" in args.variant,
            immediate_k32_pv_conversion="k32-immediate" in args.variant,
            lazy_int32_exponent_recurrence="int32-lazy" in args.variant,
            integer_exponent_headroom=(
                int(args.variant.split("-h", maxsplit=1)[1][0])
                if "int32-lazy-h" in args.variant
                else 0
            ),
            paired_int32_tiles="paired" in args.variant,
            probability_fp16="fp16p" in args.variant,
            fp16_pv_scaling="fp16-pv-scale" in args.variant,
            factored_pv_scaling=any(
                control in args.variant
                for control in (
                    "factored-pv-scale",
                    "precomputed-pv-scale",
                )
            ),
            precomputed_pv_multiplier="precomputed-pv-scale" in args.variant,
            use_pv_scale_descriptor="scale-descriptor" in args.variant,
            omit_pv_scaling="no-pv-scale" in args.variant,
            fp32_scale_forward_metadata="fp32-metadata" in args.variant,
            normalized_fp16_recurrence="fp16norm" in args.variant,
            scaled_fp16_numerator="scaled-fp16-numerator" in args.variant,
            scaled_fp16_denominator="scaled-fp16-denominator" in args.variant,
            split_pv_head_dim="split" in args.variant,
            tile_common_log_denominator="tile-common" in args.variant,
            narrow_int8_log_denominator="narrow-denom" in args.variant,
            running_max_probability_recurrence="running-max" in args.variant,
            scale_forward_log_recurrence="scale-forward" in args.variant,
            use_tensor_descriptors=args.variant.endswith("descriptor"),
            unmasked_self_attention="unmasked" in args.variant,
            maxnreg=args.maxnreg,
        )
        jit_kernel = _uint8_pv_feature_convrot_attention_kernel
    else:
        launch = _int8_launcher(
            query,
            key,
            value,
            output,
            scale,
            block_scaled=args.variant.startswith("int8-block"),
            block_global_probability="global-p" in args.variant,
            block_m=args.block_m,
            block_n=args.block_n,
            num_stages=args.num_stages,
            num_warps=args.num_warps,
            value_transposed=args.variant.endswith(("transposed", "descriptor")),
            use_tensor_descriptors=args.variant.endswith("descriptor"),
            split_pv_head_dim="split" in args.variant,
            native_unsigned_probability=(
                "fixed-uint8" in args.variant or "block-uint8" in args.variant
            ),
            integer_pv_recurrence=(
                "int32" in args.variant and "int32-raw" not in args.variant
            ),
            raw_integer_pv_recurrence="int32-raw" in args.variant,
            raw_fp32_pv_recurrence="fp32-raw" in args.variant,
            magic_score_conversion="magic-all" in args.variant,
            magic_pv_conversion="magic" in args.variant,
            fp16_pv_conversion="fp16-convert" in args.variant,
            bf16_pv_conversion="bf16-convert" in args.variant,
            unmasked_self_attention="unmasked" in args.variant,
            maxnreg=args.maxnreg,
        )
        jit_kernel = _int8_pv_attention_kernel

    launch()
    torch.cuda.synchronize()
    latency_ms = float(
        triton.testing.do_bench(
            launch,
            warmup=args.warmup_ms,
            rep=args.repeat_ms,
        )
    )
    print(
        json.dumps(
            {
                "variant": args.variant,
                "sequence": args.sequence,
                "block_m": args.block_m,
                "block_n": args.block_n,
                "num_stages": args.num_stages,
                "requested_num_warps": args.num_warps,
                "requested_maxnreg": args.maxnreg,
                "sampled_headroom_log2": args.sampled_headroom_log2,
                **(
                    _compiler_report(jit_kernel, latency_ms)
                    if args.compiler_report
                    else {"latency_ms": latency_ms}
                ),
            },
            sort_keys=True,
        )
    )
    for _ in range(args.iterations):
        launch()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
