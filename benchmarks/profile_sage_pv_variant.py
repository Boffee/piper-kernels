"""Measure one fixed-schedule Sage PV variant and inspect its generated code."""

# Profiling launchers intentionally expose each controlled schedule dimension.
# ruff: noqa: PLR0913

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
import triton.testing
from triton.compiler.compiler import CompiledKernel
from triton.runtime.jit import JITFunction
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels.attention._sage2pp.backends import triton as _sage_backend
from piper_kernels.attention._sage2pp.experiments.int8_pv import (
    _int8_pv_attention_kernel,
    _launch_int8_pv_attention,
    _prepare_block_int8_pv_inputs,
    _prepare_int8_pv_inputs,
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
            "int8-fixed",
            "int8-fixed-transposed",
            "int8-fixed-descriptor",
            "int8-block",
            "int8-block-transposed",
            "int8-block-descriptor",
            "int8-log",
            "int8-log-transposed",
            "int8-log-descriptor",
            "int8-log-unweighted-transposed",
            "int8-log-signed-transposed",
            "int8-log-signed-descriptor",
            "int8-affine-block-transposed",
            "int8-per-key-dynamic-transposed",
            "int8-per-key-tile-transposed",
        ],
    )
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--block-m", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--block-n", type=int, choices=[64, 128], default=64)
    parser.add_argument("--num-stages", type=int, choices=[2, 3], default=3)
    parser.add_argument("--num-warps", type=int, choices=[4, 8], default=4)
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

    instruction_pattern = re.compile(
        r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)"
    )
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
            "HADD2",
            "F2FP",
            "FFMA",
            "FADD",
            "FMUL",
            "MUFU",
            "PRMT",
            "LDSM",
            "LDS",
            "STS",
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
    value_fp8_shape = (
        (batch, heads, head_dim, sequence) if value_transposed else value.shape
    )
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


def _int8_launcher(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    *,
    block_scaled: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    num_warps: int,
    value_transposed: bool,
    use_tensor_descriptors: bool,
) -> Callable[[], torch.Tensor]:
    sequence = query.shape[2]
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
            value_transposed=value_transposed,
            use_tensor_descriptors=use_tensor_descriptors,
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
    weighted_log_denominator: bool,
    value_scale_axis: Literal["feature", "key"],
    probability_scale_mode: Literal["dynamic", "tile", "log"],
    affine_probability: bool,
    use_tensor_descriptors: bool,
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
            weighted_log_denominator=weighted_log_denominator,
            affine_probability=affine_probability,
            use_tensor_descriptors=use_tensor_descriptors,
        )

    return launch


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
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
            weighted_log_denominator="unweighted" not in args.variant,
            value_scale_axis=value_scale_axis,
            probability_scale_mode=probability_scale_mode,
            affine_probability="signed" not in args.variant,
            use_tensor_descriptors=args.variant.endswith("descriptor"),
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
            block_m=args.block_m,
            block_n=args.block_n,
            num_stages=args.num_stages,
            num_warps=args.num_warps,
            value_transposed=args.variant.endswith(("transposed", "descriptor")),
            use_tensor_descriptors=args.variant.endswith("descriptor"),
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
