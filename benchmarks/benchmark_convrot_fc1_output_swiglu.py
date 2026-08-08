"""Benchmark folding H3 FC1's materialized SwiGLU into its GEMM epilogue.

The paired kernel computes matching gate/up columns in one CTA.  It preserves
the two BF16 boundaries in the eager expression ``silu(gate) * up`` while
avoiding the temporary [M, 2F] projection in HBM.
"""

# The experimental JIT kernel mirrors Triton's launcher-style production
# signature, whose compile-time arguments are not represented to type checkers.
# pyright: reportCallIssue=false
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0915, PLR0917

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional
import triton
import triton.language as tl
from lib import Timing, triton_benchmark

from piper_kernels.convrot.int8.backends import triton as baseline

K = 5_376
F = 14_336


@dataclass(frozen=True, slots=True)
class Config:
    block_m: int
    block_pair_n: int
    block_k: int
    num_stages: int
    num_warps: int

    def display(self) -> str:
        return (
            f"{self.block_m}x({self.block_pair_n}+{self.block_pair_n})x"
            f"{self.block_k},s{self.num_stages},w{self.num_warps}"
        )


# The first configuration has exactly the production FC1 MMA shape and
# accumulator count.  The other two test whether less state per CTA or more M
# parallelism pays for reloading activation tiles.
_CONFIGS = (
    Config(128, 128, 128, 3, 8),
    Config(128, 64, 128, 3, 8),
    Config(64, 128, 128, 3, 8),
)


@triton.jit
def _int8_fc1_paired_swiglu_kernel(
    activation_ptr,
    weight_ptr,
    output_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    m,
    f,
    k,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    block_m: tl.constexpr,
    block_pair_n: tl.constexpr,
    block_k: tl.constexpr,
    even_m: tl.constexpr,
    even_f: tl.constexpr,
    even_k: tl.constexpr,
    group_m: tl.constexpr,
):
    """Compute [gate tile | matching up tile], then store only SwiGLU."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(f, block_pair_n)
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    actual_group_m = tl.minimum(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % num_pid_in_group) % actual_group_m
    pid_n = (pid % num_pid_in_group) // actual_group_m

    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_pair_n = pid_n * block_pair_n + tl.arange(0, block_pair_n)
    offsets_combined_n = tl.arange(0, 2 * block_pair_n)
    offsets_k = tl.arange(0, block_k)
    # Form one regular [BM, 2*BN] MMA tile whose second output half addresses
    # the distant up segment of the canonical [gate F | up F] weight tensor.
    pair_columns = pid_n * block_pair_n + offsets_combined_n % block_pair_n
    weight_columns = pair_columns + tl.where(
        offsets_combined_n < block_pair_n,
        0,
        f,
    )
    offsets_m_i64 = offsets_m.to(tl.int64)
    offsets_k_i64 = offsets_k.to(tl.int64)
    weight_columns_i64 = weight_columns.to(tl.int64)
    activation_pointers = (
        activation_ptr + offsets_m_i64[:, None] * stride_am + offsets_k_i64[None, :] * stride_ak
    )
    weight_pointers = (
        weight_ptr + weight_columns_i64[None, :] * stride_wn + offsets_k_i64[:, None] * stride_wk
    )
    accumulator = tl.zeros((block_m, 2 * block_pair_n), dtype=tl.int32)

    for k_offset in range(tl.cdiv(k, block_k)):
        if even_m and even_k:
            activation = tl.load(activation_pointers)
        else:
            activation = tl.load(
                activation_pointers,
                mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < k - k_offset * block_k),
                other=0,
            )
        if even_f and even_k:
            weight = tl.load(weight_pointers)
        else:
            weight = tl.load(
                weight_pointers,
                mask=(pair_columns[None, :] < f) & (offsets_k[:, None] < k - k_offset * block_k),
                other=0,
            )
        accumulator += tl.dot(activation, weight)
        activation_pointers += block_k * stride_ak
        weight_pointers += block_k * stride_wk

    if even_m:
        activation_scale = tl.load(activation_scale_ptr + offsets_m)
    else:
        activation_scale = tl.load(
            activation_scale_ptr + offsets_m,
            mask=offsets_m < m,
            other=0.0,
        )

    # Split the combined MMA tile into matching gate/up columns.  Casting each
    # projection first reproduces the materialized BF16 FC1 output.  Casting
    # SiLU before multiplication reproduces PyTorch's second BF16 boundary.
    grouped = tl.reshape(accumulator, (block_m, 2, block_pair_n))
    paired = tl.permute(grouped, (0, 2, 1))
    gate_accumulator, up_accumulator = tl.split(paired)
    gate_weight_scale = tl.load(
        weight_scale_ptr + offsets_pair_n,
        mask=offsets_pair_n < f,
        other=0.0,
    )
    up_weight_scale = tl.load(
        weight_scale_ptr + f + offsets_pair_n,
        mask=offsets_pair_n < f,
        other=0.0,
    )
    gate = (
        gate_accumulator.to(tl.float32) * activation_scale[:, None] * gate_weight_scale[None, :]
    ).to(tl.bfloat16)
    up = (up_accumulator.to(tl.float32) * activation_scale[:, None] * up_weight_scale[None, :]).to(
        tl.bfloat16
    )
    activated_gate = (gate.to(tl.float32) / (1.0 + tl.exp(-gate.to(tl.float32)))).to(tl.bfloat16)
    result = (activated_gate.to(tl.float32) * up.to(tl.float32)).to(tl.bfloat16)

    output_pointers = (
        output_ptr
        + offsets_m_i64[:, None] * stride_om
        + offsets_pair_n.to(tl.int64)[None, :] * stride_on
    )
    tl.store(
        output_pointers,
        result,
        mask=(offsets_m[:, None] < m) & (offsets_pair_n[None, :] < f),
    )


@triton.jit
def _materialized_swiglu_kernel(
    projection_ptr,
    output_ptr,
    f,
    output_numel,
    block_size: tl.constexpr,
):
    """Materialized BF16 equivalent of ``F.silu(gate) * up``."""
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < output_numel
    rows = offsets // f
    columns = offsets - rows * f
    offsets_i64 = offsets.to(tl.int64)
    projection_row_offsets = rows.to(tl.int64) * (2 * f)
    gate = tl.load(
        projection_ptr + projection_row_offsets + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        projection_ptr + projection_row_offsets + f + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    activated_gate = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    result = (activated_gate.to(tl.float32) * up).to(tl.bfloat16)
    tl.store(output_ptr + offsets_i64, result, mask=mask)


def _production_launcher(
    activation: torch.Tensor,
    weight: torch.Tensor,
    projection: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> Callable[[], None]:
    m, k = activation.shape
    n = weight.shape[0]
    block_m, block_n, block_k, stages, warps = baseline._int8_matmul_config(
        m,
        n,
        k,
        True,
    )
    num_m_tiles = triton.cdiv(m, block_m)
    num_n_tiles = triton.cdiv(n, block_n)
    group_m = 1 if num_n_tiles <= 32 else 64

    def launch() -> None:
        baseline._int8_matmul_kernel[(num_m_tiles * num_n_tiles,)](
            activation,
            weight,
            projection,
            activation_scale,
            weight_scale,
            projection,
            m,
            n,
            k,
            activation.stride(0),
            activation.stride(1),
            weight.stride(0),
            weight.stride(1),
            projection.stride(0),
            projection.stride(1),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            has_bias=False,
            even_m=m % block_m == 0,
            even_n=n % block_n == 0,
            even_k=k % block_k == 0,
            group_m=group_m,
            num_stages=stages,
            num_warps=warps,
            num_ctas=1,
        )

    return launch


def _standalone_launcher(
    projection: torch.Tensor,
    output: torch.Tensor,
) -> Callable[[], None]:
    f = output.shape[1]
    output_numel = output.numel()
    block_size = 1_024

    def launch() -> None:
        _materialized_swiglu_kernel[(triton.cdiv(output_numel, block_size),)](
            projection,
            output,
            f,
            output_numel,
            block_size=block_size,
            num_warps=8,
        )

    return launch


def _paired_launcher(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    config: Config,
) -> Callable[[], None]:
    m, k = activation.shape
    f = output.shape[1]
    num_m_tiles = triton.cdiv(m, config.block_m)
    num_n_tiles = triton.cdiv(f, config.block_pair_n)
    group_m = 1 if num_n_tiles <= 32 else 64

    def launch() -> None:
        _int8_fc1_paired_swiglu_kernel[(num_m_tiles * num_n_tiles,)](
            activation,
            weight,
            output,
            activation_scale,
            weight_scale,
            m,
            f,
            k,
            activation.stride(0),
            activation.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            block_m=config.block_m,
            block_pair_n=config.block_pair_n,
            block_k=config.block_k,
            even_m=m % config.block_m == 0,
            even_f=f % config.block_pair_n == 0,
            even_k=k % config.block_k == 0,
            group_m=group_m,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            num_ctas=1,
        )

    return launch


def _composite_launcher(
    first: Callable[[], None],
    second: Callable[[], None],
) -> Callable[[], None]:
    def launch() -> None:
        first()
        second()

    return launch


def _preparation_launcher(
    activation: torch.Tensor,
    quantized: torch.Tensor,
    activation_scale: torch.Tensor,
    *,
    input_act_code: int,
) -> Callable[[], None]:
    """Launch production's fused rotate/quantize preparation path."""
    input_dtype_code = baseline._input_dtype_code(activation.dtype)

    def launch() -> None:
        baseline._fused_rotate_quantize_activations(
            activation,
            quantized,
            activation_scale,
            group_size=256,
            input_dtype_code=input_dtype_code,
            input_act_code=input_act_code,
        )

    return launch


def _validation_inputs(
    m: int = 257,
    f: int = 512,
    k: int = 256,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    activation = torch.randint(
        -16,
        17,
        (m, k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    weight = torch.randint(
        -16,
        17,
        (2 * f, k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    activation_scale = torch.rand(m, device="cuda", dtype=torch.float32, generator=generator)
    activation_scale = activation_scale * 0.01 + 0.005
    weight_scale = torch.rand(2 * f, device="cuda", dtype=torch.float32, generator=generator)
    weight_scale = weight_scale * 0.01 + 0.005
    return activation, weight, activation_scale, weight_scale


def _validate(
    config: Config,
    m: int = 257,
    f: int = 512,
    k: int = 256,
) -> tuple[float, int]:
    activation, weight, activation_scale, weight_scale = _validation_inputs(m, f, k)
    m, _ = activation.shape
    f = weight.shape[0] // 2
    projection = torch.empty((m, 2 * f), device="cuda", dtype=torch.bfloat16)
    expected = torch.empty((m, f), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    _production_launcher(
        activation,
        weight,
        projection,
        activation_scale,
        weight_scale,
    )()
    _standalone_launcher(projection, expected)()
    _paired_launcher(
        activation,
        weight,
        actual,
        activation_scale,
        weight_scale,
        config,
    )()
    torch.cuda.synchronize()

    # This is the strongest comparison: separate and fused Triton paths must
    # agree bit-for-bit after reproducing both materialized BF16 boundaries.
    if not torch.equal(actual, expected):
        difference = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"fused differs from materialized path: max_abs={difference.max().item()}, "
            f"mismatches={torch.count_nonzero(difference).item()}"
        )

    torch_reference = (torch_functional.silu(projection[:, :f]) * projection[:, f:]).to(
        torch.bfloat16
    )
    torch_difference = (expected.float() - torch_reference.float()).abs()
    max_abs = float(torch_difference.max().item())
    mismatches = int(torch.count_nonzero(torch_difference).item())
    torch.testing.assert_close(expected, torch_reference, rtol=0, atol=1 / 128)

    # Validate the real current boundary as well.  FC2 preparation currently
    # consumes raw [gate | up] and fuses SwiGLU with rotation/quantization.  A
    # paired FC1 instead materializes the exact BF16 SwiGLU result, then uses
    # the ordinary rotation/quantization specialization.
    current_quantized = torch.empty((m, f), device="cuda", dtype=torch.int8)
    proposed_quantized = torch.empty_like(current_quantized)
    current_scale = torch.empty(m, device="cuda", dtype=torch.float32)
    proposed_scale = torch.empty_like(current_scale)
    _preparation_launcher(
        projection,
        current_quantized,
        current_scale,
        input_act_code=1,
    )()
    _preparation_launcher(
        actual,
        proposed_quantized,
        proposed_scale,
        input_act_code=0,
    )()
    torch.cuda.synchronize()
    if not torch.equal(proposed_quantized, current_quantized):
        mismatched_quantized = torch.count_nonzero(proposed_quantized != current_quantized).item()
        raise AssertionError(
            "paired FC1 preparation differs from current fused preparation: "
            f"quantized mismatches={mismatched_quantized}"
        )
    torch.testing.assert_close(proposed_scale, current_scale, rtol=0, atol=0)
    return max_abs, mismatches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=37_710)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--measurement-time-ms", type=int, default=300)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--reverse-provider-order", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        raise SystemExit("FC1 output-SwiGLU experiments require an NVIDIA Blackwell GPU")
    if arguments.rows <= 0:
        raise SystemExit("--rows must be positive")
    if arguments.warmup_ms < 0 or arguments.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")

    valid_configs: list[Config] = []
    print("Validating exact materialized BF16 boundaries:")
    for config in _CONFIGS:
        try:
            max_abs, mismatches = _validate(config)
        except (
            triton.compiler.errors.CompilationError,
            triton.runtime.errors.OutOfResources,
        ) as error:
            print(f"  skip {config.display()}: {error}")
            continue
        print(
            f"  {config.display()}: fused BF16 and prepared INT8 bit-exact; "
            f"vs torch max_abs={max_abs:.6g}, mismatches={mismatches}"
        )
        valid_configs.append(config)
    if _CONFIGS[0] in valid_configs:
        h3_max_abs, h3_mismatches = _validate(_CONFIGS[0], f=F, k=K)
        print(
            "  production-width H3 check: fused BF16 and prepared INT8 bit-exact; "
            f"vs torch max_abs={h3_max_abs:.6g}, mismatches={h3_mismatches}"
        )
    if arguments.validation_only or not valid_configs:
        return

    m = arguments.rows
    generator = torch.Generator(device="cuda").manual_seed(1)
    activation = torch.randint(
        -16,
        17,
        (m, K),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    weight = torch.randint(
        -16,
        17,
        (2 * F, K),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    activation_scale = torch.full((m,), 1e-3, device="cuda", dtype=torch.float32)
    weight_scale = torch.full((2 * F,), 1e-3, device="cuda", dtype=torch.float32)
    projection = torch.empty((m, 2 * F), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, F), device="cuda", dtype=torch.bfloat16)
    quantized = torch.empty((m, F), device="cuda", dtype=torch.int8)
    prepared_scale = torch.empty(m, device="cuda", dtype=torch.float32)

    production = _production_launcher(
        activation,
        weight,
        projection,
        activation_scale,
        weight_scale,
    )
    standalone = _standalone_launcher(projection, output)
    composite = _composite_launcher(production, standalone)
    production()
    torch.cuda.synchronize()

    materialized_rows: list[tuple[str, Timing]] = []
    materialized_rows.append(
        (
            "production FC1 GEMM",
            triton_benchmark(
                production,
                arguments.warmup_ms,
                arguments.measurement_time_ms,
            ),
        )
    )
    materialized_rows.append(
        (
            "standalone materialized SwiGLU",
            triton_benchmark(
                standalone,
                arguments.warmup_ms,
                arguments.measurement_time_ms,
            ),
        )
    )
    materialized_rows.append(
        (
            "production + materialized SwiGLU",
            triton_benchmark(
                composite,
                arguments.warmup_ms,
                arguments.measurement_time_ms,
            ),
        )
    )
    materialized_baseline_ms = materialized_rows[-1][1].median_ms

    paired_launchers: list[tuple[Config, Callable[[], None]]] = []
    for config in valid_configs:
        fused = _paired_launcher(
            activation,
            weight,
            output,
            activation_scale,
            weight_scale,
            config,
        )
        fused()
        torch.cuda.synchronize()
        paired_launchers.append((config, fused))
        materialized_rows.append(
            (
                f"paired fused {config.display()}",
                triton_benchmark(
                    fused,
                    arguments.warmup_ms,
                    arguments.measurement_time_ms,
                ),
            )
        )

    operations = 2 * m * (2 * F) * K
    materialized_bytes_saved = 2 * m * (2 * F) * torch.bfloat16.itemsize
    properties = torch.cuda.get_device_properties("cuda")
    print(f"\nGPU: {properties.name}")
    print(f"H3 FC1 -> SwiGLU: M={m} K={K} F={F}, projection N={2 * F}")
    print(
        "Standalone-materialization comparison: paired fusion removes "
        f"{materialized_bytes_saved / 1e9:.3f} GB of HBM traffic"
    )
    print("| provider | p50 [p20, p80] (ms) | vs composite | effective TOP/s |")
    print("|:---|---:|---:|---:|")
    for name, timing in materialized_rows:
        effective_tops = (
            "-"
            if name == "standalone materialized SwiGLU"
            else f"{operations / timing.median_ms / 1e9:.1f}"
        )
        print(
            f"| {name} | {timing.display(4)} | "
            f"{materialized_baseline_ms / timing.median_ms:.3f}x | {effective_tops} |"
        )

    current_preparation = _preparation_launcher(
        projection,
        quantized,
        prepared_scale,
        input_act_code=1,
    )
    ordinary_preparation = _preparation_launcher(
        output,
        quantized,
        prepared_scale,
        input_act_code=0,
    )
    actual_providers: list[tuple[str, Callable[[], None]]] = [
        (
            "current fused SwiGLU + rotate + quantize",
            current_preparation,
        ),
        (
            "ordinary rotate + quantize",
            ordinary_preparation,
        ),
        (
            "current FC1 + fused FC2 preparation",
            _composite_launcher(production, current_preparation),
        ),
    ]
    for config, fused in paired_launchers:
        actual_providers.append(
            (
                f"paired FC1 + ordinary preparation {config.display()}",
                _composite_launcher(fused, ordinary_preparation),
            )
        )
    if arguments.reverse_provider_order:
        actual_providers.reverse()
    actual_rows = [
        (
            name,
            triton_benchmark(
                launch,
                arguments.warmup_ms,
                arguments.measurement_time_ms,
            ),
        )
        for name, launch in actual_providers
    ]
    current_chain_ms = next(
        timing.median_ms
        for name, timing in actual_rows
        if name == "current FC1 + fused FC2 preparation"
    )

    actual_bytes_saved = 2 * m * F * torch.bfloat16.itemsize
    print(
        "\nActual FC2-preparation comparison: paired fusion removes "
        f"{actual_bytes_saved / 1e9:.3f} GB of HBM traffic"
    )
    print("| provider | p50 [p20, p80] (ms) | vs current chain |")
    print("|:---|---:|---:|")
    for name, timing in actual_rows:
        print(f"| {name} | {timing.display(4)} | {current_chain_ms / timing.median_ms:.3f}x |")


if __name__ == "__main__":
    main()
