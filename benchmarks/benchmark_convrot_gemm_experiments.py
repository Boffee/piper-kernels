"""Benchmark isolated Blackwell ConvRot INT8 GEMM formulations."""

# Triton's JIT arguments intentionally omit ordinary Python annotations and
# mirror the production kernel's wide launch signature.
# ruff: noqa: ANN001, ANN202, PLR0913, PLR0917

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from lib import Timing, triton_benchmark
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels.convrot.int8.backends import triton as baseline


@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    m: int
    n: int
    k: int


@dataclass(frozen=True, slots=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    warp_specialize: bool = False

    def display(self) -> str:
        suffix = ",ws" if self.warp_specialize else ""
        return (
            f"{self.block_m}x{self.block_n}x{self.block_k},"
            f"s{self.num_stages},w{self.num_warps}{suffix}"
        )


_SHAPES = {
    shape.name: shape
    for shape in (
        Shape("qkv", 37_710, 21_504, 5_376),
        Shape("attention-out", 37_710, 5_376, 7_168),
        Shape("mlp-fc1", 37_710, 28_672, 5_376),
        Shape("mlp-fc2", 37_710, 5_376, 14_336),
        Shape("roof", 16_384, 16_384, 8_192),
    )
}


@triton.jit
def _pid_for_tile(
    tile_id,
    num_pid_m,
    num_pid_n,
    group_m: tl.constexpr,
):
    num_pid_in_group = group_m * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_m
    actual_group_m = tl.minimum(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (tile_id % num_pid_in_group) % actual_group_m
    pid_n = (tile_id % num_pid_in_group) // actual_group_m
    return pid_m, pid_n


@triton.jit
def _int8_matmul_tma_kernel(
    activation_desc,
    weight_desc,
    output_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    m,
    n,
    k,
    stride_om,
    stride_on,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    pid_m, pid_n = _pid_for_tile(tile_id, num_pid_m, num_pid_n, group_m)
    offset_m = pid_m * block_m
    offset_n = pid_n * block_n
    accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

    for k_tile in tl.range(tl.cdiv(k, block_k), warp_specialize=warp_specialize):
        offset_k = k_tile * block_k
        activation = activation_desc.load([offset_m, offset_k])
        # Canonical ConvRot weights are [N, K]. TMA loads a contiguous
        # [BN, BK] block and the register/shared tile is transposed for MMA.
        weight = weight_desc.load([offset_n, offset_k])
        accumulator = tl.dot(
            activation,
            weight.T,
            accumulator,
            out_dtype=tl.int32,
        )

    offsets_m = offset_m + tl.arange(0, block_m)
    offsets_n = offset_n + tl.arange(0, block_n)
    activation_scale = tl.load(
        activation_scale_ptr + offsets_m,
        mask=offsets_m < m,
        other=0.0,
    )
    weight_scale = tl.load(
        weight_scale_ptr + offsets_n,
        mask=offsets_n < n,
        other=0.0,
    )
    result = accumulator.to(tl.float32) * activation_scale[:, None] * weight_scale[None, :]
    output_ptrs = (
        output_ptr
        + offsets_m.to(tl.int64)[:, None] * stride_om
        + offsets_n.to(tl.int64)[None, :] * stride_on
    )
    tl.store(
        output_ptrs,
        result.to(tl.bfloat16),
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


@triton.jit
def _int8_matmul_tma_persistent_kernel(
    activation_desc,
    weight_desc,
    output_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    m,
    n,
    k,
    stride_om,
    stride_on,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    num_sms: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    start_tile = tl.program_id(0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    num_tiles = num_pid_m * num_pid_n

    # Match Triton's Blackwell persistent-TMA formulation: specialize and
    # flatten the outer tile loop so loads/MMA/epilogue can be partitioned.
    for tile_id in tl.range(
        start_tile,
        num_tiles,
        num_sms,
        flatten=True,
        warp_specialize=warp_specialize,
    ):
        pid_m, pid_n = _pid_for_tile(tile_id, num_pid_m, num_pid_n, group_m)
        offset_m = pid_m * block_m
        offset_n = pid_n * block_n
        accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

        for k_tile in range(tl.cdiv(k, block_k)):
            offset_k = k_tile * block_k
            activation = activation_desc.load([offset_m, offset_k])
            weight = weight_desc.load([offset_n, offset_k])
            accumulator = tl.dot(
                activation,
                weight.T,
                accumulator,
                out_dtype=tl.int32,
            )

        offsets_m = offset_m + tl.arange(0, block_m)
        offsets_n = offset_n + tl.arange(0, block_n)
        activation_scale = tl.load(
            activation_scale_ptr + offsets_m,
            mask=offsets_m < m,
            other=0.0,
        )
        weight_scale = tl.load(
            weight_scale_ptr + offsets_n,
            mask=offsets_n < n,
            other=0.0,
        )
        result = accumulator.to(tl.float32) * activation_scale[:, None] * weight_scale[None, :]
        output_ptrs = (
            output_ptr
            + offsets_m.to(tl.int64)[:, None] * stride_om
            + offsets_n.to(tl.int64)[None, :] * stride_on
        )
        tl.store(
            output_ptrs,
            result.to(tl.bfloat16),
            mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
        )


def _baseline_launcher(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> Callable[[], None]:
    m, k = activation.shape
    n = weight.shape[0]
    block_m, block_n, block_k, num_stages, num_warps = baseline._int8_matmul_config(
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
            output,
            activation_scale,
            weight_scale,
            output,
            m,
            n,
            k,
            activation.stride(0),
            activation.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            has_bias=False,
            even_m=m % block_m == 0,
            even_n=n % block_n == 0,
            even_k=k % block_k == 0,
            group_m=group_m,
            num_stages=num_stages,
            num_warps=num_warps,
            num_ctas=1,
        )

    return launch


def _tma_launcher(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    config: Config,
    *,
    persistent: bool,
) -> Callable[[], None]:
    m, k = activation.shape
    n = weight.shape[0]
    activation_desc = TensorDescriptor.from_tensor(
        activation,
        [config.block_m, config.block_k],
    )
    weight_desc = TensorDescriptor.from_tensor(weight, [config.block_n, config.block_k])
    num_m_tiles = triton.cdiv(m, config.block_m)
    num_n_tiles = triton.cdiv(n, config.block_n)
    group_m = 1 if num_n_tiles <= 32 else 64
    num_sms = torch.cuda.get_device_properties(activation.device).multi_processor_count
    kernel = _int8_matmul_tma_persistent_kernel if persistent else _int8_matmul_tma_kernel
    grid = (min(num_sms, num_m_tiles * num_n_tiles) if persistent else num_m_tiles * num_n_tiles,)

    def launch() -> None:
        keyword_arguments = {
            "block_m": config.block_m,
            "block_n": config.block_n,
            "block_k": config.block_k,
            "group_m": group_m,
            "warp_specialize": config.warp_specialize,
            "num_stages": config.num_stages,
            "num_warps": config.num_warps,
            "num_ctas": 1,
        }
        if persistent:
            keyword_arguments["num_sms"] = num_sms
        kernel[grid](
            activation_desc,
            weight_desc,
            output,
            activation_scale,
            weight_scale,
            m,
            n,
            k,
            output.stride(0),
            output.stride(1),
            **keyword_arguments,
        )

    return launch


def _tops(shape: Shape, timing: Timing) -> float:
    return 2 * shape.m * shape.n * shape.k / timing.median_ms / 1e9


def _validate(config: Config, persistent: bool) -> None:
    shape = Shape("validation", 257, 512, 256)
    generator = torch.Generator(device="cuda").manual_seed(0)
    activation = torch.randint(
        -16,
        17,
        (shape.m, shape.k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    weight = torch.randint(
        -16,
        17,
        (shape.n, shape.k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    # A transposed view alone changes no bytes. Materialize [K, N], then view
    # it as [N, K] with strides (1, N) so the production pointer kernel reads
    # a physically row-major logical B tile.
    weight_kn = weight.T.contiguous()
    weight_kn_view = weight_kn.T
    activation_scale = torch.rand(shape.m, device="cuda", dtype=torch.float32)
    weight_scale = torch.rand(shape.n, device="cuda", dtype=torch.float32)
    expected = torch.empty((shape.m, shape.n), device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    physical_kn = torch.empty_like(expected)
    _baseline_launcher(activation, weight, expected, activation_scale, weight_scale)()
    _baseline_launcher(
        activation,
        weight_kn_view,
        physical_kn,
        activation_scale,
        weight_scale,
    )()
    _tma_launcher(
        activation,
        weight,
        actual,
        activation_scale,
        weight_scale,
        config,
        persistent=persistent,
    )()
    torch.cuda.synchronize()
    torch.testing.assert_close(physical_kn, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _benchmark_shape(
    shape: Shape,
    variants: Sequence[tuple[Config, bool]],
    warmup_ms: int,
    measurement_time_ms: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    activation = torch.randint(
        -127,
        128,
        (shape.m, shape.k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    weight = torch.randint(
        -127,
        128,
        (shape.n, shape.k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    # Materialize physical [K, N] once, outside every timed launch.
    weight_kn = weight.T.contiguous()
    weight_kn_view = weight_kn.T
    activation_scale = torch.full((shape.m,), 1e-4, device="cuda", dtype=torch.float32)
    weight_scale = torch.full((shape.n,), 1e-4, device="cuda", dtype=torch.float32)
    output = torch.empty((shape.m, shape.n), device="cuda", dtype=torch.bfloat16)

    providers: list[tuple[str, Callable[[], None]]] = [
        (
            "baseline",
            _baseline_launcher(activation, weight, output, activation_scale, weight_scale),
        ),
        (
            "baseline:physical-kn",
            _baseline_launcher(
                activation,
                weight_kn_view,
                output,
                activation_scale,
                weight_scale,
            ),
        ),
    ]
    for config, persistent in variants:
        providers.append(
            (
                f"{'persistent' if persistent else 'tma'}:{config.display()}",
                _tma_launcher(
                    activation,
                    weight,
                    output,
                    activation_scale,
                    weight_scale,
                    config,
                    persistent=persistent,
                ),
            )
        )

    print(f"\n{shape.name}: M={shape.m} N={shape.n} K={shape.k}")
    print("| provider | p50 [p20, p80] (ms) | TOP/s | vs baseline |")
    print("|:---|---:|---:|---:|")
    measurements: list[tuple[str, Timing, float]] = []
    for provider, launch in providers:
        timing = triton_benchmark(launch, warmup_ms, measurement_time_ms)
        measurements.append((provider, timing, _tops(shape, timing)))
    baseline_tops = measurements[0][2]
    for provider, timing, tops in measurements:
        print(f"| {provider} | {timing.display(4)} | {tops:.1f} | {tops / baseline_tops:.3f}x |")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", choices=tuple(_SHAPES), nargs="+", default=["qkv"])
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--measurement-time-ms", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        raise SystemExit("descriptor experiments require an NVIDIA Blackwell GPU")
    if arguments.warmup_ms < 0 or arguments.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")
    configs = (
        Config(128, 256, 128, 2, 8),
        Config(128, 256, 128, 2, 8, warp_specialize=True),
        Config(128, 256, 64, 3, 8),
        Config(128, 256, 64, 3, 8, warp_specialize=True),
        Config(128, 128, 128, 3, 8),
        Config(128, 128, 128, 3, 8, warp_specialize=True),
        Config(128, 128, 64, 4, 8),
        Config(128, 128, 64, 4, 8, warp_specialize=True),
    )
    variants: list[tuple[Config, bool]] = []
    for config in configs:
        for persistent in (False, True):
            try:
                _validate(config, persistent)
            except (
                triton.compiler.errors.CompilationError,
                triton.runtime.errors.OutOfResources,
            ) as error:
                name = "persistent" if persistent else "tma"
                print(f"skipping {name}:{config.display()}: {error}")
            else:
                variants.append((config, persistent))
    for name in arguments.cases:
        _benchmark_shape(
            _SHAPES[name],
            variants,
            arguments.warmup_ms,
            arguments.measurement_time_ms,
        )


if __name__ == "__main__":
    main()
