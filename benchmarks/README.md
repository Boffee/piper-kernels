# Benchmarks

Operator benchmarks live here rather than in the correctness test suite. Each benchmark
should report hardware, dtype, shapes, cold compilation time, and warmed execution time.

Run the ConvRot provider comparison with:

```shell
uv run python benchmarks/benchmark_convrot.py
```

Use `--help` to select activation rows, weight dimensions, group size, dtype, and timing
windows. The script verifies exact agreement before reporting Triton and reference timings.

Run the end-to-end SageAttention2++ comparison with:

```shell
uv run python benchmarks/benchmark_sage_attention.py
```

The Sage timing includes K smoothing and Q/K/V quantization. Use `--help` to select sequence
lengths, dtype, head dimension, and causal mode. The comparison reports warmed PyTorch SDPA
latency and error relative to SDPA as well as cold Triton compilation time. Warmed latency is
reported as the median with p20/p80 bounds; increase `--repeat-ms` for a larger timing sample.

The official SageAttention2++ CUDA implementation is an opt-in benchmark-only dependency. It
is pinned to an exact upstream revision, and its build needs NVCC plus an explicit target when
the desired GPU is not visible. Install it and include it in the comparison with:

```shell
# RTX 4090 (SM89)
TORCH_CUDA_ARCH_LIST=8.9 uv sync --group benchmark

# RTX 5090 (SM120)
TORCH_CUDA_ARCH_LIST=12.0 uv sync --group benchmark

uv run --group benchmark python benchmarks/benchmark_sage_attention.py --canonical
```

The canonical comparison explicitly selects INT8 QK + FP8 PV with `fp32+fp16` accumulation.
It uses the official production Q/K granularity for each target: per-thread on SM89 and
per-warp on SM12x. Building for SM89 requires CUDA 12.4 or newer; SM120 requires CUDA 12.8 or
newer. The dependency stays out of the default development group so CPU-only development and
CI do not need a CUDA compiler.
