# Benchmarks

Operator benchmarks live here rather than in the correctness test suite. Each benchmark
should report hardware, dtype, shapes, cold compilation time, and warmed execution time.

Run the ConvRot provider comparison with:

```shell
uv run python benchmarks/benchmark_convrot.py
```

Use `--help` to select activation rows, weight dimensions, group size, dtype, and timing
windows. The script verifies exact agreement before reporting Triton and reference timings.
