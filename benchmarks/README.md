# Benchmarks

Operator benchmarks live here rather than in the correctness test suite. Each benchmark
reports hardware, software, Git state, shapes, kernel configuration, numerical quality,
and consistently named timing phases. The support code in `_lib/` is development-only;
it is not part of the installed `piper_kernels` API.

## Common provider and timing model

A provider has two explicit callables:

- `prepare()` performs per-invocation preprocessing such as quantization, packing, or
  scale construction and returns the prepared inputs.
- `run(prepared)` executes the operator using already-prepared inputs. It may launch
  one or more kernels.

The common runner reports these phases:

- `first_call_ms`: synchronized wall time for the first operator invocation, including
  any lazy compilation. It is not compiler CPU time in isolation and does not claim
  that compiler caches were initially empty.
- `preparation`: warmed, synchronized wall latency of preparation-only work.
- `prepared_execution`: warmed device-event latency of `run(prepared)` on fixed
  prepared inputs.
- `operator_end_to_end`: warmed, synchronized wall latency of `run(prepare())`.

Synchronized wall timing captures host dispatch, allocation, packing, and device work.
Device-event timing isolates elapsed work on the accelerator stream. Every latency
distribution serializes its `clock`, and `first_call_clock` describes the scalar first call.

Warmed latencies are displayed as `p50 [p20, p80]`. A phase is `null` in machine output
when it does not apply to a provider. The configured warmup and measurement-time windows
are stored alongside every phase result. Benchmark code can use the shared model directly:

```python
provider = BenchmarkProvider(
    name="my-kernel",
    prepare=prepare_inputs,
    run=launch_kernel,
    synchronize=torch.cuda.synchronize,
    configuration={"block_m": 64, "num_warps": 4},
)
measurement = measure_provider(provider, warmup_ms=100, measurement_time_ms=500)
```

`AttentionShape` records batch size, Q/KV head counts, Q/KV sequence lengths, and head
dimension without assuming self-attention or MHA. `AttentionConfig` records dtype,
causality, scale, and an explicit QKV layout such as `BHSD`.

## Quality and reproducibility

`measure_quality()` centralizes mean/max absolute error, relative L1 and L2 error,
SQNR, cosine similarity, and actual/reference non-finite counts. Providers can attach
endpoint saturation counts for quantized tensors with `measure_saturation()`.
Integer quality inputs through 32 bits are promoted to FP64. Full-width INT64 and UINT64
inputs are rejected because no floating comparison dtype preserves every possible value;
providers needing them should use a domain-specific exact comparison.

Every `BenchmarkRecord` includes:

- GPU name, accelerator backend, and architecture;
- Python, Torch, Triton, CUDA or ROCm runtime, and available driver versions;
- Git revision and dirty-worktree state;
- logical shape, provider configuration, phase timings, quality, and optional extras.

All benchmark CLIs retain their Markdown or terminal summaries. Add `--json PATH` to
write a versioned JSON array or `--jsonl PATH` to write one compact record per line.
Serialization is strict JSON; non-finite floating-point metrics such as infinite SQNR
for an exact result are represented as `null`.

The schema starts at version 1. Consumers should check `schema_version` before relying
on field names. A shortened record looks like:

```json
{
  "schema_version": 1,
  "benchmark": "integer-pv-dot",
  "provider": "triton-native",
  "shape": {"tiles": 2048, "key_tile": 64},
  "configuration": {
    "lhs_dtype": "int8",
    "rhs_dtype": "int8",
    "accumulator_dtype": "int32",
    "implementation": "native",
    "block_m": 64,
    "block_n": 128,
    "num_warps": 4,
    "seed": 0
  },
  "timings": {
    "warmup_ms": 500,
    "measurement_time_ms": 2000,
    "first_call_ms": 310.2,
    "first_call_clock": "synchronized_wall",
    "preparation": {
      "median_ms": 0.004,
      "p20_ms": 0.004,
      "p80_ms": 0.005,
      "clock": "synchronized_wall"
    },
    "prepared_execution": {
      "median_ms": 0.031,
      "p20_ms": 0.030,
      "p80_ms": 0.032,
      "clock": "device_event"
    },
    "operator_end_to_end": {
      "median_ms": 0.036,
      "p20_ms": 0.035,
      "p80_ms": 0.037,
      "clock": "synchronized_wall"
    }
  }
}
```

## Included benchmarks

Run the ConvRot provider comparison with:

```shell
uv run python benchmarks/benchmark_convrot.py
```

Use `--help` to select activation rows, weight dimensions, group size, dtype,
deterministic input seed, and timing windows. The script verifies exact agreement before
reporting Triton and reference timings.

Run the stock-Triton signed or affine-proxy integer P x V microbenchmark with:

```shell
uv run python benchmarks/benchmark_integer_pv_dot.py s8-s8
uv run python benchmarks/benchmark_integer_pv_dot.py u8-s8-affine-proxy
```

The optional `u8-s8-native` variant requires a Triton compiler and target with
mixed-sign UINT8 x INT8 dot support. The benchmark records the LHS, RHS, and accumulator
dtypes explicitly, checks exact INT32 output, and records operand saturation.
Backend-specific PTX, SASS, and AMDGCN inspection belongs to compiler/profiling tooling
rather than this portable benchmark runner.

## Triton compiler inspection

Providers register the Triton JIT functions they launch through
`triton_jit_functions`. After the provider has run at least once, the shared inspector
discovers its compiled specialization and reports:

- registers per thread, compiler-reported spills, shared memory per Triton program,
  warps, stages, and CUDA CTAs per cluster;
- a resource-only program and warp residency ceiling per compute unit from the device
  limits exposed by PyTorch, including the limiting resource;
- static PTX instruction-family and MMA-opcode counts when PTX is available;
- static SASS instruction-family and MMA-opcode counts for NVIDIA CUDA kernels.

The residency value is a ceiling, not achieved occupancy. It does not model every
architecture's allocation granularity or replace hardware profiling. Static instruction
counts describe one compiled program, not dynamic execution counts.

The integer P x V benchmark is the executable reference integration:

```shell
uv run python benchmarks/benchmark_integer_pv_dot.py s8-s8 \
  --compiler-report \
  --compiler-json artifacts/s8-s8-compiler.json
```

SASS inspection invokes `nvdisasm` from the NVIDIA CUDA Toolkit. It is enabled
automatically for CUDA compiler reports and disabled for other Triton backends. If the
tool is absent, the inspector gives an actionable error; use `--no-sass` when only the
portable resource report and available compiler IR are needed, or use
`--nvdisasm /path/to/nvdisasm` when the toolkit binary is not on `PATH`. ROCm resource
reporting uses the same provider and specialization model, while AMDGCN disassembly
remains a separate future backend adapter.

All Triton-cache and compiled-metadata access lives in `_lib/triton_inspection.py`.
Specialized diagnostics can read an artifact without depending on Triton internals:

```python
from _lib.triton_inspection import compiled_artifact

ttgir = compiled_artifact(jit_kernel, "ttgir")
```

Compiler JSON has its own versioned `triton_compiler` record type and includes provider
configuration, environment and Git metadata, specialization fingerprints, resources,
and instruction summaries for comparison across commits. `--compiler-json` writes an
array and `--compiler-jsonl` writes one compiler record per line through the same output
machinery as benchmark records.

Compiler reporting requires each registered JIT function to have one specialization in
the current process by default. This prevents one provider from silently claiming
specializations compiled earlier by another provider. Run compiler comparisons as one
provider/configuration per process; advanced diagnostics that intentionally inspect an
entire process-wide cache must opt out explicitly.

## External profiler captures

`profile_provider()` launches either `prepared_execution` or `operator_end_to_end` for
any `BenchmarkProvider`. By default, its initial compilation call and warmup iterations
finish and synchronize before the CUDA profiler starts. Passing
`--profile-include-setup` explicitly includes them in a separate `profile/setup` NVTX
range.

For example, capture the integer P x V provider with Nsight Systems:

```shell
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  uv run python benchmarks/benchmark_integer_pv_dot.py s8-s8 \
  --profile --profile-phase prepared_execution
```

The launch loop accepts an injected capture controller so a future ROCTracer/ROCTx
adapter can reuse its provider-phase and setup-exclusion behavior. The built-in
controller intentionally reports a clear unsupported-backend error on ROCm rather than
presenting CUDA profiler APIs as portable.
