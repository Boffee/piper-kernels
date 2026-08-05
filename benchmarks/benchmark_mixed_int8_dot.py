"""Benchmark native U8 x S8 MMA against signed and affine S8 baselines.

The ``native`` mode requires a Triton compiler with mixed-sign integer-dot
support.  Stock Triton can run ``signed`` and ``affine``.  The affine kernel
models the exact UINT8 identity used by the attention experiment: its
precomputed ``128 * sum(B)`` correction is loaded as the integer MMA
accumulator.
"""

# Triton's JIT pointer arguments intentionally omit Python annotations.
# ruff: noqa: ANN001, ANN202

import argparse
import re
import subprocess
import tempfile
from collections.abc import Sequence

import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def _dot_kernel(
    a_ptr,
    b_ptr,
    correction_ptr,
    output_ptr,
    mode: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    tile = tl.program_id(0)
    offsets_m = tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    a = tl.load(
        a_ptr
        + tile * block_m * block_k
        + offsets_m[:, None] * block_k
        + offsets_k[None, :]
    )
    b = tl.load(
        b_ptr
        + tile * block_k * block_n
        + offsets_k[:, None] * block_n
        + offsets_n[None, :]
    )
    if mode == "affine":
        a = (a.to(tl.int32) - 128).to(tl.int8)
        correction = tl.load(correction_ptr + tile * block_n + offsets_n)
        accumulator = tl.zeros((block_m, block_n), tl.int32) + correction[None, :]
        result = tl.dot(a, b, accumulator, out_dtype=tl.int32)
    else:
        result = tl.dot(a, b, out_dtype=tl.int32)
    tl.store(
        output_ptr
        + tile * block_m * block_n
        + offsets_m[:, None] * block_n
        + offsets_n[None, :],
        result,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "signed", "affine"))
    parser.add_argument("--tiles", type=int, default=2048)
    parser.add_argument("--block-m", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--block-n", type=int, choices=(64, 128), default=128)
    parser.add_argument("--block-k", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--repeat-ms", type=int, default=2000)
    return parser.parse_args(argv)


def _compiled_kernel():
    device_cache = next(iter(_dot_kernel.device_caches.values()))
    specialization_cache = device_cache[0]
    return next(iter(specialization_cache.values()))


def _mma_instructions() -> tuple[list[str], list[str]]:
    compiled = _compiled_kernel()
    ptx = str(compiled.asm["ptx"])
    ptx_mma = sorted(set(re.findall(r"mma\\.sync[^;]+", ptx)))
    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
        cubin_file.write(compiled.asm["cubin"])
        cubin_file.flush()
        result = subprocess.run(
            ["/usr/local/cuda/bin/nvdisasm", "--print-code", cubin_file.name],
            check=True,
            capture_output=True,
            text=True,
        )
    sass_mma = sorted(
        {
            opcode
            for opcode in re.findall(r"[A-Z][A-Z0-9_.]+", result.stdout.upper())
            if "MMA" in opcode
        }
    )
    return ptx_mma, sass_mma


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda")
    shape_a = (args.tiles, args.block_m, args.block_k)
    shape_b = (args.tiles, args.block_k, args.block_n)
    if args.mode == "signed":
        a = torch.randint(-128, 128, shape_a, device=device, dtype=torch.int8)
    else:
        a = torch.randint(0, 256, shape_a, device=device, dtype=torch.uint8)
    b = torch.randint(-128, 128, shape_b, device=device, dtype=torch.int8)
    if args.mode == "affine":
        correction = (128 * b.to(torch.int32).sum(dim=1)).to(torch.int32)
    else:
        correction = torch.empty(1, device=device, dtype=torch.int32)
    output = torch.empty(
        (args.tiles, args.block_m, args.block_n),
        device=device,
        dtype=torch.int32,
    )

    def launch() -> None:
        _dot_kernel[(args.tiles,)](
            a,
            b,
            correction,
            output,
            mode=args.mode,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=args.num_warps,
        )

    launch()
    expected = a[0].cpu().to(torch.int32) @ b[0].cpu().to(torch.int32)
    torch.testing.assert_close(output[0].cpu(), expected, rtol=0, atol=0)
    latency_ms = float(
        triton.testing.do_bench(
            launch,
            warmup=args.warmup_ms,
            rep=args.repeat_ms,
        )
    )
    operations = 2 * args.tiles * args.block_m * args.block_n * args.block_k
    tops = operations / (latency_ms * 1e-3) / 1e12
    ptx_mma, sass_mma = _mma_instructions()
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"triton: {triton.__version__}")
    print(
        f"mode={args.mode} tiles={args.tiles} "
        f"tile={args.block_m}x{args.block_n}x{args.block_k} warps={args.num_warps}"
    )
    print(f"latency_ms={latency_ms:.6f} effective_tops={tops:.2f}")
    for instruction in ptx_mma:
        print(f"ptx={instruction}")
    for instruction in sass_mma:
        print(f"sass={instruction}")


if __name__ == "__main__":
    _main()
