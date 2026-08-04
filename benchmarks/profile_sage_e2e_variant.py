"""Launch one end-to-end Sage variant inside an Nsight Systems capture range."""

import argparse
import importlib
from collections.abc import Callable, Sequence
from typing import cast

import torch

from piper_kernels.attention import sage_attention
from piper_kernels.attention._sage2pp.backends.triton import _run_sage_attention

CanonicalSage = Callable[..., torch.Tensor]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        choices=[
            "piper",
            "piper-descriptor",
            "piper-row-major",
            "canonical2pp",
            "canonical2",
        ],
    )
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    return parser.parse_args(argv)


def _canonical_launcher(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    accumulator: str,
) -> Callable[[], torch.Tensor]:
    module = importlib.import_module("sageattention")
    canonical = cast(CanonicalSage, module.sageattn_qk_int8_pv_fp8_cuda)

    def launch() -> torch.Tensor:
        return canonical(
            query,
            key,
            value,
            tensor_layout="HND",
            is_causal=False,
            qk_quant_gran="per_warp",
            sm_scale=query.shape[-1] ** -0.5,
            pv_accum_dtype=accumulator,
            smooth_k=True,
            smooth_v=False,
            return_lse=False,
        )

    return launch


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    """Warm up, then launch the selected variant inside the ``profile`` range."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    shape = (args.batch, args.heads, args.sequence, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    if args.variant.startswith("piper"):

        def launch() -> torch.Tensor:
            if args.variant == "piper":
                return sage_attention(query, key, value, is_causal=False)
            return _run_sage_attention(
                query,
                key,
                value,
                args.head_dim**-0.5,
                False,
                qk_quantization_range=127,
                value_transposed=args.variant != "piper-row-major",
                use_tensor_descriptors=args.variant == "piper-descriptor",
            )

    else:
        launch = _canonical_launcher(
            query,
            key,
            value,
            accumulator="fp32+fp16" if args.variant == "canonical2pp" else "fp32+fp32",
        )

    for _ in range(args.warmup_iterations):
        launch()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("profile")
    for _ in range(args.iterations):
        launch()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
