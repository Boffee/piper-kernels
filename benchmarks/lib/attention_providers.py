"""Provider definitions for the full-attention development benchmark."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import cast

import torch

from piper_kernels._triton.targets import (
    supports_fp8_fp16_mma,
    supports_uint8_int8_mma,
)
from piper_kernels.attention import sage_attention_2pp
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_backend
from piper_kernels.attention.piper import triton as piper_backend
from piper_kernels.attention.piper.dispatch import _default_center_value
from piper_kernels.attention.sage2pp import triton as sage_backend

from .attention import AttentionConfig, AttentionInputs
from .providers import BenchmarkProvider

CANONICAL_VERSION = "2.2.0"
CANONICAL_REVISION = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"

PIPER = "piper"
PIPER_CENTERED = "piper-centered"
PIPER_UNCENTERED = "piper-uncentered"
PIPER_AFFINE = "piper-affine"
PURE_TRITON_SAGE2PP = "pure-triton-sage2pp"
PYTORCH_SDPA = "pytorch-sdpa"
CANONICAL_SAGE2PP = "canonical-cuda-sage2pp"
CANONICAL_SAGE2 = "canonical-cuda-sage2"

PROVIDER_NAMES = (
    PIPER,
    PIPER_CENTERED,
    PIPER_UNCENTERED,
    PIPER_AFFINE,
    PURE_TRITON_SAGE2PP,
    PYTORCH_SDPA,
    CANONICAL_SAGE2PP,
    CANONICAL_SAGE2,
)
PIPER_PROVIDERS = (PIPER, PIPER_CENTERED, PIPER_UNCENTERED, PIPER_AFFINE)
FP8_SAGE_PROVIDERS = (PURE_TRITON_SAGE2PP, CANONICAL_SAGE2PP, CANONICAL_SAGE2)
TRITON_PROVIDERS = (*PIPER_PROVIDERS, PURE_TRITON_SAGE2PP)

type AttentionProvider = BenchmarkProvider[object, torch.Tensor]
type CanonicalSage = Callable[..., torch.Tensor]


def qk_quantization_granularity(capability: tuple[int, int]) -> str:
    """Return the Sage Q/K granularity used on an NVIDIA architecture."""
    return "per_warp" if capability[0] == 12 else "per_thread"


def default_provider_names(
    *,
    piper_supported: bool,
    fp8_supported: bool,
) -> tuple[str, ...]:
    """Return a useful comparison set for the active accelerator."""
    names = [PIPER, PIPER_UNCENTERED] if piper_supported else []
    if fp8_supported:
        names.append(PURE_TRITON_SAGE2PP)
    names.append(PYTORCH_SDPA)
    return tuple(names)


def resolve_provider_names(
    requested: Sequence[str] | None,
    *,
    include_canonical: bool,
    piper_supported: bool,
    fp8_supported: bool,
) -> tuple[str, ...]:
    """Resolve explicit providers, hardware-aware defaults, and canonical controls."""
    selected = (
        requested
        if requested is not None
        else default_provider_names(
            piper_supported=piper_supported,
            fp8_supported=fp8_supported,
        )
    )
    names = list(dict.fromkeys(selected))
    if include_canonical:
        for name in (CANONICAL_SAGE2PP, CANONICAL_SAGE2):
            if name not in names:
                names.append(name)
    return tuple(names)


def validate_provider_support(
    provider_names: Sequence[str],
    device: torch.device,
) -> None:
    """Reject selected providers that cannot run on the active accelerator."""
    needs_piper = any(name in PIPER_PROVIDERS for name in provider_names)
    needs_fp8 = any(name in FP8_SAGE_PROVIDERS for name in provider_names)
    if needs_piper and not supports_uint8_int8_mma(device):
        raise SystemExit(
            "Piper Attention providers require NVIDIA SM8x or consumer Blackwell SM12x"
        )
    if needs_fp8 and not supports_fp8_fp16_mma(device):
        raise SystemExit(
            "Sage 8+8 providers require NVIDIA FP8 tensor cores; "
            "the canonical RTX 30 fallback is a different FP16-PV algorithm"
        )


def run_sdpa(inputs: AttentionInputs, config: AttentionConfig) -> torch.Tensor:
    """Run the common full-precision quality reference."""
    query, key, value = inputs
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=config.scale,
        is_causal=config.is_causal,
    )


def _load_canonical(capability: tuple[int, int]) -> CanonicalSage:
    try:
        module = importlib.import_module("sageattention")
    except (ImportError, OSError) as error:
        architecture = f"{capability[0]}.{capability[1]}"
        raise SystemExit(
            "Canonical SageAttention is unavailable. Build the pinned benchmark "
            "dependency with:\n"
            f"  TORCH_CUDA_ARCH_LIST={architecture} uv sync --group benchmark"
        ) from error
    return cast(CanonicalSage, module.sageattn_qk_int8_pv_fp8_cuda)


def _qk_jit_functions(capability: tuple[int, int]) -> dict[str, object]:
    if capability[0] == 12:
        return {
            "quantize-query-per-warp": qk_backend.quantize_query_per_warp_kernel,
            "quantize-key-per-block": qk_backend.quantize_key_per_block_kernel,
        }
    return {
        "quantize-query-per-thread": qk_backend.quantize_query_per_thread_kernel,
        "quantize-key-per-thread": qk_backend.quantize_key_per_thread_kernel,
    }


def _sage_jit_functions(capability: tuple[int, int]) -> dict[str, object]:
    quantization_kernels = (
        {"quantize-key-value-role-dispatched": sage_backend._dispatch_kv_quantization_kernel}
        if capability[0] == 12
        else {
            **_qk_jit_functions(capability),
            "quantize-value-per-channel": sage_backend._quantize_value_kernel,
        }
    )
    return {
        "kv-statistics-partial": sage_backend._kv_statistics_partial_kernel,
        "kv-statistics-finish": sage_backend._finish_kv_statistics_kernel,
        **quantization_kernels,
        "attention": sage_backend._sage_attention_2pp_kernel,
    }


def _piper_jit_functions(
    capability: tuple[int, int],
    *,
    sort_value_rows: bool,
) -> dict[str, object]:
    qk_kernels = _qk_jit_functions(capability)
    if sort_value_rows and capability[0] == 12:
        qk_kernels["quantize-key-per-block"] = piper_backend._quantize_ordered_key_per_block_kernel
    kernels = {
        "kv-mean-partial": piper_backend._kv_mean_partial_kernel,
        "kv-mean-finish": piper_backend._kv_mean_finalize_kernel,
        **qk_kernels,
        "quantize-value-per-key": piper_backend._quantize_value_per_key_kernel,
        "attention": piper_backend._piper_attention_kernel,
    }
    if sort_value_rows:
        kernels["centered-value-row-range"] = piper_backend._centered_value_row_range_kernel
    return kernels


def _make_piper_provider(
    name: str,
    inputs: AttentionInputs,
    *,
    config: AttentionConfig,
    capability: tuple[int, int],
    center_value: bool,
    native_uint8: bool,
) -> AttentionProvider:
    query, key, value = inputs
    scale = config.scale if config.scale is not None else query.shape[-1] ** -0.5
    sort_value_rows = piper_backend._should_sort_value_rows(
        center_value=center_value,
        capability=capability,
        nvidia_cuda=True,
        is_causal=config.is_causal,
        head_dim=query.shape[-1],
        key_length=key.shape[2],
    )

    def prepare() -> object:
        return piper_backend._prepare_piper_attention(
            query,
            key,
            value,
            scale,
            config.is_causal,
            center_value,
            native_uint8=native_uint8,
            sort_value_rows=sort_value_rows,
        )

    def run(prepared: object) -> torch.Tensor:
        return piper_backend._launch_piper_attention(
            cast(piper_backend._PreparedPiperAttention, prepared)
        )

    return BenchmarkProvider(
        name=name,
        prepare=prepare,
        run=run,
        synchronize=torch.cuda.synchronize,
        configuration={
            **config.as_dict(),
            "implementation": "pure_triton",
            "algorithm": "piper_attention",
            "qk_quantization": qk_quantization_granularity(capability),
            "probability_dtype": "uint8",
            "value_dtype": "int8",
            "value_scale": "per_key",
            "center_value": center_value,
            "value_row_order": "centered_range_ascending" if sort_value_rows else "original",
            "mixed_sign_mma": "native" if native_uint8 else "affine_proxy",
        },
        triton_jit_functions=_piper_jit_functions(
            capability,
            sort_value_rows=sort_value_rows,
        ),
    )


def make_attention_providers(
    inputs: AttentionInputs,
    *,
    provider_names: Sequence[str],
    config: AttentionConfig,
    capability: tuple[int, int],
) -> dict[str, AttentionProvider]:
    """Construct the selected providers in command-line order."""
    query, key, _ = inputs
    default_centering = _default_center_value(query, key, config.is_causal)
    providers: dict[str, AttentionProvider] = {}
    piper_settings = {
        PIPER: (default_centering, True),
        PIPER_CENTERED: (True, True),
        PIPER_UNCENTERED: (False, True),
        PIPER_AFFINE: (default_centering, False),
    }
    for name, (center_value, native_uint8) in piper_settings.items():
        if name in provider_names:
            providers[name] = _make_piper_provider(
                name,
                inputs,
                config=config,
                capability=capability,
                center_value=center_value,
                native_uint8=native_uint8,
            )

    def prepare_inputs() -> object:
        return inputs

    if PURE_TRITON_SAGE2PP in provider_names:
        providers[PURE_TRITON_SAGE2PP] = BenchmarkProvider(
            name=PURE_TRITON_SAGE2PP,
            prepare=prepare_inputs,
            run=lambda prepared: sage_attention_2pp(
                *cast(AttentionInputs, prepared),
                scale=config.scale,
                is_causal=config.is_causal,
            ),
            synchronize=torch.cuda.synchronize,
            configuration={
                **config.as_dict(),
                "implementation": "pure_triton",
                "algorithm": "sage_attention_2pp",
                "qk_quantization": qk_quantization_granularity(capability),
                "pv_accumulation": "fp32+fp16",
            },
            triton_jit_functions=_sage_jit_functions(capability),
        )
    if PYTORCH_SDPA in provider_names:
        providers[PYTORCH_SDPA] = BenchmarkProvider(
            name=PYTORCH_SDPA,
            prepare=prepare_inputs,
            run=lambda prepared: run_sdpa(cast(AttentionInputs, prepared), config),
            synchronize=torch.cuda.synchronize,
            configuration={
                **config.as_dict(),
                "implementation": "pytorch",
                "algorithm": "scaled_dot_product_attention",
            },
        )

    canonical = (
        _load_canonical(capability)
        if CANONICAL_SAGE2PP in provider_names or CANONICAL_SAGE2 in provider_names
        else None
    )
    if canonical is not None:
        common_configuration = {
            **config.as_dict(),
            "implementation": "canonical_cuda",
            "canonical_version": CANONICAL_VERSION,
            "canonical_revision": CANONICAL_REVISION,
            "qk_quantization": qk_quantization_granularity(capability),
        }

        def canonical_run(pv_accumulation: str) -> Callable[[object], torch.Tensor]:
            def run(prepared: object) -> torch.Tensor:
                query, key, value = cast(AttentionInputs, prepared)
                return canonical(
                    query,
                    key,
                    value,
                    tensor_layout="HND",
                    is_causal=config.is_causal,
                    qk_quant_gran=qk_quantization_granularity(capability),
                    sm_scale=config.scale,
                    pv_accum_dtype=pv_accumulation,
                    smooth_k=True,
                    smooth_v=False,
                    return_lse=False,
                )

            return run

        for name, pv_accumulation, algorithm in (
            (CANONICAL_SAGE2PP, "fp32+fp16", "sage_attention_2pp"),
            (CANONICAL_SAGE2, "fp32+fp32", "sage_attention_2"),
        ):
            if name in provider_names:
                providers[name] = BenchmarkProvider(
                    name=name,
                    prepare=prepare_inputs,
                    run=canonical_run(pv_accumulation),
                    synchronize=torch.cuda.synchronize,
                    configuration={
                        **common_configuration,
                        "algorithm": algorithm,
                        "pv_accumulation": pv_accumulation,
                    },
                )

    return {name: providers[name] for name in provider_names}
