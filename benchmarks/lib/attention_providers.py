"""Provider definitions for the full-attention development benchmark."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from collections.abc import Callable, Sequence
from typing import cast

import torch

from piper_kernels import sage_attention_2pp
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import triton as qk_backend
from piper_kernels.attention.piper_attention import triton as piper_attention_backend
from piper_kernels.attention.sage_attention_2pp import _policy as sage_attention_2pp_policy
from piper_kernels.attention.sage_attention_2pp import triton as sage_attention_2pp_backend

from .attention import AttentionConfig, AttentionInputs, run_sdpa
from .providers import BenchmarkProvider

CANONICAL_VERSION = "2.2.0"
CANONICAL_REVISION = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"

PIPER_ATTENTION = "piper_attention"
SAGE_ATTENTION_2PP = "sage_attention_2pp"
PYTORCH_SDPA = "pytorch-sdpa"
CANONICAL_CUDA_SAGE_ATTENTION_2PP = "canonical_cuda_sage_attention_2pp"
CANONICAL_CUDA_SAGE_ATTENTION_2 = "canonical_cuda_sage_attention_2"

PROVIDER_NAMES = (
    PIPER_ATTENTION,
    SAGE_ATTENTION_2PP,
    PYTORCH_SDPA,
    CANONICAL_CUDA_SAGE_ATTENTION_2PP,
    CANONICAL_CUDA_SAGE_ATTENTION_2,
)
SAGE_ATTENTION_FP8_PROVIDERS = (
    SAGE_ATTENTION_2PP,
    CANONICAL_CUDA_SAGE_ATTENTION_2PP,
    CANONICAL_CUDA_SAGE_ATTENTION_2,
)
TRITON_PROVIDERS = (PIPER_ATTENTION, SAGE_ATTENTION_2PP)

type AttentionProvider = BenchmarkProvider[object, torch.Tensor]
type CanonicalSageAttention = Callable[..., torch.Tensor]


def qk_quantization_granularity(target: AcceleratorTarget) -> str:
    """Return the SageAttention Q/K granularity used on an NVIDIA architecture."""
    return "per_warp" if target.is_cuda_capability(12) else "per_thread"


def default_provider_names(
    *,
    piper_attention_supported: bool,
    sage_attention_2pp_supported: bool,
) -> tuple[str, ...]:
    """Return a useful comparison set for the active accelerator."""
    names = [PIPER_ATTENTION] if piper_attention_supported else []
    if sage_attention_2pp_supported:
        names.append(SAGE_ATTENTION_2PP)
    names.append(PYTORCH_SDPA)
    return tuple(names)


def resolve_provider_names(
    requested: Sequence[str] | None,
    *,
    include_canonical: bool,
    piper_attention_supported: bool,
    sage_attention_2pp_supported: bool,
) -> tuple[str, ...]:
    """Resolve explicit providers, hardware-aware defaults, and canonical controls."""
    selected = (
        requested
        if requested is not None
        else default_provider_names(
            piper_attention_supported=piper_attention_supported,
            sage_attention_2pp_supported=sage_attention_2pp_supported,
        )
    )
    names = list(dict.fromkeys(selected))
    if include_canonical:
        for name in (CANONICAL_CUDA_SAGE_ATTENTION_2PP, CANONICAL_CUDA_SAGE_ATTENTION_2):
            if name not in names:
                names.append(name)
    return tuple(names)


def validate_provider_support(
    provider_names: Sequence[str],
    target: AcceleratorTarget,
) -> None:
    """Reject selected providers that cannot run on the active accelerator."""
    needs_piper_attention = PIPER_ATTENTION in provider_names
    needs_sage_attention_fp8 = any(name in SAGE_ATTENTION_FP8_PROVIDERS for name in provider_names)
    if needs_piper_attention and not target.supports_uint8_int8_mma:
        raise SystemExit(
            "Piper Attention providers require NVIDIA SM8x or consumer Blackwell SM12x"
        )
    if needs_sage_attention_fp8 and not target.supports_fp8_fp16_mma:
        raise SystemExit(
            "SageAttention 8+8 providers require NVIDIA FP8 tensor cores; "
            "the canonical RTX 30 fallback is a different FP16-PV algorithm"
        )


def _load_canonical_sage_attention(capability: tuple[int, int]) -> CanonicalSageAttention:
    try:
        distribution = importlib.metadata.distribution("sageattention")
    except importlib.metadata.PackageNotFoundError as error:
        architecture = f"{capability[0]}.{capability[1]}"
        raise SystemExit(
            "Canonical SageAttention is unavailable. Build the pinned benchmark "
            "dependency with:\n"
            f"  TORCH_CUDA_ARCH_LIST={architecture} uv sync --group benchmark"
        ) from error
    if distribution.version != CANONICAL_VERSION:
        raise SystemExit(
            "Canonical SageAttention benchmarks require "
            f"sageattention=={CANONICAL_VERSION}; found {distribution.version}"
        )

    direct_url = distribution.read_text("direct_url.json")
    try:
        direct_url_metadata = None if direct_url is None else json.loads(direct_url)
    except json.JSONDecodeError as error:
        raise SystemExit("sageattention has invalid direct_url.json provenance metadata") from error
    vcs_info = (
        direct_url_metadata.get("vcs_info") if isinstance(direct_url_metadata, dict) else None
    )
    installed_revision = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if installed_revision != CANONICAL_REVISION:
        raise SystemExit(
            "Canonical SageAttention benchmarks require revision "
            f"{CANONICAL_REVISION}; found {installed_revision or 'no VCS revision metadata'}"
        )

    try:
        module = importlib.import_module("sageattention")
    except (ImportError, OSError) as error:
        architecture = f"{capability[0]}.{capability[1]}"
        raise SystemExit(
            "Canonical SageAttention is unavailable. Build the pinned benchmark "
            "dependency with:\n"
            f"  TORCH_CUDA_ARCH_LIST={architecture} uv sync --group benchmark"
        ) from error
    return cast(CanonicalSageAttention, module.sageattn_qk_int8_pv_fp8_cuda)


def _qk_jit_functions(target: AcceleratorTarget) -> dict[str, object]:
    if target.is_cuda_capability(12):
        return {
            "quantize-query-per-warp": qk_backend.quantize_query_per_warp_kernel,
            "quantize-key-per-block": qk_backend.quantize_key_per_block_kernel,
        }
    return {
        "quantize-query-per-thread": qk_backend.quantize_query_per_thread_kernel,
        "quantize-key-per-thread": qk_backend.quantize_key_per_thread_kernel,
    }


def _sage_attention_2pp_jit_functions(
    plan: sage_attention_2pp_policy.SageAttention2ppExecutionPlan,
) -> dict[str, object]:
    key_name, key_kernel = (
        ("quantize-key-per-block", qk_backend.quantize_key_per_block_kernel)
        if plan.grouped_qk
        else ("quantize-key-per-thread", qk_backend.quantize_key_per_thread_kernel)
    )
    quantization_kernels: dict[str, object] = {
        "quantize-query-per-warp" if plan.grouped_qk else "quantize-query-per-thread": (
            qk_backend.quantize_query_per_warp_kernel
            if plan.grouped_qk
            else qk_backend.quantize_query_per_thread_kernel
        ),
        key_name: key_kernel,
        "quantize-value-per-channel": sage_attention_2pp_backend._quantize_value_kernel,
    }
    return {
        "kv-statistics-partial": sage_attention_2pp_backend._kv_statistics_partial_kernel,
        "kv-statistics-finish": sage_attention_2pp_backend._finish_kv_statistics_kernel,
        **quantization_kernels,
        "attention": sage_attention_2pp_backend._sage_attention_2pp_kernel,
    }


def _piper_attention_jit_functions(
    target: AcceleratorTarget,
) -> dict[str, object]:
    return {
        "kv-mean-partial": piper_attention_backend._kv_mean_partial_kernel,
        "kv-mean-finish": piper_attention_backend._kv_mean_finalize_kernel,
        **_qk_jit_functions(target),
        "quantize-value-per-key": piper_attention_backend._quantize_value_per_key_kernel,
        "attention": piper_attention_backend._piper_attention_kernel,
    }


def _make_piper_attention_provider(
    inputs: AttentionInputs,
    *,
    config: AttentionConfig,
    target: AcceleratorTarget,
) -> AttentionProvider:
    query, key, value = inputs
    scale = config.scale if config.scale is not None else query.shape[-1] ** -0.5
    plan = piper_attention_backend._default_piper_attention_execution_plan(
        query,
        config.is_causal,
        target=target,
    )

    def prepare() -> object:
        return piper_attention_backend._prepare_piper_attention(
            query,
            key,
            value,
            scale,
            config.is_causal,
            execution_plan=plan,
        )

    def run(prepared: object) -> torch.Tensor:
        return piper_attention_backend._launch_piper_attention(
            cast(piper_attention_backend._PreparedPiperAttention, prepared)
        )

    return BenchmarkProvider(
        name=PIPER_ATTENTION,
        prepare=prepare,
        run=run,
        synchronize=torch.cuda.synchronize,
        configuration={
            **config.as_dict(),
            "implementation": "pure_triton",
            "algorithm": "piper_attention",
            **plan.as_dict(),
            "qk_quantization": qk_quantization_granularity(target),
            "probability_dtype": "uint8",
            "value_dtype": "int8",
            "value_scale": "per_key",
        },
        triton_jit_functions=_piper_attention_jit_functions(target),
    )


def _make_sage_attention_2pp_provider(
    name: str,
    inputs: AttentionInputs,
    *,
    config: AttentionConfig,
    target: AcceleratorTarget,
) -> AttentionProvider:
    query, _, _ = inputs
    plan = sage_attention_2pp_backend._default_sage_attention_2pp_execution_plan(
        query,
        config.is_causal,
        target=target,
    )

    def prepare() -> object:
        return inputs

    def run(prepared: object) -> torch.Tensor:
        return sage_attention_2pp(
            *cast(AttentionInputs, prepared),
            scale=config.scale,
            is_causal=config.is_causal,
        )

    return BenchmarkProvider(
        name=name,
        prepare=prepare,
        run=run,
        synchronize=torch.cuda.synchronize,
        configuration={
            **config.as_dict(),
            "implementation": "pure_triton",
            "algorithm": "sage_attention_2pp",
            "qk_quantization": qk_quantization_granularity(target),
            "pv_accumulation": "fp32+fp16",
            "block_n": int(sage_attention_2pp_backend._BLOCK_N),
            **plan.as_dict(),
        },
        triton_jit_functions=_sage_attention_2pp_jit_functions(plan),
    )


def _make_sdpa_provider(
    inputs: AttentionInputs,
    config: AttentionConfig,
) -> AttentionProvider:
    """Build the full-precision PyTorch reference provider."""

    def prepare() -> object:
        return inputs

    return BenchmarkProvider(
        name=PYTORCH_SDPA,
        prepare=prepare,
        run=lambda prepared: run_sdpa(cast(AttentionInputs, prepared), config),
        synchronize=torch.cuda.synchronize,
        configuration={
            **config.as_dict(),
            "implementation": "pytorch",
            "algorithm": "scaled_dot_product_attention",
        },
    )


def _make_canonical_sage_attention_providers(
    inputs: AttentionInputs,
    provider_names: Sequence[str],
    *,
    config: AttentionConfig,
    target: AcceleratorTarget,
) -> dict[str, AttentionProvider]:
    """Build the requested revision-pinned canonical CUDA providers."""
    requested = tuple(
        specification
        for specification in (
            (CANONICAL_CUDA_SAGE_ATTENTION_2PP, "fp32+fp16", "sage_attention_2pp"),
            (CANONICAL_CUDA_SAGE_ATTENTION_2, "fp32+fp32", "sage_attention_2"),
        )
        if specification[0] in provider_names
    )
    if not requested:
        return {}
    capability = target.cuda_capability
    if capability is None:
        raise SystemExit("canonical SageAttention providers require NVIDIA CUDA")
    canonical_sage_attention = _load_canonical_sage_attention(capability)
    qk_granularity = qk_quantization_granularity(target)
    common_configuration = {
        **config.as_dict(),
        "implementation": "canonical_cuda",
        "canonical_version": CANONICAL_VERSION,
        "canonical_revision": CANONICAL_REVISION,
        "qk_quantization": qk_granularity,
    }

    def prepare() -> object:
        return inputs

    def make_run(pv_accumulation: str) -> Callable[[object], torch.Tensor]:
        def run(prepared: object) -> torch.Tensor:
            query, key, value = cast(AttentionInputs, prepared)
            return canonical_sage_attention(
                query,
                key,
                value,
                tensor_layout="HND",
                is_causal=config.is_causal,
                qk_quant_gran=qk_granularity,
                sm_scale=config.scale,
                pv_accum_dtype=pv_accumulation,
                smooth_k=True,
                smooth_v=False,
                return_lse=False,
            )

        return run

    return {
        name: BenchmarkProvider(
            name=name,
            prepare=prepare,
            run=make_run(pv_accumulation),
            synchronize=torch.cuda.synchronize,
            configuration={
                **common_configuration,
                "algorithm": algorithm,
                "pv_accumulation": pv_accumulation,
            },
        )
        for name, pv_accumulation, algorithm in requested
    }


def make_attention_providers(
    inputs: AttentionInputs,
    *,
    provider_names: Sequence[str],
    config: AttentionConfig,
    target: AcceleratorTarget,
) -> dict[str, AttentionProvider]:
    """Construct the selected providers in command-line order."""
    providers: dict[str, AttentionProvider] = {}
    if PIPER_ATTENTION in provider_names:
        providers[PIPER_ATTENTION] = _make_piper_attention_provider(
            inputs,
            config=config,
            target=target,
        )

    if SAGE_ATTENTION_2PP in provider_names:
        providers[SAGE_ATTENTION_2PP] = _make_sage_attention_2pp_provider(
            SAGE_ATTENTION_2PP,
            inputs,
            config=config,
            target=target,
        )
    if PYTORCH_SDPA in provider_names:
        providers[PYTORCH_SDPA] = _make_sdpa_provider(inputs, config)

    providers.update(
        _make_canonical_sage_attention_providers(
            inputs,
            provider_names,
            config=config,
            target=target,
        )
    )

    return {name: providers[name] for name in provider_names}
