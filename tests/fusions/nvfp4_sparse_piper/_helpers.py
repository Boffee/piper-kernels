"""Shared NVFP4 sparse-Piper test operands and materialized references."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.kernels.sparse_piper.layout import padded_sequence_length
from piper_kernels.linear.nvfp4._projection import matmul_prepared_chunk_out

_HEAD_DIM = 128
_TILE_ROWS = 64


@dataclass(frozen=True, slots=True)
class Projection:
    """One prepared activation and compatible NVFP4 projection weight."""

    input_qdata: torch.Tensor
    input_scale: torch.Tensor
    input_per_tensor_scale: torch.Tensor
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    weight_per_tensor_scale: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_qdata,
            self.input_scale,
            self.input_per_tensor_scale,
            self.weight_qdata,
            self.weight_scale,
            self.weight_per_tensor_scale,
        )


@dataclass(frozen=True, slots=True)
class Operands:
    """Canonical batch-one Q/K/V region used by the fused operator tests."""

    input: torch.Tensor
    activation_scale: torch.Tensor
    weights: tuple[TorchAONVFP4Tensor, TorchAONVFP4Tensor, TorchAONVFP4Tensor]
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    def projection(self, index: int) -> Projection:
        prepared = TorchAONVFP4Tensor.to_nvfp4(
            self.input.reshape(-1, self.input.shape[-1]),
            per_tensor_scale=self.activation_scale,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )
        weight = self.weights[index]
        assert weight.per_tensor_scale is not None
        return Projection(
            prepared.qdata,
            prepared.scale,
            prepared.per_tensor_scale,
            weight.qdata,
            weight.scale,
            weight.per_tensor_scale,
        )


def exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def make_operands(
    *,
    sequence_length: int = 193,
    input_features: int = 256,
    heads: int = 2,
    seed: int = 811,
) -> Operands:
    """Create deterministic static-scale NVFP4 operands."""
    torch.manual_seed(seed)
    input = torch.randn(  # noqa: A001
        (1, sequence_length, input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    activation_scale = per_tensor_amax_to_scale(input.abs().amax())
    quantization = QuantizeTensorToNVFP4Kwargs(
        block_size=16,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        use_dynamic_per_tensor_scale=False,
    )
    weights = []
    for _ in range(3):
        dense = torch.randn(
            (heads * _HEAD_DIM, input_features),
            device="cuda",
            dtype=torch.bfloat16,
        )
        weights.append(
            TorchAONVFP4Tensor.to_nvfp4(
                dense,
                per_tensor_scale=per_tensor_amax_to_scale(dense.abs().amax()),
                act_per_tensor_scale=activation_scale,
                is_swizzled_scales=True,
                act_quant_kwargs=quantization,
            )
        )
    norms = tuple(
        torch.rand(_HEAD_DIM, device="cuda", dtype=torch.float32).add_(0.5).bfloat16()
        for _ in range(2)
    )
    angles = torch.rand(
        (sequence_length, 96),
        device="cuda",
        dtype=torch.float32,
    ).mul_(2 * torch.pi)
    return Operands(
        input,
        activation_scale,
        tuple(weights),  # type: ignore[arg-type]
        *norms,
        angles.cos().contiguous(),
        angles.sin().contiguous(),
    )


def materialize_projection(
    projection: Projection,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the raw BF16 GEMM and FP32 fused linear epilogue."""
    rows = projection.input_qdata.shape[0]
    output = torch.empty(
        (rows, projection.weight_qdata.shape[0]),
        device=projection.input_qdata.device,
        dtype=torch.bfloat16,
    )
    raw = matmul_prepared_chunk_out(
        projection.input_qdata,
        projection.input_scale,
        projection.weight_qdata,
        projection.weight_scale,
        0,
        rows,
        output,
    )
    global_scale = projection.input_per_tensor_scale * projection.weight_per_tensor_scale
    projected = raw.float() * global_scale
    if bias is not None:
        projected += bias.float()
    return projected


def materialize_qk(
    projection: Projection,
    norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    norm_epsilon: float,
) -> torch.Tensor:
    """Apply the FP32 normalization/RoPE contract used by fused Q/K kernels."""
    sequence_length = projection.input_qdata.shape[0]
    heads = projection.weight_qdata.shape[0] // _HEAD_DIM
    projected = materialize_projection(projection, bias).view(
        sequence_length,
        heads,
        _HEAD_DIM,
    )
    normalized = F.rms_norm(
        projected.float(),
        (_HEAD_DIM,),
        norm.float(),
        norm_epsilon,
    )
    rotary_dim = cos.shape[1]
    rotary = normalized[..., :rotary_dim]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    rotary = rotary * cos[:, None, :] + rotated * sin[:, None, :]
    return torch.cat((rotary, normalized[..., rotary_dim:]), dim=-1).transpose(0, 1)[None]


def query_reference(
    query: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length = query.shape[2]
    storage_length = padded_sequence_length(sequence_length)
    padded = query.new_zeros((*query.shape[:2], storage_length, _HEAD_DIM))
    padded[:, :, :sequence_length] = query
    blocks = padded.unflatten(2, (storage_length // _TILE_ROWS, _TILE_ROWS))
    valid = torch.arange(storage_length, device=query.device) < sequence_length
    valid = valid.unflatten(0, (storage_length // _TILE_ROWS, _TILE_ROWS))
    summary = blocks.masked_fill(~valid[None, None, :, :, None], -torch.inf).amax(
        dim=3
    ) + blocks.masked_fill(~valid[None, None, :, :, None], torch.inf).amin(dim=3)
    quantized, scale = qk_quantization.prepare_query(
        query,
        softmax_scale,
        grouped=True,
        storage_query_length=storage_length,
    )
    return quantized, scale, summary


def key_reference(
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_length = key.shape[2]
    storage_length = padded_sequence_length(sequence_length)
    quantized, scale = qk_quantization.prepare_key(
        key,
        torch.zeros((*key.shape[:2], _HEAD_DIM), device=key.device, dtype=torch.float32),
        grouped=True,
        storage_key_length=storage_length,
    )
    padded = key.new_zeros((*key.shape[:2], storage_length, _HEAD_DIM))
    padded[:, :, :sequence_length] = key
    blocks = padded.unflatten(2, (storage_length // _TILE_ROWS, _TILE_ROWS))
    valid = torch.arange(storage_length, device=key.device) < sequence_length
    valid = valid.unflatten(0, (storage_length // _TILE_ROWS, _TILE_ROWS))
    maximum = blocks.masked_fill(~valid[None, None, :, :, None], -torch.inf).amax(dim=3)
    minimum = blocks.masked_fill(~valid[None, None, :, :, None], torch.inf).amin(dim=3)
    return quantized, scale, maximum, minimum


def value_reference(
    projected: torch.Tensor,
    value_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_length, heads, _head_dim = projected.shape
    storage_length = padded_sequence_length(sequence_length)
    centered = projected.new_zeros((1, heads, storage_length, _HEAD_DIM), dtype=torch.float32)
    centered[:, :, :sequence_length] = (
        projected.transpose(0, 1)[None].float() - value_mean[:, :, None, :]
    )
    blocks = centered.unflatten(2, (storage_length // _TILE_ROWS, _TILE_ROWS))
    scale = blocks.abs().amax(dim=(-1, -2)) / 127.0 + 1e-7
    normalized = blocks / scale[..., None, None]
    quantized = (
        torch.trunc(normalized + 0.5 * torch.where(normalized >= 0, 1.0, -1.0))
        .clamp(-127, 127)
        .to(torch.int8)
    )
    return (
        quantized.flatten(2, 3).permute(0, 1, 3, 2).contiguous(),
        (scale * 255.0).unsqueeze(-1),
    )
