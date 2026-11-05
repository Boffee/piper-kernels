"""Encoder vectors: the research encoder's output on portable inputs.

``build_rows`` makes Gaussian-like rows from integer hashing, so the inputs are the
same on every device and PyTorch version. The digests are SHA-256 of the stored
tensors' bytes from ``trellis_vr_fast`` in piper-research
(``projects/e8-w4a8-vq``, ``encode_rows_fast`` then ``pack``) on an RTX 5090.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

_MASK = 0xFFFFFFFF


def _hash(index: torch.Tensor, salt: int) -> torch.Tensor:
    """A 32-bit integer hash of ``index`` (int64, below 2^31); products stay below 2^63."""
    h = (index * 0x9E3779B1 + salt) & _MASK
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & _MASK
    h ^= h >> 12
    h = (h * 0x297A2D39) & _MASK
    return h ^ (h >> 15)


def build_rows(rows: int, cols: int, seed: int) -> torch.Tensor:
    """FP32 ``[rows, cols]`` rows like ConvRot-rotated weights: a sum of four uniforms
    per weight, 1% of weights four times larger, and row magnitudes over two octaves."""
    index = torch.arange(rows * cols, dtype=torch.int64)
    total = torch.zeros(rows * cols, dtype=torch.float64)
    for draw in range(4):
        total += _hash(index, seed * 7919 + draw).double() / 2**32
    values = (total - 2.0) * 3**0.5
    outlier = _hash(index, seed * 7919 + 4) % 100 == 0
    values = torch.where(outlier, 4 * values, values).view(rows, cols)
    magnitude = 0.02 * 2.0 ** ((torch.arange(rows, dtype=torch.float64) % 5 - 2) / 2)
    return (values * magnitude[:, None]).float()


@dataclass(frozen=True)
class EncoderVector:
    rows: int
    cols: int
    block_bits: int
    seed: int
    bitstreams_sha256: str
    block_scales_sha256: str
    scale_sha256: str


def digest(tensor: torch.Tensor) -> str:
    """SHA-256 of a tensor's bytes in row-major order."""
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


# Cases, as (rows, cols, block_bits, seed); the block size is the format's choice for cols:
# 4/5-bit steps at 4.5 bits per weight with scales, then 224-weight blocks at that rate;
# 4-bit steps; 3/4-bit steps at 3.5; 3-bit steps; 5/6-bit steps at 5.5; 6-bit steps; and
# 112-weight blocks, whose final overlap spans the whole sequence.
ENCODER_VECTORS = (
    EncoderVector(
        6, 512, 1144, 1,
        "cf9bf1fe88a6fd256553e8382a4aaaa68242acf5de4e5de9b0fb06cab790a304",
        "97c5e397074266ba59225d731de3c02bf0115d148475cbe8632a67cd8b0b51cd",
        "85a11e07f02d17e4a54e089fd219fb076fbee7af856cb97489f98de034760887",
    ),
    EncoderVector(
        6, 448, 1000, 2,
        "5dc8dfca6b238943876a99ac7af6be32ac17c6bfc9936ac925b748b5ef1e134e",
        "8750678cf4d18c28d78b7d5a1c2aa111efa6db2633e2939896f6f1b94241cb81",
        "569c0674675e03df8771f4e62b3c9cbbca2aba565afa2b78a0dec67bd73ee950",
    ),
    EncoderVector(
        6, 512, 1024, 3,
        "909a62fe8261a3b563a1aaab84ad5436f033c36b3b05663e17b3b8f837fca0ca",
        "a9b2cec6a344fc687d53e687acbd2434fb651e00be07ada5ae99add7e65505ca",
        "7b7876897a2e5bb4eb2f3327f393118832c484599247012076576fec03595e57",
    ),
    EncoderVector(
        6, 512, 888, 4,
        "725a91ca7daccdabcda8fcc5b524071a7ea9a64f95ef4f7c5fd893d74e82315b",
        "5885eec712ee466216d8d655e8137f1008b200e4da2c6d638e84f9707a1e32bb",
        "b9d14884148320bc96703e4ee8c412c84db380e109dbd2e51483cfbee3fc0869",
    ),
    EncoderVector(
        6, 512, 768, 5,
        "077afdfe7c1bbc78b7c2b25972f24fee40582952fec3f4ef3dbe105e2160acf9",
        "aad7cedbb71b8c82ebe81c6687ced22d56e3f3ee453b20f9144fd0a772578280",
        "f3798aad18c5c19fe6a00c1330b8fa4ad45b37e4d11315e89f54ce25fa773e23",
    ),
    EncoderVector(
        6, 512, 1400, 6,
        "cc8ae86f07ce7ddbfc69c6f2bc9e5d71b92483a6a74af5ae28c4c61c58230030",
        "88c8d47c3770516acf378fe5d86ec55e7b5fca3bcd082f34e6c4a100c51ad73e",
        "b3ca57871010b143ea392d0dff3852b8a9725eada45a05df9f6742e87cdd9ad7",
    ),
    EncoderVector(
        6, 512, 1536, 7,
        "4a2bebf0a05895bb27d8b5267749417478c34f0c94edeb8edb2ffc230c88dfb1",
        "cf5a95e335e3874f51dc1df4939ba2a180ec44beaacb0bf2e1e9f8bc45e55547",
        "2e1bfbb6b72d27e88f001c2e320c5c90e9789eb163854c41c9cac610d56d3d71",
    ),
    EncoderVector(
        4, 336, 496, 8,
        "9af4a13f7c094738d55f39bcd4b972006495be54317599c3dc229fd519056963",
        "a72aaa7a5d39c8138acdc855ccb4fbe4c9c6b181e7a9d16ed8998035a5cd0d38",
        "396486ec19d012066ca70127789a35efa7955710a039aaf26ca2eb7f6798e14e",
    ),
)  # fmt: skip
