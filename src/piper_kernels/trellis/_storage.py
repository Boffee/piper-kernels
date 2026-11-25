"""Trellis-coded storage of a ConvRot INT8 linear weight, and its encoder entry point."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels._triton import runtime
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.weights.convrot._rotation import rotate_groups

from ._layout import TrellisLayout, resolve_block_size, resolve_layout


def _require_encoder_device(device: torch.device) -> None:
    """Raise unless ``device`` is an NVIDIA SM80+ CUDA GPU with Triton 3.9+ Gluon support."""
    target = AcceleratorTarget.from_device(device)
    supported = target.is_nvidia_cuda and runtime.supports_device(device)
    if supported:
        try:
            from . import _gluon_forward  # noqa: PLC0415 - Triton/Gluon is optional

            supported = _gluon_forward.supports_device(device)
        except ImportError:
            supported = False
    if not supported:
        raise ValueError(
            "trellis encoding requires an NVIDIA CUDA device (SM80+) with Triton 3.9+, "
            f"got {device}"
        )


@dataclass(frozen=True, slots=True)
class TrellisStorage:
    """The stored tensors of one trellis-coded ConvRot INT8 linear weight.

    For a logical ``[out, in]`` weight cut into blocks of ``T = block_size``
    weights, ``bitstreams`` is uint8 ``[out, in / T, bytes]``, ``block_scales`` is
    float8_e4m3fn ``[out, in / T]`` and ``scale`` is the float32 row scale
    ``[out, 1]``. Decode them with ``ConvRotInt8Tensor.from_trellis`` or
    ``copy_from_trellis_``; the package README specifies the format.
    """

    bitstreams: torch.Tensor
    block_scales: torch.Tensor
    scale: torch.Tensor
    block_size: int

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        *,
        group_size: int,
        bits_per_weight: float | None = None,
        bytes_per_block: int | None = None,
    ) -> TrellisStorage:
        """Rotate and trellis-encode a high-precision linear weight.

        ``hp_tensor`` is a 2-D float16, bfloat16 or float32 ``[out, in]`` weight on an
        NVIDIA SM80+ CUDA device (99 KiB opt-in shared memory, Triton 3.9+); it is rotated
        in FP32 for ConvRot ``group_size`` and encoded there. Specify either
        ``bits_per_weight`` (the stored rate counting each block's FP8 scale: each block
        takes the most whole bytes within it) or an exact whole-byte ``bytes_per_block``,
        both yielding 2 to 6 code bits per weight. The block size is the largest multiple
        of 16, up to 256, that divides ``in``. For example, 4.5 bits per weight or
        ``bytes_per_block=143`` stores each 256-weight block in 143 bytes plus its scale.

        Encoding searches each block's walk and scale against the exact INT8
        decode. Unlike wrapping supplied storage, it reads the weight's values and
        raises ``ValueError`` when the rotated weight is not finite. Results are
        deterministic on a given device and software stack. ``hp_tensor`` is
        detached first.
        """
        # The INT8 package needs TorchAO, which importing this package must not.
        from piper_kernels.weights.convrot.int8._quantization import (  # noqa: PLC0415
            validate_hp_weight,
        )

        source = hp_tensor.detach()
        validate_hp_weight(source, group_size, ndims=(2,))
        if (bits_per_weight is None) == (bytes_per_block is None):
            raise ValueError("specify exactly one of bits_per_weight or bytes_per_block")
        if bits_per_weight is not None:
            layout = resolve_layout(source.shape[1], bits_per_weight)
        else:
            assert bytes_per_block is not None
            layout = TrellisLayout.from_storage(
                resolve_block_size(source.shape[1]), bytes_per_block
            )
        _require_encoder_device(source.device)
        rotated = rotate_groups(source.float(), group_size)
        # Offline conversion reads values: a non-finite weight would encode to
        # unspecified streams.
        if not bool(torch.isfinite(rotated).all()):
            raise ValueError("trellis encoding requires a finite weight")
        from ._encoder import encode_matrix  # noqa: PLC0415 - Triton is optional

        bitstreams, block_scales, scale = encode_matrix(rotated, layout)
        return cls(bitstreams, block_scales, scale, layout.block_size)


__all__ = ["TrellisStorage"]
