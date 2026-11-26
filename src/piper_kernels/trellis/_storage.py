"""Trellis-coded storage of a ConvRot INT8 linear weight, and its encoder entry point."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from piper_kernels._triton import runtime
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels._validation import validate_finite_float
from piper_kernels.weights.convrot._rotation import rotate_groups, validate_group_size

from ._layout import TrellisLayout, resolve_block_size, resolve_layout, validate_block_size

_DEFAULT_HESSIAN_DAMPING = 0.01
_HESSIAN_DAMPING_MULTIPLIERS = (1.0, 2.0, 5.0, 10.0, 25.0)
_SUPPORTED_HESSIAN_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


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


def _validate_damping(damping: float) -> float:
    validated = validate_finite_float(
        damping,
        "trellis Hessian damping must be finite and non-negative",
        type_message="trellis Hessian damping must be a real scalar",
    )
    if validated < 0.0:
        raise ValueError(f"trellis Hessian damping must be non-negative, got {damping}")
    return validated


def _validate_raw_hessian(hessian: torch.Tensor, *, group_size: int, block_size: int | None) -> int:
    if not isinstance(hessian, torch.Tensor):
        raise TypeError(f"trellis Hessian must be a torch.Tensor, got {type(hessian).__name__}")
    validate_group_size(group_size)
    if hessian.layout is not torch.strided:
        raise ValueError("trellis Hessian must use strided layout")
    if hessian.device.type == "meta":
        raise ValueError("trellis Hessian cannot factorize a meta tensor without values")
    if hessian.dtype not in _SUPPORTED_HESSIAN_DTYPES:
        raise ValueError(
            "trellis Hessian must have floating-point dtype float16, bfloat16, float32, or "
            f"float64, got {hessian.dtype}"
        )
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or hessian.shape[0] <= 0:
        raise ValueError(
            "trellis Hessian must be a non-empty square 2-D matrix [in, in], "
            f"got {tuple(hessian.shape)}"
        )
    cols = hessian.shape[0]
    resolved_block = (
        resolve_block_size(cols) if block_size is None else validate_block_size(block_size)
    )
    if cols % group_size:
        raise ValueError(
            f"trellis Hessian width {cols} is not divisible by group size {group_size}"
        )
    if cols % resolved_block:
        raise ValueError(
            f"trellis Hessian width {cols} is not divisible by block size {resolved_block}"
        )
    return resolved_block


@torch.no_grad()
def _factorize_rotated_hessian(
    hessian: torch.Tensor,
    *,
    group_size: int,
    block_size: int,
    damping: float,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Rotate raw ``[in, in]`` Hessian, damp, Cholesky-factor, and invert block diagonals."""
    # Transfer in source dtype before casting to FP64 so host H2D avoids doubling bandwidth.
    transferred = hessian.detach().to(device=device)
    h_f64 = transferred.to(torch.float64)
    del transferred
    if not bool(torch.isfinite(h_f64).all()):
        raise ValueError("trellis Hessian must be finite")
    h_scale = float(h_f64.abs().max().item())
    if h_scale == 0.0:
        if damping > 0.0:
            return torch.eye(h_f64.shape[0], dtype=torch.float32, device=device), damping
        raise ValueError("trellis Hessian is not positive definite")
    h_f64 = h_f64 / h_scale

    h_f64 = h_f64.add(h_f64.t()).mul_(0.5)
    half_rot = rotate_groups(h_f64, group_size)
    del h_f64
    h_rot = rotate_groups(half_rot.t().contiguous(), group_size)
    del half_rot
    h_rot = h_rot.add(h_rot.t()).mul_(0.5)

    cols = h_rot.shape[0]
    diag = h_rot.diagonal()
    pos = diag[diag > 0]
    if pos.numel() == 0:
        raise ValueError("trellis Hessian is not positive definite")

    mean_diag = float(pos.mean().item())
    diag_orig = diag.clone()
    l_chol: torch.Tensor | None = None
    used_damping = damping
    multipliers = (0.0,) if damping == 0.0 else _HESSIAN_DAMPING_MULTIPLIERS
    for mult in multipliers:
        cand = damping * mult
        if cand > 0.0:
            h_rot.diagonal().copy_(diag_orig + cand * mean_diag)
        cand_chol, info = torch.linalg.cholesky_ex(h_rot)
        if int(info.item()) == 0:
            l_chol = cand_chol
            used_damping = cand
            break
    del h_rot, diag_orig
    if l_chol is None:
        raise ValueError("trellis Hessian is not positive definite")

    eye_t = torch.eye(block_size, dtype=torch.float64, device=device)
    for s in range(0, cols, block_size):
        e = s + block_size
        l_bb = l_chol[s:e, s:e].clone()
        if e < cols:
            l_chol[e:, s:e].copy_(
                torch.linalg.solve_triangular(l_bb, l_chol[e:, s:e], upper=False, left=False)
            )
        l_chol[s:e, s:e].copy_(eye_t)
    feedback = l_chol.to(torch.float32).contiguous()
    del l_chol
    if not bool(torch.isfinite(feedback).all()):
        raise ValueError("trellis Hessian feedback overflowed float32; increase damping")
    return feedback, used_damping


@dataclass(frozen=True, slots=True)
class TrellisHessian:
    """Pre-factored Block-LDLQ feedback matrix in the ConvRot basis.

    Stores the float32 block unit-lower-triangular matrix ``feedback`` (``[in, in]``)
    whose strict block-lower entries equal ``L[e:, s:e] @ L[s:e, s:e]^{-1}`` for the
    damped Cholesky factor ``L @ L^T = M^T H M + damping * mean_diag * I``, alongside
    the ``group_size``, ``block_size``, and applied ``damping``. Build one with
    :meth:`from_hessian` and reuse it across weights sharing the same input covariance.
    """

    feedback: torch.Tensor
    group_size: int
    block_size: int
    damping: float = _DEFAULT_HESSIAN_DAMPING

    def __post_init__(self) -> None:
        validate_group_size(self.group_size)
        validate_block_size(self.block_size)
        _validate_damping(self.damping)
        fb = self.feedback
        if not isinstance(fb, torch.Tensor):
            raise TypeError(
                f"trellis Hessian feedback must be a torch.Tensor, got {type(fb).__name__}"
            )
        if fb.layout is not torch.strided:
            raise ValueError("trellis Hessian feedback must use strided layout")
        if fb.device.type == "meta":
            raise ValueError("trellis Hessian feedback cannot reside on a meta device")
        if fb.dtype is not torch.float32 or not fb.is_contiguous():
            raise ValueError(
                "trellis Hessian feedback must be a contiguous float32 tensor, "
                f"got {fb.dtype} (contiguous={fb.is_contiguous()})"
            )
        if fb.ndim != 2 or fb.shape[0] != fb.shape[1] or fb.shape[0] <= 0:
            raise ValueError(
                "trellis Hessian feedback must be a non-empty square 2-D matrix [in, in], "
                f"got {tuple(fb.shape)}"
            )
        if fb.shape[0] % self.group_size or fb.shape[0] % self.block_size:
            raise ValueError(
                f"trellis Hessian feedback width {fb.shape[0]} must be divisible by "
                f"group_size {self.group_size} and block_size {self.block_size}"
            )

    @classmethod
    def from_hessian(
        cls,
        hessian: torch.Tensor,
        *,
        group_size: int,
        block_size: int | None = None,
        damping: float = _DEFAULT_HESSIAN_DAMPING,
        device: torch.device | str | None = None,
    ) -> TrellisHessian:
        """Rotate and Cholesky-factor an unrotated second-moment matrix ``H [in, in]``.

        ``hessian`` is symmetrized, rotated in float64 into the ConvRot ``group_size``
        basis, damped relative to its positive diagonal mean (retried up to ``25x`` if
        ill-conditioned), and converted into the block unit-lower-triangular float32
        feedback matrix for ``block_size`` (defaulting to ``resolve_block_size(in)``).
        """
        validated_damping = _validate_damping(damping)
        resolved_block = _validate_raw_hessian(
            hessian, group_size=group_size, block_size=block_size
        )
        target_device = hessian.device if device is None else torch.device(device)
        if target_device.type == "meta":
            raise ValueError("trellis Hessian cannot factorize onto a meta device")
        feedback, used_damping = _factorize_rotated_hessian(
            hessian,
            group_size=group_size,
            block_size=resolved_block,
            damping=validated_damping,
            device=target_device,
        )
        return cls(
            feedback=feedback,
            group_size=group_size,
            block_size=resolved_block,
            damping=used_damping,
        )

    def to(self, device: torch.device | str) -> TrellisHessian:
        """Return a view or copy whose ``feedback`` resides on ``device``."""
        with torch.no_grad():
            target = torch.device(device)
            moved = self.feedback.to(device=target)
            if moved is self.feedback:
                return self
            return TrellisHessian(
                feedback=moved.contiguous(),
                group_size=self.group_size,
                block_size=self.block_size,
                damping=self.damping,
            )


def _validate_hessian_argument(
    hessian: torch.Tensor | TrellisHessian | None,
    hessian_damping: float | None,
    *,
    in_features: int,
    group_size: int,
    block_size: int,
    weight_device: torch.device,
) -> float | None:
    if hessian is None:
        if hessian_damping is not None:
            raise ValueError("hessian_damping requires hessian")
        return None
    if isinstance(hessian, TrellisHessian):
        if hessian_damping is not None:
            raise ValueError("hessian_damping cannot be passed with a pre-factored TrellisHessian")
        if hessian.group_size != group_size:
            raise ValueError(
                f"TrellisHessian group_size {hessian.group_size} does not match {group_size}"
            )
        if hessian.block_size != block_size:
            raise ValueError(
                f"TrellisHessian block_size {hessian.block_size} does not match {block_size}"
            )
        if tuple(hessian.feedback.shape) != (in_features, in_features):
            raise ValueError(
                f"TrellisHessian shape {tuple(hessian.feedback.shape)} does not match "
                f"{(in_features, in_features)}"
            )
        if hessian.feedback.device != weight_device:
            raise ValueError(
                f"TrellisHessian feedback device {hessian.feedback.device} does not match "
                f"weight device {weight_device}"
            )
        return None
    if isinstance(hessian, torch.Tensor):
        damping = (
            _DEFAULT_HESSIAN_DAMPING
            if hessian_damping is None
            else _validate_damping(hessian_damping)
        )
        _validate_raw_hessian(hessian, group_size=group_size, block_size=block_size)
        if hessian.shape[0] != in_features:
            raise ValueError(
                f"trellis Hessian shape {tuple(hessian.shape)} does not match "
                f"{(in_features, in_features)}"
            )
        return damping
    raise TypeError(
        "trellis hessian must be a torch.Tensor, TrellisHessian, or None, "
        f"got {type(hessian).__name__}"
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
        hessian: torch.Tensor | TrellisHessian | None = None,
        hessian_damping: float | None = None,
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

        Optionally pass ``hessian`` (a raw 2-D unrotated second-moment matrix ``[in, in]``
        or a reusable :class:`TrellisHessian`) and ``hessian_damping`` (default ``0.01`` for
        raw tensors) to run Block-LDLQ error feedback across blocks from right to left while
        preserving ``scale``. Encoding searches each block's walk and scale against the exact
        INT8 decode. Unlike wrapping supplied storage, it reads the weight's values and raises
        ``ValueError`` when the rotated weight (or raw Hessian) is not finite. Results are
        deterministic on a given device and software stack. ``hp_tensor`` is detached first.
        """
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
        damping = _validate_hessian_argument(
            hessian,
            hessian_damping,
            in_features=source.shape[1],
            group_size=group_size,
            block_size=layout.block_size,
            weight_device=source.device,
        )
        _require_encoder_device(source.device)
        with torch.no_grad():
            rotated = rotate_groups(source.float(), group_size)
            # Offline conversion reads values: a non-finite weight would encode to
            # unspecified streams.
            if not bool(torch.isfinite(rotated).all()):
                raise ValueError("trellis encoding requires a finite weight")

            feedback: torch.Tensor | None = None
            if isinstance(hessian, TrellisHessian):
                feedback = hessian.feedback.detach()
            elif isinstance(hessian, torch.Tensor):
                assert damping is not None
                feedback, _ = _factorize_rotated_hessian(
                    hessian,
                    group_size=group_size,
                    block_size=layout.block_size,
                    damping=damping,
                    device=source.device,
                )

            from ._encoder import encode_matrix  # noqa: PLC0415 - Triton is optional

            bitstreams, block_scales, scale = encode_matrix(rotated, layout, feedback=feedback)
        return cls(bitstreams, block_scales, scale, layout.block_size)


__all__ = ["TrellisHessian", "TrellisStorage"]
