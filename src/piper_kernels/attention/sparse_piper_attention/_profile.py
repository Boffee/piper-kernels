"""Runtime ratio configuration shared by sparse attention and its compiler fusions.

Only the head count is graph metadata. Ratio values stay opaque to Dynamo and
are read by the operator at execution, so replacing an immutable profile does
not specialize the enclosing model graph or require tensor-content readbacks.

Consumers carry ``cudagraph_unsafe``: automatic CUDA graph replay does not guard
symbolic custom-class values, and would reuse routes and budgets from capture.
Explicit CUDA graph capture remains valid for a fixed profile.
"""

from dataclasses import dataclass

from torch._custom_class_base import CustomClassBase
from torch._library.opaque_object import MemberType, register_custom_class


@dataclass(frozen=True, slots=True)
class _SparsePiperRatioProfile(CustomClassBase):
    """Normalized fixed-point ratios; construct from validated Python configuration."""

    units: tuple[int, ...]

    @property
    def num_heads(self) -> int:
        """Number of query heads covered by the profile."""
        return len(self.units)


def _profile_guards(profile: _SparsePiperRatioProfile) -> list[int]:
    return [profile.num_heads]


# Torch 2.14's custom-class boundary lifts this object as a graph input. Its
# fake counterpart exposes only the guarded head count to tracing and fusions.
register_custom_class(
    _SparsePiperRatioProfile,
    typ="symbolic",
    guard_fn=_profile_guards,
    members={"num_heads": MemberType.USE_REAL},
)
