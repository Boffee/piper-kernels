"""Trellis-coded weight storage.

The package README specifies the format. Encode a weight with
``TrellisStorage.from_hp`` and decode it with
``piper_kernels.weights.convrot.int8.ConvRotInt8Tensor.from_trellis``.
"""

from ._storage import TrellisStorage

__all__ = ["TrellisStorage"]
