"""Quantized weight formats shared by inference operators.

Import a format from its subpackage or import :class:`QuantizedWeight` for
format-independent storage transport.
"""

from ._dispatch import QuantizedWeight

__all__ = ["QuantizedWeight"]
