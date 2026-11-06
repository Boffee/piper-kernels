"""Attention residual-add operators take the add's arguments, then their output's."""

import pytest

from piper_kernels.fusions.convrot_int8_piper import _compile as dense_compile
from piper_kernels.fusions.convrot_int8_piper import output as dense
from piper_kernels.fusions.convrot_int8_sparse_piper import _compile as sparse_compile
from piper_kernels.fusions.convrot_int8_sparse_piper import output as sparse

_PAIRS = {
    "dense": (
        dense._projected_query_attention_output_op,
        dense._projected_query_attention_residual_add_op,
        dense_compile._HEAD_SLICING,
    ),
    "sparse": (
        sparse._projected_query_attention_output_op,
        sparse._projected_query_attention_residual_add_op,
        sparse_compile._HEAD_SLICING,
    ),
}


@pytest.mark.parametrize("name", _PAIRS)
def test_residual_add_takes_the_add_then_the_output_arguments(name):
    output, residual_add, slicing = _PAIRS[name]
    output_names = [
        argument.name
        for argument in output._opoverload._schema.arguments
        if argument.name != "output_dtype"
    ]

    # A head slice's position, when the add takes one, follows; its defaults mean one slice.
    assert [argument.name for argument in residual_add._opoverload._schema.arguments] == [
        "base",
        "gate",
        "gate_indices",
        "python_indexing",
        *output_names,
        *(slicing.position or ()),
    ]
