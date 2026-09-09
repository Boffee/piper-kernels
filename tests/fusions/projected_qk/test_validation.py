"""Weightless Q/K projections keep explicit head widths through their custom-op schemas."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from piper_kernels.fusions.convrot_int8_sparse_piper import key as int8_key
from piper_kernels.fusions.convrot_int8_sparse_piper import query as int8_query
from piper_kernels.fusions.nvfp4_sparse_piper import key as nvfp4_key
from piper_kernels.fusions.nvfp4_sparse_piper import query as nvfp4_query
from piper_kernels.fusions.projected_qk._validation import resolve_head_dim


@pytest.mark.parametrize("head_dim", [64, 128])
def test_head_width_can_be_inferred_only_from_an_affine_norm(head_dim):
    norm = torch.empty(head_dim)
    assert resolve_head_dim(norm, None) == head_dim
    assert resolve_head_dim(norm, head_dim) == head_dim
    assert resolve_head_dim(None, head_dim) == head_dim


@pytest.mark.parametrize(
    ("shape", "head_dim", "message"),
    [
        (None, None, "explicit head_dim"),
        (None, True, "head_dim=64 or 128"),
        (None, 32, "head_dim=64 or 128"),
        (None, 64.0, "head_dim=64 or 128"),
        ((64,), 128, "must match head_dim"),
        ((2, 64), 64, "must match head_dim"),
        ((2, 64), None, "head_dim=64 or 128"),
    ],
)
def test_invalid_or_ambiguous_head_width_is_rejected(shape, head_dim, message):
    norm = torch.empty(shape) if shape is not None else None
    with pytest.raises(ValueError, match=message):
        resolve_head_dim(norm, head_dim)


@pytest.mark.parametrize("family", ["int8", "nvfp4"])
@pytest.mark.parametrize("operation", ["query", "key"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_weightless_projection_fake_kernel_keeps_the_explicit_head_width(
    family, operation, head_dim
):
    rows, heads, width = 193, 2, 256
    with torch.device("meta"):
        if family == "int8":
            prepared = (
                torch.empty((1, rows, width), dtype=torch.int8),
                torch.empty((1, rows), dtype=torch.float32),
                torch.empty((heads * head_dim, width), dtype=torch.int8),
                torch.empty((heads * head_dim, 1), dtype=torch.float32),
            )
            op = int8_query._project_query_op if operation == "query" else int8_key._project_key_op
        else:
            prepared = (
                torch.empty((rows, width // 2), dtype=torch.uint8),
                torch.empty((64, 64), dtype=torch.float8_e4m3fn),
                torch.empty((), dtype=torch.float32),
                torch.empty((heads * head_dim, width // 2), dtype=torch.uint8),
                torch.empty((64, 64), dtype=torch.float8_e4m3fn),
                None,
                None,
            )
            op = nvfp4_query.project_query if operation == "query" else nvfp4_key.project_key
        cos = torch.empty((rows, head_dim // 2), dtype=torch.float32)
        sin = torch.empty_like(cos)
    policy = (head_dim**-0.5,) if operation == "query" else ()
    policy += (128, 0) if family == "nvfp4" else (0,)
    args = (*prepared, None, cos, sin, 1e-5, *policy)
    with pytest.raises(ValueError, match="explicit head_dim"):
        op(*args)
    output = op(*args, head_dim=head_dim)
    assert output[0].shape == (1, heads, 256, head_dim)
    assert output[0].dtype is torch.int8
    assert output[2].shape == (1, heads, 4, head_dim)


def test_symbolic_head_width_is_validated_during_fake_execution():
    mode = FakeTensorMode(shape_env=ShapeEnv())
    norm = mode.from_tensor(torch.empty(128), static_shapes=False)
    head_dim = norm.shape[0]
    assert isinstance(head_dim, torch.SymInt)
    assert resolve_head_dim(norm, None) == 128
    assert resolve_head_dim(None, head_dim) == 128
