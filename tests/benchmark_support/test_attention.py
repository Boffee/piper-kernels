import pytest
import torch
from lib.attention import AttentionConfig, AttentionShape, make_attention_inputs


def test_attention_shape_expands_implicit_kv_heads() -> None:
    shape = AttentionShape(
        batch_size=2,
        num_query_heads=16,
        num_key_value_heads=4,
        query_length=1024,
        key_value_length=2048,
        head_dim=128,
    )

    assert shape.effective_num_key_value_heads == 4
    assert shape.as_dict() == {
        "batch_size": 2,
        "num_query_heads": 16,
        "num_key_value_heads": 4,
        "query_length": 1024,
        "key_value_length": 2048,
        "head_dim": 128,
    }


@pytest.mark.parametrize(
    "shape",
    [
        AttentionShape,
        lambda **values: AttentionShape(num_key_value_heads=3, **values),
    ],
)
def test_attention_shape_rejects_invalid_dimensions(shape) -> None:
    values = {
        "batch_size": 1,
        "num_query_heads": 8,
        "query_length": 32,
        "key_value_length": 32,
        "head_dim": 64,
    }
    if shape is AttentionShape:
        values["head_dim"] = 0

    with pytest.raises(ValueError, match=r"attention dimensions|query heads"):
        shape(**values)


def test_attention_config_uses_stable_names() -> None:
    config = AttentionConfig(dtype="float16", is_causal=True, scale=0.125)

    assert config.as_dict() == {
        "dtype": "float16",
        "is_causal": True,
        "scale": 0.125,
        "qkv_layout": "BHSD",
    }


def test_attention_inputs_follow_mha_and_gqa_shapes() -> None:
    shape = AttentionShape(2, 8, 5, 7, 64, num_key_value_heads=2)

    query, key, value = make_attention_inputs(
        shape,
        dtype=torch.float16,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(10),
    )

    assert query.shape == (2, 8, 5, 64)
    assert key.shape == value.shape == (2, 2, 7, 64)
    assert query.dtype is key.dtype is value.dtype is torch.float16
