import pytest
from _lib.attention import AttentionConfig, AttentionShape


def test_attention_shape_expands_implicit_kv_heads() -> None:
    shape = AttentionShape(
        batch=2,
        query_heads=16,
        key_value_heads=4,
        query_sequence=1024,
        key_value_sequence=2048,
        head_dim=128,
    )

    assert shape.kv_heads == 4
    assert shape.as_dict() == {
        "batch": 2,
        "query_heads": 16,
        "key_value_heads": 4,
        "query_sequence": 1024,
        "key_value_sequence": 2048,
        "head_dim": 128,
    }


@pytest.mark.parametrize(
    "shape",
    [
        AttentionShape,
        lambda **values: AttentionShape(key_value_heads=3, **values),
    ],
)
def test_attention_shape_rejects_invalid_dimensions(shape) -> None:
    values = {
        "batch": 1,
        "query_heads": 8,
        "query_sequence": 32,
        "key_value_sequence": 32,
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
        "tensor_layout": "HND",
    }
