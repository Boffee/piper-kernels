"""Tests for the ConvRot benchmark shape presets."""

from benchmark_convrot import BenchmarkShape, _benchmark_shapes, _parse_args, _quality_row_indices
from benchmark_convrot_preparation import _minimum_global_bytes


def test_custom_shapes_expand_requested_rows() -> None:
    arguments = _parse_args(["--rows", "3", "7", "--out-features", "96", "--in-features", "512"])

    shapes = _benchmark_shapes(arguments)

    actual = [(shape.name, shape.rows, shape.out_features, shape.in_features) for shape in shapes]
    assert actual == [
        ("custom", 3, 96, 512),
        ("custom", 7, 96, 512),
    ]


def test_minimax_h3_preset_uses_principal_five_second_linears() -> None:
    arguments = _parse_args(["--preset", "minimax-h3-5s"])

    shapes = _benchmark_shapes(arguments)

    actual = [(shape.name, shape.rows, shape.out_features, shape.in_features) for shape in shapes]
    assert actual == [
        ("qkv", 37_710, 21_504, 5_376),
        ("attention-out", 37_710, 5_376, 7_168),
        ("mlp-fc1", 37_710, 28_672, 5_376),
        ("mlp-fc2", 37_710, 5_376, 14_336),
    ]
    assert [shape.input_act for shape in shapes] == [None, None, None, "swiglu"]
    assert not any(shape.has_bias for shape in shapes)


def test_custom_shape_can_include_swiglu_without_bias() -> None:
    arguments = _parse_args(
        [
            "--rows",
            "7",
            "--out-features",
            "96",
            "--in-features",
            "512",
            "--input-act",
            "swiglu",
            "--no-bias",
        ]
    )

    (shape,) = _benchmark_shapes(arguments)

    assert shape.input_act == "swiglu"
    assert not shape.has_bias


def test_quality_rows_are_stratified_and_include_large_address_boundaries() -> None:
    shape = BenchmarkShape("qkv", 131_072, 21_504, 5_376, has_bias=False)

    rows = _quality_row_indices(shape)

    output_boundary = ((1 << 31) + shape.out_features - 1) // shape.out_features
    assert len(rows) == 256
    assert rows[0] == 0
    assert rows[-1] == shape.rows - 1
    assert {output_boundary - 1, output_boundary, output_boundary + 1} <= set(rows)


def test_quality_rows_include_raw_swiglu_input_address_boundary() -> None:
    shape = BenchmarkShape("mlp-fc2", 131_072, 5_376, 14_336, "swiglu", False)

    rows = _quality_row_indices(shape)

    raw_width = 2 * shape.in_features
    input_boundary = ((1 << 31) + raw_width - 1) // raw_width
    assert {input_boundary - 1, input_boundary, input_boundary + 1} <= set(rows)


def test_preparation_minimum_global_traffic_accounts_for_split_intermediate() -> None:
    rows, in_features, element_size = 3, 512, 2

    rotate = _minimum_global_bytes("rotate", rows, in_features, element_size)
    quantize = _minimum_global_bytes("quantize", rows, in_features, element_size)
    split = _minimum_global_bytes("split", rows, in_features, element_size)
    fused = _minimum_global_bytes("fused", rows, in_features, element_size)
    fused_swiglu = _minimum_global_bytes(
        "fused",
        rows,
        in_features,
        element_size,
        "swiglu",
    )

    assert rotate == 4 * rows * in_features
    assert quantize == 3 * rows * in_features + 4 * rows
    assert split == rotate + quantize
    assert fused == quantize
    assert fused_swiglu == 5 * rows * in_features + 4 * rows
