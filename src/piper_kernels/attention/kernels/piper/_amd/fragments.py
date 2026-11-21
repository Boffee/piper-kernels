"""Shared four-wave Q64 fragments for RDNA4 dense and sparse Piper."""

# Gluon device parameters, layouts and static tuples are not Python values.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.rdna4 import wmma

from piper_kernels._triton import amd_compiler_workarounds

from ..gluon import split_bf16_pair
from ._wmma import int8_wmma, uint8_int8_wmma

MMA_LAYOUT: gl.constexpr = AMDWMMALayout(
    version=2, transposed=True, warp_bases=[[0, 1, 0], [0, 2, 0]], rank=3
)
VECTOR_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])

_RESCALE_ASM = gl.constexpr(
    tuple(
        "\n".join(
            [f"v_cmp_neq_f32_e64 vcc_lo, 1.0, ${2 * registers}", "s_cbranch_vccz 1f"]
            + [f"v_mul_f32 ${index}, ${index}, ${2 * registers}" for index in range(registers)]
            + ["1:"]
        )
        for registers in (32, 64)
    )
)
_RESCALE_CONSTRAINTS = gl.constexpr(
    tuple(
        ",".join(
            ["=&v"] * registers
            + [str(index) for index in range(registers)]
            + ["v"] * registers
            + ["~{vcc}"]
        )
        for registers in (32, 64)
    )
)


# Whether FMAs that LLVM must not vectorize use opaque scalar assembly; see
# amd_compiler_workarounds.USE_SCALAR_FMA. Kernels read it at compile time.
_SCALAR_FMA: gl.constexpr = gl.constexpr(amd_compiler_workarounds.USE_SCALAR_FMA)


@gluon.jit
def scalar_fma(a, b, c):
    """Compute ``gl.fma(a, b, c)`` without letting LLVM bundle it into vector FMAs.

    RDNA4 has no packed FP32 FMA, so bundled FMAs only add register pressure.
    Where that happens, one ``v_fma_f32`` per element keeps the same rounding.
    """
    if _SCALAR_FMA:
        return gl.inline_asm_elementwise(
            "v_fma_f32 $0, $1, $2, $3",
            constraints="=v,v,v,v",
            args=(a, b, c),
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    return gl.fma(a, b, c)


@gluon.jit
def _recover_scores(biased, scalar: gl.constexpr):
    """Recover exact integer QK sums from the binade the accumulator bias selects."""
    if scalar and _SCALAR_FMA:
        # A literal 2^26 and an SGPR offset keep this FMA scalar without
        # spending VGPRs on splat constants; see scalar_fma.
        offset = gl.full(biased.shape, -10680707.0, gl.float32, biased.type.layout)
        return gl.inline_asm_elementwise(
            "v_fma_f32 $0, $1, 0x4c800000, $2",
            constraints="=v,v,s",
            args=(biased, offset),
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    return gl.fma(biased, 67108864.0, -10680707.0)


@gluon.jit
def split_columns(value):
    return gl.split(
        value.reshape([value.shape[0], value.shape[1], 2, value.shape[2] // 2]).permute(
            [0, 1, 3, 2]
        )
    )


@gluon.jit
def concat_columns(left, right):
    return (
        gl.join(left, right)
        .permute([0, 1, 3, 2])
        .reshape([left.shape[0], left.shape[1], 2 * left.shape[2]])
    )


@gluon.jit
def _concat_vectors(left, right):
    return gl.join(left, right).permute([0, 2, 1]).reshape([1, 2 * left.shape[1]])


@gluon.jit
def four_fragments(value):
    a, b = split_columns(value)
    a0, a1 = split_columns(a)
    b0, b1 = split_columns(b)
    return a0, a1, b0, b1


@gluon.jit
def eight_fragments(value):
    a0, a1, b0, b1 = four_fragments(value)
    a00, a01 = split_columns(a0)
    a10, a11 = split_columns(a1)
    b00, b01 = split_columns(b0)
    b10, b11 = split_columns(b1)
    return a00, a01, a10, a11, b00, b01, b10, b11


@gluon.jit
def _matrix_fragment(acc):
    left = _concat_vectors(_concat_vectors(acc[0], acc[1]), _concat_vectors(acc[2], acc[3]))
    right = _concat_vectors(_concat_vectors(acc[4], acc[5]), _concat_vectors(acc[6], acc[7]))
    # [register, wave-M, half-wave, row] -> [query row, column].
    matrix = _concat_vectors(left, right).reshape([1, 8, 4, 2, 16]).permute([0, 2, 4, 3, 1])
    return gl.convert_layout(matrix.reshape([1, 64, 16]), MMA_LAYOUT)


@gluon.jit
def _join_matrix(fragments):
    gl.static_assert(len(fragments) == 4 or len(fragments) == 8)
    result = concat_columns(
        concat_columns(fragments[0], fragments[1]), concat_columns(fragments[2], fragments[3])
    )
    if len(fragments) == 8:
        result = concat_columns(
            result,
            concat_columns(
                concat_columns(fragments[4], fragments[5]),
                concat_columns(fragments[6], fragments[7]),
            ),
        )
    return gl.convert_layout(result, MMA_LAYOUT)


@gluon.jit
def query_fragments(
    query_ptr,
    blocks,
    use_64bit_query_offsets: gl.constexpr,
    head_dim: gl.constexpr,
):
    threads = gl.arange(0, 128, gl.SliceLayout(0, VECTOR_LAYOUT))
    lane = threads % 32
    blocks = gl.convert_layout(blocks, gl.SliceLayout(1, VECTOR_LAYOUT))
    if use_64bit_query_offsets:
        blocks = blocks.to(gl.int64)
    rows = blocks[:, None] * 64 + (threads // 32 * 16 + lane % 16)[None, :]
    words = query_ptr.to(gl.pointer_type(gl.uint64))
    row_words = words + rows * (head_dim // 8) + (lane[None, :] // 16) * 2
    fragments = ()
    for k_pair in gl.static_range(head_dim // 32):
        pair_words = row_words + k_pair * 4
        fragments += (gl.load(pair_words), gl.load(pair_words + 1))
    return fragments


@gluon.jit
def qk_tiles(
    query,
    key_ptr,
    tile_0,
    tile_1,
    use_64bit_context_offsets: gl.constexpr,
    head_dim: gl.constexpr,
    two_tiles: gl.constexpr = True,
    scalar_score_recovery: gl.constexpr = False,
):
    """Return Q64 scores for one K64 tile or two independently selected K64 tiles.

    Paired mode preserves sparse Piper's K128 layout. Single-tile mode ignores
    ``tile_1`` and returns K64 scores for dense Piper's per-tile recurrence.
    ``scalar_score_recovery`` uses :func:`scalar_fma`'s form for score recovery,
    which dense Piper needs; sparse Piper keeps plain FMAs, about 1% faster there.
    Warp-pipeline ``"qk"`` stages run at ``priority=3`` above ``"softmax"``
    (``priority=0``).
    """
    offset_type: gl.constexpr = gl.int64 if use_64bit_context_offsets else gl.uint32
    lane = gl.arange(0, 128, gl.SliceLayout(0, VECTOR_LAYOUT)) % 32
    tile_0 = gl.convert_layout(tile_0, gl.SliceLayout(1, VECTOR_LAYOUT))
    tile_1 = gl.convert_layout(tile_1, gl.SliceLayout(1, VECTOR_LAYOUT))
    results = ()
    for group in gl.static_range(4 if two_tiles else 2):
        for column in gl.static_range(group * 2, group * 2 + 2):
            tile = tile_0 if column < 4 else tile_1
            row = tile[:, None] * 64 + (column % 4) * 16 + lane[None, :] % 16
            row = row.to(offset_type)
            acc = None
            for k_pair in gl.static_range(head_dim // 32):
                offset = (row * head_dim + k_pair * 32 + (lane[None, :] // 16) * 16).to(offset_type)
                pair_words = (key_ptr + offset).to(gl.pointer_type(gl.uint64))
                acc = int8_wmma(query[2 * k_pair], gl.load(pair_words), acc, column % 2)
                acc = int8_wmma(query[2 * k_pair + 1], gl.load(pair_words + 1), acc, column % 2)
            result: gl.tensor = _matrix_fragment(acc)
            results += (
                _recover_scores(result.to(gl.float32, bitcast=True), scalar_score_recovery),
            )
        with gl.amd.warp_pipeline_stage("qk", priority=3):
            pass
    return _join_matrix(results)


@gluon.jit
def compute_chunk_key_biases(
    query,
    key_center_ptr,
    key_center_scale_ptr,
    tiles_shared,
    key_bias_shared,
    query_scale,
    use_64bit_context_offsets: gl.constexpr,
    head_dim: gl.constexpr,
):
    """Precompute Q64 @ H(K_center)^T * Q_scale score biases for 16 tiles into LDS."""
    offset_type: gl.constexpr = gl.int64 if use_64bit_context_offsets else gl.uint32
    lane = gl.arange(0, 128, gl.SliceLayout(0, VECTOR_LAYOUT)) % 32
    column_in_mma = lane % 16
    center_term = column_in_mma % 2
    scale_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, MMA_LAYOUT))
    store_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 2], [4, 1], [0, 1])
    row_query_scale = gl.convert_layout(query_scale.reshape([64]), gl.SliceLayout(1, store_layout))[
        :, None
    ]

    for half in gl.static_range(2):
        tile_slot = half * 8 + column_in_mma // 2
        tile_index = tiles_shared.gather(tile_slot, 0).to(offset_type)[None, :]
        row_offset = (tile_index * (2 * head_dim) + center_term[None, :] * head_dim).to(offset_type)
        acc = None
        for k_pair in gl.static_range(head_dim // 32):
            offset = (row_offset + k_pair * 32 + (lane[None, :] // 16) * 16).to(offset_type)
            pair_words = (key_center_ptr + offset).to(gl.pointer_type(gl.uint64))
            acc = int8_wmma(query[2 * k_pair], gl.load(pair_words), acc, 0)
            acc = int8_wmma(query[2 * k_pair + 1], gl.load(pair_words + 1), acc, 0)
        recovered = _recover_scores(_matrix_fragment(acc).to(gl.float32, bitcast=True), False)
        column = gl.arange(0, 16, scale_layout)
        column_tile_index = tiles_shared.gather(half * 8 + column // 2, 0).to(offset_type)
        center_scales = gl.load(
            key_center_scale_ptr + column_tile_index * 2 + (column % 2).to(offset_type)
        )
        weighted = recovered * center_scales[None, None, :]
        dots = gl.convert_layout(
            gl.sum(weighted.reshape([1, 64, 8, 2]), axis=3).reshape([64, 8]),
            store_layout,
        )
        key_bias_shared.slice(half * 8, 8, dim=1).store(dots * row_query_scale)


@gluon.jit
def accumulate_chunk_value_means(
    numerator,
    tile_masses_shared,
    pair_max_shared,
    tiles_shared,
    running_max,
    value_mean_ptr,
    num_valid_tiles,
    use_64bit_context_offsets: gl.constexpr,
    head_dim: gl.constexpr,
):
    """Accumulate up to 16 buffered per-tile value_mean contributions using BF16 WMMA."""
    offset_type: gl.constexpr = gl.int64 if use_64bit_context_offsets else gl.uint32
    mma_2d_layout: gl.constexpr = AMDWMMALayout(
        version=2, transposed=True, warp_bases=[[1, 0], [2, 0]], rank=2
    )
    lhs_layout: gl.constexpr = gl.DotOperandLayout(0, mma_2d_layout, k_width=8)
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, mma_2d_layout, k_width=8)
    mass_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 2], [4, 1], [0, 1])
    max_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 2], [4, 1], [0, 1])

    tile_masses = tile_masses_shared.load(mass_layout)
    pair_max = gl.convert_layout(
        pair_max_shared.load(max_layout)
        .reshape([64, 8, 1])
        .broadcast_to([64, 8, 2])
        .reshape([64, 16]),
        mass_layout,
    )
    row_running_max = gl.convert_layout(running_max.reshape([64]), gl.SliceLayout(1, mass_layout))[
        :, None
    ]
    tile_slots = gl.arange(0, 16, gl.SliceLayout(0, mass_layout))[None, :]
    has_valid_scores = (tile_slots < num_valid_tiles) & (pair_max != -float("inf"))
    pair_decay = gl.where(has_valid_scores, gl.exp2(pair_max - row_running_max), 0.0)
    # Gate invalid/masked slots outside the product so uninitialized or NaN masses become 0.0.
    scaled_masses = gl.where(has_valid_scores, tile_masses * pair_decay * 255.0, 0.0)
    masses_high, masses_low = split_bf16_pair(scaled_masses)
    lhs_high = gl.convert_layout(masses_high, lhs_layout)
    lhs_low = gl.convert_layout(masses_low, lhs_layout)

    mean_slots = gl.arange(0, 16, gl.SliceLayout(1, rhs_layout))
    tile_indices = gl.where(
        mean_slots < num_valid_tiles,
        tiles_shared.load(gl.SliceLayout(1, rhs_layout)),
        0,
    ).to(offset_type)
    numerator_slices = four_fragments(numerator) if head_dim == 64 else eight_fragments(numerator)
    results = ()
    for column_tile in gl.static_range(head_dim // 16):
        feature_offsets = column_tile * 16 + gl.arange(0, 16, gl.SliceLayout(0, rhs_layout)).to(
            offset_type
        )
        tile_means = gl.load(
            value_mean_ptr + tile_indices[:, None] * head_dim + feature_offsets[None, :]
        )
        acc_slice = gl.convert_layout(
            numerator_slices[column_tile].reshape([64, 16]), mma_2d_layout
        )
        acc_slice = wmma(lhs_high, tile_means, acc_slice)
        acc_slice = wmma(lhs_low, tile_means, acc_slice)
        results += (gl.convert_layout(acc_slice.reshape([1, 64, 16]), MMA_LAYOUT),)
    return gl.convert_layout(_join_matrix(results), MMA_LAYOUT)


@gluon.jit
def qk_fp32_center_bias(
    query_ptr,
    stored_query_block,
    key_center_ptr,
    query_scale,
    rows,
    use_64bit_query_offsets: gl.constexpr,
    head_dim: gl.constexpr,
    row_layout: gl.constexpr,
):
    """Compute Q64 @ H(K_mean)^T * Q_scale bias in FP32 via L1-cached packed Q reads."""
    offset_type: gl.constexpr = gl.int64 if use_64bit_query_offsets else gl.uint32
    words = query_ptr.to(gl.pointer_type(gl.uint32))
    row_words = (stored_query_block.to(offset_type) * 64 + rows[None, :].to(offset_type)) * (
        head_dim // 4
    )
    acc = gl.full([1, 64], 0.0, gl.float32, row_layout)
    for d_word in range(head_dim // 4):
        word32 = gl.load(words + row_words + d_word)
        d_base = d_word * 4
        for byte in gl.static_range(4):
            q_val = (word32 >> (byte * 8)).to(gl.int8).to(gl.float32)
            acc = gl.fma(q_val, gl.load(key_center_ptr + d_base + byte), acc)
    return acc * query_scale


@gluon.jit
def softmax_fragment(scores, maximum, scale, inverse_multiplier, log_multiplier, k_bias=None):
    """Pack a sparse fragment with a tile-wide V scale and unrounded denominator."""
    row_layout: gl.constexpr = maximum.type.layout
    maximum = gl.convert_layout(maximum, gl.SliceLayout(2, scores.type.layout))
    scale = gl.convert_layout(scale, maximum.type.layout)
    scalar_layout: gl.constexpr = gl.SliceLayout(1, maximum.type.layout)
    log_multiplier = gl.convert_layout(log_multiplier, scalar_layout)
    inverse_multiplier = gl.convert_layout(inverse_multiplier, scalar_layout)
    if k_bias is None:
        offset = log_multiplier[:, None] - maximum
    else:
        k_bias_local = gl.convert_layout(k_bias, maximum.type.layout)
        offset = log_multiplier[:, None] + k_bias_local - maximum
    probability = gl.exp2(gl.fma(scores, scale[:, :, None], offset[:, :, None]))
    total = gl.convert_layout(gl.sum(probability, 2) * inverse_multiplier[:, None], row_layout)
    return pack_probabilities(probability), total


@gluon.jit
def pack_probabilities(probability):
    """Round and saturate FP32 probabilities directly into eight-byte PV words."""
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 16, 2], [1, 4, 1], [1, 2, 0])
    codes = gl.convert_layout(probability, layout)
    count: gl.constexpr = probability.shape[2] // 8
    parts = eight_fragments(codes.reshape([1, 64 * count, 8]))
    args = ()
    for byte in gl.static_range(8):
        part: gl.tensor = parts[byte]
        args += (part.reshape([1, 64, count]),)
    low, high = gl.inline_asm_elementwise(
        "v_cvt_pk_u8_f32 $0, $2, 0, 0\n"
        "v_cvt_pk_u8_f32 $0, $3, 1, $0\n"
        "v_cvt_pk_u8_f32 $0, $4, 2, $0\n"
        "v_cvt_pk_u8_f32 $0, $5, 3, $0\n"
        "v_cvt_pk_u8_f32 $1, $6, 0, 0\n"
        "v_cvt_pk_u8_f32 $1, $7, 1, $1\n"
        "v_cvt_pk_u8_f32 $1, $8, 2, $1\n"
        "v_cvt_pk_u8_f32 $1, $9, 3, $1",
        constraints="=&v,=&v,v,v,v,v,v,v,v,v",
        args=args,
        dtype=(gl.uint32, gl.uint32),
        is_pure=True,
        pack=1,
    )
    return low.to(gl.uint64) | (high.to(gl.uint64) << 32)


@gluon.jit
def rescale_numerator(numerator, old_weight):
    # Every register of one thread belongs to one query row in this layout.
    # Skip only when all active wave lanes have weight==1. Otherwise perform
    # every original FP32 product. VCC is clobbered; EXEC is never modified.
    gl.static_assert(numerator.type.layout == MMA_LAYOUT)
    head_dim: gl.constexpr = numerator.shape[2]
    gl.static_assert(head_dim == 64 or head_dim == 128)  # noqa: PLR1714
    gl.static_assert(numerator.shape == [1, 64, head_dim])
    return gl.inline_asm_elementwise(
        _RESCALE_ASM[head_dim // 64 - 1],
        constraints=_RESCALE_CONSTRAINTS[head_dim // 64 - 1],
        args=(numerator, old_weight[:, :, None]),
        dtype=gl.float32,
        is_pure=True,
        pack=head_dim // 2,
    )


@gluon.jit
def pv_tiles(
    probability,
    value_ptr,
    tile_0,
    tile_1,
    numerator,
    current_weight,
    use_64bit_context_offsets: gl.constexpr,
    two_tiles: gl.constexpr = True,
):
    """Accumulate D16 products into the already-rescaled numerator.

    Probability words follow the single-K64 or paired-K128 order from ``qk_tiles``;
    V uses the packed WMMA layout. Single-tile mode ignores ``tile_1``.
    """
    head_dim: gl.constexpr = numerator.shape[2]
    packed = eight_fragments(probability) if two_tiles else four_fragments(probability)
    fragments = ()
    for k in gl.static_range(8 if two_tiles else 4):
        fragment: gl.tensor = packed[k]
        fragments += (
            gl.convert_layout(
                fragment.reshape([1, 4, 16, 2]).permute([0, 1, 3, 2]).reshape([1, 128]),
                VECTOR_LAYOUT,
            ),
        )
    lane = gl.arange(0, 128, gl.SliceLayout(0, VECTOR_LAYOUT)) % 32
    offset_type: gl.constexpr = gl.int64 if use_64bit_context_offsets else gl.uint32
    start_0 = gl.convert_layout(tile_0, gl.SliceLayout(1, VECTOR_LAYOUT)).to(offset_type) * 64
    start_1 = gl.convert_layout(tile_1, gl.SliceLayout(1, VECTOR_LAYOUT)).to(offset_type) * 64
    numerators = four_fragments(numerator) if head_dim == 64 else eight_fragments(numerator)
    results = ()
    for group in gl.static_range(head_dim // 32):
        for column_tile in gl.static_range(group * 2, group * 2 + 2):
            column = column_tile * 16 + lane % 16
            acc = None
            for k_pair in gl.static_range(4 if two_tiles else 2):
                start = start_0 if k_pair < 2 else start_1
                token = (k_pair % 2) * 32 + (lane[None, :] // 16) * 16
                offset = (start[:, None] * head_dim + column[None, :] * 64 + token).to(offset_type)
                pair_words = (value_ptr + offset).to(gl.pointer_type(gl.uint64))
                acc = uint8_int8_wmma(
                    fragments[2 * k_pair], gl.load(pair_words), acc, column_tile % 2
                )
                acc = uint8_int8_wmma(
                    fragments[2 * k_pair + 1], gl.load(pair_words + 1), acc, column_tile % 2
                )
            result: gl.tensor = _matrix_fragment(acc)
            results += (
                gl.fma(
                    result.to(gl.float32),
                    current_weight[:, :, None],
                    gl.convert_layout(numerators[column_tile], MMA_LAYOUT),
                ),
            )
        with gl.amd.warp_pipeline_stage("pv", priority=3):
            pass
    return _join_matrix(results)
