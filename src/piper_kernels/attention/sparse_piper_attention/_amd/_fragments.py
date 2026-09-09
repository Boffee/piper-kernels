"""Four-wave Q64 fragments for the RDNA4 D64/D128 attention schedule."""

# Gluon device parameters, layouts and static tuples are not Python values.
# ruff: noqa: ANN001, ANN202
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout

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
def query_fragments(query_ptr, blocks, query_storage_length: gl.constexpr, head_dim: gl.constexpr):
    threads = gl.arange(0, 128, gl.SliceLayout(0, VECTOR_LAYOUT))
    lane = threads % 32
    blocks = gl.convert_layout(blocks, gl.SliceLayout(1, VECTOR_LAYOUT))
    if query_storage_length * (head_dim // 8) > 2147483648:
        blocks = blocks.to(gl.int64)
    rows = blocks[:, None] * 64 + (threads // 32 * 16 + lane % 16)[None, :]
    words = query_ptr.to(gl.pointer_type(gl.uint64))
    fragments = ()
    for k in gl.static_range(head_dim // 16):
        fragments += (
            gl.load(
                words + rows * (head_dim // 8) + (k // 2) * 4 + (lane[None, :] // 16) * 2 + k % 2
            ),
        )
    return fragments


@gluon.jit
def qk_pair(query, key_ptr, tile_0, tile_1, storage_length: gl.constexpr, head_dim: gl.constexpr):
    offset_type: gl.constexpr = gl.uint32 if storage_length * head_dim <= 4294967296 else gl.int64
    lane = gl.arange(0, 128, gl.SliceLayout(0, VECTOR_LAYOUT)) % 32
    tile_0 = gl.convert_layout(tile_0, gl.SliceLayout(1, VECTOR_LAYOUT))
    tile_1 = gl.convert_layout(tile_1, gl.SliceLayout(1, VECTOR_LAYOUT))
    results = ()
    for group in gl.static_range(4):
        for column in gl.static_range(group * 2, group * 2 + 2):
            tile = tile_0 if column < 4 else tile_1
            row = tile[:, None] * 64 + (column % 4) * 16 + lane[None, :] % 16
            row = row.to(offset_type)
            # All D64/D128 INT8 dot products are bounded by 2^21. This integer
            # bias keeps the result bits in one FP32 binade; the FMA below
            # recovers the exact integer sum without integer-to-float conversion.
            bias = gl.full([1, 128], 0x3E22F983, gl.int32, VECTOR_LAYOUT)
            acc = (bias, bias, bias, bias, bias, bias, bias, bias)
            for k in gl.static_range(head_dim // 16):
                offset = (
                    row * head_dim + (k // 2) * 32 + (lane[None, :] // 16) * 16 + (k % 2) * 8
                ).to(offset_type)
                key = gl.load((key_ptr + offset).to(gl.pointer_type(gl.uint64)))
                acc = int8_wmma(query[k], key, acc, column % 2)
            result: gl.tensor = _matrix_fragment(acc)
            results += (gl.fma(result.to(gl.float32, bitcast=True), 67108864.0, -10680707.0),)
        with gl.amd.warp_pipeline_stage("qk"):
            pass
    return _join_matrix(results)


@gluon.jit
def softmax_fragment(scores, maximum, scale, inverse_multiplier, log_multiplier):
    row_layout: gl.constexpr = maximum.type.layout
    maximum = gl.convert_layout(maximum, gl.SliceLayout(2, scores.type.layout))
    scale = gl.convert_layout(scale, maximum.type.layout)
    scalar_layout: gl.constexpr = gl.SliceLayout(1, maximum.type.layout)
    log_multiplier = gl.convert_layout(log_multiplier, scalar_layout)
    inverse_multiplier = gl.convert_layout(inverse_multiplier, scalar_layout)
    offset = log_multiplier[:, None] - maximum
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
def pv_pair(
    probability, value_ptr, tile_0, tile_1, numerator, current_weight, storage_length: gl.constexpr
):
    """Accumulate each D16 product directly into the already-rescaled numerator."""
    head_dim: gl.constexpr = numerator.shape[2]
    packed = eight_fragments(probability)
    fragments = ()
    for k in gl.static_range(8):
        fragment: gl.tensor = packed[k]
        fragments += (
            gl.convert_layout(
                fragment.reshape([1, 4, 16, 2]).permute([0, 1, 3, 2]).reshape([1, 128]),
                VECTOR_LAYOUT,
            ),
        )
    lane = gl.arange(0, 128, gl.SliceLayout(0, VECTOR_LAYOUT)) % 32
    offset_type: gl.constexpr = gl.uint32 if storage_length * head_dim <= 4294967296 else gl.int64
    start_0 = gl.convert_layout(tile_0, gl.SliceLayout(1, VECTOR_LAYOUT)).to(offset_type) * 64
    start_1 = gl.convert_layout(tile_1, gl.SliceLayout(1, VECTOR_LAYOUT)).to(offset_type) * 64
    numerators = four_fragments(numerator) if head_dim == 64 else eight_fragments(numerator)
    results = ()
    for group in gl.static_range(head_dim // 32):
        for column_tile in gl.static_range(group * 2, group * 2 + 2):
            column = column_tile * 16 + lane % 16
            zero = gl.full([1, 128], 0, gl.int32, VECTOR_LAYOUT)
            acc = (zero, zero, zero, zero, zero, zero, zero, zero)
            for k in gl.static_range(8):
                start = start_0 if k < 4 else start_1
                token = (k % 4 // 2) * 32 + (lane[None, :] // 16) * 16 + (k % 2) * 8
                offset = (start[:, None] * head_dim + column[None, :] * 64 + token).to(offset_type)
                value = gl.load((value_ptr + offset).to(gl.pointer_type(gl.uint64)))
                acc = uint8_int8_wmma(fragments[k], value, acc, column_tile % 2)
            result: gl.tensor = _matrix_fragment(acc)
            results += (
                gl.fma(
                    result.to(gl.float32),
                    current_weight[:, :, None],
                    gl.convert_layout(numerators[column_tile], MMA_LAYOUT),
                ),
            )
        with gl.amd.warp_pipeline_stage("pv"):
            pass
    return _join_matrix(results)
