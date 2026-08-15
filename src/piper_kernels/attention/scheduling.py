"""Host-side attention scheduling policies."""

BLOCK_M_VALUES = (64, 128)
NUM_WARPS_VALUES = (2, 4, 8)
NUM_STAGES_VALUES = (1, 2, 3, 4)
LOOP_NUM_STAGES_VALUES = (None, 1, 2, 3, 4)
