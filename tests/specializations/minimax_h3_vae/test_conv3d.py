"""Candidate H3 encoder calibration coverage."""

from piper_kernels.specializations.minimax_h3_vae.conv3d import P995_ACTIVATION_SCALES


def test_calibration_covers_eligible_encoder_layers():
    assert len(P995_ACTIVATION_SCALES) == 29
    assert all(scale > 0 for scale in P995_ACTIVATION_SCALES.values())
    assert not any("conv_in" in name or "shortcut" in name for name in P995_ACTIVATION_SCALES)
