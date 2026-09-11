"""Versioned fixed activation scales for the H3 VAE INT8 encoder."""

from types import MappingProxyType

# Per-layer p99.5 token-amax scales. The calibration set contains the first 124
# frames of four independent real-camera sequences (Foreman, Coastguard,
# Football, and Mobile), cropped losslessly to the VAE's native 256x256 tile.
# Held-out quality was evaluated on Bus, Flower, Tempete, and Waterfall.
_P995_SCALES = {
    "down_blocks.0.resnets.0.conv1": 0.008035494945943356,
    "down_blocks.0.resnets.0.conv2": 0.0028028113301843405,
    "down_blocks.0.resnets.1.conv1": 0.004402220714837313,
    "down_blocks.0.resnets.1.conv2": 0.003371831960976124,
    "down_blocks.0.downsamplers.0.conv": 0.012887549586594105,
    "down_blocks.1.resnets.0.conv1": 0.004440668039023876,
    "down_blocks.1.resnets.0.conv2": 0.004167691804468632,
    "down_blocks.1.resnets.1.conv1": 0.004848209675401449,
    "down_blocks.1.resnets.1.conv2": 0.004425289109349251,
    "down_blocks.1.downsamplers.0.conv": 0.019069882109761238,
    "down_blocks.2.resnets.0.conv1": 0.004302257671952248,
    "down_blocks.2.resnets.0.conv2": 0.007220410741865635,
    "down_blocks.2.resnets.1.conv1": 0.004156157840043306,
    "down_blocks.2.resnets.1.conv2": 0.007416492328047752,
    "down_blocks.2.downsamplers.0.conv": 0.06859005987644196,
    "down_blocks.3.resnets.0.conv1": 0.003792830277234316,
    "down_blocks.3.resnets.0.conv2": 0.011980191804468632,
    "down_blocks.3.resnets.1.conv1": 0.006124661769717932,
    "down_blocks.3.resnets.1.conv2": 0.00841227825731039,
    "down_blocks.3.downsamplers.0.conv": 0.08895177394151688,
    "down_blocks.4.resnets.0.conv1": 0.0021472841035574675,
    "down_blocks.4.resnets.0.conv2": 0.007167380303144455,
    "down_blocks.4.resnets.1.conv1": 0.0025817390996962786,
    "down_blocks.4.resnets.1.conv2": 0.009342703968286514,
    "down_blocks.5.resnets.0.conv1": 0.002271111588925123,
    "down_blocks.5.resnets.0.conv2": 0.0052557517774403095,
    "down_blocks.5.resnets.1.conv1": 0.004219265654683113,
    "down_blocks.5.resnets.1.conv2": 0.007066621445119381,
    "conv_out": 0.004467581398785114,
}

P995_ACTIVATION_SCALES = MappingProxyType(_P995_SCALES)

__all__ = ["P995_ACTIVATION_SCALES"]
