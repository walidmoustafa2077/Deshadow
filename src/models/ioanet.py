"""IOANet: the low-resolution shadow removal core of LP-IOANet.

Architecture (from FastDepth + LRA&LDRA + Coordinate Attention sources):
  - Encoder: ImageNet-pretrained MobileNetV2 feature layers.
  - Decoder: FBNet-style blocks with 5 skip connections from encoder.
  - IOA: Input/Output Attention (LRA on input, LDRA on output) combined via
    a long residual connection:
        I_out = LDRA(R(x)) + LRA(x)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from .coordinate_attention import CoordinateAttention


class FBNetDecoderBlock(nn.Module):
    """FBNet-style decoder block (from FastDepth source).

    Correct order of operations:
      1. Pointwise expansion 1x1 (Cin -> Cmid = 3*Cin)
      2. BN + ReLU
      3. Nearest-neighbor upsampling x2
      4. Depthwise 5x5 conv (stride 1, padding 2, groups=Cmid)
      5. BN + ReLU
      6. Pointwise projection 1x1 (Cmid -> Cout, matching skip channels)
      7. BN (no activation yet; skip connection added after)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = 3 * in_channels
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.dw = nn.Conv2d(mid, mid, kernel_size=5, padding=2, groups=mid)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1-2. Pointwise expansion + BN + ReLU
        x = self.act(self.bn1(self.conv1(x)))
        # 3. Upsample x2 (inside the block)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # 4-5. Depthwise 5x5 + BN + ReLU
        x = self.act(self.bn2(self.dw(x)))
        # 6-7. Pointwise projection + BN (no activation)
        x = self.bn3(self.conv2(x))
        return x


class IOANet(nn.Module):
    """Shadow removal network with Input/Output Attention.

    Args:
        pretrained: whether to use ImageNet-pretrained MobileNetV2 encoder.
        reduction: reduction ratio for Coordinate Attention.
    """

    def __init__(self, pretrained=True, reduction=32):
        super().__init__()

        # ---- Encoder: MobileNetV2 feature layers ----
        if pretrained:
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            backbone = mobilenet_v2(weights=None)
        self.encoder = backbone.features

        # ---- Skip connection feature maps (channels at each scale) ----
        # MobileNetV2 downsampling points: after layers 1,3,6,13,17
        # (scales 1/2, 1/4, 1/8, 1/16, 1/32).
        self.skip_indices = [1, 3, 6, 13, 17]
        skip_channels = [16, 24, 32, 96, 320]

        # ---- Decoder: FBNet-style blocks (bottom-up) ----
        # Encoder output is at 1/32 scale (1280 channels). We project it down
        # to 320 channels and fuse with the 5th skip (320ch at 1/32), giving
        # the decoder input at 1/32 scale with 320 channels. Each block then
        # upsamples x2 internally and outputs channels matching the skip at
        # the upsampled resolution:
        #   1/32->1/16 (skip 96), 1/16->1/8 (skip 32), 1/8->1/4 (skip 24),
        #   1/4->1/2 (skip 16). The final 1/2->1/1 upsampling is done by
        #   final_conv (16 -> 3 channels).
        decoder_channels = [96, 32, 24, 16]
        self.decoder_blocks = nn.ModuleList()
        in_ch = 320
        for out_ch in decoder_channels:
            self.decoder_blocks.append(FBNetDecoderBlock(in_ch, out_ch))
            in_ch = out_ch

        # ---- 5th skip connection (1/32 scale) ----
        # The paper/FastDepth uses FIVE skip connections. The 5th skip is the
        # 320-channel feature at 1/32 scale (features[17]), fused into the
        # decoder input (1280ch from features[18]) via a 1x1 projection.
        self.skip_proj = nn.Conv2d(1280, 320, kernel_size=1)

        # ---- Final 1x1 conv to 3-channel output ----
        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)

        # ---- Input/Output Attention (LRA and LDRA) ----
        self.input_attn = CoordinateAttention(3, reduction=reduction)
        self.output_attn = CoordinateAttention(3, reduction=reduction)

    def forward(self, x):
        # Input attention (parallelizable with the network).
        xa = self.input_attn(x)

        # Encoder forward, collecting skip features.
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_indices:
                skips.append(x)

        # Decoder forward with skip connections (bottom-up).
        # skips are ordered [1/2, 1/4, 1/8, 1/16, 1/32] (indices 0..4).
        # The encoder output (1280 ch) is already at 1/32 scale. We fuse the
        # 1/32 skip (index 4, 320ch) into the decoder input via skip_proj,
        # then fuse with skips at 1/16, 1/8, 1/4, 1/2 (indices 3, 2, 1, 0) as
        # each block upsamples x2.
        # 5th skip: fuse 1/32 feature (320ch) into the 1280ch decoder input.
        skip_32 = skips[4]
        x = self.skip_proj(x) + skip_32

        for idx, block in enumerate(self.decoder_blocks):
            # Block upsamples x2 internally and outputs matching channels.
            x = block(x)
            # Fuse with the corresponding skip (1/16, 1/8, 1/4, 1/2).
            skip = skips[3 - idx]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = x + skip

        # Final conv to 3 channels (at 1/2 scale).
        out = self.final_conv(x)
        # Upsample to full resolution (1/2 -> 1/1).
        out = F.interpolate(out, scale_factor=2, mode="nearest")

        # Output attention + long residual connection.
        out = self.output_attn(out)
        return xa + out
