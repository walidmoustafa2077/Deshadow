"""LP-IOANet: full pipeline combining IOANet + Laplacian pyramid upsampler.

Pipeline:
  - Downscale high-res input (768x1024) to low-res (192x256).
  - IOANet removes shadows at low resolution.
  - Laplacian pyramid upsampler upscales 4x back to high resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ioanet import IOANet
from .upsampler import LaplacianPyramidUpsampler


class LPIOANet(nn.Module):
    """Full LP-IOANet pipeline.

    Args:
        low_res: (H, W) low-resolution core size (default 192x256).
        high_res: (H, W) target high-resolution size (default 768x1024).
        pretrained: use ImageNet-pretrained MobileNetV2 encoder.
        levels: number of Laplacian pyramid levels (2).
        hidden: width of upsampler residual blocks.
    """

    def __init__(self, low_res=(192, 256), high_res=(768, 1024),
                 pretrained=True, levels=2, hidden=16):
        super().__init__()
        self.low_res = low_res
        self.high_res = high_res
        self.ioanet = IOANet(pretrained=pretrained)
        self.upsampler = LaplacianPyramidUpsampler(levels=levels, hidden=hidden)

    def forward(self, x):
        """x: high-res input at (B, 3, H, W)."""
        # Downscale to low resolution.
        x_low = F.interpolate(x, size=self.low_res, mode="bilinear",
                              align_corners=False)
        # Shadow removal at low resolution.
        out_low = self.ioanet(x_low)
        # Upsample back to high resolution.
        out_high = self.upsampler(out_low, x)
        return out_high

    def forward_low(self, x_low):
        """Run only the low-res shadow removal core (for Stage 1 training)."""
        return self.ioanet(x_low)
