"""Laplacian Pyramid Upsampling module for LP-IOANet.

From the LPTN source: the low-frequency (shadow-removed) component is
translated by IOANet, and the high-frequency residuals are refined via
lightweight masks. The mask at level L-1 is generated from the concatenation
of [h_{L-1}, up(I_L), up(Î_L)], then progressively upsampled and fine-tuned.

Reconstruction path uses nearest-neighbor x2 upsampling (LP-IOANet diagram);
mask/low-frequency upsampling uses bilinear (LPTN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.pyramid_utils import downsample, expand, get_laplacian_pyramid


class DepthwiseSeparableConv(nn.Module):
    """Cheap depthwise separable convolution (DW conv + 1x1 conv)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.pw(self.act(self.dw(x)))


class LightweightResidualBlock(nn.Module):
    """Lightweight residual block using depthwise separable convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class MaskGenerator(nn.Module):
    """Generates the per-pixel mask M for a high-frequency residual.

    Input: concatenation of [h, up(I_L), up(Î_L)] (9 channels for RGB).
    Output: single-channel mask M (a scaling multiplier, NOT Tanh-bounded).

    Note: The mask acts as a local scaling multiplier on the high-frequency
    residual (ĥ = h * M). Tanh would restrict it to [-1,1] and cause
    phase/sign flipping on the directional residual, so we use a linear
    output (the Tanh in the LPTN diagram is on the translated low-frequency
    output, not the mask).
    """

    def __init__(self, in_channels=9, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            LightweightResidualBlock(hidden),
            LightweightResidualBlock(hidden),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class LaplacianPyramidUpsampler(nn.Module):
    """2-level Laplacian pyramid upsampler: 192x256 -> 384x512 -> 768x1024.

    Args:
        levels: number of pyramid levels (2 for LP-IOANet).
        hidden: width of the lightweight residual blocks.
    """

    def __init__(self, levels=2, hidden=16):
        super().__init__()
        self.levels = levels
        # Base mask generator: only runs at the lowest level (L-1).
        self.mask_generator = MaskGenerator(in_channels=9, hidden=hidden)
        # Lightweight fine-tuning convs for progressively upsampled masks
        # (one per higher level, l = L-2 down to 0).
        self.mask_finetune = nn.ModuleList(
            [nn.Conv2d(1, 1, kernel_size=3, padding=1) for _ in range(levels - 1)]
        )

    def forward(self, low_res_out, high_res_input):
        """Upsample the low-res shadow-free output to high resolution.

        Args:
            low_res_out: IOANet output at low resolution.
            high_res_input: original high-res input (for residuals).
        Returns:
            high-res shadow-free output.
        """
        # Decompose the high-res input into a Laplacian pyramid.
        residuals, low_freq = get_laplacian_pyramid(high_res_input, self.levels)
        # residuals: [h_{L-1}, ..., h_0] (smallest to largest)
        # low_freq: I_L at low resolution

        # The low-res shadow-free output is our translated low-frequency Î_L.
        translated_low = low_res_out

        # --- Level L-1 (smallest): generate the base mask ---
        residual = residuals[0]
        target_size = residual.shape[2:]
        up_il = F.interpolate(low_freq, size=target_size, mode="bilinear",
                              align_corners=False)
        up_il_hat = F.interpolate(translated_low, size=target_size,
                                  mode="bilinear", align_corners=False)
        mask_input = torch.cat([residual, up_il, up_il_hat], dim=1)
        mask = self.mask_generator(mask_input)
        refined_residuals = [residual * mask]

        # --- Higher levels: progressively upsample + refine the mask ---
        for idx in range(1, self.levels):
            residual = residuals[idx]
            # Upsample the previous mask x2 (bilinear) and fine-tune.
            mask = F.interpolate(mask, scale_factor=2, mode="bilinear",
                                 align_corners=False)
            mask = self.mask_finetune[idx - 1](mask)
            # Match resolution exactly (in case of rounding).
            if mask.shape[2:] != residual.shape[2:]:
                mask = F.interpolate(mask, size=residual.shape[2:],
                                     mode="bilinear", align_corners=False)
            refined_residuals.append(residual * mask)

        # Reconstruct: start from translated low-freq, add refined residuals.
        output = translated_low
        for idx in range(self.levels - 1, -1, -1):
            # Upsample output x2 (nearest, per LP-IOANet reconstruction path).
            output = F.interpolate(output, scale_factor=2, mode="nearest")
            output = output + refined_residuals[idx]
        return output
