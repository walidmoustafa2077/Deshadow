"""Laplacian pyramid utilities for LP-IOANet.

Implements the closed-form, reversible frequency decomposition used by the
upsampling module (from the LPTN source). The low-pass kernel [1,4,6,4,1]
approximates average pooling with a receptive field of 5.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel_2d(device=None, dtype=torch.float32):
    """Return a 2D low-pass kernel derived from [1,4,6,4,1] (Burt-Adelson)."""
    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=dtype)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    # shape: (out_ch, in_ch, kh, kw) for conv2d
    kernel_2d = kernel_2d.view(1, 1, 5, 5).repeat(1, 1, 1, 1)
    if device is not None:
        kernel_2d = kernel_2d.to(device)
    return kernel_2d


def downsample(x, kernel=None):
    """Low-pass filter + downsample by 2 (REDUCE operation)."""
    if kernel is None:
        kernel = gaussian_kernel_2d(x.device, x.dtype)
    # apply per-channel low-pass
    b, c, h, w = x.shape
    k = kernel.expand(c, 1, 5, 5)
    x = F.conv2d(x, k, padding=2, groups=c)
    return x[:, :, ::2, ::2]


def expand(x, kernel=None):
    """Upsample by 2 + low-pass filter (EXPAND operation)."""
    if kernel is None:
        kernel = gaussian_kernel_2d(x.device, x.dtype)
    b, c, h, w = x.shape
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    k = kernel.expand(c, 1, 5, 5)
    x = F.conv2d(x, k, padding=2, groups=c)
    return x


def get_laplacian_pyramid(x, levels=2):
    """Decompose image into a Laplacian pyramid.

    Returns (residuals, low_freq):
      - residuals: list of high-frequency bands [h_{L-1}, ..., h_0]
      - low_freq:  the lowest-frequency image I_L
    """
    residuals = []
    current = x
    for _ in range(levels):
        low = downsample(current)
        up = expand(low)
        residual = current - up
        residuals.append(residual)
        current = low
    return residuals, current


def reconstruct_from_pyramid(residuals, low_freq):
    """Invert the Laplacian pyramid (exact reconstruction).

    residuals: list ordered [h_{L-1}, ..., h_0] (smallest to largest).
    """
    current = low_freq
    for residual in reversed(residuals):
        current = expand(current) + residual
    return current


class LaplacianPyramid(nn.Module):
    """nn.Module wrapper around the Laplacian decomposition."""

    def __init__(self, levels=2):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        return get_laplacian_pyramid(x, self.levels)
