"""Transforms for the Mixed_Shadow_Dataset.

Includes Laplacian decomposition transforms (shared between dataloader and
model to prevent train/inference distribution shift) and augmentation.
"""
import torch
import torch.nn.functional as F

from ..utils.pyramid_utils import get_laplacian_pyramid


class ToTensor:
    """Convert PIL image to float tensor in [0,1]."""

    def __call__(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class Resize:
    """Resize image to (H, W)."""

    def __init__(self, size):
        self.size = size  # (H, W)

    def __call__(self, img):
        return F.interpolate(img.unsqueeze(0), size=self.size,
                             mode="bilinear", align_corners=False).squeeze(0)


class RandomHorizontalFlip:
    """Random horizontal flip applied consistently to a batch of tensors."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *tensors):
        if torch.rand(1).item() < self.p:
            return [torch.flip(t, dims=[2]) for t in tensors]
        return list(tensors)


class LaplacianDecompose:
    """Decompose a high-res image into (residuals, low_freq) Laplacian pyramid.

    Used in Stage 2 so the dataloader and model share the same decomposition.
    """

    def __init__(self, levels=2):
        self.levels = levels

    def __call__(self, img):
        residuals, low_freq = get_laplacian_pyramid(img.unsqueeze(0), self.levels)
        residuals = [r.squeeze(0) for r in residuals]
        low_freq = low_freq.squeeze(0)
        return residuals, low_freq
