"""Metrics for LP-IOANet training.

Provides MAE, PSNR, SSIM, and region-based (shadow vs non-shadow) analysis,
matching the reference DocShadow-Lite train.py.

Metrics guide:
  - MAE  (target < 0.02): Mean absolute error on normalized [0,1] scale
  - PSNR (target > 28 dB): Peak signal-to-noise ratio
  - SSIM (target > 0.95):  Structural similarity
"""
import torch
import torch.nn.functional as F


def mae(pred, target):
    """Mean absolute error on [0,1] scale."""
    return F.l1_loss(pred, target).item()


def psnr(pred, target, max_val=1.0):
    """Peak signal-to-noise ratio in dB."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val**2 / mse)).item()


def ssim(pred, target, window_size=5, max_val=1.0):
    """Structural similarity index (simplified, single-scale)."""
    # Gaussian window (1D), then outer product to 2D.
    gauss = torch.tensor(
        [0.3989, 0.2419, 0.0539, 0.0044, 0.0001], dtype=pred.dtype, device=pred.device
    )
    gauss = gauss / gauss.sum()
    kernel_1d = gauss.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1,1,5,1)
    kernel_2d = kernel_1d * kernel_1d.transpose(2, 3)  # (1,1,5,5)
    kernel_2d = kernel_2d.expand(pred.shape[1], 1, window_size, window_size).contiguous()

    padding = window_size // 2
    mu1 = F.conv2d(pred, kernel_2d, padding=padding, groups=pred.shape[1])
    mu2 = F.conv2d(target, kernel_2d, padding=padding, groups=target.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel_2d, padding=padding, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel_2d, padding=padding, groups=target.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel_2d, padding=padding, groups=pred.shape[1]) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


def region_metrics(pred, target, mask):
    """Compute MAE/PSNR/SSIM separately for shadow and non-shadow regions.

    Args:
        pred, target: (B, 3, H, W) tensors in [0,1].
        mask: (B, 3, H, W) or (B, 1, H, W) shadow mask (1 = shadow).
    Returns:
        dict with 'shadow' and 'nonshadow' sub-dicts of metrics.
    """
    # Reduce mask to single channel and binarize.
    if mask.shape[1] > 1:
        mask = mask.mean(dim=1, keepdim=True)
    mask = (mask > 0.5).float()

    results = {}
    for region, region_mask in [("shadow", mask), ("nonshadow", 1.0 - mask)]:
        # Expand mask to 3 channels for element-wise masking.
        region_mask_3 = region_mask.expand_as(pred)
        pred_r = pred * region_mask_3
        target_r = target * region_mask_3
        # Only count pixels in the region.
        n_pixels = region_mask_3.sum()
        if n_pixels == 0:
            results[region] = {"mae": float("nan"), "psnr": float("nan"), "ssim": float("nan")}
            continue
        mae_r = (pred_r - target_r).abs().sum() / n_pixels
        mse_r = ((pred_r - target_r) ** 2).sum() / n_pixels
        psnr_r = 10.0 * torch.log10(1.0 / (mse_r + 1e-8)).item()
        results[region] = {
            "mae": mae_r.item(),
            "psnr": psnr_r,
            "ssim": ssim(pred_r, target_r),
        }
    return results


class MetricsCalculator:
    """Static helper to compute all metrics at once."""

    @staticmethod
    def compute_all(pred, target, mask=None):
        """Compute MAE, PSNR, SSIM (and region metrics if mask provided)."""
        metrics = {
            "mae": mae(pred, target),
            "psnr": psnr(pred, target),
            "ssim": ssim(pred, target),
        }
        if mask is not None:
            metrics["regions"] = region_metrics(pred, target, mask)
        return metrics
