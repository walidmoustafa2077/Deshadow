"""Losses for LP-IOANet training.

Stage 1: L1 * 10 + LPIPS * 5 (from the paper).
Stage 2: L1 only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """L1 pixel loss."""

    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity loss.

    Uses a pretrained VGG network. Highly sensitive to blur, which helps
    preserve crisp text contrast.
    """

    def __init__(self, net="alex", device="cuda"):
        super().__init__()
        try:
            import lpips
        except ImportError as e:
            raise ImportError(
                "LPIPS requires the 'lpips' package. Install with: pip install lpips"
            ) from e
        self.lpips = lpips.LPIPS(net=net).to(device)
        # Freeze LPIPS weights.
        for p in self.lpips.parameters():
            p.requires_grad = False
        self.lpips.eval()

    def forward(self, pred, target):
        # LPIPS expects inputs in [-1, 1].
        pred = pred * 2.0 - 1.0
        target = target * 2.0 - 1.0
        return self.lpips(pred, target).mean()


class CombinedLoss(nn.Module):
    """Stage 1 loss: L1 * w_l1 + LPIPS * w_lpips."""

    def __init__(self, w_l1=10.0, w_lpips=5.0, device="cuda"):
        super().__init__()
        self.w_l1 = w_l1
        self.w_lpips = w_lpips
        self.l1 = L1Loss()
        self.lpips = LPIPSLoss(device=device)

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_lpips = self.lpips(pred, target)
        return self.w_l1 * loss_l1 + self.w_lpips * loss_lpips, {
            "l1": loss_l1.item(),
            "lpips": loss_lpips.item(),
        }
