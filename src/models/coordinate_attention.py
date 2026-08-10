"""Coordinate Attention module (from the Coordinate Attention source).

Used to implement the Input/Output Attention (LRA and LDRA) in IOANet.

The coordinate attention factorizes channel attention into two 1D feature
encoding processes (horizontal + vertical), preserving positional information
that 2D global pooling (SE attention) loses.

Equations (from source):
    z_h(h) = (1/W) sum_i x(h, i)
    z_w(w) = (1/H) sum_j x(j, w)
    f = delta(F1([z_h, z_w]))
    g_h = sigma(F_h(f_h)),  g_w = sigma(F_w(f_w))
    y_c(i,j) = x_c(i,j) * g_h_c(i) * g_w_c(j)
"""
import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """Coordinate attention block.

    Args:
        in_channels: number of input channels C.
        reduction: reduction ratio r for the bottleneck (C/r).
        min_bottleneck: minimum bottleneck width to avoid collapse on small C.
    """

    def __init__(self, in_channels, reduction=32, min_bottleneck=8):
        super().__init__()
        # Avoid bottleneck collapse when C is small (e.g. C=3 for LRA/LDRA).
        mid = max(min_bottleneck, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape

        # Coordinate information embedding: two 1D global poolings.
        x_h = torch.mean(x, dim=3, keepdim=True)          # (b, c, h, 1)
        x_w = torch.mean(x, dim=2, keepdim=True)          # (b, c, 1, w)
        x_w = x_w.permute(0, 1, 3, 2)                     # (b, c, w, 1)

        # Concatenate along spatial dim, then shared 1x1 conv.
        x_cat = torch.cat([x_h, x_w], dim=2)              # (b, c, h+w, 1)
        f = self.act(self.bn1(self.conv1(x_cat)))         # (b, mid, h+w, 1)

        # Split back into horizontal and vertical.
        f_h, f_w = torch.split(f, [h, w], dim=2)          # (b, mid, h, 1), (b, mid, w, 1)
        f_w = f_w.permute(0, 1, 3, 2)                     # (b, mid, 1, w)

        # Coordinate attention generation.
        g_h = torch.sigmoid(self.conv_h(f_h))             # (b, c, h, 1)
        g_w = torch.sigmoid(self.conv_w(f_w))             # (b, c, 1, w)

        # Re-weight: y = x * g_h * g_w
        return x * g_h * g_w
