"""Completion head: a small fully-convolutional UNet.

Input  (B, 5, H, W): splat RGB (3) + splat depth (1, /4) + splat mask (1)
Output rgb (B, 3, H, W) in [0,1] via sigmoid; depth (B, 1, H, W) >= 0 via
softplus.  Fully convolutional -- trains on random crops, evaluates on
full frames padded to a multiple of 16 (4 downsamplings)."""
import torch
import torch.nn as nn


def _block(cin, cout):
    g = min(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.SiLU())


class CompletionUNet(nn.Module):
    def __init__(self, in_ch=5, base=48):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc0 = _block(in_ch, c[0])
        self.down = nn.ModuleList(
            [nn.Conv2d(c[i], c[i + 1], 4, stride=2, padding=1)
             for i in range(3)])
        self.enc = nn.ModuleList([_block(c[i + 1], c[i + 1])
                                  for i in range(3)])
        self.mid = _block(c[3], c[3])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(c[i + 1], c[i], 4, stride=2, padding=1)
             for i in reversed(range(3))])
        self.dec = nn.ModuleList([_block(c[i] * 2, c[i])
                                  for i in reversed(range(3))])
        self.head_rgb = nn.Conv2d(c[0], 3, 3, padding=1)
        self.head_depth = nn.Conv2d(c[0], 1, 3, padding=1)

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc[0](self.down[0](s0))
        s2 = self.enc[1](self.down[1](s1))
        h = self.mid(self.down[2](s2))
        h = self.dec[0](torch.cat([self.up[0](h), s2], 1))
        h = self.dec[1](torch.cat([self.up[1](h), s1], 1))
        h = self.dec[2](torch.cat([self.up[2](h), s0], 1))
        return torch.sigmoid(self.head_rgb(h)), \
            nn.functional.softplus(self.head_depth(h))
