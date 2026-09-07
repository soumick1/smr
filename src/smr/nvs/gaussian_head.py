"""Per-pixel 3D Gaussian head over frozen-backbone geometry (torch; server-verified by the smoke run).

Input per view: 8 channels from smr.nvs.geometry.head_input -- rgb(3), world xyz(3), depth(1),
confidence(1) -- on the render grid.  A small U-Net predicts, per pixel, a position offset,
log-scales, a rotation, an opacity and a colour residual; valid pixels (object mask AND finite
geometry) become Gaussians; gsplat rasterises them into any target camera; the image is
composited on white (LVSM/GS-LRM convention).  Geometry comes from the backbone (raw or read);
the head is shared by every row of a backbone, so a row difference is a geometry difference.

    head = GaussianHead(); g = head(x, xyz, rgb, valid); img = render(g, c2w_tgt, K, res)

gsplat: `pip install gsplat` (CUDA build at first import).  LPIPS: `pip install lpips`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNetSmall(nn.Module):
    def __init__(self, in_ch, out_ch, base=32):
        super().__init__()
        self.e1, self.e2, self.e3 = _block(in_ch, base), _block(base, 2 * base), _block(2 * base, 4 * base)
        self.mid = _block(4 * base, 4 * base)
        self.d3, self.d2, self.d1 = _block(8 * base, 2 * base), _block(4 * base, base), _block(2 * base, base)
        self.out = nn.Conv2d(base, out_ch, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)     # start at the identity Gaussians

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.avg_pool2d(e1, 2)); e3 = self.e3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([up(m, e3), e3], 1)); d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return self.out(d1)


class GaussianHead(nn.Module):
    """x (V,8,H,W) -> Gaussians of the valid pixels.  base_scale ~ one pixel footprint at the
    object (radius 0.5 spans ~180 px at 256^2 -> 0.0056); offsets bounded to 0.02 (4 % of the
    radius) so the backbone's geometry stays in charge."""

    def __init__(self, in_ch=8, base=32, base_scale=0.006, max_offset=0.02, opacity_bias=3.0):
        super().__init__()
        self.net = UNetSmall(in_ch, 14, base)
        self.base_scale, self.max_offset, self.opacity_bias = base_scale, max_offset, opacity_bias

    def forward(self, x, xyz, rgb, valid):
        """x (V,8,H,W); xyz (V,3,H,W) world points; rgb (V,3,H,W); valid (V,H,W) bool."""
        o = self.net(x)
        means = xyz + self.max_offset * torch.tanh(o[:, 0:3])
        scales = self.base_scale * torch.exp(o[:, 3:6].clamp(-3, 3))
        quats = o[:, 6:10] + torch.tensor([1.0, 0, 0, 0], device=x.device).view(1, 4, 1, 1)
        quats = quats / quats.norm(dim=1, keepdim=True).clamp_min(1e-6)
        opac = torch.sigmoid(o[:, 10] + self.opacity_bias)
        colors = (rgb + 0.5 * torch.tanh(o[:, 11:14])).clamp(0, 1)
        sel = valid.reshape(-1)
        f = lambda t: t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])[sel]
        return dict(means=f(means), scales=f(scales), quats=f(quats), opacities=opac.reshape(-1)[sel], colors=f(colors))


def render(g, c2w, K, res, bg=1.0):
    """Rasterise Gaussians `g` into cameras c2w (T,4,4 OpenCV, y-up world) with intrinsics K (3,3).
    Returns images (T,3,res,res) composited on a constant background, and alphas (T,1,res,res)."""
    from gsplat import rasterization
    T = c2w.shape[0]
    finite = torch.isfinite(g["means"]).all(1) & torch.isfinite(g["scales"]).all(1) & torch.isfinite(g["quats"]).all(1) \
        & torch.isfinite(g["opacities"]) & torch.isfinite(g["colors"]).all(1)
    if finite.sum() == 0:                                   # nothing to draw: background only, no kernel launch
        img = torch.full((T, 3, res, res), float(bg), device=c2w.device); return img, torch.zeros((T, 1, res, res), device=c2w.device)
    if not bool(finite.all()):
        g = {k: v[finite] for k, v in g.items()}
    viewmats = torch.linalg.inv(c2w).float().contiguous()
    Ks = K.float().unsqueeze(0).expand(T, 3, 3).contiguous()
    # No `backgrounds` argument: its expected layout differs across gsplat versions. gsplat returns
    # premultiplied colours over black plus alpha, so the constant background is composited here.
    colors, alphas, _ = rasterization(g["means"].float(), g["quats"].float(), g["scales"].float(), g["opacities"].float(),
                                      g["colors"].float(), viewmats, Ks, res, res, near_plane=0.05, far_plane=20.0,
                                      render_mode="RGB")
    img = colors + (1.0 - alphas) * float(bg)
    return img.permute(0, 3, 1, 2).clamp(0, 1), alphas.permute(0, 3, 1, 2)


def psnr(a, b):
    mse = F.mse_loss(a, b, reduction="none").flatten(1).mean(1)
    return -10 * torch.log10(mse.clamp_min(1e-10))


class Perceptual(nn.Module):
    """LPIPS (VGG) wrapper; inputs in [0,1]."""

    def __init__(self):
        super().__init__()
        import lpips
        self.net = lpips.LPIPS(net="vgg", verbose=False).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, a, b):
        return self.net(a * 2 - 1, b * 2 - 1).mean()


def ssim(a, b):
    """Mean SSIM over a batch (torchmetrics if available, else a Gaussian-window implementation)."""
    try:
        from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
        return tm_ssim(a, b, data_range=1.0)
    except ImportError:
        pass
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    g = torch.exp(-(torch.arange(11, device=a.device) - 5.0) ** 2 / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    conv = lambda t: F.conv2d(t, w, padding=5, groups=3)
    mu_a, mu_b = conv(a), conv(b)
    s_aa, s_bb, s_ab = conv(a * a) - mu_a ** 2, conv(b * b) - mu_b ** 2, conv(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    return s.mean()
