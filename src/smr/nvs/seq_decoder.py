"""Sequence-scale Gaussian decoder (v202) for novel view synthesis from the memory map.

Per source frame and pixel (11 channels): rgb (3), map xyz (3, sample-normalised), depth (1), log-confidence (1),
viewing-ray direction (3).  A 4-level U-Net predicts a per-pixel Gaussian: position offset bounded by two pixel
footprints, scales relative to the footprint, rotation, opacity and a colour residual.  Gaussians of all source frames
are rasterised into the target camera at the native aspect (gsplat, RGB + expected depth); an optional image-space
refiner (small U-Net on rendered rgb, alpha and depth) sharpens the result and fills small holes.  The same decoder
architecture and training recipe are used for every geometry; only the geometry it is trained and tested on differs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, groups=8):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU())


class UNet4(nn.Module):
    def __init__(self, in_ch, out_ch, base=48, zero_out=True):
        super().__init__()
        c = [base, 2 * base, 4 * base, 8 * base]
        self.e = nn.ModuleList([_block(in_ch, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3])])
        self.mid = _block(c[3], c[3])
        self.d = nn.ModuleList([_block(c[3] + c[3], c[2]), _block(c[2] + c[2], c[1]), _block(c[1] + c[1], c[0]), _block(c[0] + c[0], c[0])])
        self.out = nn.Conv2d(c[0], out_ch, 1)
        if zero_out:
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = []
        h = x
        for i, blk in enumerate(self.e):
            h = blk(h if i == 0 else F.avg_pool2d(h, 2))
            skips.append(h)
        h = self.mid(F.avg_pool2d(h, 2))
        for blk, s in zip(self.d, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, s], 1))
        return self.out(h)


class SeqGaussianDecoder(nn.Module):
    IN_CH = 11

    def __init__(self, base=48, opacity_bias=2.0):
        super().__init__()
        self.net = UNet4(self.IN_CH, 14, base)
        self.opacity_bias = opacity_bias

    def forward(self, x, xyz, rgb, foot, valid):
        """x (V,11,H,W); xyz (V,3,H,W) normalised map points; rgb (V,3,H,W); foot (V,1,H,W) pixel footprint in map
        units; valid (V,H,W) bool.  Returns flat Gaussian parameters of the valid pixels."""
        o = self.net(x)
        means = xyz + 2.0 * foot * torch.tanh(o[:, 0:3])
        scales = foot * torch.exp(o[:, 3:6].clamp(-2.5, 2.5))
        quats = o[:, 6:10] + torch.tensor([1.0, 0, 0, 0], device=x.device).view(1, 4, 1, 1)
        quats = quats / quats.norm(dim=1, keepdim=True).clamp_min(1e-6)
        opac = torch.sigmoid(o[:, 10] + self.opacity_bias)
        colors = (rgb + 0.5 * torch.tanh(o[:, 11:14])).clamp(0, 1)
        sel = valid.reshape(-1)
        f = lambda t: t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])[sel]
        return dict(means=f(means), scales=f(scales), quats=f(quats), opacities=opac.reshape(-1)[sel], colors=f(colors))


class Refiner(nn.Module):
    """Image-space refinement: rendered rgb (3) + alpha (1) + normalised depth (1) -> rgb residual."""

    def __init__(self, base=32):
        super().__init__()
        self.net = UNet4(5, 3, base)

    def forward(self, img, alpha, depth):
        return (img + 0.5 * torch.tanh(self.net(torch.cat([img, alpha, depth], 1)))).clamp(0, 1)


def render_hw(g, c2w, K, H, W, bg=0.0):
    """Rasterise into cameras c2w (T,4,4, OpenCV) with K (3,3) at H x W. Returns rgb (T,3,H,W), alpha (T,1,H,W),
    expected depth (T,1,H,W; zeros if the gsplat build lacks RGB+ED)."""
    from gsplat import rasterization
    T = c2w.shape[0]
    finite = torch.isfinite(g["means"]).all(1) & torch.isfinite(g["scales"]).all(1) & torch.isfinite(g["quats"]).all(1) \
        & torch.isfinite(g["opacities"]) & torch.isfinite(g["colors"]).all(1)
    if finite.sum() == 0:
        z = torch.zeros((T, 1, H, W), device=c2w.device)
        return torch.full((T, 3, H, W), float(bg), device=c2w.device), z, z
    if not bool(finite.all()):
        g = {k: v[finite] for k, v in g.items()}
    viewmats = torch.linalg.inv(c2w).float().contiguous()
    Ks = K.float().unsqueeze(0).expand(T, 3, 3).contiguous()
    args = (g["means"].float(), g["quats"].float(), g["scales"].float(), g["opacities"].float(), g["colors"].float(), viewmats, Ks, W, H)
    try:
        out, alphas, _ = rasterization(*args, near_plane=0.01, far_plane=100.0, render_mode="RGB+ED")
        rgb, depth = out[..., :3], out[..., 3:4]
    except Exception:  # noqa: BLE001
        rgb, alphas, _ = rasterization(*args, near_plane=0.01, far_plane=100.0, render_mode="RGB")
        depth = torch.zeros_like(alphas)
    img = rgb + (1.0 - alphas) * float(bg)
    return img.permute(0, 3, 1, 2).clamp(0, 1), alphas.permute(0, 3, 1, 2), depth.permute(0, 3, 1, 2)
