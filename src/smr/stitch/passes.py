"""Backbone passes: running, caching and simulating them.

A *pass* is one backbone forward over an ORDERED list of frame indices.
Its output is a set of camera-to-world poses in the pass's own frame and
gauge.  Every stitcher in this package consumes passes through PassCache,
so (i) a backbone is never run twice on the same frame set, (ii) a run can
be resumed after an SSH drop, and (iii) rows that share passes (chained vs
the per-chunk probe, SMR-streaming vs SMR->PGO) are guaranteed to see
identical backbone outputs.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np

from ..backbones.base import rgb_descriptor
from ..eval.trajectory import rotation_angle_deg
from .chunks import regauge


# ---------------------------------------------------------------- runners --
class BackboneRunner:
    """Runs a real backbone on image paths.  First execution happens on the
    GPU server; this class cannot be exercised where torch is absent."""

    def __init__(self, backbone_name, paths, device="cuda"):
        from ..backbones import get_backbone
        self.bb = get_backbone(backbone_name, device=device)
        self.paths = [str(p) for p in paths]
        self.name = backbone_name

    def __call__(self, idx):
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:                     # pragma: no cover
            torch, cuda = None, False
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = self.bb.infer([self.paths[i] for i in idx])
        secs = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
        return np.asarray(out.poses, float), secs, peak


class SimulatedRunner:
    """Stands in for a backbone when validating the harness without a GPU.

    Slices ground-truth poses, re-gauges them to the pass's own frame and
    scale (as every real backbone does), then applies

      * one random similarity per pass (frame/gauge error).  Sim(3)
        alignment absorbs this EXACTLY, so on its own it produces no drift
        -- which is precisely why it is not the interesting error;
      * a within-pass DISTORTION that grows with the frame's distance from
        the pass's reference frame: rotation about a random axis,
        translation along a random direction and a scale ramp, all
        reaching `distortion` (rad / units / log-scale) at the last frame.
        This mimics reference-anchored drift (VGGT reaches ~29 deg at the
        antipode of an orbit) and is what chaining compounds;
      * per-pose jitter (`noise`).

    Never a result: any report built on this is marked SIMULATED.
    """

    def __init__(self, gt_poses, noise=0.02, distortion=0.0, seed=0,
                 covis_deg=None, n_core=16):
        self.gt = np.asarray(gt_poses, float)
        self.noise = float(noise)
        self.distortion = float(distortion)
        self.rng = np.random.default_rng(seed)
        self.covis_deg = covis_deg
        self.n_core = n_core

    def __call__(self, idx):
        from .sim3 import rotmat, apply
        t0 = time.time()
        p = regauge(self.gt[list(idx)])
        K = len(p)
        if self.covis_deg is not None and K >= 3:
            # A frame that shares no view direction (within covis_deg)
            # with any CORE frame -- the chunk: the first n_core frames of
            # a full pass, the first half of a small verification pass --
            # has nothing the backbone can relate; a real model then
            # emits an arbitrary pose.  Emulate that with a fresh random
            # pose per call, so two passes never agree on it.
            # core = the chunk: every frame of a chunk-sized pass, the
            # first n_core of an anchored pass, the first half of a small
            # verification pass (<= 6 frames)
            nc = (max(1, K // 2) if K <= 6 else
                  K if K <= self.n_core else self.n_core)
            fw = p[:, :3, 2].copy()
            core = fw[:nc]
            for li in range(nc, K):
                c = np.clip(core @ fw[li], -1, 1)
                if np.degrees(np.arccos(c)).min() > self.covis_deg:
                    p[li, :3, :3] = rotmat(self.rng.normal(size=3) * 2.0)
                    p[li, :3, 3] = self.rng.normal(size=3) * 2.0
        if self.distortion > 0 and K > 1:
            # distortion grows with the GEOMETRIC distance of a frame from
            # the pass's reference frame (frame 0), reaching `distortion`
            # at 90 degrees of rotation or one gauge unit of translation
            # -- so appending an anchor to a pass never changes the
            # chunk's own error, and an anchor that revisits the place of
            # the reference frame is nearly undistorted (as it is for a
            # real reference-anchored backbone).
            ax = self.rng.normal(size=3); ax /= np.linalg.norm(ax)
            di = self.rng.normal(size=3); di /= np.linalg.norm(di)
            sg = self.rng.choice([-1.0, 1.0])
            R0 = p[0, :3, :3]
            for li in range(K):
                ang = np.radians(rotation_angle_deg(R0.T @ p[li, :3, :3]))
                f = min(2.0, ang / (np.pi / 2) + np.linalg.norm(p[li, :3, 3]))
                p[li, :3, :3] = rotmat(f * self.distortion * ax) @ p[li, :3, :3]
                p[li, :3, 3] = p[li, :3, 3] * np.exp(sg * f * self.distortion) \
                    + f * self.distortion * di
        if self.noise > 0:
            ang = self.rng.normal(scale=self.noise)
            ax = self.rng.normal(size=3); ax /= np.linalg.norm(ax)
            S = (1.0 + self.rng.normal(scale=self.noise), rotmat(ang * ax),
                 self.rng.normal(scale=self.noise, size=3))
            p = apply(S, p)
            p[:, :3, 3] += self.rng.normal(scale=self.noise * 0.25,
                                           size=p[:, :3, 3].shape)
            for i in range(K):
                jr = self.rng.normal(size=3) * self.noise * 0.25
                p[i, :3, :3] = rotmat(jr) @ p[i, :3, :3]
        return p, time.time() - t0, 0.0


# ------------------------------------------------------------------ cache --
class PassCache:
    """Memoises passes by their ORDERED frame tuple; persists to disk."""

    def __init__(self, path=None):
        self.path = pathlib.Path(path) if path else None
        self.data = {}
        self.n_runs = 0
        if self.path is not None and self.path.exists():
            self.data = np.load(self.path, allow_pickle=True).item()

    def key(self, idx):
        return tuple(int(i) for i in idx)

    def has(self, idx):
        return self.key(idx) in self.data

    def get(self, idx, runner):
        k = self.key(idx)
        if k not in self.data:
            poses, secs, peak = runner(list(k))
            self.data[k] = dict(poses=np.asarray(poses, float),
                                secs=float(secs), peak_gb=float(peak))
            self.n_runs += 1
            self.save()
        return self.data[k]

    def save(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.path, self.data, allow_pickle=True)

    def totals(self, keys=None):
        """Wall time and peak memory over the given passes (all if None)."""
        items = [self.data[self.key(k)] for k in keys] if keys is not None \
            else list(self.data.values())
        secs = sum(d["secs"] for d in items)
        peak = max([d["peak_gb"] for d in items] + [0.0])
        return secs, peak


# ------------------------------------------------------------ descriptors --
def _load_small(path, max_side=160):
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        w, h = im.size
        f = max_side / max(w, h)
        if f < 1:
            im = im.resize((max(1, int(w * f)), max(1, int(h * f))),
                           Image.BILINEAR)
        return np.asarray(im, float) / 255.0
    except ImportError:                                       # pragma: no cover
        import imageio.v2 as iio
        a = np.asarray(iio.imread(path), float)[..., :3] / 255.0
        st = max(1, int(np.ceil(max(a.shape[:2]) / max_side)))
        return a[::st, ::st]


def image_descriptors(paths, max_side=160):
    """The bind-time view descriptor for every image, computed from the
    raw file at reduced size.  Backbone-independent, so every row of the
    table proposes anchors from identical evidence."""
    return np.stack([rgb_descriptor(_load_small(p, max_side)) for p in paths])


def array_descriptors(rgbs):
    return np.stack([rgb_descriptor(r) for r in rgbs])


def dino_descriptors(paths, device="cuda", model="dinov2_vits14", size=224):
    """Global place descriptor from DINOv2's CLS token (384-d for ViT-S/14),
    L2-normalised.  The pooled-RGB descriptor could not separate a revisit
    from any other view of the same room on TUM fr1_room (true revisits
    at cosine 0.33-0.58, look-alikes up to 0.77); a self-supervised ViT
    embedding is the standard remedy and is backbone-agnostic.

    FIRST EXECUTION ON THE GPU SERVER: needs torch + torch.hub access to
    facebookresearch/dinov2 (weights download once).  Fails loud.
    """
    import torch
    from PIL import Image
    m = torch.hub.load("facebookresearch/dinov2", model).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            ims = []
            for pth in paths[i:i + 32]:
                im = Image.open(pth).convert("RGB").resize((size, size), Image.BICUBIC)
                ims.append(torch.from_numpy(np.asarray(im, np.float32) / 255.0)
                           .permute(2, 0, 1))
            x = (torch.stack(ims).to(device) - mean) / std
            f = m(x).float().cpu().numpy()            # (B, 384) CLS features
            out.append(f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9))
    return np.concatenate(out)
