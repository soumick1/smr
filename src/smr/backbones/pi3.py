"""Pi3 / Pi3X adapter (server-side, GPU) -- written against the cloned
official repo github.com/yyfz/Pi3 (API verified from README + example.py
+ pi3/utils/basic.py on 2026-08):

  * model:  pi3.models.pi3.Pi3 / pi3.models.pi3x.Pi3X (recommended),
            .from_pretrained("yyfz233/Pi3" / "yyfz233/Pi3X")
  * input:  (B, N, 3, H, W) in [0, 1]; their load_images_as_tensor sorts
            a directory and resizes ALL frames to the first frame's
            aspect at <= 255000 px, dims rounded to multiples of 14
  * output: dict with local_points (B,N,H,W,3) camera-frame points,
            conf (B,N,H,W,1) RAW LOGITS (sigmoid -> prob, ex. thr 0.1),
            camera_poses (B,N,4,4) CAMERA-TO-WORLD, OpenCV -- i.e. OUR
            world-from-camera convention: NO inversion (unlike VGGT).

Depth := local_points[..., 2].  Intrinsics are NOT output; they are
estimated per view by least squares from local points (u = fx X/Z + cx),
an estimator unit-verified offline against a synthetic pinhole.
Scale: per-scene median confident depth := 1, camera centres scaled
identically (plan Sec. Portability).
"""
from __future__ import annotations

import pathlib

import numpy as np

from .base import Backbone, BackboneOutput, register


def estimate_K(local_pts, mask, z_min=1e-4):
    """Least-squares pinhole fit from camera-frame points on a pixel grid.

    local_pts: (H, W, 3); returns K (3, 3).  For each axis, regress the
    pixel coordinate on (coord/Z, 1): u = fx * X/Z + cx."""
    H, W = local_pts.shape[:2]
    vs, us = np.mgrid[0:H, 0:W]
    Z = local_pts[..., 2]
    m = mask & (Z > z_min)
    if m.sum() < 100:
        m = Z > z_min
    xz = (local_pts[..., 0] / np.where(Z > z_min, Z, 1.0))[m]
    yz = (local_pts[..., 1] / np.where(Z > z_min, Z, 1.0))[m]
    A = np.stack([xz, np.ones_like(xz)], 1)
    fx, cx = np.linalg.lstsq(A, us[m].astype(float), rcond=None)[0]
    B = np.stack([yz, np.ones_like(yz)], 1)
    fy, cy = np.linalg.lstsq(B, vs[m].astype(float), rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


@register("pi3")
class Pi3Backbone(Backbone):
    name = "pi3"

    def __init__(self, device="cuda", variant="pi3x", ckpt=None,
                 conf_thr=0.1, use_features=False):
        # use_features accepted for harness parity; pi3 features are a
        # later adapter revision.
        self.device, self.variant, self.ckpt = device, variant, ckpt
        self.conf_thr = conf_thr
        self._model = None

    def _load(self):
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch missing: setup_env.sh --with-backbones"
                              ) from e
        try:
            if self.variant == "pi3x":
                from pi3.models.pi3x import Pi3X as M
                model_id = "yyfz233/Pi3X"
            else:
                from pi3.models.pi3 import Pi3 as M
                model_id = "yyfz233/Pi3"
        except ImportError as e:
            raise ImportError(
                "pi3 not importable: server/setup_backbones.sh clones the "
                "verified repo; then `pip install -e third_party/Pi3` (or "
                "add it to PYTHONPATH).") from e
        self._torch = torch
        if self.ckpt:                                  # example.py pattern
            self._model = M()
            if str(self.ckpt).endswith(".safetensors"):
                from safetensors.torch import load_file
                self._model.load_state_dict(load_file(self.ckpt))
            else:
                self._model.load_state_dict(
                    torch.load(self.ckpt, map_location=self.device,
                               weights_only=False))
        else:
            self._model = M.from_pretrained(model_id)
        self._model = self._model.to(self.device).eval()
        self._dtype = (torch.bfloat16
                       if torch.cuda.is_available()
                       and torch.cuda.get_device_capability()[0] >= 8
                       else torch.float16)

    def infer(self, image_paths) -> BackboneOutput:
        if self._model is None:
            self._load()
        torch = self._torch
        from pi3.utils.basic import load_images_as_tensor

        paths = [pathlib.Path(p) for p in image_paths]
        parents = {p.parent for p in paths}
        assert len(parents) == 1, "pi3 adapter expects one image directory"
        d = parents.pop()
        imgs = load_images_as_tensor(str(d), interval=1).to(self.device)
        assert imgs.shape[0] == len(paths), (
            f"their sorted-directory loader found {imgs.shape[0]} images but "
            f"{len(paths)} paths were passed -- pass the full directory")
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self._dtype):
                res = self._model(imgs[None])
        local = res["local_points"].squeeze(0).float().cpu().numpy()
        conf = torch.sigmoid(res["conf"]).squeeze(0).squeeze(-1)
        conf = conf.float().cpu().numpy()
        poses = res["camera_poses"].squeeze(0).float().cpu().numpy()  # w-f-c
        rgb = imgs.permute(0, 2, 3, 1).float().cpu().numpy()

        depth = local[..., 2].copy()
        mask = (conf > self.conf_thr) & (depth > 0)
        try:                                           # their example filter
            from pi3.utils.geometry import depth_normal_edge
            ne = ~depth_normal_edge(
                res["local_points"],
                rtol=0.03,
                mask=torch.from_numpy(mask)[None].to(res["conf"].device)
            ).squeeze(0).cpu().numpy()
            mask &= ne
        except Exception:
            pass                                       # optional refinement

        Ks = np.stack([estimate_K(local[i], mask[i])
                       for i in range(local.shape[0])])
        K = np.median(Ks, axis=0)

        s = float(np.median(depth[mask]))
        depth /= s
        poses = poses.copy()
        poses[:, :3, 3] /= s

        return BackboneOutput(poses=poses, intrinsics=K, depth=depth,
                              rgb=np.clip(rgb, 0, 1), mask=mask,
                              extras=dict(conf=conf, scene_scale=s,
                                          intrinsics_all=Ks,
                                          variant=self.variant))
