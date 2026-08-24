"""CUT3R adapter (continuous updating 3D reconstruction, streaming).

Entry points verified against github.com/CUT3R/CUT3R demo.py (2026-08-24):

    add_path_to_dust3r(model_path)                 # their sys.path helper
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.inference import inference
    model  = ARCroco3DStereo.from_pretrained(model_path)
    views  = prepare_input(...)                    # demo.py:82
    outputs, state_args = inference(views, model, device)
    pts3ds_other, colors, conf, cam_dict = prepare_output(outputs, outdir,
                                                          1, True)  # demo.py:189

demo.py lives at the repo ROOT (not inside a package), so we add
third_party/CUT3R to sys.path and import it as a top-level module.

WEIGHTS: Google Drive only -- no direct URL, so fetch_weights.py cannot
automate it.  Run, once:

    pip install gdown
    cd third_party/checkpoints
    gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view

which lands cut3r_512_dpt_4_64.pth (their final checkpoint).

STATUS: experimental.  `prepare_output`'s cam_dict layout is the one thing
I could not pin from the repo, so `_poses_from_cam_dict` accepts several
shapes and otherwise raises listing exactly what it found.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .pointmap import PointmapBackbone, RawViews


def _poses_from_cam_dict(cam_dict, n):
    """Extract (n,4,4) camera-to-world poses from CUT3R's cam_dict."""
    if isinstance(cam_dict, dict):
        for k in ("c2w", "cam2world", "poses", "camera_poses", "T_c2w"):
            if k in cam_dict:
                P = np.asarray([np.asarray(p, dtype=float)
                                for p in cam_dict[k]])
                if P.shape == (n, 4, 4):
                    return P
        raise RuntimeError(
            f"cut3r: no (N,4,4) pose array in cam_dict; keys={list(cam_dict)}"
            " -- inspect prepare_output in third_party/CUT3R/demo.py and "
            "extend _poses_from_cam_dict.")
    P = np.asarray(cam_dict, dtype=float)
    if P.shape == (n, 4, 4):
        return P
    raise RuntimeError(f"cut3r: unexpected cam_dict of shape {P.shape}")


@register("cut3r")
class CUT3RBackbone(PointmapBackbone):
    name = "cut3r"
    pose_convention = "c2w"
    third_party_paths = ("CUT3R",)
    requires = ("src.dust3r.model",)
    weights = "cut3r_512_dpt_4_64.pth"
    weights_file = "cut3r_512_dpt_4_64.pth"
    weights_hub = None
    weights_url = None                       # Google Drive; see module docs
    weights_note = ("Google Drive only: pip install gdown && gdown --fuzzy "
                    "https://drive.google.com/file/d/"
                    "1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view")
    experimental = True

    def __init__(self, device="cuda", conf_keep=0.8, image_size=512,
                 revisit=1, model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self.image_size, self.revisit = image_size, revisit
        self._weights_override = model_id

    def _load(self):
        self.model_id = self.resolve_weights(self._weights_override)
        from add_ckpt_path import add_path_to_dust3r      # CUT3R repo root
        add_path_to_dust3r(self.model_id)
        from src.dust3r.model import ARCroco3DStereo
        self._model = ARCroco3DStereo.from_pretrained(self.model_id).to(
            self.device).eval()

    def _raw(self, image_paths) -> RawViews:
        import tempfile
        from demo import prepare_input, prepare_output        # repo root
        from src.dust3r.inference import inference

        views = prepare_input(list(image_paths), size=self.image_size)
        outputs, _ = inference(views, self._model, self.device)
        with tempfile.TemporaryDirectory() as td:
            pts3ds_other, colors, conf, cam_dict = prepare_output(
                outputs, td, self.revisit, True)

        def np_(x):
            return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach")
                              else x, dtype=float)

        pts_w = np.stack([np_(p).squeeze() for p in pts3ds_other])
        rgb = np.stack([np_(c).squeeze() for c in colors])
        cf = np.stack([np_(c).squeeze() for c in conf])
        poses = _poses_from_cam_dict(cam_dict, pts_w.shape[0])

        pts_local = np.empty_like(pts_w)                  # global -> camera
        for i in range(poses.shape[0]):
            Rt = np.linalg.inv(poses[i])
            pts_local[i] = pts_w[i] @ Rt[:3, :3].T + Rt[:3, 3]
        return RawViews(poses=poses, rgb=np.clip(rgb, 0, 1), conf=cf,
                        pts_local=pts_local)
