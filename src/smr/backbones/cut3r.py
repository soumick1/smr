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
    python scripts/fetch_weights.py --backbones cut3r
    # (manual equivalent: gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD -O <dest>)

which lands cut3r_512_dpt_4_64.pth (their final checkpoint).

STATUS: experimental.  `prepare_output`'s cam_dict layout is the one thing
I could not pin from the repo, so `_poses_from_cam_dict` accepts several
shapes and otherwise raises listing exactly what it found.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .pointmap import (PointmapBackbone, RawViews,
                       isolate_third_party,
                       patch_torch_load_legacy_checkpoints)


def _poses_from_cam_dict(cam_dict, n):
    """(n,4,4) camera-to-world from CUT3R's cam_dict.

    Verified in demo.py's prepare_output:
        cam_dict = {"focal": (B,), "pp": (B,2),
                    "R": (B,3,3) c2w, "t": (B,3) c2w}
    """
    if not isinstance(cam_dict, dict) or not {"R", "t"} <= set(cam_dict):
        raise RuntimeError(
            f"cut3r: expected R/t in cam_dict, got "
            f"{list(cam_dict) if isinstance(cam_dict, dict) else type(cam_dict)}")
    R = np.asarray(cam_dict["R"], dtype=float).reshape(-1, 3, 3)
    tt = np.asarray(cam_dict["t"], dtype=float).reshape(-1, 3)
    assert R.shape[0] == n == tt.shape[0], (R.shape, tt.shape, n)
    P = np.tile(np.eye(4), (n, 1, 1))
    P[:, :3, :3], P[:, :3, 3] = R, tt
    return P


@register("cut3r")
class CUT3RBackbone(PointmapBackbone):
    name = "cut3r"
    pose_convention = "c2w"
    # add_ckpt_path.add_path_to_dust3r() inserts the checkpoint's own
    # directory so that `import dust3r` finds CUT3R's fork -- i.e. their
    # code refers to it as TOP-LEVEL dust3r, not src.dust3r.  So CUT3R/src
    # must lead, and the repo root follows for `import demo`.
    # CUT3R/src/croco must lead: their model.py does
    #   `from models.croco import CroCoNet, CrocoConfig`
    # and DUSt3R's croco (which another adapter may already have put on
    # sys.path) has no CrocoConfig.
    third_party_paths = ("CUT3R/src/croco", "CUT3R/src", "CUT3R")
    requires = ("dust3r",)
    weights = "cut3r_512_dpt_4_64.pth"
    weights_file = "cut3r_512_dpt_4_64.pth"
    weights_hub = None
    weights_url = None                       # Google Drive; see module docs
    # Their official Drive link is quota-blocked ("too many users have
    # viewed or downloaded this file recently"), so this points at our own
    # mirror of the same checkpoint.  Original:
    #   https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view
    weights_gdrive = ("https://drive.google.com/file/d/"
                      "1V7TEIpHncGBpgsSx8R2fxYISCgE2w9Bl/view")
    weights_note = ("Google Drive only: pip install gdown, then "
                    "fetch_weights.py handles it")
    experimental = True

    def __init__(self, device="cuda", conf_keep=0.8, image_size=512,
                 revisit=1, model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self.image_size, self.revisit = image_size, revisit
        self._weights_override = model_id

    def _load(self):
        self.model_id = self.resolve_weights(self._weights_override)
        isolate_third_party(("dust3r", "croco", "models"),
                            *self.third_party_paths)
        patch_torch_load_legacy_checkpoints()
        from dust3r.model import ARCroco3DStereo
        self._model = ARCroco3DStereo.from_pretrained(self.model_id).to(
            self.device).eval()

    def _raw(self, image_paths) -> RawViews:
        import tempfile
        isolate_third_party(("dust3r", "croco", "models"),
                            *self.third_party_paths)
        from demo import prepare_input, prepare_output        # repo root
        from dust3r.inference import inference

        paths = [str(p) for p in image_paths]
        # signature verified in demo.py: (img_paths, img_mask, size,
        # raymaps=None, raymap_mask=None, revisit=1, update=True); their own
        # main passes img_mask=[True]*len(img_paths).
        views = prepare_input(img_paths=paths,
                              img_mask=[True] * len(paths),
                              size=self.image_size,
                              revisit=self.revisit)
        outputs, _ = inference(views, self._model, self.device)
        with tempfile.TemporaryDirectory() as td:
            pts3ds_other, colors, conf, cam_dict = prepare_output(
                outputs, td, self.revisit, True)

        def np_(x):
            return np.asarray(x.detach().float().cpu().numpy()
                              if hasattr(x, "detach") else x, dtype=float)

        pts_w = np.stack([np_(p).squeeze() for p in pts3ds_other])
        rgb = np.stack([np_(c).squeeze() for c in colors])
        cf = np.stack([np_(c).squeeze() for c in conf])
        poses = _poses_from_cam_dict(cam_dict, pts_w.shape[0])

        # use_pose=True puts the point maps in the world frame; the socket
        # wants camera-frame points, so undo the pose per view.
        pts_local = np.empty_like(pts_w)
        for i in range(poses.shape[0]):
            Rt = np.linalg.inv(poses[i])
            pts_local[i] = pts_w[i] @ Rt[:3, :3].T + Rt[:3, 3]

        H, W = pts_w.shape[1], pts_w.shape[2]
        f = float(np.median(np.asarray(cam_dict["focal"], dtype=float)))
        intr = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]])
        return RawViews(poses=poses, rgb=np.clip(rgb, 0, 1), conf=cf,
                        pts_local=pts_local, intrinsics=intr,
                        extras=dict(focal=f))
