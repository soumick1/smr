"""MV-DUSt3R+ adapter.

Repo: github.com/facebookresearch/mvdust3r
Checkpoints: huggingface.co/Zhenggang/MV-DUSt3R/tree/main/checkpoints
    MVD.pth      -- MV-DUSt3R
    MVDp_s1.pth  -- MV-DUSt3R+ (stage 1, 8 views)   <- our default

Flow reconstructed from their demo.py (verified 2026-08-24) rather than
called through it: `get_reconstructed_scene` contains a literal
`input('press enter to continue')` and writes a .glb, so it is unusable
as a library call.  We reproduce only the parts that matter:

    from dust3r.model import AsymmetricCroCo3DStereoMultiView   # NOT the
    from dust3r.inference import inference_mv                   # plain class
    from dust3r.losses import (calibrate_camera_pnpransac,
                               estimate_focal_knowing_depth)

    imgs   = load_images(filelist, size, n_frame=...)
    output = inference_mv(imgs, model, device)
    pts3d  = [output['pred1']['pts3d'][0]] + \
             [x['pts3d_in_other_view'][0] for x in output['pred2s']]
    conf   = [output['pred1']['conf'][0]]  + [x['conf'][0] for x in pred2s]
    focal  = estimate_focal_knowing_depth(pts3d[0], valid_first)
    c2w_i  = calibrate_camera_pnpransac(pts3d_i, pixel_coords, valid_i, K)

POSE CONVENTION: camera-to-world (their variable is `cams2world`, recovered
by PnP-RANSAC against a shared pinhole).

All point maps are expressed in the FIRST view's frame, so the adapter
transforms them into each camera's own frame for the socket, exactly as in
the fast3r adapter.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .pointmap import (PointmapBackbone, RawViews,
                       isolate_third_party)


@register("mvdust3r")
class MVDUSt3RBackbone(PointmapBackbone):
    name = "mvdust3r"
    pose_convention = "c2w"
    third_party_paths = ("mvdust3r",)
    requires = ("dust3r",)
    weights = "MVDp_s1.pth"
    weights_file = "MVDp_s1.pth"
    weights_hub = None
    weights_url = ("https://huggingface.co/Zhenggang/MV-DUSt3R/resolve/main/"
                   "checkpoints/MVDp_s1.pth")
    experimental = True

    def __init__(self, device="cuda", conf_keep=0.8, image_size=512,
                 conf_pct_focal=0.03, model_id=None, **kw):
        super().__init__(device=device, conf_keep=conf_keep, **kw)
        self.image_size = image_size
        self.conf_pct_focal = conf_pct_focal
        self._weights_override = model_id

    def _load(self):
        import torch
        self.model_id = self.resolve_weights(self._weights_override)
        # mvdust3r vendors its OWN dust3r fork (the only one defining
        # AsymmetricCroCo3DStereoMultiView).  If a previous backbone in this
        # process already imported plain dust3r, the cached module wins and
        # the class looks missing -- so purge and re-prioritise first.
        isolate_third_party(("dust3r", "croco"), "mvdust3r")
        from dust3r.model import AsymmetricCroCo3DStereoMultiView
        self._model = AsymmetricCroCo3DStereoMultiView.from_pretrained(
            self.model_id).to(self.device).eval()
        self._torch = torch

    def _raw(self, image_paths) -> RawViews:
        import torch
        isolate_third_party(("dust3r", "croco"), "mvdust3r")
        from dust3r.inference import inference_mv
        from dust3r.losses import (calibrate_camera_pnpransac,
                                   estimate_focal_knowing_depth)
        from dust3r.utils.image import load_images

        paths = [str(p) for p in image_paths]
        try:                                   # their loader takes n_frame
            imgs = load_images(paths, size=self.image_size, verbose=False,
                               n_frame=len(paths))
        except TypeError:
            imgs = load_images(paths, size=self.image_size, verbose=False)
        if len(imgs) == 1:                     # their single-view handling
            import copy
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]["idx"] = 1
        for im in imgs:
            if not torch.is_tensor(im["true_shape"]):
                im["true_shape"] = torch.from_numpy(im["true_shape"]).long()

        with torch.no_grad():
            out = inference_mv(imgs, self._model, self.device, verbose=False)

            pts = [out["pred1"]["pts3d"][0]] + \
                  [x["pts3d_in_other_view"][0] for x in out["pred2s"]]
            conf = torch.stack([out["pred1"]["conf"][0]] +
                               [x["conf"][0] for x in out["pred2s"]], 0)
            h, w = pts[0].shape[0], pts[0].shape[1]

            # focal from the first view's confident pixels (their recipe)
            c0 = conf[0].reshape(-1)
            thr = c0.sort()[0][int(c0.shape[0] * self.conf_pct_focal)]
            valid_first = (c0 >= thr).reshape(h, w)
            focal = float(estimate_focal_knowing_depth(
                pts[0][None].to(self.device),
                valid_first[None].to(self.device)).cpu().item())

            K = torch.eye(3, device=self.device)
            K[0, 0] = K[1, 1] = focal
            K[0, 2], K[1, 2] = w / 2.0, h / 2.0

            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w),
                                    indexing="ij")
            pix = torch.stack([xx, yy], -1).float().to(self.device)

            # PnP needs a per-view validity mask; reuse the global threshold
            cs = conf.reshape(-1).sort()[0]
            gthr = cs[int(cs.shape[0] * 0.03)]
            msk = conf >= gthr

            c2ws = []
            for p, v in zip(pts, msk):
                c2ws.append(calibrate_camera_pnpransac(
                    p.to(self.device).flatten(0, 1)[None],
                    pix.flatten(0, 1)[None],
                    v.to(self.device).flatten(0, 1)[None], K[None])[0])
            poses = torch.stack(c2ws, 0).cpu().numpy().astype(float)

            rgbs = [out["pred1"]["rgb"][0]] if "rgb" in out["pred1"] else None
            if rgbs is None:                   # demo injects rgb after the
                rgbs = [imgs[0]["img"][0].permute(1, 2, 0)]  # forward pass
                rgbs += [im["img"][0].permute(1, 2, 0) for im in imgs[1:]]
            rgb = np.stack([((r.float().cpu().numpy() + 1) / 2)
                            for r in rgbs])
            pts_w = np.stack([p.float().cpu().numpy() for p in pts])
            conf_np = conf.float().cpu().numpy()

        pts_local = np.empty_like(pts_w)        # first-view frame -> camera i
        for i in range(poses.shape[0]):
            Rt = np.linalg.inv(poses[i])
            pts_local[i] = pts_w[i] @ Rt[:3, :3].T + Rt[:3, 3]

        intr = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]])
        return RawViews(poses=poses, rgb=np.clip(rgb, 0, 1), conf=conf_np,
                        pts_local=pts_local, intrinsics=intr,
                        extras=dict(focal=focal))
