"""Path-integrated view change (T4): predict what the camera would see at a
pose reached by integrating motion commands from one localised image --
without ever showing the backbone the target.

The memory holds keyframes (descriptor + metric pose) and their surfels
(points + colours lifted once, at bind time).  A path starts from ONE image:
localise it against memory (the T2 machinery), then apply relative motion
commands; at the integrated pose the memory is read out by splatting the
bank.  Baselines bracket it from below (nearest stored view; a single stored
view's reprojection) and above (a fresh backbone pass on the recalled
keyframes at query time).

Metrics are the established ones, each scored against the real frame at the
target pose: PSNR / SSIM / LPIPS (the novel-view-synthesis convention),
depth abs-rel against the dataset's depth (the monocular-depth convention),
plus coverage (fraction of target pixels the readout can answer at all) --
the column a generative model wins by construction and a memory must earn.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ splat --
def splat_points(points_w, colors, T_wc, K, hw, point_px=2):
    """Z-buffer splat of world points into a pinhole view.

    T_wc: camera-to-world 4x4 (OpenCV: x right, y down, z forward).
    Returns (rgb HxWx3 in [0,1], depth HxW, mask HxW bool)."""
    H, W = hw
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    Pc = (points_w - t) @ R                       # world -> camera
    z = Pc[:, 2]
    ok = z > 1e-6
    Pc, C, z = Pc[ok], colors[ok], z[ok]
    u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    img = np.zeros((H, W, 3)); dep = np.full((H, W), np.inf); r = point_px // 2
    order = np.argsort(-z)                        # far first, near overwrites
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    for i in order[inside[order]]:
        y0, y1 = max(0, vi[i] - r), min(H, vi[i] + r + 1)
        x0, x1 = max(0, ui[i] - r), min(W, ui[i] + r + 1)
        img[y0:y1, x0:x1] = C[i]
        dep[y0:y1, x0:x1] = z[i]
    mask = np.isfinite(dep)
    dep[~mask] = 0.0
    return img, dep, mask


# ---------------------------------------------------------------- metrics --
def psnr(a, b, mask=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        d = ((a - b) ** 2)[mask]
    else:
        d = (a - b) ** 2
    mse = float(d.mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def _gauss1d(sig=1.5, n=11):
    x = np.arange(n) - n // 2
    k = np.exp(-0.5 * (x / sig) ** 2)
    return k / k.sum()


def ssim(a, b):
    """Standard grayscale SSIM, 11x11 Gaussian window, K1/K2 = 0.01/0.03."""
    a = np.asarray(a, float).mean(-1) if a.ndim == 3 else np.asarray(a, float)
    b = np.asarray(b, float).mean(-1) if b.ndim == 3 else np.asarray(b, float)
    k = _gauss1d()
    def f(x):
        x = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, x)
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, x)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = f(a), f(b)
    va, vb = f(a * a) - ma * ma, f(b * b) - mb * mb
    cab = f(a * b) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2) + 1e-12)
    return float(s.mean())


def depth_absrel(d_est, d_gt, mask):
    """abs-rel on pixels where both the estimate and the ground truth are
    valid (the Eigen convention)."""
    m = mask & (d_gt > 1e-6) & (d_est > 1e-6)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(d_est[m] - d_gt[m]) / d_gt[m]))


def lpips_fn():
    """Returns an LPIPS(alex) callable on HxWx3 [0,1] arrays, or None."""
    try:
        import lpips, torch
        net = lpips.LPIPS(net="alex", verbose=False)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = net.to(dev).eval()

        def f(a, b):
            ta = torch.from_numpy(np.asarray(a, np.float32).transpose(2, 0, 1))[None] * 2 - 1
            tb = torch.from_numpy(np.asarray(b, np.float32).transpose(2, 0, 1))[None] * 2 - 1
            with torch.no_grad():
                return float(net(ta.to(dev), tb.to(dev)).item())
        return f
    except Exception:
        return None


def score_view(rgb_est, dep_est, mask, rgb_gt, dep_gt, lp=None):
    out = dict(psnr=psnr(rgb_est, rgb_gt, mask), psnr_full=psnr(rgb_est, rgb_gt),
               ssim=ssim(rgb_est, rgb_gt), coverage=float(mask.mean()),
               absrel=depth_absrel(dep_est, dep_gt, mask))
    if lp is not None:
        out["lpips"] = lp(rgb_est, rgb_gt)
    return out


def build_bank(infer_fn, map_paths, chunks, pose_of, owner_of, fit_robust,
               pixel_stride=5, conf_keep=0.3):
    """Lift every map keyframe ONCE (its owner chunk's pass) into metric world
    points, pixel-indexed, so dense 3D-3D placement and view readout share one
    store.  pose_of/owner_of are keyed by KEYFRAME POSITION.  Returns
    dict(kf -> dict(pix=(M,2) int yx, pts=(M,3) world, col=(M,3)))."""
    bank = {}
    for ci, ch in enumerate(chunks):
        rv = infer_fn([map_paths[i] for i in ch])
        A = rv.poses
        B = np.stack([pose_of[i] for i in ch])
        S, _, _ = fit_robust(A, B, min_inliers=3)
        H, W = rv.depth.shape[1:3]
        ys, xs = np.mgrid[0:H:pixel_stride, 0:W:pixel_stride]
        for li, gi in enumerate(ch):
            if owner_of[gi] != ci:
                continue
            d = rv.depth[li][ys, xs]
            ok = d > 1e-6
            conf = getattr(rv, "conf", None)
            if conf is not None and ok.any():
                c = conf[li][ys, xs]
                ok &= c >= np.quantile(c[ok], conf_keep)
            K_ = rv.intrinsics[li] if rv.intrinsics.ndim == 3 else rv.intrinsics
            X = (xs[ok] - K_[0, 2]) / K_[0, 0] * d[ok]
            Y = (ys[ok] - K_[1, 2]) / K_[1, 1] * d[ok]
            Pc = np.stack([X, Y, d[ok]], 1)
            Pw = Pc @ rv.poses[li][:3, :3].T + rv.poses[li][:3, 3]
            Pm = S[0] * (Pw @ S[1].T) + S[2]
            col = rv.rgb[li][ys, xs][ok]
            bank[gi] = dict(pix=np.stack([ys[ok], xs[ok]], 1).astype(np.int32),
                            pts=Pm.astype(np.float64),
                            col=(col / (255.0 if col.max() > 2 else 1.0)).astype(np.float32))
    return bank
