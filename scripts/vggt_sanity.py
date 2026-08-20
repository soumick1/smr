#!/usr/bin/env python3
"""Backbone-swap sanity (plan week 3): run VGGT on the synthetic frames and
compare its geometry to ground truth after Sim(3) alignment.

Usage (server, after export_synthetic_frames.py):
    python scripts/vggt_sanity.py --frames outputs/synth_frames
Reports per-view rotation / centre errors after Umeyama alignment and
scale-aligned depth error; writes a comparison figure.  Soft thresholds
(rot < 3 deg, centre < 3% scene scale, depth medAE < 5%) are printed as
PASS/WARN -- this scene is easy, so misses indicate wiring problems, not
model limits.
"""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt                          # noqa: E402
from smr.backbones import get_backbone                   # noqa: E402
from smr.viz import TEAL, CORAL, savefig                 # noqa: E402


def self_consistency(out, pairs=None, conf_mask=True):
    """Cross-view reprojection using ONLY the backbone's own outputs:
    unproject depth_i with pose_i, reproject into view j, compare against
    depth_j.  Median relative error over adjacent pairs.  Distinguishes
    'backbone failed on this input' (inconsistent) from 'adapter/GT
    comparison is wrong' (self-consistent yet mismatching GT)."""
    K_all = out.extras.get("intrinsics_all")
    S = out.poses.shape[0]
    pairs = pairs or [(i, (i + 1) % S) for i in range(S)]
    errs = []
    for i, j in pairs:
        Ki = K_all[i] if K_all is not None else out.intrinsics
        Kj = K_all[j] if K_all is not None else out.intrinsics
        H, W = out.depth[i].shape
        step = max(1, H // 130)
        vs, us = np.mgrid[0:H:step, 0:W:step]
        d = out.depth[i][::step, ::step]
        m = d > 0
        if conf_mask:
            m &= out.mask[i][::step, ::step]
        x = (us - Ki[0, 2]) / Ki[0, 0] * d
        y = (vs - Ki[1, 2]) / Ki[1, 1] * d
        Pc = np.stack([x[m], y[m], d[m]], -1)
        Pw = Pc @ out.poses[i][:3, :3].T + out.poses[i][:3, 3]
        Rj, tj = out.poses[j][:3, :3], out.poses[j][:3, 3]
        Pj = (Pw - tj) @ Rj
        z = Pj[:, 2]; ok = z > 1e-3
        uj = (Kj[0, 0] * Pj[ok, 0] / z[ok] + Kj[0, 2]).round().astype(int)
        vj = (Kj[1, 1] * Pj[ok, 1] / z[ok] + Kj[1, 2]).round().astype(int)
        ib = (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
        dj = out.depth[j][vj[ib], uj[ib]]
        good = dj > 0
        if good.sum() < 50:
            continue
        errs.append(np.median(np.abs(z[ok][ib][good] - dj[good]) / dj[good]))
    print("  per-pair self-consistency:",
          np.array2string(np.array(errs), precision=4))
    return float(np.mean(errs)) if errs else float("nan")


def umeyama_sim3(X, Y):
    """Similarity aligning X -> Y (both (N,3)): returns s, R, t."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (Xc ** 2).sum() * len(X)
    t = my - s * R @ mx
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="outputs/synth_frames")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    fr = pathlib.Path(a.frames)
    if not (fr / "gt.npz").exists():
        # GT-free mode (real photos): adapter validation via VGGT's own
        # internal consistency + qualitative depth / top-down trajectory.
        paths = sorted(sum([[str(p) for p in fr.glob(pat)]
                            for pat in ("*.png", "*.jpg", "*.jpeg")], []))
        assert paths, f"no images found in {fr}"
        bb = get_backbone("vggt", device=a.device)
        out = bb.infer(paths)
        sc = self_consistency(out)
        print(f"  [{'PASS' if sc < 0.05 else 'WARN'}] SELF-consistency"
              f"       {sc:.4f} (soft thr 0.05, no GT)")
        C = out.poses[:, :3, 3]
        fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
        axs[0].imshow(np.clip(out.rgb[0], 0, 1)); axs[0].set_title("view 0")
        axs[1].imshow(out.depth[0], cmap="viridis")
        axs[1].set_title("VGGT depth view 0")
        axs[2].plot(C[:, 0], C[:, 2], "o-", color=TEAL)
        axs[2].set_title("camera centres (top-down)"); axs[2].axis("equal")
        for ax in axs[:2]: ax.axis("off")
        savefig(fig, pathlib.Path("outputs/figures/vggt_sanity_nogt.png"))
        return
    gt = np.load(fr / "gt.npz", allow_pickle=True)
    paths = [str(p) for p in gt["paths"]]
    T_gt = gt["poses"]; D_gt = gt["depth"]

    bb = get_backbone("vggt", device=a.device)
    out = bb.infer(paths)

    C_est = out.poses[:, :3, 3]
    C_gt = T_gt[:, :3, 3]
    s, R, t = umeyama_sim3(C_est, C_gt)
    scene = np.linalg.norm(C_gt.max(0) - C_gt.min(0))
    rot_e, cen_e = [], []
    for i in range(len(paths)):
        Ra = R @ out.poses[i, :3, :3]
        dR = Ra.T @ T_gt[i, :3, :3]
        rot_e.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2,
                                                  -1, 1))))
        cen_e.append(np.linalg.norm(s * R @ C_est[i] + t - C_gt[i]) / scene)
    # depth: nearest-resize est to GT grid, align scale on valid pixels
    dep_errs = []
    for i in range(len(paths)):
        de = out.depth[i]
        if de.shape != D_gt[i].shape:
            yi = (np.linspace(0, de.shape[0] - 1, D_gt[i].shape[0])).astype(int)
            xi = (np.linspace(0, de.shape[1] - 1, D_gt[i].shape[1])).astype(int)
            de = de[np.ix_(yi, xi)]
        v = (D_gt[i] > 0) & (de > 0)
        sc = np.median(D_gt[i][v] / de[v])
        dep_errs.append(np.median(np.abs(sc * de[v] - D_gt[i][v])
                                  / D_gt[i][v]))
    rot_m, cen_m, dep_m = map(lambda x: float(np.mean(x)),
                              (rot_e, cen_e, dep_errs))
    sc = self_consistency(out)
    for name, val, thr in (("rotation (deg)", rot_m, 3.0),
                           ("centre (frac scene)", cen_m, 0.03),
                           ("depth rel medAE", dep_m, 0.05),
                           ("SELF-consistency", sc, 0.05)):
        print(f"  [{'PASS' if val < thr else 'WARN'}] {name:22s} "
              f"{val:.4f} (soft thr {thr})")
    # reference-anchored drift: align frames exactly at view 0, then plot
    # per-view rotation error against GT angular distance from view 0 --
    # reference-anchored backbones show error growing with this distance
    # (the inconsistency SMR's binding is designed to absorb).
    R0a = T_gt[0, :3, :3] @ out.poses[0, :3, :3].T
    drift, dist0 = [], []
    for i in range(len(paths)):
        dR = (R0a @ out.poses[i, :3, :3]).T @ T_gt[i, :3, :3]
        drift.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2,
                                                  -1, 1))))
        dG = T_gt[0, :3, :3].T @ T_gt[i, :3, :3]
        dist0.append(np.degrees(np.arccos(np.clip((np.trace(dG) - 1) / 2,
                                                  -1, 1))))
    fig, axs = plt.subplots(1, 4, figsize=(12.6, 3.0))
    axs[0].imshow(D_gt[0], cmap="viridis"); axs[0].set_title("GT depth v0")
    axs[1].imshow(out.depth[0], cmap="viridis"); axs[1].set_title("VGGT depth v0")
    axs[2].bar(range(len(rot_e)), rot_e, color=TEAL)
    axs[2].axhline(3.0, color=CORAL, ls="--")
    axs[2].set_title("per-view rot err (deg)")
    order = np.argsort(dist0)
    axs[3].plot(np.array(dist0)[order], np.array(drift)[order], "o-",
                color=CORAL)
    axs[3].set_xlabel("GT angular distance from view 0 (deg)")
    axs[3].set_title("reference-anchored drift")
    for ax in axs[:2]: ax.axis("off")
    savefig(fig, pathlib.Path("outputs/figures/vggt_sanity.png"))


if __name__ == "__main__":
    main()
