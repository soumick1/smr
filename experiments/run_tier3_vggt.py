#!/usr/bin/env python3
"""Tier 3 through an ESTIMATED-geometry backbone: bind what the backbone
believes, measure what the scaffold machinery preserves and what error is
inherited from the backbone.

    python experiments/run_tier3_vggt.py --backbone vggt        # server
    python experiments/run_tier3_vggt.py --backbone synthetic   # logic check

With --backbone synthetic the exported GT itself plays the "estimate"
(exact), which validates every code path of this script on CPU; with vggt
the same checks quantify the real swap.  V-checks:
  V1 place->decode consistency against the backbone's own poses (our
     machinery must be sub-degree regardless of whose poses they are).
  V3 relocalisation from corrupted descriptors.
  V5 hold out one view; place at the backbone's pose for it; recall+splat;
     compare depth against (a) the backbone's own held-out depth (frame-
     consistent target, thresholded) and (b) true GT after per-view median
     scale alignment (reported: the honest inherited-error number).
"""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import imageio.v2 as imageio                             # noqa: E402
import matplotlib.pyplot as plt                          # noqa: E402
from smr.backbones import get_backbone, BackboneOutput   # noqa: E402
from smr.dynamics import ScaffoldState                   # noqa: E402
from smr.pipeline import bind                            # noqa: E402
from smr.scene import Camera, splat, transform           # noqa: E402
from smr.utils import Report, project_root, rng          # noqa: E402
from smr.utils.geometry import make_T, pose_errors       # noqa: E402
from smr.viz import TEAL, CORAL, savefig                 # noqa: E402

OUT = project_root() / "outputs"


def load_gt(frames_dir):
    fr = pathlib.Path(frames_dir)
    if not (fr / "gt.npz").exists():          # real photos: GT-free mode
        paths = sorted(sum([[str(p) for p in fr.glob(pat)]
                            for pat in ("*.png", "*.jpg", "*.jpeg")], []))
        assert paths, f"no images found in {fr}"
        return None, None, None, paths
    gt = np.load(fr / "gt.npz", allow_pickle=True)
    return gt["poses"], gt["K"], gt["depth"], [str(p) for p in gt["paths"]]


def backbone_output(name, frames_dir, device):
    T_gt, K, D_gt, paths = load_gt(frames_dir)
    if name == "synthetic":
        assert T_gt is not None, "--backbone synthetic needs gt.npz"
        rgb = np.stack([imageio.imread(p) for p in paths]).astype(float) / 255
        out = BackboneOutput(poses=T_gt.copy(), intrinsics=K, depth=D_gt.copy(),
                             rgb=rgb, mask=D_gt > 0)
    else:
        out = get_backbone(name, device=device).infer(paths)
    return out, T_gt, D_gt


def subset(out, keep):
    return BackboneOutput(poses=out.poses[keep], intrinsics=out.intrinsics,
                          depth=out.depth[keep], rgb=out.rgb[keep],
                          mask=out.mask[keep], extras=out.extras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vggt")
    ap.add_argument("--frames", default="outputs/synth_frames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--holdout", type=int, default=2)
    a = ap.parse_args()
    rep = Report(f"Tier 3 / estimated geometry ({a.backbone})")
    print(rep.title)

    out, T_gt, D_gt = backbone_output(a.backbone, a.frames, a.device)
    K_v = out.poses.shape[0]
    H, W = out.depth.shape[1:3]
    Kmat = out.intrinsics
    cam = Camera(H=H, W=W, f=float(Kmat[0, 0]))   # cx,cy ~ centre (checked ok)

    ss = ScaffoldState(periods=[2.4, 3.2, 4.0], ring_N=128, torus_N=32,
                       seed=0, omega_max=0.16)
    ss.calibrate()

    # V1: machinery consistency against the backbone's own frame
    rot_e, pos_e = [], []
    for i in range(K_v):
        ss.place_pose(out.poses[i])
        T_est = make_T(ss.decode_R(), ss.decode_position())
        re_, pe_ = pose_errors(T_est, out.poses[i])
        rot_e.append(re_); pos_e.append(pe_)
    rep.check("V1 place->decode vs backbone poses (rot,pos max)",
              (round(float(np.max(rot_e)), 4), round(float(np.max(pos_e)), 4)),
              np.max(rot_e) < 0.02 and np.max(pos_e) < 0.05,
              "sub-degree/1% regardless of whose poses")

    keep = [i for i in range(K_v) if i != a.holdout]
    bound = bind(subset(out, keep), ss, cam, formation_extent=1.2,
                 formation_spacing=0.4, torus_N=32, N_h=1024, k=64, seed=0)

    # V3: relocalisation
    g0 = rng(77); correct = 0
    Hs = [bound.store.content[j]["h"] for j in range(len(bound.store.states))]
    sub = subset(out, keep)
    for i in range(len(keep)):
        st = sub.descriptor(i) + 0.2 * g0.standard_normal(448)
        st /= np.linalg.norm(st)
        hh = bound.mem.cue_h(st)
        correct += int(int(np.argmax([hh @ hj for hj in Hs])) == i)
    rep.check(f"V3 relocalisation top-1 (of {len(keep)})", correct,
              correct >= len(keep) - 2, "corrupted-cue id (allow 2 miss)")
    # pose-space reloc quality (id metric is harsh on ring neighbours)
    perr = []
    for i in range(len(keep)):
        st = sub.descriptor(i) + 0.2 * g0.standard_normal(448)
        st /= np.linalg.norm(st)
        hh = bound.mem.cue_h(st)
        j = int(np.argmax([hh @ hj for hj in Hs]))
        re_, pe_ = pose_errors(sub.poses[j], sub.poses[i])
        perr.append(re_)
    rep.check("V3 reloc pose err (rad, median, report)",
              round(float(np.median(perr)), 4), True,
              "informational: distance of relocalised pose")

    # V5: held-out view via recall + splat
    T_q = out.poses[a.holdout]
    ss.place_pose(T_q)
    xi = ss.state()
    ids = bound.store.nearest(xi, k=2)
    P, C = [], []
    for j in ids:
        pay = bound.store.content[j]
        P.append(transform(pay["pts"], pay["T"], T_q)); C.append(pay["cols"])
    rgb_s, dep_s, msk_s = splat(np.concatenate(P), np.concatenate(C), T_q, cam)
    d_bb = out.depth[a.holdout]
    both = msk_s & (d_bb > 0) & out.mask[a.holdout]
    rel_bb = float(np.median(np.abs(dep_s[both] - d_bb[both]) / d_bb[both]))
    thr = 0.05 if a.backbone == "synthetic" else 0.12
    rep.check("V5 held-out depth vs backbone-self (rel med)",
              round(rel_bb, 4), rel_bb < thr and both.mean() > 0.10,
              f"recalled splat matches backbone frame (<{thr})")
    # inherited error vs TRUE GT (report-only): per-view median scale align
    if D_gt is None:
        savefig_only = True
        d_gt = None
    else:
        savefig_only = False
        d_gt = D_gt[a.holdout]
    if not savefig_only and d_gt.shape != dep_s.shape:
        yi = np.linspace(0, d_gt.shape[0] - 1, dep_s.shape[0]).astype(int)
        xi2 = np.linspace(0, d_gt.shape[1] - 1, dep_s.shape[1]).astype(int)
        d_gt = d_gt[np.ix_(yi, xi2)]
    if not savefig_only:
        v = msk_s & (d_gt > 0)
        sc = np.median(d_gt[v] / dep_s[v])
        rel_gt = float(np.median(np.abs(sc * dep_s[v] - d_gt[v]) / d_gt[v]))
        rep.check("V5 inherited error vs true GT (rel med, report)",
                  round(rel_gt, 4), True,
                  "informational: backbone+SMR vs world")

    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    axs[0].imshow(np.clip(rgb_s, 0, 1)); axs[0].set_title("recalled splat @ held-out")
    axs[1].imshow(d_bb, cmap="viridis"); axs[1].set_title("backbone depth (held-out)")
    err = np.where(both, np.abs(dep_s - d_bb), np.nan)
    im = axs[2].imshow(err, cmap="magma"); axs[2].set_title(f"|dz|, med rel {rel_bb:.3f}")
    for ax in axs: ax.axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    savefig(fig, OUT / "figures" / f"V5_{a.backbone}.png")
    rep.save(OUT / "reports" / f"tier3_{a.backbone}.json")
    sys.exit(0 if rep.all_passed else 1)


if __name__ == "__main__":
    main()
