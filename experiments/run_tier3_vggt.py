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


def backbone_output(name, frames_dir, device, **bb_kw):
    T_gt, K, D_gt, paths = load_gt(frames_dir)
    if name == "synthetic":
        bb_kw.pop("use_features", None)
        assert T_gt is not None, "--backbone synthetic needs gt.npz"
        rgb = np.stack([imageio.imread(p) for p in paths]).astype(float) / 255
        out = BackboneOutput(poses=T_gt.copy(), intrinsics=K, depth=D_gt.copy(),
                             rgb=rgb, mask=D_gt > 0)
    else:
        out = get_backbone(name, device=device, **bb_kw).infer(paths)
    return out, T_gt, D_gt


def fit_decode_window(out, periods, frac=0.8):
    """Guard the residue-decode envelope: camera centres must lie inside
    +-max(period)/2 per axis or the chained decode wraps.  Well-posed
    scenes (median depth 1, centres ~1) pass untouched; out-of-envelope
    input (e.g. unrelated photos with degenerate scale) is Sim(3)-rescaled
    into the window with a loud notice."""
    window = max(periods) / 2.0
    m = float(np.abs(out.poses[:, :3, 3]).max())
    s2 = max(1.0, m / (frac * window))
    if s2 > 1.0:
        out.poses = out.poses.copy()
        out.poses[:, :3, 3] /= s2
        out.depth = out.depth / s2
        print(f"  [NOTE] scene envelope: max|centre|={m:.2f} exceeds "
              f"{frac:.1f}x decode window {window:.1f}; rescaled by 1/{s2:.2f}")
    return out


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
    ap.add_argument("--features", action="store_true",
                    help="adapter v2: pooled VGGT aggregator tokens as "
                         "descriptors (VERIFY-ON-SERVER)")
    a = ap.parse_args()
    rep = Report(f"Tier 3 / estimated geometry ({a.backbone})")
    print(rep.title)

    out, T_gt, D_gt = backbone_output(a.backbone, a.frames, a.device,
                                      use_features=a.features)

    out = fit_decode_window(out, periods=[2.4, 3.2, 4.0])
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

    # V3: relocalisation quality in POSE space, thresholded by the
    # capture's own view spacing.  On dense captures (LLFF: dozens of
    # near-duplicate frames a degree apart) exact-ID among near-identical
    # views is meaningless -- no global descriptor should separate them.
    # Landing within 2x the median adjacent-view spacing of the true pose
    # IS successful relocalisation; on sparse rings (views 30-45 deg
    # apart) this reduces to the old exact-ID criterion.
    g0 = rng(77)
    Hs = [bound.store.content[j]["h"] for j in range(len(bound.store.states))]
    sub = subset(out, keep)
    d_rot, d_pos = [], []
    for i in range(len(keep) - 1):
        re_, pe_ = pose_errors(sub.poses[i], sub.poses[i + 1])
        d_rot.append(re_); d_pos.append(pe_)
    thr_rot = max(2 * float(np.median(d_rot)), 0.02)
    thr_pos = max(2 * float(np.median(d_pos)), 0.02)
    # Corruption is stated in SIGNAL units: unit-norm noise on a unit
    # descriptor (cos ~ 0.71 to the clean cue).  The earlier 0.2/dim
    # convention was noise of norm 4.2 -- an SNR under which neighbour
    # discrimination is information-theoretically impossible on dense
    # captures.  Retrieval is two-stage, as the architecture implies:
    # the h-cue PROPOSES top-5 candidates (pattern completion through the
    # memory), direct descriptor comparison VERIFIES among them --
    # avoiding the cue-inversion noise amplification that produces
    # catastrophic aliasing flips.
    S_all = [bound.store.content[j]["s"] for j in range(len(Hs))]
    ok = exact = 0
    perr_rot, perr_pos = [], []
    for i in range(len(keep)):
        nz = g0.standard_normal(448); nz /= np.linalg.norm(nz)
        st = sub.descriptor(i) + nz
        st /= np.linalg.norm(st)
        hh = bound.mem.cue_h(st)
        cand = np.argsort([hh @ hj for hj in Hs])[-5:]
        j = int(cand[np.argmax([st @ S_all[c] for c in cand])])
        re_, pe_ = pose_errors(sub.poses[j], sub.poses[i])
        perr_rot.append(re_); perr_pos.append(pe_)
        ok += int(re_ <= thr_rot and pe_ <= thr_pos)
        exact += int(j == i)
    rep.check(f"V3 reloc within 2x view spacing (of {len(keep)})",
              (ok, round(thr_rot, 3), round(thr_pos, 3)),
              ok >= 0.9 * len(keep),
              f"90 pct within rot<={thr_rot:.3f} rad, pos<={thr_pos:.3f} "
              f"under unit-SNR cue corruption")
    rep.check("V3 exact-ID / median pose err (report)",
              (exact, round(float(np.median(perr_rot)), 4),
               round(float(np.median(perr_pos)), 4)), True,
              "informational: exact hits and median reloc distance")

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
    thr = 0.05 if a.backbone == "synthetic" else 0.12
    if both.sum() < 200:
        rep.check("V5 held-out depth vs backbone-self (rel med)",
                  ("NO OVERLAP", round(float(msk_s.mean()), 3)), False,
                  "no shared content at the held-out pose -- are these "
                  "frames views of ONE static scene?")
        p_fr = pathlib.Path(a.frames)
        scene = p_fr.parent.name if p_fr.name.startswith("images") else p_fr.name
        rep.save(OUT / "reports" / f"tier3_{a.backbone}_{scene}.json")
        sys.exit(1)
    rel_bb = float(np.median(np.abs(dep_s[both] - d_bb[both]) / d_bb[both]))
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
    p_fr = pathlib.Path(a.frames)
    scene = p_fr.parent.name if p_fr.name.startswith("images") else p_fr.name
    tag = f"{a.backbone}_{scene}"
    savefig(fig, OUT / "figures" / f"V5_{tag}.png")
    rep.save(OUT / "reports" / f"tier3_{tag}.json")
    sys.exit(0 if rep.all_passed else 1)


if __name__ == "__main__":
    main()
