#!/usr/bin/env python3
"""Harvest (partial recalled splat -> actual frame) supervision pairs for
the completion head (I6, the only trained component).

    python scripts/harvest_completion_pairs.py --frames lab_photos/room/images_8 \
           --backbone vggt --out data/completion

For every holdout view h: bind the remaining views, recall the k nearest
bound views at h's pose, splat, and save
    {scene}_{backbone}_h{h:03d}.npz  with
    splat_rgb, splat_depth, splat_mask   (the partial render -- input)
    tgt_rgb,  tgt_depth,  tgt_mask       (the actual frame -- target)
    pose (4x4), K (3x3)
Disocclusions (mask False) are exactly the pixels the head must fill."""
import argparse, importlib.util, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smr.dynamics import ScaffoldState                  # noqa: E402
from smr.pipeline import bind                           # noqa: E402
from smr.scene import Camera, splat, transform          # noqa: E402

spec = importlib.util.spec_from_file_location(
    "t3", ROOT / "experiments" / "run_tier3_vggt.py")
t3 = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ["t3"]; spec.loader.exec_module(t3)
sys.argv = _argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--backbone", default="vggt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data/completion")
    ap.add_argument("--holdouts", default="all",
                    help="'all', 'every:N', or comma-separated view indices")
    ap.add_argument("--context", default="all",
                    help="comma list of context sizes per holdout, e.g. "
                         "'1,4,all': nearest-N-by-pose views are bound. "
                         "Trains one head across the 1-view..N-view regime.")
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()

    out, T_gt, D_gt = t3.backbone_output(a.backbone, a.frames, a.device)
    out = t3.fit_decode_window(out, periods=[2.4, 3.2, 4.0])
    K_v = out.poses.shape[0]
    if a.holdouts == "all":
        hold = list(range(K_v))
    elif a.holdouts.startswith("every:"):
        hold = list(range(0, K_v, int(a.holdouts.split(":")[1])))
    else:
        hold = [int(x) for x in a.holdouts.split(",")]
    ctxs = a.context.split(",")
    p_fr = pathlib.Path(a.frames)
    scene = p_fr.parent.name if p_fr.name.startswith("images") else p_fr.name
    dst = pathlib.Path(a.out); dst.mkdir(parents=True, exist_ok=True)
    cam = Camera(H=out.depth.shape[1], W=out.depth.shape[2],
                 f=float(out.intrinsics[0, 0]))

    from smr.utils.geometry import pose_errors as _pe

    def nearest_ctx(h, n_ctx):
        others = [i for i in range(K_v) if i != h]
        d = [sum(_pe(out.poses[i], out.poses[h])) for i in others]
        order = [others[i] for i in np.argsort(d)]
        return sorted(order[:n_ctx])

    n = 0
    for h in hold:
        for cs in ctxs:
            tagc = "all" if cs == "all" else f"{int(cs):02d}"
            fn = dst / f"{scene}_{a.backbone}_h{h:03d}_c{tagc}.npz"
            if fn.exists():
                print(f"  h{h:03d} c{tagc}: exists, skipping")
                continue
            keep = ([i for i in range(K_v) if i != h] if cs == "all"
                    else nearest_ctx(h, int(cs)))
            ss = ScaffoldState(periods=[2.4, 3.2, 4.0], ring_N=128,
                               torus_N=32, seed=0, omega_max=0.16)
            ss.calibrate()
            sub = t3.subset(out, keep)
            bound = bind(sub, ss, cam, formation_extent=1.2,
                         formation_spacing=0.4, torus_N=32, N_h=1024, k=64,
                         seed=0)
            T_q = out.poses[h]
            ss.place_pose(T_q)
            ids = bound.store.nearest(ss.state(), k=min(a.k, len(keep)))
            P, C = [], []
            for j in ids:
                pay = bound.store.content[j]
                P.append(transform(pay["pts"], pay["T"], T_q))
                C.append(pay["cols"])
            rgb_s, dep_s, msk_s = splat(np.concatenate(P), np.concatenate(C),
                                        T_q, cam)
            np.savez_compressed(
                fn,
                splat_rgb=rgb_s.astype(np.float16),
                splat_depth=dep_s.astype(np.float16),
                splat_mask=msk_s,
                tgt_rgb=out.rgb[h].astype(np.float16),
                tgt_depth=out.depth[h].astype(np.float16),
                tgt_mask=out.mask[h],
                pose=T_q, K=out.intrinsics, context=len(keep))
            n += 1
            print(f"  h{h:03d} c{tagc}: coverage {msk_s.mean():.2f} -> "
                  f"{fn.name}")
    print(f"harvested {n} pairs -> {dst}")


if __name__ == "__main__":
    main()
