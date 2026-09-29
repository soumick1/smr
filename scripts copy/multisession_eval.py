#!/usr/bin/env python3
"""Gauge-fair evaluation of a multi-session run (v177, plan Task 7).

    python scripts/multisession_eval.py --est outputs/ms/chess_s0106_streamvggt.npz --gt data/gt/7scenes_chess_s0106.npz

The GT npz carries frame_ids with the session in the 100000s digit (as pilot_a expects).  For every trajectory row
in the --save-est npz (est_chained, est_smr, est_smr_pgo) this reports
  * joint ATE: one Sim(3) over all sessions (what Table 2's multi-session cells report),
  * per-session ATE: one Sim(3) per session, so a query session that starts in an arbitrary frame is judged on its own
    trajectory quality and not on where the memory placed it,
  * frame recovery: the residual similarity between the per-session alignments of session A and session B
    (rotation deg, translation in units of the GT extent, |log scale|) -- zero means the second session was placed in the
    first session's frame; this is the quantity persistence actually claims.
The four conditions of the plan (empty bank / retained bank / saved+reloaded bank / unrelated-scene bank) are separate
runs; this script scores each of them with the same rule.
"""
import argparse, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from smr.eval.trajectory import align_to_gt, ate_rmse, rotation_angle_deg  # noqa: E402
from smr.stitch.chunks import keyframe_indices  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--est", required=True); ap.add_argument("--gt", required=True)
ap.add_argument("--rows", nargs="*", default=None)
a = ap.parse_args()
z = np.load(a.est, allow_pickle=True); g = np.load(a.gt, allow_pickle=True)
key = [int(k) for k in z["key"]]
gt = np.asarray(g["poses"], float)[key]
fid = np.asarray(g["frame_ids"])
sess = (fid[key] // 100000).astype(int)
ids = sorted(set(sess.tolist()))
extent = float(np.linalg.norm(np.ptp(gt[:, :3, 3], axis=0)))
print(f"{len(key)} keyframes, sessions {ids} with sizes {[int((sess == s).sum()) for s in ids]}, GT extent {extent:.3f}")
rows = a.rows or [k[4:] for k in z.files if k.startswith("est_")]
for r in rows:
    est = np.asarray(z[f"est_{r}"], float)
    joint = ate_rmse(est, gt)
    per, fits = [], []
    for s in ids:
        m = sess == s
        per.append(ate_rmse(est[m], gt[m]))
        _, S = align_to_gt(est[m], gt[m], with_scale=True)
        fits.append(S)
    line = f"{r:<10} joint ATE {joint:.3f}   per-session ATE " + " ".join(f"{p:.3f}" for p in per)
    if len(fits) >= 2:
        for s_i in range(1, len(fits)):
            Sa, Sb = fits[0], fits[s_i]
            # residual similarity between the two per-session alignments: identity if B sits in A's frame
            sa, Ra, ta = Sa if isinstance(Sa, tuple) else (Sa["s"], Sa["R"], Sa["t"])
            sb, Rb, tb = Sb if isinstance(Sb, tuple) else (Sb["s"], Sb["R"], Sb["t"])
            rot = rotation_angle_deg(Ra.T @ Rb)
            trans = float(np.linalg.norm(ta - tb)) / max(extent, 1e-9)
            ls = abs(float(np.log(sb / sa)))
            line += f"   frame recovery s{ids[0]}->s{ids[s_i]}: rot {rot:.2f} deg, trans {trans:.3f} extent, |log s| {ls:.3f}"
    print(line)
