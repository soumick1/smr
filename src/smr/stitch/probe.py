"""Per-chunk probe: is the BACKBONE any good before anything is stitched?

RPE at delta=1 is dominated by pairs inside a chunk, which stitching
never touches.  So each chunk is Sim(3)-aligned to GT on its own and
scored (ATE, AUC@30, median RRA), and compared with a REFERENCE pass of
the same size spread over the whole sequence.  If the backbone fails
inside chunks, nothing downstream is a measurement of any stitcher.
"""
from __future__ import annotations

import numpy as np

from ..eval.trajectory import ate_rmse, auc_at, pairwise_pose_errors


def chunk_probe(chunks, cache, runner, gt):
    rows = []
    for k, idx in enumerate(chunks):
        P = cache.get(idx, runner)["poses"]
        g = gt[idx]
        rra, rta = pairwise_pose_errors(P, g)
        rows.append(dict(chunk=k, n=len(idx), ate=ate_rmse(P, g),
                         auc30=auc_at(rra, rta),
                         auc30_rot=auc_at(rra, np.zeros_like(rta)),
                         rra_med=float(np.median(rra)),
                         rta_med=float(np.median(rta))))
    return rows


def reference_probe(n_frames, cache, runner, gt, size):
    """One pass of `size` frames spread evenly over the sequence."""
    step = max(1, n_frames // size)
    idx = list(range(0, n_frames, step))[:size]
    P = cache.get(idx, runner)["poses"]
    g = gt[idx]
    rra, rta = pairwise_pose_errors(P, g)
    return dict(idx=idx, n=len(idx), ate=ate_rmse(P, g), auc30=auc_at(rra, rta),
                rra_med=float(np.median(rra)), rta_med=float(np.median(rta)))


def gate(probe_rows, reference, ratio=0.6, floor=40.0, pct=10.0):
    """True when the chunks are a usable regime.

    The median per-chunk AUC must reach `ratio` of the reference pass's
    AUC (the backbone's own wide-baseline ability on THIS sequence) and
    the `pct`-th percentile chunk must clear `floor`.  Percentile, not
    worst: a 74-chunk walkthrough contains the odd near-pure-rotation
    chunk whose translation directions are undefined (AUC 29 with RTA
    median 25 deg on chess x6), and one such chunk, shared by every row,
    does not invalidate a comparison between rows.  Chunks under the
    floor are listed so they can be inspected.
    """
    aucs = np.array([r["auc30"] for r in probe_rows])
    med, worst = float(np.median(aucs)), float(aucs.min())
    low = float(np.percentile(aucs, pct))
    ok = med >= ratio * reference["auc30"] and low >= floor
    # Rotation-dominant sequences (TUM rpy, 360) have undefined translation
    # directions, so the joint AUC collapses while the poses are fine
    # (ATE 0.02 m on rpy).  Judge those on rotation-only AUC.
    if not ok and all("auc30_rot" in r for r in probe_rows):
        aucr = np.array([r["auc30_rot"] for r in probe_rows])
        ok = float(np.median(aucr)) >= 60.0 and float(np.percentile(aucr, pct)) >= floor
    return ok, dict(median_auc=med, worst_auc=worst, pct_auc=low, pct=pct,
                    below_floor=[int(r.get("chunk", i)) for i, r in enumerate(probe_rows)
                                 if r["auc30"] < floor],
                    reference_auc=reference["auc30"], ratio=ratio, floor=floor)


def print_probe(rows, reference, verdict):
    print(f"\n  reference pass ({reference['n']} frames spread over the "
          f"sequence): AUC@30 {reference['auc30']:.1f}  ATE {reference['ate']:.4f}"
          f"  RRA med {reference['rra_med']:.2f}")
    print(f"  per-chunk quality (backbone only, each aligned separately):")
    print(f"  {'chunk':>6}{'frames':>8}{'ATE':>10}{'AUC@30':>9}{'RRA med':>10}"
          f"{'RTA med':>10}")
    for r in rows:
        print(f"  {r['chunk']:>6}{r['n']:>8}{r['ate']:>10.4f}{r['auc30']:>9.1f}"
              f"{r['rra_med']:>10.2f}{r['rta_med']:>10.2f}")
    ok, d = verdict
    if ok:
        print(f"  gate PASSED: median chunk AUC {d['median_auc']:.1f} >= "
              f"{d['ratio']:.0%} of reference {d['reference_auc']:.1f}, "
              f"{d['pct']:.0f}th percentile {d['pct_auc']:.1f} >= {d['floor']:.0f}"
              + (f" (below floor: chunks {d['below_floor']} -- shared by every row)"
                 if d['below_floor'] else ""))
    else:
        print(f"\n  *** GATE FAILED: median chunk AUC {d['median_auc']:.1f} vs "
              f"reference {d['reference_auc']:.1f}, {d['pct']:.0f}th percentile "
              f"{d['pct_auc']:.1f}, worst {d['worst_auc']:.1f}.")
        print(f"  *** The BACKBONE is failing inside chunks. Stitching cannot "
              f"repair that and nothing below is a\n  *** measurement of any "
              f"stitcher. Change --keyframe-stride / --chunk until this passes.\n")
