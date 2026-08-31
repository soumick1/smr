#!/usr/bin/env python3
"""VGGT's own Table 1 (camera pose, AUC@30) with the memory-consensus row.

    python experiments/vggt_suite.py --gt-glob 'data/gt/co3d/*.npz' \
        --backbone vggt_omega --k-passes 8 --json outputs/reports/suite_pose_co3d_omega.json

Per sequence: the protocol's frames in their given order are ONE pass (the
backbone's own row, bit-identical to running it bare).  The +SMR row runs
k-1 additional passes over permutations of the SAME frames, binds every
pass, and reads each pose out by robust consensus across contexts
(smr.stitch.consensus_pose).  A permutation-equivariant backbone gains
nothing by construction; an order-sensitive one converts its variance into
accuracy.  Only the protocol frames are ever seen.
"""
import argparse, glob, json, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smr.stitch import BackboneRunner, PassCache                      # noqa: E402
from smr.stitch.consensus_pose import (auc_at, consensus_poses,       # noqa: E402
                                       pairwise_errors, pairwise_median_errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-glob", required=True)
    ap.add_argument("--backbone", default="vggt_omega")
    ap.add_argument("--k-passes", type=int, default=8)
    ap.add_argument("--rows", default="single,consensus")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--gt-convention", default="auto",
                    choices=["auto", "as-is", "inverted", "right-flip", "inv+flip"],
                    help="'auto': score the backbone's first pass against all four "
                         "hypotheses, pick the best, print it, apply to every sequence")
    ap.add_argument("--diag", action="store_true",
                    help="convention diagnosis on the FIRST sequence: score the "
                         "backbone's own pass against four GT hypotheses "
                         "(as-is / inverted / right-flip / inverted+flip) and print "
                         "which one the backbone agrees with")
    a = ap.parse_args()

    files = sorted(glob.glob(a.gt_glob))
    assert files, f"nothing matches {a.gt_glob}"

    F = np.diag([-1.0, -1.0, 1.0, 1.0])
    CONV = {"as-is": lambda G: G, "inverted": lambda G: np.linalg.inv(G),
            "right-flip": lambda G: G @ F, "inv+flip": lambda G: np.linalg.inv(G) @ F}
    conv = a.gt_convention
    if conv == "auto":
        npz0 = np.load(files[0], allow_pickle=True)
        paths0 = [str(s) for s in npz0["image_paths"]]
        gt0 = np.asarray(npz0["poses"], float)
        r0 = BackboneRunner(a.backbone, paths0, device=a.device)
        import hashlib
        h0 = hashlib.sha1("|".join(paths0).encode()).hexdigest()[:10]
        c0 = PassCache((ROOT / "outputs" / "cache" / "suite" /
                        (pathlib.Path(files[0]).stem + f"_{h0}_{a.backbone}.npy")))
        P0 = c0.get(list(range(len(paths0))), r0)["poses"]
        scores = {}
        for name, fn in CONV.items():
            r, t_ = pairwise_errors(P0, fn(gt0))
            scores[name] = float(np.median(np.maximum(r, t_)))
        conv = min(scores, key=scores.get)
        print("GT convention (auto): " +
              "  ".join(f"{k} {v:.1f}deg" for k, v in scores.items()) +
              f"  -> using '{conv}'")
        if scores[conv] > 30:
            print("WARNING: no hypothesis fits (median > 30 deg); the GT or the "
                  "frame association is broken -- do not trust this run")
    rows = a.rows.split(",")
    per_seq = {}
    cache_dir = ROOT / "outputs" / "cache" / "suite"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for fi, f in enumerate(files):
        npz = np.load(f, allow_pickle=True)
        paths = [str(p) for p in npz["image_paths"]]
        gt = CONV[conv](np.asarray(npz["poses"], float))
        n = len(paths)
        runner = BackboneRunner(a.backbone, paths, device=a.device)
        import hashlib
        content = hashlib.sha1("|".join(paths).encode()).hexdigest()[:10]
        cache = PassCache(cache_dir / (pathlib.Path(f).stem + f"_{content}_{a.backbone}.npy"))
        rng = np.random.default_rng(a.seed)
        single_ids = list(range(n))
        P0 = cache.get(single_ids, runner)["poses"]
        if a.diag:
            F = np.diag([-1.0, -1.0, 1.0, 1.0])
            hyps = {"as-is": gt,
                    "inverted": np.linalg.inv(gt),
                    "right-flip": gt @ F,
                    "inv+flip": np.linalg.inv(gt) @ F}
            print(f"convention diagnosis on {pathlib.Path(f).stem} ({a.backbone}):")
            for name, G in hyps.items():
                r, t_ = pairwise_errors(P0, G)
                print(f"  {name:<11} median rot {np.median(r):6.1f} deg   "
                      f"median trans-angle {np.median(t_):6.1f} deg   AUC@30 {auc_at(r, t_):.3f}")
            adj = [np.degrees(np.arccos(np.clip((np.trace(gt[i][:3,:3].T @ gt[i+1][:3,:3])-1)/2,-1,1)))
                   for i in range(n-1)]
            print(f"  adjacent-frame GT rotation (as-is): median {np.median(adj):.1f} deg "
                  f"(a smooth orbit sampled 10x should be well under 45)")
            return
        res = {}
        if "single" in rows:
            r, t = pairwise_errors(P0, gt)
            res["single"] = dict(auc30=auc_at(r, t, 30), auc15=auc_at(r, t, 15))
        if "consensus" in rows:
            passes = [(single_ids, P0)]
            for _ in range(a.k_passes - 1):
                ids = [int(x) for x in rng.permutation(n)]
                P = cache.get(ids, runner)["poses"]
                passes.append((ids, P))
            r, t = pairwise_median_errors(passes, gt)
            res["consensus"] = dict(auc30=auc_at(r, t, 30), auc15=auc_at(r, t, 15))
        per_seq[pathlib.Path(f).stem] = res
        if fi % 20 == 0:
            print(f"  [{fi+1}/{len(files)}] {pathlib.Path(f).stem}: " +
                  " ".join(f"{k} {v['auc30']:.3f}" for k, v in res.items()), flush=True)
    worst = sorted(per_seq, key=lambda k: per_seq[k].get("single", per_seq[k].get("consensus", {})).get("auc30", 1))[:10]
    print("\n  worst-10 sequences (eyeball for broken GT): " +
          ", ".join(f"{w} {per_seq[w].get('single', {}).get('auc30', float('nan')):.2f}" for w in worst))
    print(f"== {a.backbone} over {len(files)} sequences ({time.time()-t0:.0f}s) ==")
    summary = {}
    for row in rows:
        for m in ("auc30", "auc15"):
            v = float(np.mean([per_seq[s][row][m] for s in per_seq if row in per_seq[s]]))
            summary[f"{row}_{m}"] = v
        print(f"  {row:<10} AUC@30 {summary[f'{row}_auc30']*100:6.2f}   AUC@15 {summary[f'{row}_auc15']*100:6.2f}")
    out = pathlib.Path(a.json or ROOT / "outputs" / "reports" / f"suite_pose_{a.backbone}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(backbone=a.backbone, k_passes=a.k_passes,
                                   n_seq=len(files), summary=summary, per_seq=per_seq),
                              indent=1))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
