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
                                       pairwise_errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-glob", required=True)
    ap.add_argument("--backbone", default="vggt_omega")
    ap.add_argument("--k-passes", type=int, default=8)
    ap.add_argument("--rows", default="single,consensus")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    files = sorted(glob.glob(a.gt_glob))
    assert files, f"nothing matches {a.gt_glob}"
    rows = a.rows.split(",")
    per_seq = {}
    cache_dir = ROOT / "outputs" / "cache" / "suite"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for fi, f in enumerate(files):
        npz = np.load(f, allow_pickle=True)
        paths = [str(p) for p in npz["image_paths"]]
        gt = np.asarray(npz["poses"], float)
        n = len(paths)
        runner = BackboneRunner(a.backbone, paths, device=a.device)
        cache = PassCache(cache_dir / (pathlib.Path(f).stem + f"_{a.backbone}.npy"))
        rng = np.random.default_rng(a.seed)
        single_ids = list(range(n))
        P0 = cache.get(single_ids, runner)["poses"]
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
            fused, ctx = consensus_poses(passes, n)
            r, t = pairwise_errors(fused, gt)
            res["consensus"] = dict(auc30=auc_at(r, t, 30), auc15=auc_at(r, t, 15),
                                    min_ctx=int(min(ctx.values())))
        per_seq[pathlib.Path(f).stem] = res
        if fi % 20 == 0:
            print(f"  [{fi+1}/{len(files)}] {pathlib.Path(f).stem}: " +
                  " ".join(f"{k} {v['auc30']:.3f}" for k, v in res.items()), flush=True)
    print(f"\n== {a.backbone} over {len(files)} sequences ({time.time()-t0:.0f}s) ==")
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
