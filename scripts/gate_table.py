#!/usr/bin/env python3
"""Aggregate scripts/gate_sweep.sh (or any variant tree) into one table with closure reliability (v177, plan Task 5).

    python scripts/gate_table.py --root outputs/ablate/gate --gt-dir data/gt --md outputs/gate_sweep.md
Layout expected: <root>/<dataset>/<variant>/<scene>.json.  Per variant and dataset: mean raw / +SMR / +PGO
(ATE for 7scenes, AUC@30 for co3d), accepted closures per sequence, and (with --gt-dir) pooled closure
precision / recall against the GT revisit definition of scripts/closure_reliability.py.
"""
import argparse, glob, json, pathlib, subprocess, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import closure_reliability as CR  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/ablate/gate"); ap.add_argument("--gt-dir", default=None)
ap.add_argument("--md", default=None); ap.add_argument("--row", default="smr"); ap.add_argument("--variants", nargs="*", default=None)
ap.add_argument("--gt-pattern", default=None, help="e.g. '7scenes=7scenes_{stem}_seq01.npz,co3d=co3d_full/co3d_{stem}.npz'")
a = ap.parse_args()
root = pathlib.Path(a.root)
lines = ["| dataset | variant | n | raw | +SMR | +SMR+PGO | closures/seq | precision covis | recall covis | false (covis) | useful | harmful | anchored passes |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for ds in sorted(p.name for p in root.iterdir() if p.is_dir()):
    metric = "ate_rmse" if "7scenes" in ds else "auc30"
    variants = a.variants or sorted(p.name for p in (root / ds).iterdir() if p.is_dir())
    for v in variants:
        files = sorted(glob.glob(str(root / ds / v / "*.json")))
        if not files:
            continue
        raw, smr, pgo, loops, recs = [], [], [], [], []
        for f in files:
            R = json.load(open(f)); rows = {r["method"]: r for r in R["rows"]}
            if "chained" not in rows or a.row not in rows:
                continue
            raw.append(rows["chained"][metric]); smr.append(rows[a.row][metric]); pgo.append(rows.get(a.row + "_pgo", rows[a.row])[metric])
            loops.append(rows[a.row].get("n_loops", 0))
            truth = None
            if a.gt_dir:
                gtp = CR.find_gt(R, f, argparse.Namespace(gt=None, gt_dir=a.gt_dir, gt_pattern=a.gt_pattern))
                if gtp is not None:
                    try:
                        estp = pathlib.Path(f).with_suffix(".npz")
                        gt, chunks, _ = CR.gt_and_chunks(R, gtp, estp if estp.exists() else None)
                        truth, owner, _ = CR.revisit_truth(gt, chunks)
                        est = None
                        if estp.exists():
                            z = np.load(estp, allow_pickle=True)
                            est = {k[4:]: np.asarray(z[k], float) for k in z.files if k.startswith("est_")}
                        recs += CR.analyse(R, truth, a.row, est, gt, owner); continue
                    except Exception as ex:  # noqa: BLE001
                        print(f"  {f}: GT unusable ({ex})", file=sys.stderr)
            recs += CR.analyse(R, truth, a.row)
        has_gt = bool(recs) and all("gt_revisit" in r for r in recs)
        S = CR.summarise(recs, has_gt) if recs else {}
        fm = (lambda x: f"{x:.3f}") if metric == "ate_rmse" else (lambda x: f"{x:.1f}")
        lines.append(f"| {ds} | {v} | {len(raw)} | {fm(np.mean(raw))} | {fm(np.mean(smr))} | {fm(np.mean(pgo))} | {np.mean(loops):.1f} | "
                     f"{CR.fmt(S.get('closure_precision_covis'))} | {CR.fmt(S.get('closure_recall_covis'))} | {CR.fmt(S.get('false_closures_covis'))} | "
                     f"{CR.fmt(S.get('useful'))}/{CR.fmt(S.get('closures_scored'))} | {CR.fmt(S.get('harmful'))} | {S.get('anchored_passes_attempted', '-')} |")
out = "\n".join(lines); print(out)
if a.md:
    pathlib.Path(a.md).write_text(out + "\n"); print("->", a.md)
