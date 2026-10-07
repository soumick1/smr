#!/usr/bin/env python3
"""Rank acceptance rules from the acceptance sweep.

    python scripts/acceptance_pick.py --root outputs/ablate/acceptance [--gt-dir data/gt]

Per arm: 7-Scenes mean ATE (smr, smr_pgo), CO3D mean AUC@30 (smr, smr_pgo), closures per sequence, and, when GT is
available, closure precision (co-visibility) and useful/harmful counts from closure_reliability. Ranking: sum of the
arm's rank in 7-Scenes ATE (lower better) and CO3D AUC (higher better), ties broken by fewer harmful closures. The raw
rows are printed once so every arm is read against the same baseline.
"""
import argparse, glob, json, pathlib, subprocess, sys
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/ablate/acceptance"); ap.add_argument("--gt-dir", default="data/gt")
ap.add_argument("--no-reliability", action="store_true")
a = ap.parse_args()
root = pathlib.Path(a.root)

def rows(ds, arm):
    out = {}
    for f in sorted((root / ds / arm).glob("*.json")):
        r = json.load(open(f))
        for row in r.get("rows", []):
            nm = next((row[k] for k in ("name", "row", "variant", "label", "method") if isinstance(row.get(k), str)), None)
            if nm:
                out.setdefault(nm, []).append(row)
    return out

arms = sorted({p.name for ds in ("7scenes", "co3d") for p in (root / ds).glob("*") if p.is_dir()} if (root / "7scenes").exists() or (root / "co3d").exists() else [])
table = []
for arm in arms:
    rec = dict(arm=arm)
    for ds, metric, key in (("7scenes", "ate_rmse", "ate"), ("co3d", "auc_all", "auc")):
        R = rows(ds, arm)
        if not R:
            continue
        for row in ("chained", "smr", "smr_pgo"):
            if row in R:
                rec[f"{key}_{row}"] = float(np.mean([x.get(metric, x.get("auc30", np.nan)) for x in R[row]])); rec[f"n_{ds}"] = len(R[row])
        if "smr" in R:
            rec[f"loops_{ds}"] = float(np.mean([next((x[k] for k in ("n_loops", "loops", "accepted", "closures") if isinstance(x.get(k), (int, float))), 0) for x in R["smr"]]))
    if not a.no_reliability:
        for ds, pat in (("7scenes", "7scenes=7scenes_{stem}_seq01.npz"), ("co3d", None)):
            if not (root / ds / arm).exists():
                continue
            cmd = [sys.executable, "scripts/closure_reliability.py", "--glob", str(root / ds / arm / "*.json"), "--gt-dir", a.gt_dir, "--est-dir", str(root / ds / arm)]
            if pat: cmd += ["--gt-pattern", pat]
            try:
                txt = subprocess.run(cmd, capture_output=True, text=True, timeout=900).stdout
                for line in txt.splitlines():
                    for k_out, k_in in (("prec", "closure_precision_covis"), ("useful", "useful"), ("harmful", "harmful")):
                        if line.strip().startswith(k_in + " "):
                            try: rec[f"{k_out}_{ds}"] = float(line.split()[-1])
                            except ValueError: pass
            except Exception:  # noqa: BLE001
                pass
    table.append(rec)
if not table:
    sys.exit("no arms found")
have7 = [t for t in table if "ate_smr" in t]; haveC = [t for t in table if "auc_smr" in t]
r7 = {t["arm"]: i for i, t in enumerate(sorted(have7, key=lambda t: t["ate_smr"]))}
rC = {t["arm"]: i for i, t in enumerate(sorted(haveC, key=lambda t: -t["auc_smr"]))}
for t in table:
    t["score"] = r7.get(t["arm"], len(have7)) + rC.get(t["arm"], len(haveC))
table.sort(key=lambda t: (t["score"], t.get("harmful_7scenes", 0) + t.get("harmful_co3d", 0)))
ref = table[0]
print(f"raw: 7-Scenes ATE {ref.get('ate_chained', float('nan')):.4f} (n={ref.get('n_7scenes', 0)}), CO3D AUC {ref.get('auc_chained', float('nan')):.2f} (n={ref.get('n_co3d', 0)})\n")
print(f"{'arm':<16} {'7S smr':>7} {'7S pgo':>7} {'loops':>5} {'prec':>5} {'use/harm':>9} | {'CO3D smr':>8} {'CO3D pgo':>8} {'loops':>5} {'prec':>5} {'use/harm':>9} | rank")
for t in table:
    uh7 = f"{int(t.get('useful_7scenes', -1))}/{int(t.get('harmful_7scenes', -1))}" if "useful_7scenes" in t else "-"
    uhc = f"{int(t.get('useful_co3d', -1))}/{int(t.get('harmful_co3d', -1))}" if "useful_co3d" in t else "-"
    print(f"{t['arm']:<16} {t.get('ate_smr', float('nan')):>7.4f} {t.get('ate_smr_pgo', float('nan')):>7.4f} {t.get('loops_7scenes', float('nan')):>5.1f} {t.get('prec_7scenes', float('nan')):>5.2f} {uh7:>9} | "
          f"{t.get('auc_smr', float('nan')):>8.2f} {t.get('auc_smr_pgo', float('nan')):>8.2f} {t.get('loops_co3d', float('nan')):>5.1f} {t.get('prec_co3d', float('nan')):>5.2f} {uhc:>9} | {t['score']}")
json.dump(table, open(root / "acceptance_pick.json", "w"), indent=1)
print("\n->", root / "acceptance_pick.json")
