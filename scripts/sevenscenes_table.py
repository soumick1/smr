#!/usr/bin/env python3
"""Aggregate 7-Scenes runs two ways -- seq-01 only (the current Table 2 protocol) and the full test split -- and
per index kind (template = scaffold, flat = key-value baseline).

    python scripts/sevenscenes_table.py --root outputs/7scenes_test --bb vggt_omega
Prints per-scene ATE (m) for chained / +SMR / +SMR+PGO with the number of trajectories, then the scene means.
"""
import argparse, glob, json, pathlib, re
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/7scenes_test"); ap.add_argument("--bb", default="vggt_omega")
a = ap.parse_args()
rows = {}
for f in glob.glob(f"{a.root}/*_{a.bb}_*.json"):
    m = re.match(rf"(\w+?)_seq(\d+)_{a.bb}_(\w+)\.json", pathlib.Path(f).name)
    if not m: continue
    scene, seq, idx = m.group(1), int(m.group(2)), m.group(3)
    d = json.load(open(f)); r = {x["method"]: x for x in d.get("rows", [])}
    if "smr" in r:
        rows.setdefault(idx, {}).setdefault(scene, {})[seq] = r
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
for idx, per in rows.items():
    print(f"\n=== index = {idx} ({a.bb}) ===")
    for proto, pick in (("seq-01 only", lambda seqs: [s for s in seqs if s == 1]), ("full test split", lambda seqs: sorted(seqs))):
        print(f"--- {proto}: scene  n  raw / +SMR / +SMR+PGO  (loops)")
        means = []
        for sc in SCENES:
            seqs = pick(per.get(sc, {}).keys())
            if not seqs: print(f"  {sc:<11} --"); continue
            ate = np.array([[per[sc][s][k]["ate_rmse"] for k in ("chained", "smr", "smr_pgo")] for s in seqs]).mean(0)
            loops = np.mean([per[sc][s]["smr"]["n_loops"] for s in seqs]); means.append(ate)
            print(f"  {sc:<11} {len(seqs)}  {ate[0]:.3f} / {ate[1]:.3f} / {ate[2]:.3f}  ({loops:.1f})")
        if means:
            m = np.mean(means, 0); print(f"  {'MEAN':<11} {len(means)}  {m[0]:.3f} / {m[1]:.3f} / {m[2]:.3f}")
