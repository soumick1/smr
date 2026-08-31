#!/usr/bin/env python3
"""Zero-GPU verdict: cross-model agreement + GT smoothness per sequence.

    python scripts/co3d_gt_verdict.py outputs/reports/suite_pose_co3d_*_v95.json
Prints, per sequence: AUC@30 for each backbone, and whether all models agree
it's near-zero (broken GT signature).  Then the filtered averages."""
import json, sys

import numpy as np

reports = {p: json.load(open(p)) for p in sys.argv[1:]}
seqs = sorted(next(iter(reports.values()))["per_seq"])
print(f"{'sequence':<34}" + "".join(f"{p.split('_co3d_')[1].split('_v9')[0]:>12}" for p in reports))
broken = []
for s in seqs:
    vals = [reports[p]["per_seq"][s]["single"]["auc30"] for p in reports]
    flag = all(v < 0.20 for v in vals)
    if flag:
        broken.append(s)
    print(f"{s:<34}" + "".join(f"{v:>12.3f}" for v in vals) + ("   << broken-GT" if flag else ""))
print(f"\nbroken-GT by cross-model consensus: {len(broken)}/{len(seqs)}")
for p, r in reports.items():
    keep = [s for s in seqs if s not in broken]
    a30 = np.mean([r["per_seq"][s]["single"]["auc30"] for s in keep])
    a15 = np.mean([r["per_seq"][s]["single"]["auc15"] for s in keep])
    print(f"  {p.split('_co3d_')[1].split('_v9')[0]:<12} filtered AUC@30 {a30*100:6.2f}  AUC@15 {a15*100:6.2f}  ({len(keep)} seqs)")
