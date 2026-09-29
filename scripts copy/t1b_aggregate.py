#!/usr/bin/env python3
"""Table 1b aggregator: named-key AUC/ATE rows from pilot_a reports.

    python scripts/t1b_aggregate.py 'outputs/reports/t1b_*_omega.json'
    python scripts/t1b_aggregate.py 'outputs/reports/pilotA_kitti_0[069]*.json'

Prints, per report set: mean auc_all / auc_within_pass / auc_cross_pass /
ATE for each method row, plus loop counts -- no column guessing."""
import glob, json, sys

import numpy as np

files = sorted(sum((glob.glob(g) for g in sys.argv[1:]), []))
assert files, "no reports match"
agg = {}
for f in files:
    for row in json.load(open(f))["rows"]:
        m = row["method"]
        agg.setdefault(m, []).append(row)
print(f"{len(files)} reports")
print(f"{'method':<14}{'n':>4}{'ATE':>9}{'auc_all':>9}{'within':>9}{'cross':>9}{'loops':>7}")
for m, rows in agg.items():
    def mv(k):
        vals = [r[k] for r in rows if k in r and r[k] == r[k]]
        return float(np.mean(vals)) if vals else float("nan")
    print(f"{m:<14}{len(rows):>4}{mv('ate_rmse'):>9.3f}{mv('auc_all'):>9.1f}"
          f"{mv('auc_within_pass'):>9.1f}{mv('auc_cross_pass'):>9.1f}{mv('n_loops'):>7.1f}")
