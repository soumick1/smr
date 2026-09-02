#!/usr/bin/env python3
"""Aggregate eth3d_pointmap_eval.py json-lines into the Table-3 style mean.

    python scripts/eth3d_pm_summary.py outputs/reports/eth_pm_official.jsonl
Prints per-scene rows and the 13-scene mean for each alignment key
(pre-align / umeyama / umeyama-trim), next to the published VGGT rows.
"""
import json, sys, pathlib, collections
import numpy as np

rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
by_scene = collections.OrderedDict()
for r in rows:                                   # last entry per (scene, source) wins
    scene = pathlib.Path(r["maps"]).parent.name
    by_scene[(scene, r["source"], r["gt"])] = r
keys = [k for k in rows[-1] if isinstance(rows[-1][k], dict) and "acc" in rows[-1][k]]
print(f"{'scene':<26}" + "".join(f"{k:>30}" for k in keys))
agg = {k: [] for k in keys}
for (scene, src, gt), r in by_scene.items():
    line = f"{scene+'/'+src+'/'+gt:<26}"
    for k in keys:
        v = r[k]; agg[k].append([v["acc"], v["comp"], v["overall"]])
        line += f"{v['acc']:>9.3f}{v['comp']:>10.3f}{v['overall']:>11.3f}"
    print(line)
print(f"{'MEAN (n=%d)' % len(by_scene):<26}" + "".join(
    f"{np.mean(a, 0)[0]:>9.3f}{np.mean(a, 0)[1]:>10.3f}{np.mean(a, 0)[2]:>11.3f}" for a in (agg[k] for k in keys)))
print(f"{'published VGGT Depth+Cam':<26}{0.873:>9.3f}{0.482:>10.3f}{0.677:>11.3f}   (Point head 0.901/0.518/0.709; MASt3R 0.968/0.684/0.826)")
