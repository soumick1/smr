#!/usr/bin/env python3
"""Summarise dtu_eval.py logs written as '== <tag> scanN ==' blocks.

    python scripts/dtu_summary.py outputs/reports/dtu_baseline_v123.log [more logs]
Per tag: n, mean and median Acc/Comp/Overall, mean |ICP scale-1| and |t|,
the four worst scans, and the published/independent anchors.
"""
import re, sys, collections, pathlib
import numpy as np

res = collections.defaultdict(dict)
pat = re.compile(r"== (?P<tag>.+?) scan(?P<scan>\d+) ==\n(?:icp\(\d+\): scale (?P<sc>[\d.]+)  \|t\| (?P<t>[\d.]+) mm\n)?"
                 r"scan\d+: Acc (?P<acc>[\d.]+)  Comp (?P<comp>[\d.]+)  Overall (?P<ov>[\d.]+)")
for fn in sys.argv[1:]:
    for m in pat.finditer(pathlib.Path(fn).read_text()):
        g = m.groupdict()
        res[g["tag"]][int(g["scan"])] = (float(g["acc"]), float(g["comp"]), float(g["ov"]),
                                         float(g["sc"]) if g["sc"] else np.nan, float(g["t"]) if g["t"] else np.nan)
for tag, d in res.items():
    A = np.array([d[s] for s in sorted(d)])
    print(f"{tag:<28} n={len(d):2d}  mean {A[:,0].mean():.3f}/{A[:,1].mean():.3f}/{A[:,2].mean():.3f}"
          f"  median {np.median(A[:,0]):.3f}/{np.median(A[:,1]):.3f}/{np.median(A[:,2]):.3f}"
          f"  |icp scale-1| {np.nanmean(np.abs(A[:,3]-1))*100:.2f}%  |t| {np.nanmean(A[:,4]):.1f} mm"
          f"  worst {[(s, round(d[s][2], 2)) for s in sorted(d, key=lambda s: -d[s][2])[:4]]}")
print("anchors: published VGGT 0.389/0.374/0.382 | KIT'26 feed-forward VGGT-p 0.74/0.44/0.59, VGGT-d 1.53/0.40/0.97 (15 scenes) | MASt3R 0.403/0.344/0.374 | DUSt3R 2.677/0.805/1.741")
