#!/usr/bin/env python3
"""Summarise dtu_eval.py logs written as '== <tag> scanN ==' blocks.

    python scripts/dtu_summary.py outputs/reports/dtu_baseline_v123.log [more logs]
Per tag: n, mean and median Acc/Comp/Overall, mean |ICP scale-1| and |t|,
the four worst scans, and the published/independent anchors.
"""
import re, sys, collections, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np

res = collections.defaultdict(dict)
hdr = re.compile(r"== (?P<tag>.+?) scan(?P<scan>\d+) ==")
for fn in sys.argv[1:]:
    txt = pathlib.Path(fn).read_text()
    heads = list(hdr.finditer(txt))
    for i, h in enumerate(heads):
        block = txt[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(txt)]
        m = re.search(r"scan\d+: Acc ([\d.]+)  Comp ([\d.]+)  Overall ([\d.]+)", block)
        if not m:
            continue
        ok = re.findall(r"icp\([^)]*\): scale ([\d.]+)  \|t\| ([\d.]+) mm", block)   # successful refinement(s)
        rejected = ("REJECTED" in block) and not ok
        sc, t = (float(ok[-1][0]), float(ok[-1][1])) if ok else (np.nan, np.nan)
        res[h.group("tag")][int(h.group("scan"))] = (float(m.group(1)), float(m.group(2)), float(m.group(3)), sc, t, 1.0 if rejected else 0.0)
for tag, d in res.items():
    A = np.array([d[s] for s in sorted(d)])
    print(f"{tag:<28} n={len(d):2d}  mean {A[:,0].mean():.3f}/{A[:,1].mean():.3f}/{A[:,2].mean():.3f}"
          f"  median {np.median(A[:,0]):.3f}/{np.median(A[:,1]):.3f}/{np.median(A[:,2]):.3f}"
          f"  |icp scale-1| {np.nanmean(np.abs(A[:,3]-1))*100:.2f}%  |t| {np.nanmean(A[:,4]):.1f} mm  rejected {int(A[:,5].sum())}"
          f"  worst {[(s, round(d[s][2], 2)) for s in sorted(d, key=lambda s: -d[s][2])[:4]]}")
print("anchors: published VGGT 0.389/0.374/0.382 | KIT'26 feed-forward VGGT-p 0.74/0.44/0.59, VGGT-d 1.53/0.40/0.97 (15 scenes) | MASt3R 0.403/0.344/0.374 | DUSt3R 2.677/0.805/1.741")
