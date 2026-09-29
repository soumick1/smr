#!/usr/bin/env python3
"""Summarise one resource_sample.sh run (or all, as a table):  python scripts/resource_summary.py [label ...] [--table]"""
import glob, pathlib, re, sys
import numpy as np
out = pathlib.Path("outputs/resource")
labels = [x for x in sys.argv[1:] if not x.startswith("--")] or sorted({p.name.rsplit(".", 2)[0] for p in out.glob("*.time.txt")})
rows = []
for lab in labels:
    t = (out / f"{lab}.time.txt").read_text() if (out / f"{lab}.time.txt").exists() else ""
    g = lambda pat: float(re.search(pat, t).group(1)) if re.search(pat, t) else float("nan")
    user, sys_ = g(r"User time \(seconds\): ([\d.]+)"), g(r"System time \(seconds\): ([\d.]+)")
    rss = g(r"Maximum resident set size \(kbytes\): (\d+)") / 1e6
    cpu_pct = g(r"Percent of CPU this job got: (\d+)%")
    m = re.search(r"Elapsed \(wall clock\) time.*?: (?:(\d+):)?(\d+):([\d.]+)", t)
    wall = (int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else float("nan")
    util, mem = [], []
    for line in (out / f"{lab}.util.csv").read_text().splitlines() if (out / f"{lab}.util.csv").exists() else []:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            try: util.append(float(parts[1])); mem.append(float(parts[2]))
            except ValueError: pass
    util, mem = np.array(util), np.array(mem)
    active = util[util > 0]
    rows.append(dict(label=lab, cpu_min=(user + sys_) / 60, cpu_pct=cpu_pct, rss_gb=rss, wall_min=wall / 60,
                     util_mean=util.mean() if len(util) else float("nan"), util_active=active.mean() if len(active) else float("nan"),
                     util_max=util.max() if len(util) else float("nan"), gpu_mem_gb=mem.max() / 1024 if len(mem) else float("nan"), n_samples=len(util)))
print(f"{'run':<20} {'CPU (min)':>9} {'CPU %':>6} {'RAM peak (GB)':>13} {'GPU util mean %':>15} {'active %':>8} {'max %':>6} {'GPU mem (GB)':>12} {'wall (min)':>10} {'samples':>7}")
for r in rows:
    print(f"{r['label']:<20} {r['cpu_min']:>9.2f} {r['cpu_pct']:>6.0f} {r['rss_gb']:>13.2f} {r['util_mean']:>15.1f} {r['util_active']:>8.1f} {r['util_max']:>6.0f} {r['gpu_mem_gb']:>12.1f} {r['wall_min']:>10.2f} {r['n_samples']:>7}")
print("CPU % = CPU time / wall time (100 % = one core busy); GPU util = nvidia-smi 200 ms samples (mean over the run, mean over active samples, max); GPU mem = device memory used, max")
