#!/usr/bin/env python3
"""Compute-resources tables: MEASURED costs for raw / +SMR / +SMR+PGO
harvested from pilot_a reports, plus native single-pass costs and FLOPs
from flops_probe.py. Same hardware throughout.

    python scripts/compute_table.py outputs/reports/compute_probe.json \
        --reports-dir outputs/reports --ref-n 100 > outputs/reports/compute_tables.tex

Per backbone at --ref-n:
  (native)      probe row at N: CPU s | GPU GB | GFLOPs | RAM GB | wall s
  (raw)         pilot_a 'chained':  Time = backbone_secs + stitch_secs,
                GPU = peak_mem_gb  (measured, mean over sequences)
  +SMR          pilot_a 'smr' row   (its passes INCLUDE anchored passes)
  +SMR+PGO      pilot_a 'smr_pgo':  Time adds pgo_secs
  GFLOPs for windowed rows = n_windows x GF(32) + n_loops x GF(36),
  marked composed (only FLOPs is composed; time/memory are measured).
"""
import argparse, glob, json, math
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("probe"); ap.add_argument("--reports-dir", default="outputs/reports")
ap.add_argument("--ref-n", type=int, default=100)
a = ap.parse_args()
P = json.load(open(a.probe))
probe = {(r["backbone"], r.get("N")): r for r in P["rows"]}
NS = {10: "20", 50: "4", 100: "2", 200: "1"}
S = NS[a.ref_n]
BBS = ["dust3r", "fast3r", "mast3r", "vggt", "vggt_omega", "pi3", "streamvggt", "stream3r"]

def harvest(bb, S=None):
    S = S or globals()["S"]
    rows = {"chained": [], "smr": [], "smr_pgo": []}
    for f in glob.glob(f"{a.reports_dir}/t1*_{bb}_s{S}.json") + \
             glob.glob(f"{a.reports_dir}/t1b_*_{bb}.json" if S == "1" else "/nonexistent"):
        try: rep = json.load(open(f))
        except Exception: continue
        for r in rep.get("rows", []):
            if r.get("method") in rows: rows[r["method"]].append(r)
    def agg(rs, pgo=False):
        if not rs: return None
        t = [r.get("backbone_secs", 0) + r.get("stitch_secs", 0) +
             (r.get("pgo_secs", 0) if pgo else 0) for r in rs]
        g = [r.get("peak_mem_gb") for r in rs if r.get("peak_mem_gb")]
        L = [r.get("n_loops", 0) for r in rs]
        return dict(wall_s=float(np.mean(t)), gpu=float(np.mean(g)) if g else None,
                    loops=float(np.mean(L)), n=len(rs))
    return agg(rows["chained"]), agg(rows["smr"]), agg(rows["smr_pgo"], pgo=True)

def gf(bb, N):
    r = probe.get((bb, N))
    return r.get("gflops") if r and r.get("status") == "ok" else None

def c(v, f="{:.1f}"):
    return "--" if v is None else (v if isinstance(v, str) else f.format(v))

nwin = max(1, math.ceil((a.ref_n - 16) / 16))
print(f"% compute table @ N={a.ref_n}; windowed rows measured from pilot_a s{S} reports")
print("\\begin{tabular}{@{}l rrrrrr@{}}\n\\toprule")
print("Model & CPU (s) & GPU (GB) & GFLOPs & RAM (GB) & Time (s) & s/frame \\\\\n\\midrule")
for bb in BBS:
    nat = probe.get((bb, a.ref_n))
    if nat and nat["status"] == "OOM":
        print(f"{bb} (native) & \\multicolumn{{5}}{{c}}{{OOM ({nat.get('gpu_req','').strip()})}} \\\\")
    else:
        print(f"{bb} (native) & {c(nat and nat.get('cpu_s'))} & "
              f"{c(nat and nat.get('gpu_peak_gb'),'{:.2f}')} & {c(gf(bb,a.ref_n),'{:,.0f}')} & "
              f"{c(nat and nat.get('ram_peak_gb'),'{:.2f}')} & {c(nat and nat.get('wall_s'))} & {c(nat and nat.get('wall_s') and nat['wall_s']/a.ref_n,'{:.2f}')} \\\\")
    raw, smr, pgo = harvest(bb)
    g32, g36 = gf(bb, 32), gf(bb, 36)
    for lab, r, add_cls in (("(raw, windowed)", raw, 0),
                            ("+SMR", smr, 1), ("+SMR+PGO", pgo, 1)):
        if r is None:
            print(f"\\;{lab} & -- & -- & -- & -- & -- \\\\"); continue
        gfl = None
        if g32: gfl = nwin * g32 + (add_cls * r["loops"] * (g36 or g32))
        print(f"\\;{lab} & -- & {c(r['gpu'],'{:.2f}')} & "
              f"{(c(gfl,'{:,.0f}') + '$^c$') if gfl else '--'} & -- & {c(r['wall_s'])} & {c(r['wall_s']/a.ref_n,'{:.2f}')} \\\\")
    print("\\midrule")
print("\\bottomrule\n\\end{tabular}")
print("% $^c$ composed: n_win x GF(32) + loops x GF(36); time/GPU measured.")

# ---- Table C: GPU peak (GB) vs N -- native (probe) vs +SMR (MEASURED from
# pilot_a peak_mem_gb at each stride). The claim in one table: native grows
# to the wall; the scaffold-memory system is flat at the window cost.
print("\n% ---- Table C: GPU peak (GB) vs N (native measured per N; +SMR measured per N)")
ns_all = [10, 50, 100, 200]
print("\\begin{tabular}{@{}l " + "rr " * len(ns_all) + "@{}}\n\\toprule")
print("Model " + "".join(f"& \\multicolumn{{2}}{{c}}{{$N{{=}}{n}$}} " for n in ns_all) + "\\\\")
print("      " + "& nat. & +\\textsc{SMR} " * len(ns_all) + "\\\\\n\\midrule")
for bb in BBS:
    cells = ""
    for n in ns_all:
        nat = probe.get((bb, n))
        nc = ("OOM" if nat and nat["status"] == "OOM"
              else c(nat and nat.get("gpu_peak_gb"), "{:.1f}"))
        _, smr_r, _ = harvest(bb, NS[n])
        sc = c(smr_r and smr_r["gpu"], "{:.1f}")
        cells += f"& {nc} & {sc} "
    print(f"{bb} {cells}\\\\")
print("\\bottomrule\n\\end{tabular}")
print("% native = one N-frame inference (probe); +SMR = measured peak over its")
print("% actual 32/36-frame passes at that N (pilot_a peak_mem_gb).")
