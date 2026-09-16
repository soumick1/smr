#!/usr/bin/env python3
"""Export the accounting the compute appendix needs, from existing pilot_a reports (no re-runs):
per report and per backbone: passes (count, frames per pass), anchored passes (passes wider than the window = a
proposal was examined, accepted or not), proposals examined / accepted / rejected, measured backbone seconds (sum of the
inference time recorded when each pass was first computed; identical on warm-cache re-runs), stitch seconds
(retrieval + verification + relaxation, without the backbone), PGO seconds, peak GPU GB, and the composed GFLOPs
Σ_k GF(|pass_k|) using the probe's per-size unit costs when given.

    python scripts/compute_export.py 'outputs/reports/t1*_vggt_s2.json' [--probe outputs/reports/compute_probe.json --chunk 32]
"""
import argparse, glob, json, pathlib
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("reports"); ap.add_argument("--probe", default=""); ap.add_argument("--chunk", type=int, default=32)
a = ap.parse_args()
unit = {}
if a.probe and pathlib.Path(a.probe).exists():
    for r in json.load(open(a.probe))["rows"]:
        if r.get("status") == "ok" and r.get("gflops") and r.get("frames"): unit[int(r["frames"])] = float(r["gflops"])
def gf(frames):
    if not unit: return None
    ks = sorted(unit); k = min(ks, key=lambda x: abs(x - frames)); return unit[k] * frames / k     # nearest probed size, scaled linearly
rows = []
for f in sorted(glob.glob(a.reports)):
    d = json.load(open(f)); R = {r["method"]: r for r in d.get("rows", [])}
    if "smr" not in R: continue
    ev = (d.get("events") or {}).get("smr") or []
    examined = sum(1 for e in ev if isinstance(e, dict) and e.get("sites"))          # windows where >=1 site was appended to the pass
    proposals = sum(len(e.get("candidates") or []) for e in ev if isinstance(e, dict))
    s = R["smr"]; c = R.get("chained", {}); p = R.get("smr_pgo", {})
    n_pass, fpp = s.get("n_passes"), s.get("frames_per_pass")
    n_anch = round((fpp - a.chunk) / 4 * n_pass) if fpp and n_pass else None       # 2 frames per site, mean over passes -> anchored-pass equivalents
    rows.append(dict(file=pathlib.Path(f).stem, backbone=d.get("backbone"), K=d.get("keyframes"), n_pass=n_pass, frames_per_pass=fpp,
                     windows_with_sites=examined, candidates=proposals, accepted=s.get("n_loops"), rejected=s.get("n_rejected"),
                     backbone_s=s.get("backbone_secs"), raw_backbone_s=c.get("backbone_secs"), stitch_s=s.get("stitch_secs"), pgo_s=p.get("pgo_secs"),
                     peak_gb=s.get("peak_mem_gb"), raw_peak_gb=c.get("peak_mem_gb"),
                     gflops=(n_pass * gf(fpp) if n_pass and fpp and gf(fpp) else None)))
if not rows: raise SystemExit("no reports with an smr row")
print(f"{'report':<42} {'K':>4} {'passes':>6} {'f/pass':>6} {'w/sites':>7} {'cands':>5} {'acc':>4} {'rej':>4} {'bb_s':>8} {'raw_bb_s':>8} {'stitch_s':>8} {'pgo_s':>6} {'peak':>5}")
for r in rows:
    f = lambda v, w=8, p=1: f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) and v is not None else f"{'--':>{w}}"
    print(f"{r['file'][:42]:<42} {str(r['K']):>4} {str(r['n_pass']):>6} {f(r['frames_per_pass'],6)} {str(r['windows_with_sites']):>7} {str(r['candidates']):>5} {str(r['accepted']):>4} {str(r['rejected']):>4} {f(r['backbone_s'])} {f(r['raw_backbone_s'])} {f(r['stitch_s'])} {f(r['pgo_s'],6)} {f(r['peak_gb'],5,1)}")
by_bb = {}
for r in rows: by_bb.setdefault(r["backbone"], []).append(r)
print("\nper backbone (means): windows-with-sites / accepted / rejected per sequence; backbone s; stitch s; pgo s; peak GB; composed GFLOPs")
for bb, rs in by_bb.items():
    m = lambda k: np.mean([r[k] for r in rs if r.get(k) is not None]) if any(r.get(k) is not None for r in rs) else float("nan")
    print(f"  {bb:<11} n={len(rs):<3} sites {m('windows_with_sites'):.1f} acc {m('accepted'):.1f} rej {m('rejected'):.1f} | bb {m('backbone_s'):.1f}s (raw {m('raw_backbone_s'):.1f}) stitch {m('stitch_s'):.1f}s pgo {m('pgo_s'):.1f}s | peak {m('peak_gb'):.1f} GB | GFLOPs {m('gflops'):,.0f}")
print("\nbank storage per keyframe (stitching index): descriptor 384 x 4 B + address N_h x 8 B + state 12 x 8 B + pose 16 x 8 B; RLS matrices fixed 2 x N_h x 384 x 8 B + (N_h^2 + 384^2) x 8 B")
for N_h in (1024, 2048):
    per = 384 * 4 + N_h * 8 + 12 * 8 + 16 * 8; fixed = 2 * N_h * 384 * 8 + (N_h ** 2 + 384 ** 2) * 8
    print(f"  N_h={N_h}: {per/1024:.1f} KB per keyframe, {fixed/1e6:.1f} MB fixed; 200 keyframes -> {(fixed + 200*per)/1e6:.1f} MB (poses of every pass are also cached: 16 x 8 B per frame per pass)")
