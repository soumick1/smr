#!/usr/bin/env python3
"""LaTeX body and paired statistics for the sequence-NVS table (v205).

    python scripts/nvs_table.py --roots outputs/nvs_seq_v2 outputs/nvs_seq_v2_co3d --names 7-Scenes CO3Dv2 --out outputs/tab_nvs_seq.tex

Per dataset block and backbone: raw and +SMoRe rows with PSNR / SSIM / LPIPS as mean{\\scriptsize$\\pm$s.e.m.} over
held-out targets; the better value of each pair in bold when the paired mean difference exceeds 0.01 dB (PSNR) or 0.001
(SSIM / LPIPS). Also prints, per dataset and backbone, the paired PSNR gain with its s.e.m. and 95 % bootstrap interval,
the fraction of targets improved, the gain on revisited targets, and the GT-placement reference.
"""
import argparse, json, pathlib
import numpy as np

BBS = ["vggt_omega", "vggt", "pi3", "stream3r", "streamvggt", "fast3r"]
TEX = {"vggt_omega": r"VGGT-$\Omega$~\citep{vggtomega}", "vggt": r"VGGT~\citep{vggt}", "pi3": r"$\pi^3$~\citep{pi3}",
       "stream3r": r"STream3R~\citep{stream3r}", "streamvggt": r"StreamVGGT~\citep{streamvggt}", "fast3r": r"Fast3R~\citep{fast3r}"}
TIE = dict(psnr=0.01, ssim=0.001, lpips=0.001); HIGHER = dict(psnr=True, ssim=True, lpips=False)
FMT = dict(psnr="{:.2f}", ssim="{:.3f}", lpips="{:.3f}")


def load(root, bb, m):
    p = pathlib.Path(root) / bb / f"{m}.jsonl"
    return {(r["seq"], r["target_kf"]): r for r in map(json.loads, open(p))} if p.exists() else {}


def cell(v, sem, metric, bold):
    t = f"{FMT[metric].format(v)}{{\\scriptsize$\\pm${FMT[metric].format(sem)}}}"
    return f"\\textbf{{{t}}}" if bold else t


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0); rows_tex = []; stats = []
for bb in BBS:
    raw_cells, smr_cells = [], []
    for root, name in zip(a.roots, a.names):
        R, S, G = load(root, bb, "raw"), load(root, bb, "smr"), load(root, bb, "gt")
        keys = sorted(set(R) & set(S))
        if not keys:
            raw_cells += ["--"] * 3; smr_cells += ["--"] * 3; continue
        for metric in ("psnr", "ssim", "lpips"):
            r = np.array([R[k][metric] for k in keys]); s = np.array([S[k][metric] for k in keys]); d = s - r
            better_smr = d.mean() > TIE[metric] if HIGHER[metric] else d.mean() < -TIE[metric]
            better_raw = d.mean() < -TIE[metric] if HIGHER[metric] else d.mean() > TIE[metric]
            raw_cells.append(cell(r.mean(), r.std(ddof=1) / np.sqrt(len(r)), metric, better_raw))
            smr_cells.append(cell(s.mean(), s.std(ddof=1) / np.sqrt(len(s)), metric, better_smr))
        d = np.array([S[k]["psnr"] - R[k]["psnr"] for k in keys]); boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        dv = np.array([S[k]["psnr"] - R[k]["psnr"] for k in keys if S[k].get("revisit")])
        g = np.mean([G[k]["psnr"] for k in keys if k in G]) if G else float("nan")
        stats.append(f"{name:<9} {bb:<11} n={len(d):<4} dPSNR {d.mean():+.2f} +- {d.std(ddof=1) / np.sqrt(len(d)):.2f} s.e.m. [95% {np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]"
                     f"  improved {100 * (d > 0).mean():.0f}%  revisited {dv.mean() if len(dv) else float('nan'):+.2f} (n={len(dv)})  GT placement {g:.2f} dB")
    rows_tex.append(f"{TEX[bb]}\n& Baseline\n& " + " & ".join(raw_cells) + " \\\\\n& $+\\smr{}$\n& " + " & ".join(smr_cells) + " \\\\")
body = "\n\\addlinespace[1pt]\n".join(rows_tex)
print("\n".join(stats)); print("\n" + body)
if a.out:
    pathlib.Path(a.out).write_text(body + "\n"); print("->", a.out)
