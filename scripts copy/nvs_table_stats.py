#!/usr/bin/env python3
"""Table 3 (NVS) with between-object standard deviations, generated from the per-object jsonl records.

    python scripts/nvs_table_stats.py --root outputs/nvs --out outputs/tab_nvs_sd.tex
    python scripts/nvs_table_stats.py --root outputs/nvs --backbones vggt_omega vggt pi3 stream3r fast3r streamvggt --sets val gso

For each backbone and set, joins <root>/<backbone>/<set>_reads1.jsonl (raw) and <set>_reads4.jsonl (+SMR) by object id
and reports mean +- SD (over objects, ddof = 1) of PSNR, SSIM and LPIPS for both, the paired difference mean +- SD with a
95 % bootstrap interval, and writes LaTeX rows "mean{\\scriptsize$\\pm$sd}" in the layout of the manuscript's Table 3.
Bold marks the better member of each raw/+SMR pair when the mean difference exceeds the tie threshold (0.01 dB PSNR,
0.001 SSIM / LPIPS); ties are left unbolded. The means printed here are the values the table must carry.
"""
import argparse, json, pathlib
import numpy as np

LABEL = {"vggt_omega": r"VGGT-$\Omega$~\citep{vggtomega}", "vggt": r"VGGT~\citep{vggt}", "pi3": r"$\pi^3$~\citep{pi3}",
         "stream3r": r"STream3R~\citep{stream3r}", "fast3r": r"Fast3R~\citep{fast3r}", "streamvggt": r"StreamVGGT~\citep{streamvggt}"}
TIE = dict(psnr=0.01, ssim=0.001, lpips=0.001)
HIGHER = dict(psnr=True, ssim=True, lpips=False)
FMT = dict(psnr=("{:.2f}", "{:.2f}"), ssim=("{:.3f}", "{:.3f}"), lpips=("{:.3f}", "{:.3f}"))


def load(p):
    out = {}
    if not pathlib.Path(p).exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            if all(k in r for k in ("psnr", "ssim", "lpips")):
                out[r["id"]] = r
    return out


def stats(a, b, n_boot=5000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].mean(1)
    return dict(mean_a=a.mean(), sd_a=a.std(ddof=1), mean_b=b.mean(), sd_b=b.std(ddof=1), d_mean=d.mean(), d_sd=d.std(ddof=1),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)), n=len(d))


def cell(mean, sd, metric, bold):
    m = FMT[metric][0].format(mean); s = FMT[metric][1].format(sd)
    txt = f"{m}{{\\scriptsize$\\pm${s}}}"
    return f"\\textbf{{{txt}}}" if bold else txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/nvs"); ap.add_argument("--out", default=None)
    ap.add_argument("--backbones", nargs="+", default=["vggt_omega", "vggt", "pi3", "stream3r", "fast3r", "streamvggt"])
    ap.add_argument("--sets", nargs="+", default=["val", "gso"], help="val = held-out Objaverse-LVIS, gso = GSO")
    ap.add_argument("--raw-reads", type=int, default=1); ap.add_argument("--smr-reads", type=int, default=4)
    a = ap.parse_args()
    rows_tex, summary = [], []
    for bb in a.backbones:
        cells_raw, cells_smr, ok = [], [], True
        for st in a.sets:
            A = load(pathlib.Path(a.root) / bb / f"{st}_reads{a.raw_reads}.jsonl")
            B = load(pathlib.Path(a.root) / bb / f"{st}_reads{a.smr_reads}.jsonl")
            ids = sorted(set(A) & set(B))
            if not ids:
                print(f"{bb} {st}: no paired objects (files missing?)"); ok = False
                cells_raw += ["--"] * 3; cells_smr += ["--"] * 3; continue
            for metric in ("psnr", "ssim", "lpips"):
                s = stats([A[i][metric] for i in ids], [B[i][metric] for i in ids])
                better_smr = (s["d_mean"] > TIE[metric]) if HIGHER[metric] else (s["d_mean"] < -TIE[metric])
                better_raw = (s["d_mean"] < -TIE[metric]) if HIGHER[metric] else (s["d_mean"] > TIE[metric])
                cells_raw.append(cell(s["mean_a"], s["sd_a"], metric, better_raw))
                cells_smr.append(cell(s["mean_b"], s["sd_b"], metric, better_smr))
                summary.append(f"{bb:<11} {st:<4} {metric:<5} n={s['n']:<5} raw {s['mean_a']:.3f}+-{s['sd_a']:.3f}  +SMR {s['mean_b']:.3f}+-{s['sd_b']:.3f}  "
                               f"delta {s['d_mean']:+.4f}+-{s['d_sd']:.4f} [95% {s['lo']:+.4f}, {s['hi']:+.4f}]")
        rows_tex.append(f"{LABEL.get(bb, bb)} (raw)\n& " + " & ".join(cells_raw) + " \\\\\n\\;+\\smr{}\n& " + " & ".join(cells_smr) + " \\\\")
    body = "\n\\midrule\n".join(rows_tex)
    print("\n".join(summary)); print("\n" + body)
    if a.out:
        pathlib.Path(a.out).write_text(body + "\n"); print("->", a.out)


if __name__ == "__main__":
    main()