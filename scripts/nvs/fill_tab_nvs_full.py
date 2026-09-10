#!/usr/bin/env python3
"""Write paper/tab_nvs_full.tex: all eight backbones, held-out Objaverse-LVIS and GSO column groups, raw and +SMR rows,
from outputs/nvs/<bb>/{gso,val}_reads{1,4}.jsonl.  Cells without results print '--'.  Bold = better of the pair
(ties within 0.01 dB / 0.001 unbolded).  Run whenever a head finishes.
    python scripts/nvs/fill_tab_nvs_full.py [--nvs outputs/nvs] [--out paper/tab_nvs_full.tex]
"""
import argparse, json, pathlib
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--nvs", default="outputs/nvs"); ap.add_argument("--out", default="paper/tab_nvs_full.tex")
a = ap.parse_args()
BBS = [("vggt_omega", r"VGGT-$\Omega$", 30000), ("vggt", "VGGT", 30000), ("pi3", r"$\pi^3$", 30000), ("stream3r", "STream3R", 30000),
       ("fast3r", "Fast3R", 30000), ("streamvggt", "StreamVGGT", 30000), ("dust3r", "DUSt3R", 10000), ("mast3r", "MASt3R", 10000)]
def load(bb, split, r):
    p = pathlib.Path(a.nvs) / bb / f"{split}_reads{r}.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in open(p) if l.strip()]; rows = [x for x in rows if "psnr" in x]
    if not rows: return None
    return {k: float(np.mean([x[k] for x in rows])) for k in ("psnr", "ssim", "lpips")} | {"n": len(rows)}
def cells(m1, m4):
    if m1 is None and m4 is None: return ["--"] * 3, ["--"] * 3
    out1, out4 = [], []
    for k, fmt, hi, tol in (("psnr", "{:.2f}", True, 0.015), ("ssim", "{:.3f}", True, 0.0015), ("lpips", "{:.3f}", False, 0.0015)):
        v1 = fmt.format(m1[k]) if m1 else "--"; v4 = fmt.format(m4[k]) if m4 else "--"
        if m1 and m4 and abs(m1[k] - m4[k]) > tol:
            b4 = (m4[k] > m1[k]) if hi else (m4[k] < m1[k])
            v1, v4 = (v1, f"\\textbf{{{v4}}}") if b4 else (f"\\textbf{{{v1}}}", v4)
        out1.append(v1); out4.append(v4)
    return out1, out4
lines = []; summary = []
for bb, name, steps in BBS:
    g1, g4, v1, v4 = load(bb, "gso", 1), load(bb, "gso", 4), load(bb, "val", 1), load(bb, "val", 4)
    (vg1, vg4), (gg1, gg4) = cells(v1, v4), cells(g1, g4)
    steps_txt = f"{steps // 1000}K"
    lines.append(f"{name} (raw) & {steps_txt} & " + " & ".join(vg1) + " & " + " & ".join(gg1) + r" \\")
    lines.append(r"\;+\smr{} & & " + " & ".join(vg4) + " & " + " & ".join(gg4) + r" \\")
    summary.append(f"{bb:<11} val n={v1['n'] if v1 else 0}/{v4['n'] if v4 else 0} gso n={g1['n'] if g1 else 0}/{g4['n'] if g4 else 0}  " +
                   (f"gso {g1['psnr']:.2f}->{g4['psnr']:.2f}" if g1 and g4 else "gso pending") + (f"  val {v1['psnr']:.2f}->{v4['psnr']:.2f}" if v1 and v4 else "  val pending"))
tex = r'''\begin{table}[t]
\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{Novel view synthesis for all eight backbones} (same protocol as Table~\ref{tab:nvs}: four inputs, ten targets,
$256^2$, PSNR$\uparrow$/SSIM$\uparrow$/LPIPS$\downarrow$ on the full image). One light Gaussian head per backbone, trained on the same
${\approx}19$K Objaverse-LVIS objects; \emph{steps} is the head's training budget (DUSt3R and MASt3R run a global alignment
inside every pass, so their heads were trained for 10K steps, at which the first four heads were within $0.3$\,dB of their
final validation PSNR). raw = single pass; +\smr{} = the four-ordering read through the same head. Bold: better of each pair
(differences within $0.01$\,dB / $0.001$ are ties).}
\label{tab:nvs-full}
\begin{tabular}{@{}l c ccc ccc@{}}
\toprule
& & \multicolumn{3}{c}{Objaverse-LVIS (held out, 174)} & \multicolumn{3}{c}{GSO (1{,}033)} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Backbone & steps & PSNR & SSIM & LPIPS & PSNR & SSIM & LPIPS \\
\midrule
''' + "\n".join(lines) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text(tex)
print("\n".join(summary)); print(f"wrote {a.out}")
