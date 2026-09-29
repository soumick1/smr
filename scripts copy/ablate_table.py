#!/usr/bin/env python3
"""Aggregate scripts/ablate_all.sh reports into one LaTeX table (and a console summary).

    python scripts/ablate_table.py --root outputs/ablate --out paper/tab_ablation.tex
Rows = variants; columns = 7-Scenes ATE (m, mean over scenes) for chained / +SMR / +SMR+PGO and accepted closures per
sequence, then CO3D AUC@30 (mean over orbits) for the same three rows.  Sequences missing for a variant are dropped
from that variant's mean and the count is shown, so partial runs still read correctly.
"""
import argparse, glob, json, pathlib
import numpy as np

LABEL = {"default": "default (2 sites, relax, robust batch)", "sites1": "1 anchor site per window", "sites3": "3 anchor sites",
         "corr_jump": "closure applied as a jump", "corr_dist": "closure distributed over the window", "corr_none": "closure not applied (proposal only)",
         "no_robust": "batch solve without outlier-edge rejection", "remeasure": "closure sites re-measured (4-frame pass)",
         "desc_rgb": "scaffold keyed and cued by the engineered RGB cue",
         "desc_dino": "scaffold keyed and cued by DINOv2 (default)", "desc_feat": "scaffold keyed and cued by pooled VGGT features",
         "index_flat": "no scaffold: flat key--value memory (cosine over cues)", "index_dynamics": "scaffold addresses from settled attractor bumps", "Nh512": "scaffold $N_h{=}512$", "Nh8192": "scaffold $N_h{=}8192$",
         "torus16": "torus $16^2$ per module", "torus64": "torus $64^2$ per module", "w16": "windows $W{=}16$, overlap 8", "w64": "windows $W{=}64$, overlap 32"}
ORDER = list(LABEL)


def load(root, ds, v):
    out = {}
    for f in glob.glob(f"{root}/{ds}/{v}/*.json"):
        d = json.load(open(f)); rows = {r["method"]: r for r in d.get("rows", [])}
        if rows:
            out[pathlib.Path(f).stem] = rows
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/ablate"); ap.add_argument("--out", default="paper/tab_ablation.tex")
    ap.add_argument("--variants", nargs="*", default=None, help="subset and order of variants (default: all known)")
    ap.add_argument("--caption", default=None); ap.add_argument("--label", default="tab:ablation")
    a = ap.parse_args()
    lines = []; console = []
    for v in (a.variants or ORDER):
        s7 = load(a.root, "7scenes", v); co = load(a.root, "co3d", v)
        if not s7 and not co:
            continue
        def mean7(row, key): return np.mean([r[row][key] for r in s7.values() if row in r]) if s7 else float("nan")
        def meanco(row, key): return np.mean([r[row][key] for r in co.values() if row in r]) if co else float("nan")
        f = lambda x, p=3: ("--" if np.isnan(x) else f"{x:.{p}f}")
        ate = [mean7(r, "ate_rmse") for r in ("chained", "smr", "smr_pgo")]; loops = mean7("smr", "n_loops")
        auc = [meanco(r, "auc30") for r in ("chained", "smr", "smr_pgo")]
        lines.append(f"{LABEL[v]} & {f(ate[0])} & {f(ate[1])} & {f(ate[2])} & {f(loops, 1)} & {f(auc[0], 1)} & {f(auc[1], 1)} & {f(auc[2], 1)} \\\\")
        console.append(f"{v:<10} 7scenes n={len(s7)} ATE {f(ate[0])}/{f(ate[1])}/{f(ate[2])} loops {f(loops,1)} | co3d n={len(co)} AUC {f(auc[0],1)}/{f(auc[1],1)}/{f(auc[2],1)}")
    cap = a.caption or r"\textbf{Ablations of the memory's policy.} One knob changed at a time from the default; backbone passes fixed. 7-Scenes: ATE RMSE (m), mean over the seven seq-01 sequences (VGGT-$\Omega$, 200 keyframes, $W{=}32$); \emph{loops} = accepted closures per sequence. CO3D: AUC@30 at $N{=}200$, mean over the first twelve orbits (VGGT). raw = chained windows. No closure was rejected by the gate in the default runs, so the robust batch solve had nothing to reject."
    tex = r'''\begin{table}[t]
\centering\scriptsize\setlength{\tabcolsep}{4pt}
\caption{''' + cap + r'''}
\label{''' + a.label + r'''}
\begin{tabular}{@{}l ccc c ccc@{}}
\toprule
& \multicolumn{4}{c}{7-Scenes ATE (m)$\downarrow$} & \multicolumn{3}{c}{CO3D AUC@30$\uparrow$} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
Variant & raw & +\smr{} & +\smr{}+PGO & loops & raw & +\smr{} & +\smr{}+PGO \\
\midrule
''' + "\n".join(lines) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text(tex)
    print("\n".join(console)); print(f"wrote {a.out} ({len(lines)} variants)")


if __name__ == "__main__":
    main()
