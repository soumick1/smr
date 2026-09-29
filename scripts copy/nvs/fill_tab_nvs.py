#!/usr/bin/env python3
"""Fill the Objaverse held-out cells of paper/tab_nvs.tex from outputs/nvs/<bb>/val_reads{1,4}.jsonl.

    python scripts/nvs/fill_tab_nvs.py [--tab paper/tab_nvs.tex] [--nvs outputs/nvs]
Bold = the better of each raw/+SMR pair per metric (ties: neither).  Prints the numbers it wrote.
"""
import argparse, json, pathlib, re
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--tab", default="paper/tab_nvs.tex"); ap.add_argument("--nvs", default="outputs/nvs")
a = ap.parse_args()
pre = {"vggt_omega": "VO", "vggt": "VG", "pi3": "PI", "stream3r": "ST"}
tab = pathlib.Path(a.tab); s = tab.read_text()
def load(bb, r):
    p = pathlib.Path(a.nvs) / bb / f"val_reads{r}.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in open(p) if l.strip()]; rows = [x for x in rows if "psnr" in x]
    return {k: np.mean([x[k] for x in rows]) for k in ("psnr", "ssim", "lpips")}, len(rows)
for bb, pfx in pre.items():
    A, B = load(bb, 1), load(bb, 4)
    if A is None or B is None:
        print(f"{bb}: missing val jsonl -> cells left as --"); continue
    (ma, na), (mb, nb) = A, B
    ids = {k: [x["id"] for x in [json.loads(l) for l in open(pathlib.Path(a.nvs) / bb / f"val_reads{k}.jsonl") if l.strip()]] for k in (1, 4)}
    print(f"{bb}: n={na}/{nb}  raw {ma['psnr']:.2f}/{ma['ssim']:.3f}/{ma['lpips']:.3f}  read {mb['psnr']:.2f}/{mb['ssim']:.3f}/{mb['lpips']:.3f}")
    for k, suf, fmt in (("psnr", "r", "{:.2f}"), ("ssim", "s", "{:.3f}"), ("lpips", "l", "{:.3f}")):
        va, vb = fmt.format(ma[k]), fmt.format(mb[k])
        better_b = (mb[k] > ma[k]) if k != "lpips" else (mb[k] < ma[k]); better_a = (ma[k] > mb[k]) if k != "lpips" else (ma[k] < mb[k])
        if va == vb or abs(ma[k] - mb[k]) <= (0.015 if k == 'psnr' else 0.0015): better_a = better_b = False   # ties
        ca = f"\\textbf{{{va}}}" if better_a else va; cb = f"\\textbf{{{vb}}}" if better_b else vb
        s = re.sub(r"\\newcommand\{\\%s%s\}\{[^}]*\}" % (pfx, suf), lambda m: "\\newcommand{\\%s%s}{%s}" % (pfx, suf, ca), s)
        s = re.sub(r"\\newcommand\{\\%s%sR\}\{[^}]*\}" % (pfx, suf), lambda m: "\\newcommand{\\%s%sR}{%s}" % (pfx, suf, cb), s)
tab.write_text(s); print(f"wrote {tab}")
