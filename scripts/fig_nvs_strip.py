#!/usr/bin/env python3
"""NVS qualitative strip: rows = objects, columns = 4 inputs | GT target | raw render | +SMR render (+ PSNR labels).

    python scripts/fig_nvs_strip.py --dir outputs/nvs/vggt/images --objects OBJ1 OBJ2 --target 3 --out outputs/figures/nvs_vggt
The directory is what experiments/eval_gso.py --save-images wrote (run once with --reads 1 and once with --reads 4
into the same directory).  Only PIL + matplotlib.
"""
import argparse, json, pathlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--objects", nargs="+", required=True)
ap.add_argument("--target", type=int, default=0, help="which of the ten targets to show")
ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
a = ap.parse_args()
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
root = pathlib.Path(a.dir)
cols = ["input 1", "input 2", "input 3", "input 4", "target (GT)", "raw", "+SMR read"]
fig, axes = plt.subplots(len(a.objects), len(cols), figsize=(1.55 * len(cols), 1.55 * len(a.objects) + 0.35))
axes = np.atleast_2d(axes)
for r, obj in enumerate(a.objects):
    d = root / obj
    ims = [d / f"input_{j}.png" for j in range(4)] + [d / f"gt_{a.target}.png", d / f"pred_reads1_{a.target}.png", d / f"pred_reads4_{a.target}.png"]
    m1 = json.load(open(d / "metrics_reads1.json")) if (d / "metrics_reads1.json").exists() else {}
    m4 = json.load(open(d / "metrics_reads4.json")) if (d / "metrics_reads4.json").exists() else {}
    for c, p in enumerate(ims):
        ax = axes[r, c]; ax.axis("off")
        if p.exists():
            ax.imshow(Image.open(p))
        if r == 0:
            ax.set_title(cols[c], fontsize=8)
        if c == 5 and m1: ax.text(4, 250, f"{m1['psnr']:.2f} dB", fontsize=7, color="w", bbox=dict(fc="k", alpha=0.6, pad=1.5, lw=0))
        if c == 6 and m4: ax.text(4, 250, f"{m4['psnr']:.2f} dB", fontsize=7, color="w", bbox=dict(fc="k", alpha=0.6, pad=1.5, lw=0))
    axes[r, 0].text(-0.08, 0.5, obj.replace("_", " ")[:22], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=6)
if a.title: fig.suptitle(a.title, fontsize=9)
plt.subplots_adjust(wspace=0.03, hspace=0.06)
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out) + ".pdf", bbox_inches="tight"); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight")
print(f"wrote {out}.pdf/.png")
