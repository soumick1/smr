#!/usr/bin/env python3
"""NVS qualitative strip: rows = objects; columns = 4 inputs | target | raw | +SMR [| |raw-target| | |+SMR-target|].

    python scripts/fig_nvs_strip.py --dir outputs/nvs/vggt/images_pick --objects OBJ1 OBJ2 OBJ3 --targets 3 7 1 \
        --error-maps --out outputs/figures/nvs_vggt_pick
--targets gives one target index per object (from nvs_render_objects.py's "best target"); --target gives one for all.
--error-maps appends two columns with |render - target| (mean over RGB) on a shared colour scale, which shows where the
read changes the geometry even when the renders look alike. Labels show PSNR (and LPIPS with --lpips) of the shown target
when per_target.json exists, else the object mean. Only PIL + matplotlib.
"""
import argparse, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", required=True); ap.add_argument("--objects", nargs="+", required=True)
ap.add_argument("--target", type=int, default=0); ap.add_argument("--targets", nargs="*", default=None,
                help="one target index per object, or the single word auto to take each object's best target from <dir>/summary.json")
ap.add_argument("--auto-by", default="psnr", choices=["psnr", "lpips"], help="with --targets auto: best target by PSNR gain or LPIPS reduction")
ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
ap.add_argument("--error-maps", action="store_true"); ap.add_argument("--lpips", action="store_true")
ap.add_argument("--vmax", type=float, default=0.5, help="error-map colour scale maximum (mean absolute RGB error, 0-1)")
a = ap.parse_args()
root = pathlib.Path(a.dir)
if a.targets and len(a.targets) == 1 and a.targets[0] == "auto":
    summ = json.load(open(root / "summary.json"))
    key = "best_psnr_target" if a.auto_by == "psnr" else "best_lpips_target"
    targets = [int(summ[o][key]) if o in summ else a.target for o in a.objects]
    print("targets:", dict(zip(a.objects, targets)))
else:
    targets = [int(t) for t in a.targets] if a.targets else [a.target] * len(a.objects)
assert len(targets) == len(a.objects), "--targets needs one index per object (or the word auto)"
cols = ["input 1", "input 2", "input 3", "input 4", "target", "raw", "+SMR read"] + (["|raw − target|", "|+SMR − target|"] if a.error_maps else [])
fig, axes = plt.subplots(len(a.objects), len(cols), figsize=(1.55 * len(cols), 1.55 * len(a.objects) + 0.35))
axes = np.atleast_2d(axes)
err = lambda p, g: np.abs(np.asarray(Image.open(p).convert("RGB"), float) - np.asarray(Image.open(g).convert("RGB"), float)).mean(2) / 255.0
for r, (obj, t) in enumerate(zip(a.objects, targets)):
    d = root / obj
    gt = d / f"gt_{t}.png"; p1 = d / f"pred_reads1_{t}.png"; p4 = d / f"pred_reads4_{t}.png"
    ims = [d / f"input_{j}.png" for j in range(4)] + [gt, p1, p4]
    per = json.load(open(d / "per_target.json")) if (d / "per_target.json").exists() else None
    def label(reads):
        if per and str(reads) in per:
            s = f"{per[str(reads)]['psnr'][t]:.2f} dB"
            return s + (f"  {per[str(reads)]['lpips'][t]:.3f}" if a.lpips else "")
        m = d / f"metrics_reads{reads}.json"
        return f"{json.load(open(m))['psnr']:.2f} dB" if m.exists() else ""
    for c, p in enumerate(ims):
        ax = axes[r, c]; ax.axis("off")
        if p.exists():
            ax.imshow(Image.open(p))
        if r == 0:
            ax.set_title(cols[c], fontsize=8)
        if c in (5, 6):
            ax.text(4, 250, label(1 if c == 5 else 4), fontsize=7, color="w", bbox=dict(fc="k", alpha=0.6, pad=1.5, lw=0))
    if a.error_maps:
        for c, p in ((7, p1), (8, p4)):
            ax = axes[r, c]; ax.axis("off")
            if p.exists() and gt.exists():
                ax.imshow(err(p, gt), cmap="magma", vmin=0, vmax=a.vmax)
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
    axes[r, 0].text(-0.08, 0.5, obj.replace("_", " ")[:22], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=6)
if a.title:
    fig.suptitle(a.title, fontsize=9)
plt.subplots_adjust(wspace=0.03, hspace=0.06)
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out) + ".pdf", bbox_inches="tight"); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight")
print("wrote", str(out) + ".pdf/.png")
