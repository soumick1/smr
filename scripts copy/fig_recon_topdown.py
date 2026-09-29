#!/usr/bin/env python3
"""Top-down reconstruction comparison with zoom insets: raw (chained windows) vs +SMR fused clouds of one scene.

    python scripts/fig_recon_topdown.py --raw outputs/points/pumpkin_raw/fused.ply --smr outputs/points/pumpkin_smr/fused.ply \
        --zoom 0.3 0.2 0.25 --out outputs/figures/recon_pumpkin
Both clouds are in the same (GT-aligned) frame; --zoom X Y HALF picks the inset window (scene units).  Points are
coloured by height (or by their own colour if the PLY carries rgb).  The analogue of VGGT-SLAM's Sim(3)-vs-SL(4)
figure: ghosted geometry in a revisited region on the left, aligned on the right.
"""
import argparse, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dtu_eval import read_ply  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True); ap.add_argument("--smr", required=True)
ap.add_argument("--zoom", nargs=3, type=float, default=None, metavar=("X", "Y", "HALF"))
ap.add_argument("--max-points", type=int, default=400000); ap.add_argument("--out", required=True)
ap.add_argument("--labels", nargs=2, default=["raw (chained windows)", "+SMR"])
a = ap.parse_args()
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
clouds = [read_ply(a.raw), read_ply(a.smr)]
rng = np.random.default_rng(0)
clouds = [c[rng.choice(len(c), min(len(c), a.max_points), replace=False)] for c in clouds]
allp = np.concatenate(clouds); up = np.argmin(allp.var(0)); keep = [i for i in range(3) if i != up]
lo, hi = np.percentile(allp[:, keep], 1, axis=0), np.percentile(allp[:, keep], 99, axis=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, P, lab in zip(axes, clouds, a.labels):
    ax.scatter(P[:, keep[0]], P[:, keep[1]], c=P[:, up], s=0.15, cmap="viridis", linewidths=0, rasterized=True)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal"); ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7)
    if a.zoom:
        x, y, h = a.zoom
        ins = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=0.6)
        sel = (np.abs(P[:, keep[0]] - x) < h) & (np.abs(P[:, keep[1]] - y) < h)
        ins.scatter(P[sel, keep[0]], P[sel, keep[1]], c=P[sel, up], s=0.5, cmap="viridis", linewidths=0, rasterized=True)
        ins.set_xlim(x - h, x + h); ins.set_ylim(y - h, y + h); ins.set_xticks([]); ins.set_yticks([]); ins.set_aspect("equal")
        for sp in ins.spines.values(): sp.set_edgecolor("tab:orange"); sp.set_linewidth(1.2)
        mark_inset(ax, ins, loc1=2, loc2=4, fc="none", ec="tab:orange", lw=0.8)
out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out) + ".pdf", bbox_inches="tight", dpi=200); fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight")
print(f"wrote {out}.pdf/.png ({len(clouds[0]):,} / {len(clouds[1]):,} points drawn)")
