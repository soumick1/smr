#!/usr/bin/env python3
"""Training figures for the Gaussian heads (one per backbone).

    python scripts/fig_nvs_training.py --nvs outputs/nvs --backbones vggt_omega vggt pi3 stream3r --out outputs/figures/nvs_training
Reads <nvs>/<bb>/metrics.csv (step, loss, psnr, n_gauss, s_per_step, lr; every 50 steps) and <nvs>/<bb>/val.csv
(tag, step, PSNR, SSIM, LPIPS, n_objects; every 2000 steps).  Writes <out>.pdf (vector) + <out>.png (600 dpi):
  (a) training loss (L1 + LPIPS), EMA-smoothed, log-y          (b) training PSNR, EMA-smoothed
  (c) validation PSNR vs step with the untrained-head floor      (d) validation SSIM        (e) validation LPIPS
  (f) learning-rate schedule (warm-up + cosine) and Gaussians per step (right axis)
The untrained floor is the smoke test's val@10/20 (18.55 dB for VGGT: the head at zero init = splat the backbone's points).
"""
import argparse, csv, pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.2, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
                     "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"vggt": "VGGT", "vggt_omega": "VGGT-Ω", "pi3": "π³", "stream3r": "STream3R"}
COLS = {"vggt": "tab:blue", "vggt_omega": "tab:orange", "pi3": "tab:green", "stream3r": "tab:red"}


def read_metrics(p):
    rows = list(csv.DictReader(open(p)))
    g = lambda k: np.array([float(r[k]) for r in rows])
    return dict(step=g("step"), loss=g("loss"), psnr=g("psnr"), n_gauss=g("n_gauss"), s_per_step=g("s_per_step"), lr=g("lr"))


def read_val(p):
    rows = [r for r in csv.reader(open(p)) if r and r[0] == "val"]
    a = np.array([[float(x) for x in r[1:5]] for r in rows])
    return dict(step=a[:, 0], psnr=a[:, 1], ssim=a[:, 2], lpips=a[:, 3])


def ema(x, alpha=0.1):
    y = np.empty_like(x); m = x[0]
    for i, v in enumerate(x):
        m = (1 - alpha) * m + alpha * v; y[i] = m
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvs", default="outputs/nvs"); ap.add_argument("--backbones", nargs="+", default=["vggt_omega", "vggt", "pi3", "stream3r"])
    ap.add_argument("--floor", type=float, default=18.55, help="untrained-head validation PSNR (smoke test); 0 to hide")
    ap.add_argument("--width-in", type=float, default=8.27, help="figure width in inches (8.27 = A4 width)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default="outputs/figures/nvs_training")
    a = ap.parse_args()
    M = {bb: read_metrics(pathlib.Path(a.nvs) / bb / "metrics.csv") for bb in a.backbones if (pathlib.Path(a.nvs) / bb / "metrics.csv").exists()}
    V = {bb: read_val(pathlib.Path(a.nvs) / bb / "val.csv") for bb in a.backbones if (pathlib.Path(a.nvs) / bb / "val.csv").exists()}
    fig, axes = plt.subplots(2, 3, figsize=(a.width_in, a.width_in * 0.56)); ax = axes.ravel()
    for bb, m in M.items():
        c = COLS.get(bb, None); lab = NAMES.get(bb, bb)
        ax[0].plot(m["step"], m["loss"], color=c, alpha=0.18, lw=0.6); ax[0].plot(m["step"], ema(m["loss"]), color=c, lw=1.4, label=lab)
        ax[1].plot(m["step"], m["psnr"], color=c, alpha=0.18, lw=0.6); ax[1].plot(m["step"], ema(m["psnr"]), color=c, lw=1.4, label=lab)
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("training loss  ($\\ell_1$ + 0.5 LPIPS)"); ax[0].set_title("(a) training loss (EMA; raw in light)"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("training PSNR (dB), 4 targets / step"); ax[1].set_title("(b) training PSNR (EMA; raw in light)")
    for bb, v in V.items():
        c = COLS.get(bb, None); lab = NAMES.get(bb, bb)
        ax[2].plot(v["step"], v["psnr"], color=c, marker="o", ms=3, lw=1.3, label=lab)
        ax[3].plot(v["step"], v["ssim"], color=c, marker="o", ms=3, lw=1.3, label=lab)
        ax[4].plot(v["step"], v["lpips"], color=c, marker="o", ms=3, lw=1.3, label=lab)
    if a.floor > 0:
        ax[2].axhline(a.floor, color="0.4", ls="--", lw=0.9, label=f"untrained head ({a.floor:.2f} dB)")
    ax[2].set_xlabel("step"); ax[2].set_ylabel("validation PSNR (dB)"); ax[2].set_title("(c) held-out PSNR (20 objects, 10 targets each)")
    ax[2].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), columnspacing=1.0, handlelength=1.6)
    ax[3].set_xlabel("step"); ax[3].set_ylabel("validation SSIM"); ax[3].set_title("(d) held-out SSIM")
    ax[4].set_xlabel("step"); ax[4].set_ylabel("validation LPIPS ↓"); ax[4].set_title("(e) held-out LPIPS")
    # (f) LR schedule + gaussians per step
    bb0 = next(iter(M)); m0 = M[bb0]
    ax[5].plot(m0["step"], m0["lr"], color="0.25", lw=1.3); ax[5].set_xlabel("step"); ax[5].set_ylabel("learning rate"); ax[5].set_title("(f) LR schedule; Gaussians per step (right axis)")
    ax2 = ax[5].twinx()
    ax2.plot(m0["step"], ema(m0["n_gauss"], 0.05) / 1e3, color="tab:purple", lw=0.9, alpha=0.9)      # identical for every head: one Gaussian per visible input pixel
    ax2.set_ylabel("Gaussians / step (K, EMA)", color="tab:purple"); ax2.tick_params(axis="y", colors="tab:purple"); ax2.spines["top"].set_visible(False)
    xmax = max(m["step"].max() for m in M.values())
    for x in list(ax) + [ax2]:
        x.set_xlim(0, xmax)
    for x in ax:
        x.set_xticks([0, 10000, 20000, 30000]); x.set_xticklabels(["0", "10k", "20k", "30k"])
    fig.tight_layout(w_pad=2.2, h_pad=2.0)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png", dpi=a.dpi)      # no bbox cropping: the PNG is exactly width_in x dpi pixels wide
    for bb in M:
        m, v = M[bb], V.get(bb)
        print(f"{bb:<11} steps {int(m['step'].max())}  final loss {ema(m['loss'])[-1]:.4f}  s/step {np.median(m['s_per_step']):.2f}  gaussians {np.median(m['n_gauss']):,.0f}"
              + (f"  val best PSNR {v['psnr'].max():.2f} @ {int(v['step'][v['psnr'].argmax()])}  final {v['psnr'][-1]:.2f}/{v['ssim'][-1]:.4f}/{v['lpips'][-1]:.4f}" if v else ""))
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    main()
