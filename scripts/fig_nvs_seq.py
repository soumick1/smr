#!/usr/bin/env python3
"""Figures for sequence NVS (v203): replacement Fig. 2B, Fig. S2 and Fig. S3.

    python scripts/fig_nvs_seq.py --root outputs/nvs_seq_v2 --caches cache/nvsseq --out outputs/figures/nvs_seq \
        --old-fig2 images/Fig3_mod.png --fig2-seq office_seq01 --fig2-target 128 --bb vggt_omega

Outputs (PNG at 300 dpi + PDF):
    fig2B_nvs.png        panel B: four input keyframes of the sequence, then ground truth / backbone / backbone+SMoRe for one
                         held-out target with a zoom on the region where the memory changes the render most
    Fig2_new.png         the old panel A (cropped from --old-fig2) next to the new panel B, same height
    figS2_nvs_backbones  rows = six backbones, columns = ground truth / raw / +SMoRe / +SMoRe+PGO for one target, PSNR labels
    figS2_nvs_scenes     rows = test sequences (one saved target each), same columns, for --bb
    figS3_nvs_training   (a) decoder training loss (EMA), raw vs SMoRe decoders, six backbones; (b) paired PSNR gain of +SMoRe
                         over raw, all and revisited targets, 95 % bootstrap CIs; (c) per-sequence raw vs +SMoRe PSNR
Needs: outputs/nvs_seq_v2/<bb>/{raw,smr,smr_pgo,gt}.jsonl, train_{raw,smr}.log, images/<seq>/<method>_tNNN.png
(eval --save-images; rerun eval with --save-images 24 to have every target available), and the test caches for the
input-keyframe strip (in_paths.json; optional).
"""
from __future__ import annotations

import argparse, glob, json, pathlib, re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

BBS = ["vggt_omega", "vggt", "pi3", "stream3r", "streamvggt", "fast3r"]
LABEL = {"vggt_omega": "VGGT-Ω", "vggt": "VGGT", "pi3": "π³", "stream3r": "STream3R", "streamvggt": "StreamVGGT", "fast3r": "Fast3R"}
INK = "#222222"


def load_jsonl(p):
    p = pathlib.Path(p)
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def recs(root, bb):
    out = {}
    for m in ("raw", "smr", "smr_pgo", "gt"):
        out[m] = {(r["seq"], r["target_kf"]): r for r in load_jsonl(pathlib.Path(root) / bb / f"{m}.jsonl")}
    return out


def img(root, bb, seq, m, t):
    p = pathlib.Path(root) / bb / "images" / seq / f"{m}_t{t:03d}.png"
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0 if p.exists() else None


def saved_targets(root, bb, seq):
    return sorted(int(re.search(r"_t(\d+)\.png$", p.name).group(1)) for p in (pathlib.Path(root) / bb / "images" / seq).glob("gt_t*.png"))


def box_where_fixed(gt, raw, smr, size=(96, 72)):
    """Top-left of the size=(w,h) window where |raw-gt| exceeds |smr-gt| the most (box-filtered)."""
    e = np.abs(raw - gt).mean(-1) - np.abs(smr - gt).mean(-1)
    w, h = size; H, W = e.shape
    c = np.cumsum(np.cumsum(np.pad(e, ((1, 0), (1, 0))), 0), 1)
    s = c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return int(x), int(y), w, h


def put(ax, im, title=None, fs=10):
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fs, color=INK, pad=4)


def psnr_label(ax, v, fs=9):
    ax.text(0.98, 0.04, f"{v:.2f} dB", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs, color="white",
            bbox=dict(facecolor="black", edgecolor="none", pad=2.5))


# ------------------------------------------------------------------ figure 2B
def fig2B(a):
    bb, seq, t = a.bb, a.fig2_seq, a.fig2_target
    R = recs(a.root, bb)
    gt, raw, smr = (img(a.root, bb, seq, m, t) for m in ("gt", "raw", "smr"))
    if gt is None:
        raise SystemExit(f"no saved images for {bb}/{seq} target {t}; rerun eval with --save-images 24")
    x, y, w, h = box_where_fixed(gt, raw, smr)
    # input keyframes: 4 of the target's sources if the cache is there, else evenly spaced inputs
    cdir = pathlib.Path(a.caches) / bb / "test" / seq
    strip = []
    if (cdir / "in_paths.json").exists():
        paths = json.load(open(cdir / "in_paths.json"))
        in_gt = np.load(cdir / "in_gt.npy"); tgt_kf = np.load(cdir / "tgt_kf.npy"); tgt_gt = np.load(cdir / "tgt_gt.npy")
        ti = int(np.where(tgt_kf == t)[0][0]) if (tgt_kf == t).any() else 0
        c = in_gt[:, :3, 3]; T = tgt_gt[ti]
        score = np.linalg.norm(c - T[:3, 3], axis=1) + np.degrees(np.arccos(np.clip(in_gt[:, :3, 2] @ T[:3, 2], -1, 1))) / 15.0 * float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        order = np.argsort(score)[:8]
        pick = sorted(order[:: max(1, len(order) // 4)][:4])          # spread over the eight nearest, keeps both visits when present
        strip = [np.asarray(Image.open(paths[j]).convert("RGB").resize((gt.shape[1], gt.shape[0]), Image.LANCZOS), np.float32) / 255.0 for j in pick]
    if a.fig2_box:
        x, y, w, h = [int(v) for v in a.fig2_box.split(",")]
    fig = plt.figure(figsize=(6.4, 5.1), dpi=300)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.72, 1.0, 0.95], hspace=0.42, wspace=0.05, left=0.01, right=0.99, top=0.88, bottom=0.01)
    fig.text(0.005, 0.965, "B", fontsize=17, weight="bold", color=INK)
    n_in = len(json.load(open(cdir / "in_paths.json"))) if (cdir / "in_paths.json").exists() else None
    fig.text(0.52, 0.955, f"Input: {n_in} keyframes of a long sequence (7-Scenes {seq.split('_')[0]}); four shown" if n_in else f"Input keyframes (7-Scenes {seq.split('_')[0]})",
             ha="center", fontsize=12, color=INK)
    if strip:
        inner = gs[0, :].subgridspec(1, 4, wspace=0.04)
        for i, s_ in enumerate(strip):
            ax = fig.add_subplot(inner[0, i]); put(ax, s_)
    cols = [("Ground truth", gt, None), (LABEL[bb], raw, R["raw"].get((seq, t), {}).get("psnr")), (f"{LABEL[bb]}+SMoRe", smr, R["smr"].get((seq, t), {}).get("psnr"))]
    axes_row = []
    for i, (title, im, v) in enumerate(cols):
        ax = fig.add_subplot(gs[1, i]); put(ax, im, title, fs=12); axes_row.append(ax)
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, lw=1.4, color="black"))
        if v is not None:
            psnr_label(ax, v, fs=10)
        axz = fig.add_subplot(gs[2, i]); put(axz, im[y:y + h, x:x + w])
        for sp in axz.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.2); sp.set_color("black")
    ytop = axes_row[0].get_position().y1
    fig.text(0.52, ytop + 0.085, "Synthesized held-out view", ha="center", fontsize=12, color=INK)
    fig.add_artist(plt.Line2D([0.03, 0.97], [ytop + 0.125, ytop + 0.125], color=INK, lw=0.8))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig2B_nvs.png", dpi=300); fig.savefig(out / "fig2B_nvs.pdf"); plt.close(fig)
    print("fig2B:", seq, "target", t, "zoom box", (x, y, w, h), "PSNR raw/smr", cols[1][2], cols[2][2])
    if a.old_fig2 and pathlib.Path(a.old_fig2).exists():
        old = Image.open(a.old_fig2).convert("RGB"); A = old.crop((0, 0, a.fig2_split, old.height))
        B = Image.open(out / "fig2B_nvs.png").convert("RGB"); B = B.resize((int(B.width * old.height / B.height), old.height), Image.LANCZOS)
        new = Image.new("RGB", (A.width + 40 + B.width, old.height), "white"); new.paste(A, (0, 0)); new.paste(B, (A.width + 40, 0))
        new.save(out / "Fig2_new.png"); print("Fig2_new:", new.size)


# ------------------------------------------------------------------ figure S2
def figS2(a):
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seq, t = a.fig2_seq, a.fig2_target
    methods = [("gt", "Ground truth"), ("raw", "{bb}"), ("smr", "{bb}+SMoRe")]
    # (i) backbones on one target
    rows = [bb for bb in BBS if img(a.root, bb, seq, "gt", t) is not None]
    if rows:
        fig, axs = plt.subplots(len(rows), 3, figsize=(7.0, 1.75 * len(rows)), dpi=300, gridspec_kw=dict(wspace=0.03, hspace=0.08, left=0.07, right=0.995, top=0.95, bottom=0.005))
        axs = np.atleast_2d(axs)
        for r, bb in enumerate(rows):
            R = recs(a.root, bb)
            for c, (m, title) in enumerate(methods):
                im = img(a.root, bb, seq, m, t); ax = axs[r, c]
                put(ax, im, title.format(bb="") .replace("+", "+", 1).strip("+") if False else (title.format(bb=LABEL[bb]) if r == 0 else None), fs=9)
                if m != "gt" and (seq, t) in R[m]:
                    psnr_label(ax, R[m][(seq, t)]["psnr"], fs=7.5)
            axs[r, 0].text(-0.04, 0.5, LABEL[bb], transform=axs[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9, color=INK)
        for c, (m, title) in enumerate(methods):
            axs[0, c].set_title(title.format(bb="geometry model").replace("geometry model+", "+"), fontsize=9.5, color=INK, pad=4)
        fig.savefig(out / "figS2_nvs_backbones.png", dpi=300); fig.savefig(out / "figS2_nvs_backbones.pdf"); plt.close(fig)
        print("figS2_nvs_backbones:", rows, seq, t)
    # (ii) scenes for one backbone
    bb = a.bb; R = recs(a.root, bb)
    seqs = sorted(p.name for p in (pathlib.Path(a.root) / bb / "images").glob("*") if p.is_dir())
    rows = []
    for s in seqs:
        ts = saved_targets(a.root, bb, s)
        if not ts:
            continue
        # the saved target with the largest +SMoRe gain among revisited ones, else the largest gain
        cand = [(R["smr"].get((s, tt), {}).get("psnr", np.nan) - R["raw"].get((s, tt), {}).get("psnr", np.nan), R["smr"].get((s, tt), {}).get("revisit", False), tt) for tt in ts]
        cand = [c for c in cand if np.isfinite(c[0])] or [(0, False, ts[0])]
        rev = [c for c in cand if c[1]]
        rows.append((s, max(rev or cand)[2]))
    if rows:
        fig, axs = plt.subplots(len(rows), 3, figsize=(7.0, 1.75 * len(rows)), dpi=300, gridspec_kw=dict(wspace=0.03, hspace=0.08, left=0.07, right=0.995, top=0.95, bottom=0.005))
        axs = np.atleast_2d(axs)
        for r, (s, tt) in enumerate(rows):
            for c, (m, title) in enumerate(methods):
                ax = axs[r, c]; put(ax, img(a.root, bb, s, m, tt), (title.format(bb=LABEL[bb]) if r == 0 else None), fs=9.5)
                if m != "gt" and (s, tt) in R[m]:
                    psnr_label(ax, R[m][(s, tt)]["psnr"], fs=7.5)
            axs[r, 0].text(-0.04, 0.5, f"{s.split('_')[0]}  (kf {tt})", transform=axs[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8.5, color=INK)
        fig.savefig(out / "figS2_nvs_scenes.png", dpi=300); fig.savefig(out / "figS2_nvs_scenes.pdf"); plt.close(fig)
        print("figS2_nvs_scenes:", rows)


# ------------------------------------------------------------------ figure S3
def ema(v, k=0.9):
    out, s = [], None
    for x in v:
        s = x if s is None else k * s + (1 - k) * x; out.append(s)
    return np.array(out)


def gain_panel(ax, root, title, rng):
    """Mean +- s.e.m. of the per-target PSNR gain (+SMoRe - raw), all targets and revisited targets, per backbone."""
    xs = np.arange(len(BBS)); wbar = 0.38; any_ = False
    for sel, off, col, lab in (("all", -wbar / 2, "#4c72b0", "all held-out targets"), ("rev", wbar / 2, "#dd8452", "revisited targets")):
        means, sems = [], []
        for bb in BBS:
            R = recs(root, bb); keys = sorted(set(R["raw"]) & set(R["smr"]))
            if sel == "rev":
                keys = [kk for kk in keys if R["smr"][kk].get("revisit")]
            d = np.array([R["smr"][kk]["psnr"] - R["raw"][kk]["psnr"] for kk in keys])
            means.append(d.mean() if len(d) else np.nan); sems.append(d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan)
            any_ |= len(d) > 0
        ax.bar(xs + off, means, wbar, color=col, label=lab, yerr=sems, capsize=2.5, error_kw=dict(lw=0.9))
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels([LABEL[b] for b in BBS], fontsize=8, rotation=20)
    ax.set_ylabel("PSNR gain of +SMoRe over raw (dB)"); ax.set_title(title, fontsize=9); ax.legend(fontsize=7.5, frameon=False)
    return any_


def figS3(a):
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    has_co3d = a.root_co3d and any(pathlib.Path(a.root_co3d).glob("*/raw.jsonl"))
    ncol = 4 if has_co3d else 3
    fig, axs = plt.subplots(1, ncol, figsize=(3.9 * ncol, 3.6), dpi=300, gridspec_kw=dict(wspace=0.32, left=0.05, right=0.99, top=0.9, bottom=0.18))
    cmap = plt.get_cmap("tab10"); rng = np.random.default_rng(0)
    ax = axs[0]
    for i, bb in enumerate(BBS):
        for m, ls in (("raw", "--"), ("smr", "-")):
            p = pathlib.Path(a.root) / bb / f"train_{m}.log"
            if not p.exists():
                continue
            st, lo = [], []
            for line in open(p):
                mm = re.match(r"step (\d+)/\d+\s+loss ([\d.]+)", line)
                if mm:
                    st.append(int(mm.group(1))); lo.append(float(mm.group(2)))
            if st:
                ax.plot(st, ema(lo, 0.8), ls=ls, lw=1.2, color=cmap(i), label=f"{LABEL[bb]} ({'raw' if m == 'raw' else '+SMoRe'})")
    ax.set_xlabel("step"); ax.set_ylabel("training loss (EMA)"); ax.set_title("(a) decoder training: raw (dashed) vs SMoRe (solid) geometry", fontsize=9)
    ax.legend(fontsize=5.5, ncol=2, frameon=False)
    gain_panel(axs[1], a.root, "(b) 7-Scenes: paired PSNR gain, mean ± s.e.m.", rng)
    k = 2
    if has_co3d:
        gain_panel(axs[2], a.root_co3d, "(c) CO3Dv2 (test only): paired PSNR gain, mean ± s.e.m.", rng); k = 3
    ax = axs[k]; allv = []
    for i, bb in enumerate(BBS):
        R = recs(a.root, bb)
        seqs = sorted({kk[0] for kk in R["raw"]})
        for s_ in seqs:
            kr = [kk for kk in R["raw"] if kk[0] == s_ and kk in R["smr"]]
            if not kr:
                continue
            xr = np.mean([R["raw"][kk]["psnr"] for kk in kr]); ys = np.mean([R["smr"][kk]["psnr"] for kk in kr])
            ax.scatter(xr, ys, s=22, color=cmap(i), edgecolor="white", linewidth=0.5, label=LABEL[bb] if s_ == seqs[0] else None); allv += [xr, ys]
    if allv:
        lo_, hi_ = min(allv) - 0.5, max(allv) + 0.5; ax.plot([lo_, hi_], [lo_, hi_], "k--", lw=0.8); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
    ax.set_xlabel("raw geometry (PSNR, dB)"); ax.set_ylabel("+SMoRe (PSNR, dB)"); ax.set_title(f"({'d' if has_co3d else 'c'}) 7-Scenes: per-sequence mean PSNR", fontsize=9)
    ax.set_aspect("equal"); ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axs:
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.savefig(out / "figS3_nvs_training.png", dpi=300); fig.savefig(out / "figS3_nvs_training.pdf"); plt.close(fig)
    print("figS3_nvs_training written" + (" (with CO3D panel)" if has_co3d else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/nvs_seq_v2"); ap.add_argument("--root-co3d", default="outputs/nvs_seq_v2_co3d", help="CO3D test-only results root (for panel c)"); ap.add_argument("--caches", default="cache/nvsseq"); ap.add_argument("--out", default="outputs/figures/nvs_seq")
    ap.add_argument("--bb", default="vggt_omega"); ap.add_argument("--fig2-seq", default="office_seq01"); ap.add_argument("--fig2-target", type=int, default=128)
    ap.add_argument("--old-fig2", default=None, help="the current Fig. 2 PNG; its panel A is cropped and placed next to the new panel B"); ap.add_argument("--fig2-split", type=int, default=2010)
    ap.add_argument("--fig2-box", default=None, help="zoom box x,y,w,h in pixels of the 384x288 render (default: where +SMoRe reduces the error most)")
    ap.add_argument("--only", nargs="*", default=None, help="subset: fig2B figS2 figS3")
    a = ap.parse_args()
    want = set(a.only) if a.only else {"fig2B", "figS2", "figS3"}
    if "fig2B" in want: fig2B(a)
    if "figS2" in want: figS2(a)
    if "figS3" in want: figS3(a)


if __name__ == "__main__":
    main()
