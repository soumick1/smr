#!/usr/bin/env python3
"""One composed demo: real footage, the orientation rings and the spatial modules in a single synchronized frame.

    python scripts/demo_video.py --gt data/gt/7scenes_office_seq01.npz --report outputs/reports/fig_7scenes_office_seq01.json \
        --out docs/assets/demo --kf-range 0 200 --frames-per-kf 2 --fps 20 --width 1280

Layout (one 16:9 frame, paper terminology):
    left   : the camera's view (the real keyframe) with a small top-down trajectory inset; accepted revisits drawn as closure edges
    middle : the three orientation rings (yaw, pitch, roll), 256 units each
    right  : the three spatial modules (periods 2.4, 3.2, 4.0 scene units), each a torus with its height ring
Every panel is drawn for the same frame of the same clock, so the GIF is synchronized by construction. The fields are the
repository's Amari attractors driven by the trajectory (see scripts/anim_real_scene_gifs.py); accepted revisits come
from the run report: a hollow marker appears on every field at the stored frame's state, and turns green when the
revisit is accepted and the trajectory is revised.
Writes <out>.gif (every frame, constant delay) and <out>.mp4 if ffmpeg is available (smaller; use the mp4 on the project
page and the GIF in the README). --frames START END renders only that range (dynamics still run from frame 0) and
--assemble builds the outputs from the PNGs, for machines with short job limits.
"""
from __future__ import annotations

import argparse, json, pathlib, shutil, subprocess, sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from PIL import Image  # noqa: E402
from smr.dynamics.fields import Ring, Torus2D  # noqa: E402
from smr.stitch.chunks import keyframe_indices, make_chunks  # noqa: E402
import anim_scaffold_gifs as A  # noqa: E402
import anim_real_scene_gifs as R  # noqa: E402

PERIODS = (2.4, 3.2, 4.0)
GREEN = "#2a9d4b"
INK = "#1c1b19"


def ring_panel(ax, act, decoded, screen_of, title, readout, ticks, markers):
    A.draw_ring(ax, act, decoded, screen_of, "", "", ticks)
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3)
    for t in ax.texts:                                   # tick labels drawn by draw_ring
        t.set_fontsize(8)
    if title:
        ax.text(-2.2, 2.05, title, fontsize=11, color=INK, weight="bold", ha="left", va="center")
    if readout:
        ax.text(2.2, 2.05, readout, fontsize=11, color=INK, ha="right", va="center")
    for angle, state in markers:
        sa = screen_of(angle); col = GREEN if state == "closed" else "#7a7a7a"
        ax.scatter([np.cos(sa)], [np.sin(sa)], s=150, facecolor="none" if state == "stored" else GREEN, edgecolor=col, linewidth=2.0, zorder=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--report", required=True); ap.add_argument("--out", required=True, help="output prefix (no extension)")
    ap.add_argument("--stride", type=int, default=5); ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--kf-range", nargs=2, type=int, default=None); ap.add_argument("--frames-per-kf", type=int, default=2)
    ap.add_argument("--fps", type=int, default=20); ap.add_argument("--width", type=int, default=1280, help="frame width in pixels (16:9)")
    ap.add_argument("--unit", type=float, default=1.5); ap.add_argument("--title", default="SMoRe on a 7-Scenes sequence")
    ap.add_argument("--frames", nargs=2, type=int, default=None); ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fdir = pathlib.Path(str(out) + "_frames"); fdir.mkdir(exist_ok=True)
    if a.assemble:
        assemble(out, fdir, a.fps); return

    # ---- data and trajectory (same conventions as anim_real_scene_gifs)
    z = np.load(a.gt, allow_pickle=True)
    poses_all = np.asarray(z["poses"], float); paths_all = [str(p) for p in z["image_paths"]]
    key = keyframe_indices(len(poses_all), a.stride); poses = poses_all[key]; paths = [paths_all[i] for i in key]
    chunks = make_chunks(len(key), a.chunk, a.overlap)
    closures = R.closures_from_report(json.load(open(a.report)), chunks)
    print(f"{len(key)} keyframes, {len(chunks)} windows, {len(closures)} accepted revisits")
    pos_m = poses[:, :3, 3]; up, others = R.scene_frame(pos_m); hx, hz = others
    eul = np.array([R.euler_from_R(P[:3, :3], up, others) for P in poses])
    yaw, pitch, roll = (R.unwrap_series(eul[:, i]) for i in range(3)); yaw -= yaw[0]; roll -= roll[0]
    pos_u = pos_m / a.unit
    kf0, kf1 = a.kf_range if a.kf_range else (0, len(key)); kf1 = min(kf1, len(key))
    n = (kf1 - kf0 - 1) * a.frames_per_kf + 1
    kf_t = kf0 + np.arange(n) / a.frames_per_kf
    interp = lambda s: np.interp(kf_t, np.arange(len(key)), s)
    yaw_t, pitch_t, roll_t = interp(yaw), interp(pitch), interp(roll)
    pos_t = np.stack([interp(pos_u[:, i]) for i in range(3)], 1)
    OFF_XY = [(-1.9, -0.5)] * 3; OFF_Z = 1.2
    torus_target = lambda p: [((2 * np.pi / lam) * p[hx] + off[0], (2 * np.pi / lam) * p[hz] + off[1]) for lam, off in zip(PERIODS, OFF_XY)]
    height_target = lambda p: [(2 * np.pi / lam) * p[up] + OFF_Z for lam in PERIODS]

    # ---- fields
    rings = {k: A.Tracked(Ring(N=256, seed=i), 1, steps=40) for i, k in enumerate(("yaw", "pitch", "roll"))}
    tori = [A.Tracked(Torus2D(N=48, seed=10 + m), 2, steps=60) for m in range(3)]
    heights = [A.Tracked(Ring(N=48, seed=20 + m), 1, steps=40) for m in range(3)]
    for f in list(rings.values()) + tori + heights:
        f.f.calibrate()
    for k, v in (("yaw", yaw_t[0]), ("pitch", pitch_t[0]), ("roll", roll_t[0])):
        rings[k].place(v)
    for m in range(3):
        tori[m].place(torus_target(pos_t[0])[m]); heights[m].place(height_target(pos_t[0])[m])
    stage = [dict(anchor_kf=c["anchor_kf"], accept_kf=c["accept_kf"], yaw=yaw[c["anchor_kf"]], pitch=pitch[c["anchor_kf"]], roll=roll[c["anchor_kf"]],
                  tor=torus_target(pos_u[c["anchor_kf"]]), hei=height_target(pos_u[c["anchor_kf"]]), pos=pos_m[c["anchor_kf"]]) for c in closures]
    active = lambda kf: [(s, "closed" if kf >= s["accept_kf"] else "stored") for s in stage if kf >= s["anchor_kf"]]
    screen = {"yaw": lambda th: np.pi / 2 - th, "pitch": lambda th: th, "roll": lambda th: np.pi / 2 - th}
    ticks = {"yaw": [(0, "start"), (90, "right"), (180, ""), (270, "left")], "pitch": [(0, "level"), (90, "up"), (180, ""), (270, "down")], "roll": [(0, "start"), (90, ""), (180, ""), (270, "")]}
    names = {"yaw": "Yaw ring", "pitch": "Pitch ring", "roll": "Roll ring"}
    img_cache = {}
    def load_img(kf):
        if kf not in img_cache:
            im = Image.open(paths[kf]).convert("RGB"); im.thumbnail((800, 800)); img_cache[kf] = im
        return img_cache[kf]

    W = a.width; H = int(W * 9 / 16); dpi = 100
    lo, hi = (a.frames if a.frames else (0, n))
    X, Y, Z, Rr, rr = A.torus_xyz(48)
    for i in range(n):
        kf = kf_t[i]; kfi = int(round(kf))
        tt, ht = torus_target(pos_t[i]), height_target(pos_t[i])
        rings["yaw"].advance(yaw_t[i]); rings["pitch"].advance(pitch_t[i]); rings["roll"].advance(roll_t[i])
        for m in range(3):
            tori[m].advance(tt[m]); heights[m].advance(ht[m])
        if not (lo <= i < hi):
            continue
        st = active(kf); just = [s for s, state in st if state == "closed" and kf < s["accept_kf"] + 3]
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
        # header
        fig.text(0.012, 0.955, a.title, fontsize=15, color=INK, weight="bold", va="center")
        fig.text(0.988, 0.955, "revisit accepted, trajectory revised" if just else f"keyframe {kfi}", fontsize=12.5, color=GREEN if just else "#777777", ha="right", va="center", weight="bold" if just else "normal")
        # left: camera view + trajectory inset
        fig.text(0.212, 0.905, "Camera view", fontsize=12.5, color=INK, ha="center", va="center")
        axi = fig.add_axes([0.012, 0.09, 0.40, 0.79]); axi.set_axis_off(); axi.imshow(load_img(kfi), aspect="auto")
        axm = fig.add_axes([0.285, 0.105, 0.12, 0.26]); axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([]); axm.patch.set_alpha(0.85)
        axm.plot(pos_m[:, hx], pos_m[:, hz], color="#dddddd", lw=1.0); axm.plot(pos_m[kf0:kfi + 1, hx], pos_m[kf0:kfi + 1, hz], color=INK, lw=1.2)
        for s, state in st:
            axm.scatter([s["pos"][hx]], [s["pos"][hz]], s=55, facecolor="none" if state == "stored" else GREEN, edgecolor=GREEN if state == "closed" else "#7a7a7a", linewidth=1.6, zorder=5)
            if state == "closed":
                axm.plot([s["pos"][hx], pos_m[min(kfi, len(key) - 1), hx]], [s["pos"][hz], pos_m[min(kfi, len(key) - 1), hz]], color=GREEN, lw=1.3, ls="--")
        axm.scatter([pos_m[kfi, hx]], [pos_m[kfi, hz]], s=40, color=A.ACCENT, zorder=6)
        for sp in axm.spines.values():
            sp.set_color("#bbbbbb")
        axm.set_title("trajectory", fontsize=8.5, color="#555555", pad=2)
        # middle: orientation rings
        fig.text(0.545, 0.905, "Orientation rings", fontsize=12.5, color=INK, ha="center", va="center")
        for r_i, k in enumerate(("yaw", "pitch", "roll")):
            ax = fig.add_axes([0.435, 0.60 - r_i * 0.27, 0.22, 0.265])
            val = {"yaw": yaw_t[i], "pitch": pitch_t[i], "roll": roll_t[i]}[k]; shown = (np.degrees(val) + 180) % 360 - 180
            ring_panel(ax, rings[k].activity(), rings[k].decoded()[0], screen[k], names[k], f"{shown:+.0f}°", ticks[k], [(s[k], state) for s, state in st])
        # right: spatial modules (torus + height ring)
        fig.text(0.825, 0.905, "Spatial modules", fontsize=12.5, color=INK, ha="center", va="center")
        for m in range(3):
            ax1 = fig.add_axes([0.655, 0.585 - m * 0.27, 0.24, 0.285], projection="3d", facecolor="white")
            A.draw_torus(ax1, tori[m].activity(), tori[m].decoded(), "", OFF_XY[m])
            fig.text(0.66, 0.855 - m * 0.27, f"Module {m + 1}  (period {PERIODS[m]})", fontsize=11, color=INK, weight="bold", ha="left", va="center")
            for s, state in st:
                px, py = s["tor"][m]
                ax1.scatter([(Rr + rr * np.cos(py)) * np.cos(px)], [(Rr + rr * np.cos(py)) * np.sin(px)], [rr * np.sin(py)], s=150,
                            facecolor="none" if state == "stored" else GREEN, edgecolor=GREEN if state == "closed" else "#7a7a7a", linewidth=2.0, depthshade=False, zorder=12)
            ax2 = fig.add_axes([0.895, 0.615 - m * 0.27, 0.1, 0.23])
            ring_panel(ax2, heights[m].activity(), heights[m].decoded()[0], lambda th: np.pi / 2 - th, "", "", [(0, ""), (90, ""), (180, ""), (270, "")],
                       [(s["hei"][m], state) for s, state in st])
            ax2.text(0, 2.05, "height", fontsize=9, color="#555555", ha="center", va="center")
        fig.text(0.5, 0.03, "hollow marker: stored frame   ·   green: revisit accepted, earlier windows revised", fontsize=10.5, color="#666666", ha="center")
        arr = A.frame_of(fig)
        Image.fromarray(arr).save(fdir / f"f_{i:05d}.png")
        if i % 25 == 0:
            print(f"  frame {i}/{n}", flush=True)
    if a.frames:
        print(f"rendered {lo}..{hi - 1}; run --assemble when all ranges are done"); return
    assemble(out, fdir, a.fps)


def assemble(out, fdir, fps):
    files = sorted(fdir.glob("f_*.png"))
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    A.write_gif(frames, pathlib.Path(str(out) + ".gif"), fps)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(fdir / "f_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "22", "-movflags", "+faststart", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out) + ".mp4"], check=False)
        print("wrote", str(out) + ".mp4")
    print("wrote", str(out) + f".gif ({len(frames)} frames, {len(frames) / fps:.1f} s)")


if __name__ == "__main__":
    main()
