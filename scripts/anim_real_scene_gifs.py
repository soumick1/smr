#!/usr/bin/env python3
"""Scaffold fields on a REAL sequence (7-Scenes), with the loop closure shown (presentation GIFs, synchronised).

    python scripts/anim_real_scene_gifs.py --gt data/gt/7scenes_pumpkin_seq01.npz \
        --report outputs/reports/fig_7scenes_pumpkin_seq01.json --out outputs/anim_pumpkin --dpi 300
    python scripts/anim_real_scene_gifs.py ... --kf-range 0 150 --frames-per-kf 3          # stop after the first closure, slower

What it shows, one frame clock for every GIF (constant 50 ms per frame):
    ring_yaw / ring_pitch / ring_roll   the orientation rings following the camera's ground-truth rotation
    module1 / module2 / module3         torus + height ring following the ground-truth position (scene units = metres / --unit)
    camera                              the real RGB keyframe the camera sees, and a top-down map of the trajectory so far
Loop closure staging (from the SMR run report's accepted closures): once the camera has passed the historical frame that a
later window will retrieve, a hollow marker "stored address" appears on every field at that frame's phases and on the
map at its position; when the camera comes back, the bump lands on the marker; at the window where the closure is
accepted, the marker turns green, the camera panel shows the remembered view next to the current one, and the closure
edge is drawn on the map. Fields are the repo's Amari attractors (smr.dynamics.fields), driven by the trajectory's
angular / translational velocity in phase units with a weak anchoring term (as pose anchoring in the pipeline).
Outputs go to --out; with --frames START END only that range is rendered to PNGs (dynamics still run from frame 0), and
--assemble builds the GIFs from the PNGs. timeline.json records per-frame keyframe, pose and decoded phases.
"""
from __future__ import annotations

import argparse, json, math, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from smr.dynamics.fields import Ring, Torus2D  # noqa: E402
import anim_scaffold_gifs as A  # noqa: E402  (same directory: drawing helpers, Tracked, write_gif, pad_to)

PERIODS = (2.4, 3.2, 4.0)
GREEN = "#2a9d4b"


# ------------------------------------------------------------------ data
def keyframe_indices(n, stride):
    return np.arange(0, n, stride)


def make_chunks(n, W, O):
    step = W - O; starts = list(range(0, max(1, n - O), step))
    chunks = [np.arange(s, min(s + W, n)) for s in starts]
    if len(chunks) > 1 and len(chunks[-1]) <= O:
        chunks.pop()
    return chunks


def closures_from_report(report, chunks, row="smr"):
    out = []
    events = report.get("events", {})
    ev_list = (events.get(row) or events.get("smr") or next(iter(events.values()), [])) if isinstance(events, dict) else events
    for ev in ev_list or []:
        loop = ev.get("loop") if isinstance(ev, dict) else None
        if not loop or not loop.get("accepted") or "revoked_at" in loop:
            continue
        k = int(ev.get("chunk", -1))
        if k < 0 or k >= len(chunks):
            continue
        anchor = None
        for s in (ev.get("sites") or loop.get("sites") or []):
            if isinstance(s, dict) and s.get("ok", True):
                for key in ("view", "anchor", "frame", "g"):
                    if key in s:
                        anchor = int(s[key]); break
            if anchor is not None:
                break
        if anchor is None and "anchor_chunk" in loop and 0 <= int(loop["anchor_chunk"]) < len(chunks):
            ac = chunks[int(loop["anchor_chunk"])]; anchor = int(ac[len(ac) // 2])
        if anchor is not None:
            out.append(dict(window=k, anchor_kf=anchor, accept_kf=int(chunks[k][-1]), window_kfs=[int(x) for x in chunks[k]]))
    return out


def scene_frame(pos):
    """Right / up / forward world axes for a room: up = axis with the least positional spread (rooms are flat),
    sign chosen so that most cameras look slightly downward... we simply keep the GT's sign."""
    spread = np.ptp(pos, axis=0)
    up = int(np.argmin(spread)); others = [i for i in range(3) if i != up]
    return up, others


def euler_from_R(R, up, others):
    """yaw (heading of the optical axis in the horizontal plane), pitch (elevation of the optical axis), roll
    (rotation about the optical axis), from a camera-to-world rotation with the camera looking along its +z."""
    fwd = R[:, 2]; right = R[:, 0]
    h = fwd.copy(); h[up] = 0.0
    yaw = math.atan2(h[others[0]], h[others[1]])
    pitch = math.atan2(fwd[up], np.linalg.norm(h))
    # roll: angle between the camera's right axis and the horizontal plane
    roll = math.atan2(right[up], np.linalg.norm(np.delete(right, up)))
    return yaw, pitch, roll


def unwrap_series(a):
    return np.unwrap(np.asarray(a))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--report", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=5); ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--kf-range", nargs=2, type=int, default=None, help="keyframe range to animate (default: all)")
    ap.add_argument("--frames-per-kf", type=int, default=3, help="animation frames per keyframe (3 at 20 fps = 6.7 keyframes/s)")
    ap.add_argument("--fps", type=int, default=20); ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--unit", type=float, default=1.5, help="metres per scaffold unit (median depth)")
    ap.add_argument("--closure-index", type=int, default=None, help="stage only this accepted closure (default: all)")
    ap.add_argument("--only", nargs="*", default=None, help="subset: rings modules camera")
    ap.add_argument("--frames", nargs=2, type=int, default=None); ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.assemble:
        A.assemble(out, a.fps); return

    z = np.load(a.gt, allow_pickle=True)
    poses_all = np.asarray(z["poses"], float); paths_all = [str(p) for p in z["image_paths"]] if "image_paths" in z else None
    key = keyframe_indices(len(poses_all), a.stride)
    poses = poses_all[key]; paths = [paths_all[i] for i in key] if paths_all else None
    chunks = make_chunks(len(key), a.chunk, a.overlap)
    report = json.load(open(a.report))
    closures = closures_from_report(report, chunks)
    if a.closure_index is not None:
        closures = closures[a.closure_index:a.closure_index + 1]
    print(f"{len(key)} keyframes, {len(chunks)} windows, {len(closures)} accepted closures: " +
          ", ".join(f"window {c['window']} (kf {c['window_kfs'][0]}-{c['window_kfs'][-1]}) <- kf {c['anchor_kf']}" for c in closures))

    pos_m = poses[:, :3, 3]; up, others = scene_frame(pos_m)
    eul = np.array([euler_from_R(P[:3, :3], up, others) for P in poses])
    yaw, pitch, roll = (unwrap_series(eul[:, i]) for i in range(3))
    yaw -= yaw[0]; roll -= roll[0]                                     # start the rings at 0
    pos_u = pos_m / a.unit
    kf0, kf1 = a.kf_range if a.kf_range else (0, len(key))
    kf1 = min(kf1, len(key))
    fpk = a.frames_per_kf
    n = (kf1 - kf0 - 1) * fpk + 1
    kf_t = kf0 + np.arange(n) / fpk                                     # fractional keyframe index per animation frame
    interp = lambda s: np.interp(kf_t, np.arange(len(key)), s)
    yaw_t, pitch_t, roll_t = interp(yaw), interp(pitch), interp(roll)
    pos_t = np.stack([interp(pos_u[:, i]) for i in range(3)], 1)
    print(f"{n} frames at {a.fps} fps = {n / a.fps:.1f} s; yaw range {np.degrees(yaw_t.min()):+.0f}..{np.degrees(yaw_t.max()):+.0f} deg, "
          f"path length {np.sum(np.linalg.norm(np.diff(pos_m[kf0:kf1], axis=0), axis=1)):.1f} m")

    hx, hz = others                                                     # horizontal axes
    def torus_target(p):
        return [((2 * np.pi / lam) * p[hx] + off[0], (2 * np.pi / lam) * p[hz] + off[1]) for lam, off in zip(PERIODS, OFF_XY)]
    def height_target(p):
        return [(2 * np.pi / lam) * p[up] + OFF_Z for lam in PERIODS]
    OFF_XY = [(-1.9, -0.5)] * 3; OFF_Z = 1.2

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

    # ---- loop-closure staging: stored-address phases of each anchor keyframe, and when the marker appears / turns green
    stage = []
    for c in closures:
        j = c["anchor_kf"]
        stage.append(dict(anchor_kf=j, accept_kf=c["accept_kf"], window=c["window"],
                          yaw=yaw[j], pitch=pitch[j], roll=roll[j], tor=torus_target(pos_u[j]), hei=height_target(pos_u[j]), pos=pos_m[j]))
    def active_stages(kf):
        """stages whose anchor has been passed; state 'stored' before acceptance, 'closed' after."""
        res = []
        for s in stage:
            if kf >= s["anchor_kf"]:
                res.append((s, "closed" if kf >= s["accept_kf"] else "stored"))
        return res

    screen = {"yaw": lambda th: np.pi / 2 - th, "pitch": lambda th: th, "roll": lambda th: np.pi / 2 - th}
    ticks = {"yaw": [(0, "start"), (90, "right"), (180, "180°"), (270, "left")], "pitch": [(0, "level"), (90, "up"), (180, ""), (270, "down")],
             "roll": [(0, "start"), (90, "90°"), (180, "180°"), (270, "270°")]}
    titles = {"yaw": "Yaw ring", "pitch": "Pitch ring", "roll": "Roll ring"}
    want = set(a.only) if a.only else {"rings", "modules", "camera"}
    lo, hi = (a.frames if a.frames else (0, n)); fdir = out / "frames"; fdir.mkdir(exist_ok=True)
    frames = {k: [] for k in ("ring_yaw", "ring_pitch", "ring_roll", "module1", "module2", "module3", "camera")}
    def keep(name, arr, i):
        if a.frames:
            Image.fromarray(arr).save(fdir / f"{name}_{i:04d}.png")
        else:
            frames[name].append(arr)
    img_cache = {}
    def load_img(kf):
        if paths is None:
            return None
        if kf not in img_cache:
            im = Image.open(paths[kf]).convert("RGB"); im.thumbnail((640, 640)); img_cache[kf] = im
        return img_cache[kf]

    def marker_on_ring(ax, angle, screen_of, state, r0=1.0):
        sa = screen_of(angle); col = GREEN if state == "closed" else "#7a7a7a"
        ax.scatter([r0 * np.cos(sa)], [r0 * np.sin(sa)], s=170, facecolor="none" if state == "stored" else GREEN, edgecolor=col, linewidth=2.2, zorder=6)

    rec = []
    for i in range(n):
        kf = kf_t[i]; kfi = int(round(kf))
        tor_t = torus_target(pos_t[i]); hei_t = height_target(pos_t[i])
        rings["yaw"].advance(yaw_t[i]); rings["pitch"].advance(pitch_t[i]); rings["roll"].advance(roll_t[i])
        for m in range(3):
            tori[m].advance(tor_t[m]); heights[m].advance(hei_t[m])
        st = active_stages(kf)
        closed_now = [s for s, state in st if state == "closed" and kf < s["accept_kf"] + 2.5]      # highlight ~2.5 keyframes after acceptance
        rec.append(dict(frame=i, kf=float(kf), yaw_deg=float(np.degrees(yaw_t[i])), pitch_deg=float(np.degrees(pitch_t[i])), pos_m=[float(v) for v in pos_t[i] * a.unit],
                        ring_decoded={k: float(r.decoded()[0]) for k, r in rings.items()}, torus_decoded=[[float(v) for v in t.decoded()] for t in tori]))
        if not (lo <= i < hi):
            continue
        label = f"keyframe {kfi}" + ("   LOOP CLOSURE ACCEPTED" if closed_now else "")
        if "rings" in want:
            for k, r in rings.items():
                fig = A.new_fig(3.6, 3.6, a.dpi); ax = fig.add_axes([0, 0, 1, 1])
                val = {"yaw": yaw_t[i], "pitch": pitch_t[i], "roll": roll_t[i]}[k]
                shown = (np.degrees(val) + 180.0) % 360.0 - 180.0
                A.draw_ring(ax, r.activity(), r.decoded()[0], screen[k], f"{titles[k]}  (256 units)", f"{k} = {shown:+.1f}°", ticks[k])
                for s, state in st:
                    marker_on_ring(ax, s[k], screen[k], state)
                ax.set_ylim(-2.55, 2.3)
                ax.text(0, -2.38, label, ha="center", va="center", fontsize=9.5, color=GREEN if closed_now else "#777777")
                keep(f"ring_{k}", A.frame_of(fig), i)
        if "modules" in want:
            for m in range(3):
                fig = A.new_fig(5.2, 3.6, a.dpi)
                ax1 = fig.add_axes([0.0, 0.02, 0.66, 0.98], projection="3d", facecolor="white")
                A.draw_torus(ax1, tori[m].activity(), tori[m].decoded(), f"Module {m + 1}: torus  (period {PERIODS[m]} units)", OFF_XY[m])
                X, Y, Z, R, rr = A.torus_xyz(48)
                for s, state in st:
                    px, py = s["tor"][m]
                    ax1.scatter([(R + rr * np.cos(py)) * np.cos(px)], [(R + rr * np.cos(py)) * np.sin(px)], [rr * np.sin(py)], s=170,
                                facecolor="none" if state == "stored" else GREEN, edgecolor=GREEN if state == "closed" else "#7a7a7a", linewidth=2.2, depthshade=False, zorder=12)
                ax2 = fig.add_axes([0.66, 0.08, 0.34, 0.84])
                A.draw_ring(ax2, heights[m].activity(), heights[m].decoded()[0], lambda th: np.pi / 2 - th, "height ring", f"height = {pos_t[i][up] * a.unit:+.2f} m",
                            [(0, ""), (90, ""), (180, ""), (270, "")])
                for s, state in st:
                    marker_on_ring(ax2, s["hei"][m], lambda th: np.pi / 2 - th, state)
                fig.text(0.33, 0.045, f"x = {pos_t[i][hx] * a.unit:+.2f} m    z = {pos_t[i][hz] * a.unit:+.2f} m    keyframe {kfi}", ha="center", fontsize=10.5, color=A.INK)
                if closed_now:
                    fig.text(0.83, 0.045, "loop closure accepted", ha="center", fontsize=10.5, color=GREEN, weight="bold")
                keep(f"module{m + 1}", A.frame_of(fig), i)
        if "camera" in want:
            fig = A.new_fig(7.2, 3.6, a.dpi)
            axi = fig.add_axes([0.02, 0.08, 0.46, 0.80]); axi.set_axis_off()
            im = load_img(kfi)
            if im is not None:
                axi.imshow(im)
            axi.set_title(f"what the camera sees  (keyframe {kfi})", fontsize=12, color=A.INK, weight="bold")
            if closed_now:
                s = closed_now[0]; imr = load_img(s["anchor_kf"])
                if imr is not None:
                    axr = fig.add_axes([0.30, 0.10, 0.17, 0.30]); axr.set_axis_off(); axr.imshow(imr)
                    for sp in axr.spines.values():
                        sp.set_visible(True)
                    axr.set_title(f"remembered view (kf {s['anchor_kf']})", fontsize=8, color=GREEN)
            axm = fig.add_axes([0.55, 0.10, 0.43, 0.78]); axm.set_aspect("equal")
            axm.plot(pos_m[:, hx], pos_m[:, hz], color="#e3e3e3", lw=1.2)
            axm.plot(pos_m[kf0:kfi + 1, hx], pos_m[kf0:kfi + 1, hz], color=A.INK, lw=1.4)
            for s, state in st:
                axm.scatter([s["pos"][hx]], [s["pos"][hz]], s=110, facecolor="none" if state == "stored" else GREEN, edgecolor=GREEN if state == "closed" else "#7a7a7a", linewidth=2, zorder=5)
                if state == "closed":
                    axm.plot([s["pos"][hx], pos_m[min(kfi, len(key) - 1), hx]], [s["pos"][hz], pos_m[min(kfi, len(key) - 1), hz]], color=GREEN, lw=1.6, ls="--")
            fwd = poses[kfi][:3, 2]; axm.scatter([pos_m[kfi, hx]], [pos_m[kfi, hz]], s=60, color=A.ACCENT, zorder=6)
            axm.annotate("", xy=(pos_m[kfi, hx] + 0.25 * fwd[hx], pos_m[kfi, hz] + 0.25 * fwd[hz]), xytext=(pos_m[kfi, hx], pos_m[kfi, hz]),
                         arrowprops=dict(arrowstyle="->", color=A.ACCENT, lw=1.8))
            axm.set_xlabel("x (m)", fontsize=9); axm.set_ylabel("z (m)", fontsize=9); axm.tick_params(labelsize=8)
            for sp in axm.spines.values():
                sp.set_color(A.GREY)
            axm.set_title("trajectory (top view)", fontsize=12, color=A.INK, weight="bold")
            fig.text(0.5, 0.955, label if closed_now else f"7-Scenes {pathlib.Path(a.gt).stem.replace('7scenes_', '').replace('_', ' ')}   keyframe {kfi}",
                     ha="center", fontsize=12.5, color=GREEN if closed_now else A.INK, weight="bold")
            keep("camera", A.frame_of(fig), i)
        if i % 25 == 0:
            print(f"  frame {i}/{n}", flush=True)
    json.dump(dict(fps=a.fps, n_frames=n, stride=a.stride, unit=a.unit, closures=closures, frames=rec), open(out / "timeline.json", "w"))
    if a.frames:
        print(f"rendered frames {lo}..{hi - 1}; run --assemble when all segments are done"); return
    for k, fr in frames.items():
        if fr:
            A.write_gif(fr, out / f"{k}.gif", a.fps)
    print("done ->", out)


if __name__ == "__main__":
    main()
