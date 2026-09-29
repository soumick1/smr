#!/usr/bin/env python3
"""Synchronised GIFs of the scaffold fields driven by one camera motion (presentation figures).

    python scripts/anim_scaffold_gifs.py --out outputs/anim                # all GIFs, 20 fps, ~12 s each
    python scripts/anim_scaffold_gifs.py --out outputs/anim --no-translation --overview

Timeline (one clock for every GIF, so they stay in sync when started together):
    hold | yaw 0 -> +30 deg (turn right) | hold | pitch 0 -> +15 deg (tilt up) | hold | translate forward + up | hold
The translation phase exists so that the spatial modules move too (a pure rotation leaves position, and hence the tori
and height rings, unchanged); drop it with --no-translation.

Fields are the repo's Amari attractors (smr.dynamics.fields.Ring / Torus2D: J0 + J1 cos kernel, ReLU rate, tau = 1,
dt = 0.05, velocity-modulated drive), calibrated so a commanded rate is a phase velocity, and driven each frame by the
camera's angular / translational velocity in phase units (2 pi / period for the modules). A weak proportional term keeps
the decoded bump on the camera's true phase, as pose anchoring does in the pipeline. Outputs:
    ring_yaw.gif  ring_pitch.gif  ring_roll.gif            (i)   256-unit orientation rings
    module1.gif   module2.gif     module3.gif              (ii)  48x48 torus + 48-unit height ring per module (periods 2.4 / 3.2 / 4.0)
    camera.gif                                             (iii) world view with the camera frustum + what the camera sees
    overview.gif (with --overview)                         montage of everything, half size
    timeline.json                                          per-frame yaw / pitch / roll / position / phases
"""
from __future__ import annotations

import argparse, json, math, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from smr.dynamics.fields import Ring, Torus2D  # noqa: E402

PERIODS = (2.4, 3.2, 4.0)
ACCENT = "#d9541e"          # bump colour (deep orange)
INK = "#333333"
GREY = "#bfbfbf"
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("bump", ["#f1f1f1", "#fbd9bd", "#f7a25f", ACCENT, "#7a1e00"])


# ------------------------------------------------------------------ timeline
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def build_timeline(fps, yaw_deg, pitch_deg, forward, up, translate, hold=0.8, t_yaw=3.0, t_pitch=2.5, t_move=3.0, t_end=1.2):
    phases = [("hold", hold), ("yaw", t_yaw), ("hold", 0.7), ("pitch", t_pitch), ("hold", 0.7)]
    if translate:
        phases += [("move", t_move)]
    phases += [("hold", t_end)]
    n = int(round(sum(d for _, d in phases) * fps))
    t = np.arange(n) / fps
    yaw = np.zeros(n); pitch = np.zeros(n); dist = np.zeros(n); label = [""] * n
    t0 = 0.0
    for name, dur in phases:
        sel = (t >= t0) & (t < t0 + dur)
        s = smoothstep((t[sel] - t0) / dur)
        if name == "yaw":
            yaw[sel] = s * yaw_deg
        elif name == "pitch":
            pitch[sel] = s * pitch_deg
        elif name == "move":
            dist[sel] = s
        after = t >= t0 + dur
        if name == "yaw":
            yaw[after] = yaw_deg
        if name == "pitch":
            pitch[after] = pitch_deg
        if name == "move":
            dist[after] = 1.0
        for i in np.where(sel)[0]:
            label[i] = {"yaw": "turning right", "pitch": "tilting up", "move": "moving forward and up", "hold": ""}[name]
        t0 += dur
    yaw_r, pitch_r = np.radians(yaw), np.radians(pitch)
    # camera pose: world x right, y up, z forward; yaw right = rotation about +y, pitch up = rotation about camera x
    R = np.zeros((n, 3, 3)); pos = np.zeros((n, 3))
    for i in range(n):
        cy, sy, cp, sp = math.cos(yaw_r[i]), math.sin(yaw_r[i]), math.cos(pitch_r[i]), math.sin(pitch_r[i])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])           # pitch up tilts forward toward +y
        R[i] = Ry @ Rx
        fwd_h = np.array([sy, 0.0, cy])                                  # horizontal forward after the yaw
        pos[i] = dist[i] * (forward * fwd_h + np.array([0.0, up, 0.0]))
    return dict(n=n, fps=fps, t=t, yaw=yaw, pitch=pitch, roll=np.zeros(n), R=R, pos=pos, label=label)


# ------------------------------------------------------------------ fields
class Tracked:
    """A field plus the phase it must follow; each frame advances the dynamics by the phase increment (velocity drive)
    with a weak correction toward the target, in K Euler steps."""

    def __init__(self, field, ndim, steps=20, gain=0.35, vmax=0.12):
        self.f, self.ndim, self.K, self.gain, self.vmax = field, ndim, steps, gain, vmax

    @staticmethod
    def wrap(x):
        return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi

    def place(self, target):
        self.f.ignite(target if self.ndim == 2 else float(target)); self.f.settle(250)
        self.prev = np.array(target, float).reshape(-1)

    def advance(self, target):
        target = np.array(target, float).reshape(-1)
        dphi = self.wrap(target - self.prev)
        cur = np.array(self.f.decode(), float).reshape(-1)
        err = self.wrap(target - cur)
        rate = (dphi + self.gain * err) / (self.K * self.f.dt)
        rate = np.clip(rate, -self.vmax, self.vmax)
        for _ in range(self.K):
            self.f.step(tuple(rate) if self.ndim == 2 else float(rate[0]))
        self.prev = target

    def activity(self):
        a = self.f.f(self.f.u) / self.f.A_pred
        return np.clip(a, 0, 1)

    def decoded(self):
        return np.array(self.f.decode(), float).reshape(-1)


# ------------------------------------------------------------------ drawing helpers
def frame_of(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return arr


def new_fig(w_in, h_in, dpi):
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi, facecolor="white")
    return fig


def draw_ring(ax, act, decoded, screen_of, title, readout, ticks, r0=1.0, lift=0.55):
    """Ring field: unit circle with the activity as a coloured flame; decoded angle as a marker."""
    N = len(act)
    ang = 2 * np.pi * np.arange(N) / N
    sa = screen_of(ang)
    r = r0 + lift * act
    ax.set_aspect("equal"); ax.set_axis_off(); ax.set_xlim(-2.15, 2.15); ax.set_ylim(-2.25, 2.25)
    ax.add_patch(plt.Circle((0, 0), r0, fill=False, lw=2.0, color=GREY, zorder=1))
    xs, ys = r * np.cos(sa), r * np.sin(sa)
    poly = np.concatenate([np.stack([xs, ys], 1), np.stack([r0 * np.cos(sa[::-1]), r0 * np.sin(sa[::-1])], 1)])
    ax.add_patch(plt.Polygon(poly, closed=True, color=ACCENT, alpha=0.18, lw=0, zorder=2))
    ax.scatter(xs, ys, c=act, cmap=CMAP, vmin=0, vmax=1, s=16, lw=0, zorder=3)
    for deg, lab in ticks:
        a = screen_of(np.radians(deg)); ax.plot([r0 * 0.94 * np.cos(a), r0 * 1.0 * np.cos(a)], [r0 * 0.94 * np.sin(a), r0 * 1.0 * np.sin(a)], color=GREY, lw=1.2, zorder=1)
        ax.text(1.72 * np.cos(a), 1.72 * np.sin(a), lab, ha="center", va="center", fontsize=9.5, color="#777777")
    a = screen_of(decoded)
    ax.plot([0, r0 * np.cos(a)], [0, r0 * np.sin(a)], color=INK, lw=1.4, zorder=4)
    ax.scatter([r0 * np.cos(a)], [r0 * np.sin(a)], s=70, color=INK, zorder=5, edgecolor="white", linewidth=1.5)
    ax.text(0, 2.02, title, ha="center", va="center", fontsize=12.5, color=INK, weight="bold")
    ax.text(0, -2.02, readout, ha="center", va="center", fontsize=12, color=INK)


def torus_xyz(N, R=1.0, r=0.42):
    u = 2 * np.pi * np.arange(N + 1) / N
    U, V = np.meshgrid(u, u, indexing="ij")                                # U along phi_x (major), V along phi_y (minor)
    X = (R + r * np.cos(V)) * np.cos(U); Y = (R + r * np.cos(V)) * np.sin(U); Z = r * np.sin(V)
    return X, Y, Z, R, r


def draw_torus(ax, act, decoded, title, offsets):
    N = act.shape[0]
    X, Y, Z, R, r = torus_xyz(N)
    A = np.pad(act, ((0, 1), (0, 1)), mode="wrap")
    # display offset so the bump starts on the visible front
    colors = CMAP(A[:-1, :-1])
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True,
                    lightsource=matplotlib.colors.LightSource(azdeg=225, altdeg=82), zorder=1)
    px, py = decoded
    ax.scatter([(R + r * np.cos(py)) * np.cos(px)], [(R + r * np.cos(py)) * np.sin(px)], [r * np.sin(py)], s=60, color=INK,
               edgecolor="white", linewidth=1.5, depthshade=False, zorder=10)
    ax.set_axis_off(); ax.view_init(elev=42, azim=-65)
    ax.set_box_aspect((1, 1, 0.42)); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_zlim(-0.55, 0.55)
    ax.text2D(0.5, 0.93, title, transform=ax.transAxes, ha="center", fontsize=12.5, color=INK, weight="bold")


# ------------------------------------------------------------------ camera scene
def scene_boxes():
    """(centre, size, colour) of a few boxes in front of the start pose; world x right, y up, z forward."""
    return [((-1.4, -0.35, 3.6), (0.8, 0.9, 0.8), "#4c72b0"), ((0.9, -0.5, 3.0), (0.7, 0.6, 0.7), "#55a868"),
            ((2.2, 0.15, 4.6), (0.9, 1.9, 0.9), "#c44e52"), ((-0.2, 0.9, 5.4), (0.6, 0.6, 0.6), "#8172b2"),
            ((3.4, -0.4, 6.2), (1.0, 0.8, 1.0), "#ccb974")]


def box_edges(c, s):
    cx, cy, cz = c; sx, sy, sz = [v / 2 for v in s]
    P = np.array([[cx + dx * sx, cy + dy * sy, cz + dz * sz] for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
    E = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return P, E


def box_faces(P):
    idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [P[i] for i in idx]


def draw_world(ax, R, pos, trail, hfov=60.0):
    ax.set_axis_off(); ax.view_init(elev=22, azim=-150)
    ax.set_xlim(-3.2, 4.0); ax.set_ylim(-1.5, 3.0); ax.set_zlim(-1.2, 7.6); ax.set_box_aspect((7.2, 4.5, 8.8))
    # note: matplotlib's z is up; we map world (x, y_up, z_fwd) -> plot (x, z_fwd, y_up)
    def P(v):
        v = np.asarray(v); return np.stack([v[..., 0], v[..., 2], v[..., 1]], -1)
    for gx in np.arange(-3, 4.01, 1.0):
        ax.plot(*P(np.array([[gx, -1.0, 0.0], [gx, -1.0, 7.5]])).T, color="#e6e6e6", lw=0.8)
    for gz in np.arange(0, 7.51, 1.0):
        ax.plot(*P(np.array([[-3, -1.0, gz], [4, -1.0, gz]])).T, color="#e6e6e6", lw=0.8)
    for c, s, col in scene_boxes():
        Pb, _ = box_edges(c, s)
        ax.add_collection3d(Poly3DCollection([P(f) for f in box_faces(Pb)], facecolor=col, edgecolor="white", lw=0.6, alpha=0.9))
    # frustum
    d = 0.9; half = d * math.tan(math.radians(hfov / 2)); corners_c = np.array([[-half, -half, d], [half, -half, d], [half, half, d], [-half, half, d]])
    corners_w = pos + corners_c @ R.T
    ax.add_collection3d(Poly3DCollection([P(corners_w)], facecolor="#9ecae1", edgecolor=INK, lw=1.0, alpha=0.35))
    for cw in corners_w:
        ax.plot(*P(np.array([pos, cw])).T, color=INK, lw=1.0)
    ax.scatter(*P(pos), s=45, color=INK, depthshade=False)
    fwd = pos + R @ np.array([0, 0, 1.6])
    ax.plot(*P(np.array([pos, fwd])).T, color=ACCENT, lw=2.2)
    if len(trail) > 1:
        ax.plot(*P(np.array(trail)).T, color=ACCENT, lw=1.2, ls="--", alpha=0.7)


def draw_view(ax, R, pos, hfov=60.0):
    ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GREY); sp.set_linewidth(1.5)
    f = 1.0 / math.tan(math.radians(hfov / 2))
    def proj(Pw):
        Pc = (np.asarray(Pw) - pos) @ R                       # world -> camera (R columns are camera axes in world)
        z = Pc[:, 2]; ok = z > 0.15
        x = f * Pc[:, 0] / np.maximum(z, 0.15); y = f * Pc[:, 1] / np.maximum(z, 0.15)
        return x, y, ok
    for gx in np.arange(-3, 4.01, 1.0):
        x, y, ok = proj(np.linspace([gx, -1.0, 0.2], [gx, -1.0, 7.5], 40))
        if ok.any():
            ax.plot(np.where(ok, x, np.nan), np.where(ok, y, np.nan), color="#e0e0e0", lw=0.8)
    for gz in np.arange(1, 7.51, 1.0):
        x, y, ok = proj(np.linspace([-3.0, -1.0, gz], [4.0, -1.0, gz], 40))
        if ok.any():
            ax.plot(np.where(ok, x, np.nan), np.where(ok, y, np.nan), color="#e0e0e0", lw=0.8)
    for c, s, col in sorted(scene_boxes(), key=lambda b: -np.linalg.norm(np.array(b[0]) - pos)):
        Pb, E = box_edges(c, s)
        x, y, ok = proj(Pb)
        if ok.all():
            for fpts in box_faces(Pb):
                fx, fy, _ = proj(fpts); ax.fill(fx, fy, color=col, alpha=0.85, lw=0)
            for i, j in E:
                ax.plot([x[i], x[j]], [y[i], y[j]], color="white", lw=0.8)
    ax.plot([-0.05, 0.05], [0, 0], color=INK, lw=1); ax.plot([0, 0], [-0.05, 0.05], color=INK, lw=1)
    ax.set_title("what the camera sees", fontsize=12.5, color=INK, weight="bold", pad=6)


# ------------------------------------------------------------------ gif writing
def write_gif(frames, path, fps):
    """Every frame kept, fixed duration (1000/fps ms), no frame merging: GIFs written with the same fps stay in sync."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    ims = []
    for i, f in enumerate(frames):
        arr = np.array(f.convert("RGB") if isinstance(f, Image.Image) else f, dtype=np.uint8, copy=True)
        arr[-1, -1] = (191, 191, 191) if i % 2 else (255, 255, 255)   # one corner pixel alternates white / the ring grey (a colour every frame contains, so it survives
        #                                                                 palette quantisation): consecutive frames are never identical, so PIL cannot merge them
        ims.append(Image.fromarray(arr).quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE))
    ims[0].save(str(path), save_all=True, append_images=ims[1:], duration=int(round(1000.0 / fps)), loop=0, optimize=False, disposal=1)
    print(f"  {path}  ({len(ims)} frames, {len(ims) / fps:.1f} s)")


def pad_to(a, w, h):
    H, W = a.shape[:2]
    out = np.full((h, w, 3), 255, np.uint8)
    out[(h - H) // 2:(h - H) // 2 + H, (w - W) // 2:(w - W) // 2 + W] = a
    return out


def assemble(out, fps):
    """GIFs from PNG frames written by --frames segments (all panels, plus the half-size overview)."""
    from PIL import Image
    fdir = out / "frames"
    names = sorted({p.name.rsplit("_", 1)[0] for p in fdir.glob("*.png")})
    files = {name: sorted(fdir.glob(f"{name}_*.png")) for name in names}
    for name in names:
        write_gif([Image.open(f).convert("RGB") for f in files[name]], out / f"{name}.gif", fps)
    need = ("ring_yaw", "ring_pitch", "ring_roll", "module1", "module2", "module3", "camera")
    if all(k in files for k in need):
        n = min(len(files[k]) for k in need); ov = []
        for i in range(n):
            load = lambda k: np.asarray(Image.open(files[k][i]).convert("RGB"))
            row1 = np.concatenate([load(f"ring_{k}") for k in ("yaw", "pitch", "roll")], 1)
            row2 = np.concatenate([load(f"module{m}") for m in (1, 2, 3)], 1)
            cam = load("camera"); w = max(row1.shape[1], row2.shape[1], cam.shape[1])
            full = np.concatenate([pad_to(row1, w, row1.shape[0]), pad_to(row2, w, row2.shape[0]), pad_to(cam, w, cam.shape[0])], 0)
            im = Image.fromarray(full); im = im.resize((int(im.width * 0.45), int(im.height * 0.45)), Image.LANCZOS)
            ov.append(im)
        write_gif(ov, out / "overview.gif", fps)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/anim"); ap.add_argument("--fps", type=int, default=20); ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--yaw-deg", type=float, default=30.0); ap.add_argument("--pitch-deg", type=float, default=15.0)
    ap.add_argument("--forward", type=float, default=0.8, help="translation along the view direction (scene units)"); ap.add_argument("--up", type=float, default=0.3)
    ap.add_argument("--no-translation", action="store_true"); ap.add_argument("--overview", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="<1 slows every phase down proportionally")
    ap.add_argument("--ring-N", type=int, default=256); ap.add_argument("--torus-N", type=int, default=48)
    ap.add_argument("--only", nargs="*", default=None, help="subset: rings modules camera")
    ap.add_argument("--frames", nargs=2, type=int, default=None, metavar=("START", "END"),
                    help="render only frames [START, END) to PNGs under --out/frames (dynamics still run from frame 0); then --assemble")
    ap.add_argument("--assemble", action="store_true", help="build the GIFs from --out/frames/*.png written by earlier --frames runs")
    a = ap.parse_args()
    if a.assemble:
        assemble(pathlib.Path(a.out), a.fps); return
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s = 1.0 / a.speed
    tl = build_timeline(a.fps, a.yaw_deg, a.pitch_deg, a.forward, a.up, not a.no_translation,
                        hold=0.8 * s, t_yaw=3.0 * s, t_pitch=2.5 * s, t_move=3.0 * s, t_end=1.2 * s)
    n = tl["n"]; print(f"{n} frames at {a.fps} fps = {n / a.fps:.1f} s per GIF")
    want = set(a.only) if a.only else {"rings", "modules", "camera"}

    # ---- fields: calibrate once, place at the start phases
    rings = {k: Tracked(Ring(N=a.ring_N, seed=i), 1, steps=20) for i, k in enumerate(("yaw", "pitch", "roll"))}
    for r in rings.values():
        r.f.calibrate()
    tori = [Tracked(Torus2D(N=a.torus_N, seed=10 + m), 2, steps=20) for m in range(3)]
    heights = [Tracked(Ring(N=a.torus_N, seed=20 + m), 1, steps=20) for m in range(3)]
    for t in tori + heights:
        t.f.calibrate()
    # phase offsets (the scaffold's scene-specific offsets) chosen so every bump starts on the visible side
    off_xy = [(-1.9, -0.5), (-1.9, -0.5), (-1.9, -0.5)]; off_z = [1.2, 1.2, 1.2]

    def targets(i):
        yaw, pitch, roll = np.radians(tl["yaw"][i]), np.radians(tl["pitch"][i]), 0.0
        x, y, z = tl["pos"][i]
        tor = [((2 * np.pi / lam) * x + off_xy[m][0], (2 * np.pi / lam) * z + off_xy[m][1]) for m, lam in enumerate(PERIODS)]
        hei = [(2 * np.pi / lam) * y + off_z[m] for m, lam in enumerate(PERIODS)]
        return dict(yaw=yaw, pitch=pitch, roll=roll), tor, hei

    ang0, tor0, hei0 = targets(0)
    for k, r in rings.items():
        r.place(ang0[k])
    for m in range(3):
        tori[m].place(tor0[m]); heights[m].place(hei0[m])

    # screen mappings for the rings: yaw 0 at top, right = clockwise; pitch 0 at right, up = counter-clockwise; roll like yaw
    screen = {"yaw": lambda th: np.pi / 2 - th, "pitch": lambda th: th, "roll": lambda th: np.pi / 2 - th}
    ticks = {"yaw": [(0, "0°"), (90, "right"), (180, "180°"), (270, "left")],
             "pitch": [(0, "level"), (90, "up"), (180, "180°"), (270, "down")],
             "roll": [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]}
    titles = {"yaw": "Yaw ring", "pitch": "Pitch ring", "roll": "Roll ring"}

    frames = {k: [] for k in ("ring_yaw", "ring_pitch", "ring_roll", "module1", "module2", "module3", "camera")}
    rec = []
    trail = []
    lo, hi = (a.frames if a.frames else (0, n))
    fdir = out / "frames"; fdir.mkdir(exist_ok=True)
    from PIL import Image
    def keep(name, arr, i):
        if a.frames:
            Image.fromarray(arr).save(fdir / f"{name}_{i:04d}.png")
        else:
            frames[name].append(arr)
    for i in range(n):
        ang, tor, hei = targets(i)
        for k, r in rings.items():
            r.advance(ang[k])
        for m in range(3):
            tori[m].advance(tor[m]); heights[m].advance(hei[m])
        trail.append(tl["pos"][i].copy())
        rec.append(dict(t=float(tl["t"][i]), yaw_deg=float(tl["yaw"][i]), pitch_deg=float(tl["pitch"][i]), pos=[float(v) for v in tl["pos"][i]],
                        ring_decoded_deg={k: float(np.degrees(r.decoded()[0])) for k, r in rings.items()},
                        torus_decoded=[[float(v) for v in t.decoded()] for t in tori], height_decoded=[float(h.decoded()[0]) for h in heights]))
        if not (lo <= i < hi):
            continue
        if "rings" in want:
            for k, r in rings.items():
                fig = new_fig(3.6, 3.6, a.dpi); ax = fig.add_axes([0, 0, 1, 1])
                val = {"yaw": tl["yaw"][i], "pitch": tl["pitch"][i], "roll": 0.0}[k]
                sign = "right" if k == "yaw" else ("up" if k == "pitch" else "")
                draw_ring(ax, r.activity(), r.decoded()[0], screen[k], f"{titles[k]}  ({a.ring_N} units)",
                          f"{k} = {val:+.1f}°" + (f" {sign}" if val > 0.05 and sign else ""), ticks[k])
                keep(f"ring_{k}", frame_of(fig), i)
        if "modules" in want:
            for m in range(3):
                fig = new_fig(5.2, 3.6, a.dpi)
                ax1 = fig.add_axes([0.0, 0.02, 0.66, 0.98], projection="3d", facecolor="white")
                draw_torus(ax1, tori[m].activity(), tori[m].decoded(), f"Module {m + 1}: torus  (period {PERIODS[m]} units)", off_xy[m])
                ax2 = fig.add_axes([0.66, 0.08, 0.34, 0.84])
                z_val = tl["pos"][i][1]
                draw_ring(ax2, heights[m].activity(), heights[m].decoded()[0], lambda th: np.pi / 2 - th, "height ring",
                          f"height = {z_val:+.2f}", [(0, ""), (90, ""), (180, ""), (270, "")], r0=1.0, lift=0.55)
                fig.text(0.5, 0.045, f"position  x = {tl['pos'][i][0]:+.2f}   z = {tl['pos'][i][2]:+.2f}   y = {tl['pos'][i][1]:+.2f}      {tl['label'][i] or 'holding'}",
                         ha="center", fontsize=10.5, color=INK)
                keep(f"module{m + 1}", frame_of(fig), i)
        if "camera" in want:
            fig = new_fig(7.2, 3.6, a.dpi)
            axw = fig.add_axes([0.0, 0.0, 0.56, 0.92], projection="3d", facecolor="white")
            draw_world(axw, tl["R"][i], tl["pos"][i], trail)
            axv = fig.add_axes([0.60, 0.10, 0.37, 0.74])
            draw_view(axv, tl["R"][i], tl["pos"][i])
            fig.text(0.5, 0.94, f"camera   yaw {tl['yaw'][i]:+.1f}°   pitch {tl['pitch'][i]:+.1f}°   {tl['label'][i]}", ha="center", fontsize=12.5, color=INK, weight="bold")
            keep("camera", frame_of(fig), i)
        if i % 20 == 0:
            print(f"  frame {i}/{n}", flush=True)

    json.dump(dict(fps=a.fps, n_frames=n, frames=rec), open(out / "timeline.json", "w"))
    if a.frames:
        print(f"rendered frames {lo}..{hi - 1} to {fdir}; run with --assemble when all segments are done"); return
    for k, fr in frames.items():
        if fr:
            write_gif(fr, out / f"{k}.gif", a.fps)
    if a.overview and all(frames[k] for k in frames):
        ov = []
        for i in range(n):
            row1 = np.concatenate([frames[f"ring_{k}"][i] for k in ("yaw", "pitch", "roll")], 1)
            row2 = np.concatenate([frames[f"module{m}"][i] for m in (1, 2, 3)], 1)
            w = max(row1.shape[1], row2.shape[1], frames["camera"][i].shape[1])
            ov.append(np.concatenate([pad_to(row1, w, row1.shape[0]), pad_to(row2, w, row2.shape[0]), pad_to(frames["camera"][i], w, frames["camera"][i].shape[0])], 0)[::2, ::2])
        write_gif(ov, out / "overview.gif", a.fps)
    print("done ->", out)


if __name__ == "__main__":
    main()
