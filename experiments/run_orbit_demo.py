#!/usr/bin/env python3
"""The flagship on real data: mental rotation along an orbital capture.

    python experiments/run_orbit_demo.py --backbone vggt --frames orbit_frames

Binds every view except a held-out target, places the scaffold at a start
view, PHYSICALLY drives the bumps to the target pose (recording the
trajectory), then renders recalled content at intermediate decoded poses
(sweep GIF) and at arrival (V5-style comparison against the backbone's
own held-out frame).  GT-free by design."""
import argparse, importlib.util, json, pathlib, sys

import numpy as np
import matplotlib; matplotlib.use("Agg")               # noqa: E402
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smr.dynamics import ScaffoldState                  # noqa: E402
from smr.pipeline import bind                           # noqa: E402
from smr.pipeline.rotate import coherent_reanchor       # noqa: E402
from smr.scene import Camera, splat, transform          # noqa: E402
from smr.utils.geometry import (euler_zyx_to_R, make_T,  # noqa: E402
                                pose_errors)
from smr.viz import TEAL, CORAL, GRAY, savefig, write_gif, fig_to_frame  # noqa: E402

def decode_pos_from_phases(ph, periods):
    order = np.argsort(-np.asarray(periods))
    x = np.zeros(3)
    for j_i, j in enumerate(order):
        lam, frac = periods[j], ph[j] / (2 * np.pi)
        if j_i == 0:
            x = ((frac + 0.5) % 1.0 - 0.5) * lam
        else:
            x = x + lam * (((frac - x / lam) + 0.5) % 1.0 - 0.5)
    return x


spec = importlib.util.spec_from_file_location(
    "t3", ROOT / "experiments" / "run_tier3_vggt.py")
t3 = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ["t3"]; spec.loader.exec_module(t3)
sys.argv = _argv

OUT = ROOT / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--backbone", default="vggt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--path", choices=["direct", "orbit"], default="direct",
                    help="direct: chord in pose space (imagination off the "
                         "data manifold).  orbit: waypoint through the "
                         "capture poses -- the model mentally walks the "
                         "orbit, object held in view.")
    ap.add_argument("--scale", choices=["compact", "full"], default="compact")
    a = ap.parse_args()

    out, T_gt, D_gt = t3.backbone_output(a.backbone, a.frames, a.device)
    out = t3.fit_decode_window(out, periods=[2.4, 3.2, 4.0])
    K = out.poses.shape[0]
    tgt = a.target if a.target is not None else (a.start + K // 3) % K
    keep = [i for i in range(K) if i != tgt]
    print(f"orbit demo [{a.backbone}]: {K} views, start {a.start}, "
          f"held-out target {tgt}")

    cam = Camera(H=out.depth.shape[1], W=out.depth.shape[2],
                 f=float(out.intrinsics[0, 0]))
    periods = [2.4, 3.2, 4.0]
    ring_N, torus_N, N_h, k = ((256, 64, 8192, 410) if a.scale == "full"
                               else (128, 32, 1024, 64))
    ss = ScaffoldState(periods=periods, ring_N=ring_N, torus_N=torus_N,
                       seed=0, omega_max=0.16)
    ss.calibrate()
    sub = t3.subset(out, keep)
    bound = bind(sub, ss, cam, formation_extent=1.2,
                 formation_spacing=0.3 if a.scale == "full" else 0.4,
                 torus_N=torus_N, N_h=N_h, k=k, seed=0)

    # ---- the drive: start view state -> held-out target pose ----
    i_start = keep.index(a.start) if a.start in keep else 0
    ss.place_pose(sub.poses[i_start])
    snaps = []
    def snap(s):
        snaps.append(dict(yaw_u=s.rings["yaw"].u.copy(),
                          tor=s.modules[0].torus.f(s.modules[0].torus.u).copy(),
                          xi=s.state().copy()))
    snap(ss)
    every = 8 if a.fast else 5
    if a.path == "orbit":
        # waypoint through the experienced poses around the ring to the
        # target: the mental path follows the orbit, object held in view.
        i0 = keep.index(a.start) if a.start in keep else 0
        way = []
        j = i0
        while keep[j] != (tgt - 1) % K and len(way) < K:
            j = (j + 1) % len(keep)
            way.append(keep[j])
        n_steps = 0
        for w in way:
            n_w, _ = ss.drive_to(out.poses[w], snapshot_fn=snap,
                                 snapshot_every=every)
            n_steps += n_w
        n_w, _ = ss.drive_to(out.poses[tgt], snapshot_fn=snap,
                             snapshot_every=every)
        n_steps += n_w
    else:
        n_steps, _ = ss.drive_to(out.poses[tgt], snapshot_fn=snap,
                                 snapshot_every=every)
    diag = coherent_reanchor(bound)
    xi = ss.state()
    T_hat = make_T(euler_zyx_to_R(*xi[:3]),
                   decode_pos_from_phases(xi[3:].reshape(len(periods), 3),
                                          periods))
    re_, pe_ = pose_errors(T_hat, out.poses[tgt])
    print(f"  drive: {n_steps} steps; arrival vs backbone target pose: "
          f"rot {re_:.4f} rad, pos {pe_:.4f}"
          + (f"; re-anchor fired ({diag['score_before']:.3f}->"
             f"{diag['score_after']:.3f})" if diag.get("corrected") else ""))

    # ---- arrival render at the DECODED pose, vs backbone held-out ----
    ids = bound.store.nearest(xi, k=2)
    P, C = [], []
    for j in ids:
        pay = bound.store.content[j]
        P.append(transform(pay["pts"], pay["T"], T_hat))
        C.append(pay["cols"])
    rgb_s, dep_s, msk_s = splat(np.concatenate(P), np.concatenate(C),
                                T_hat, cam)
    d_bb = out.depth[tgt]
    both = msk_s & (d_bb > 0) & out.mask[tgt]
    rel = (float(np.median(np.abs(dep_s[both] - d_bb[both]) / d_bb[both]))
           if both.sum() >= 200 else float("nan"))
    print(f"  arrival splat vs backbone held-out: rel med {rel:.4f} "
          f"(coverage {both.mean():.2f})")

    scene = pathlib.Path(a.frames).name
    scene = pathlib.Path(a.frames).parent.name if scene.startswith("images") \
        else scene
    tag = f"{a.backbone}_{scene}" + ("_orbitpath" if a.path == "orbit"
                                       else "") +         ("_full" if a.scale == "full" else "")

    fig, axs = plt.subplots(1, 4, figsize=(14.6, 3.1),
                            gridspec_kw=dict(wspace=0.32))
    axs[0].imshow(np.clip(rgb_s, 0, 1))
    axs[0].set_title("recalled render @ decoded arrival")
    axs[1].imshow(np.clip(out.rgb[tgt], 0, 1))
    axs[1].set_title("actual held-out frame")
    err = np.where(both, np.abs(dep_s - d_bb) / np.maximum(d_bb, 1e-6), np.nan)
    im = axs[2].imshow(err, cmap="magma", vmax=0.3)
    axs[2].set_title(f"rel |dz|, med {rel:.3f}")
    fig.colorbar(im, ax=axs[2], fraction=0.040, pad=0.02)
    Cc = out.poses[:, :3, 3]
    axs[3].plot(Cc[:, 0], Cc[:, 2], "o-", color=GRAY, ms=3, label="capture")
    xs = np.array([make_T(euler_zyx_to_R(*s["xi"][:3]),
                   decode_pos_from_phases(s["xi"][3:].reshape(len(periods), 3),
                                          periods))[:3, 3] for s in snaps])
    axs[3].plot(xs[:, 0], xs[:, 2], "-", color=CORAL, lw=2, label="mental path")
    axs[3].plot(*Cc[tgt][[0, 2]], "*", color=TEAL, ms=14, label="target")
    axs[3].legend(fontsize=7); axs[3].axis("equal")
    axs[3].set_title("top-down: drive trajectory")
    for ax in axs[:3]: ax.axis("off")
    savefig(fig, OUT / "figures" / f"orbit_{tag}.png")

    frames = []
    stride = max(1, len(snaps) // (45 if a.fast else 80))
    for sn in snaps[::stride]:
        xf = sn["xi"]
        T_f = make_T(euler_zyx_to_R(*xf[:3]),
                     decode_pos_from_phases(xf[3:].reshape(len(periods), 3),
                                            periods))
        ids = bound.store.nearest(xf, k=2)
        P, C = [], []
        for j in ids:
            pay = bound.store.content[j]
            P.append(transform(pay["pts"], pay["T"], T_f))
            C.append(pay["cols"])
        rgb, _, _ = splat(np.concatenate(P), np.concatenate(C), T_f, cam)
        fig = plt.figure(figsize=(9.6, 3.1))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.25)
        ax0 = fig.add_subplot(gs[0]); ax0.imshow(np.clip(rgb, 0, 1))
        ax0.axis("off")
        ax0.set_title(f"mental rotation  yaw {np.degrees(xf[0]):6.1f}$^\\circ$")
        ax1 = fig.add_subplot(gs[1])
        ang = np.linspace(0, 2 * np.pi, len(sn["yaw_u"]), endpoint=False)
        ax1.plot(ang, sn["yaw_u"], color=TEAL, lw=1.4)
        ax1.axvline(xf[0] % (2 * np.pi), color=CORAL, ls="--", lw=1.1)
        ax1.set_ylim(-3.5, 1.6); ax1.set_title("HD yaw ring")
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(sn["tor"].T, origin="lower", cmap="viridis")
        ax2.axis("off"); ax2.set_title("grid module 0")
        frames.append(fig_to_frame(fig))
    write_gif(frames, OUT / "gifs" / f"orbit_sweep_{tag}.gif", fps=12)

    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / f"orbit_{tag}.json").write_text(json.dumps(dict(
        backbone=a.backbone, scene=scene, K=K, start=a.start, target=tgt,
        n_steps=int(n_steps), arrival_rot=float(re_), arrival_pos=float(pe_),
        splat_rel_med=rel, coverage=float(both.mean()),
        reanchor=diag), indent=2))
    print(f"  figure -> outputs/figures/orbit_{tag}.png")
    print(f"  report -> outputs/reports/orbit_{tag}.json")


if __name__ == "__main__":
    main()
