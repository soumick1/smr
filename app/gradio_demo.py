#!/usr/bin/env python3
"""SMR interactive demo: single-image mental rotation, live path integration.

    pip install gradio plotly
    python app/gradio_demo.py                 # local:   http://localhost:7860
    python app/gradio_demo.py --share         # public 72h link for remote viewing

Upload one photo -> choose a target viewpoint on the 3-D selector ->
watch the scaffold physically drive there (rings + grid modules animated
in sync with the rendered sweep), disocclusions filled by the completion
head, next to what the bare backbone can do alone."""
import argparse, json, pathlib, shutil, sys, tempfile, time, zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PERIODS = [2.4, 3.2, 4.0]
CKPT = ROOT / "outputs/logs/completion/co3d_v1/ckpt_best.pt"
SPEED = dict(slow=(2, 90, 8), normal=(4, 70, 11), fast=(7, 50, 14))
_CACHE = {}


# ---------------------------------------------------------------- geometry --
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


def _rodrigues(axis, ang):
    a = np.asarray(axis, float); a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def target_pose_from_sliders(T0, az_deg, el_deg, dist, subj_dist=1.0):
    """Orbit the camera about the subject point subj_dist along the optical
    axis (median depth := 1 pre-rescale), OpenCV convention."""
    R0, t0 = T0[:3, :3], T0[:3, 3]
    centre = t0 + R0 @ np.array([0.0, 0.0, subj_dist])
    up_w = -R0[:, 1]
    right0 = R0[:, 0]
    v = t0 - centre
    v = _rodrigues(up_w, np.radians(az_deg)) @ \
        _rodrigues(right0, np.radians(-el_deg)) @ v
    t_new = centre + dist * v
    z = centre - t_new; z = z / (np.linalg.norm(z) + 1e-12)
    down = -up_w
    y = down - z * float(down @ z); y = y / (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, t_new
    return T, centre


# ------------------------------------------------------------ 3-D selector --
def pose_preview(az, el, dist, backbone):
    import plotly.graph_objects as go
    fig = go.Figure()
    th = np.linspace(0, 2 * np.pi, 90)
    fig.add_trace(go.Scatter3d(x=np.cos(th), y=np.sin(th), z=0 * th,
                               mode="lines", line=dict(color="#93A7B8", width=3),
                               name="orbit (dist 1)"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text",
                               marker=dict(size=7, color="#17273B"),
                               text=["subject"], textposition="top center",
                               name="subject"))
    def cam(px, py, pz, color, name):
        fig.add_trace(go.Scatter3d(x=[px], y=[py], z=[pz], mode="markers+text",
                                   marker=dict(size=6, color=color, symbol="diamond"),
                                   text=[name], textposition="bottom center",
                                   name=name))
        fig.add_trace(go.Scatter3d(x=[px, 0], y=[py, 0], z=[pz, 0], mode="lines",
                                   line=dict(color=color, width=2, dash="dot"),
                                   showlegend=False))
    cam(1, 0, 0, "#2E6E8E", "reference (your photo)")
    a, e = np.radians(az), np.radians(el)
    tx, ty, tz = dist * np.cos(e) * np.cos(a), dist * np.cos(e) * np.sin(a), dist * np.sin(e)
    cam(tx, ty, tz, "#A0455A", "target")
    arc = np.linspace(0, a, 40)
    fig.add_trace(go.Scatter3d(x=np.interp(np.linspace(0,1,40),[0,1],[1,dist*np.cos(e)])*np.cos(arc),
                               y=np.interp(np.linspace(0,1,40),[0,1],[1,dist*np.cos(e)])*np.sin(arc),
                               z=np.linspace(0, tz, 40), mode="lines",
                               line=dict(color="#B8552F", width=5),
                               name="mental path"))
    fig.update_layout(template="plotly_white", height=420,
                      uirevision="smr-keep-camera",
                      scene_camera=dict(eye=dict(x=1.45, y=1.35, z=0.75),
                                        center=dict(x=0, y=0, z=-0.08)),
                      margin=dict(l=0, r=0, t=28, b=0),
                      title=dict(text=f"target: {az:+.0f}° az, {el:+.0f}° el, "
                                      f"dist ×{dist:.2f}  ·  backbone: {backbone}",
                                 font=dict(size=13)),
                      scene=dict(aspectmode="data",
                                 uirevision="smr-keep-camera",
                                 xaxis_visible=False, yaxis_visible=False,
                                 zaxis_visible=False),
                      legend=dict(orientation="h", y=0.0, font=dict(size=10)))
    return fig


# ------------------------------------------------------------- frame draw ---
def _polar_ring(ax, u, dec_angle, name):
    """The Amari ring as a ring: rectified activity drawn as a radial bump
    travelling around the circle; dashed spoke = decoded heading."""
    import numpy as np
    from smr.viz import TEAL, CORAL
    ang = np.linspace(0, 2 * np.pi, len(u), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    rate = np.maximum(u, 0.0)
    rate = rate / (rate.max() + 1e-9)
    r = 1.0 + 0.62 * rate
    r_c = np.concatenate([r, r[:1]])
    ax.plot(ang_c, np.ones_like(ang_c), color="#B9C6D2", lw=1.0)
    ax.plot(ang_c, r_c, color=TEAL, lw=1.6)
    ax.fill_between(ang_c, 1.0, r_c, color=TEAL, alpha=0.30)
    ax.plot([dec_angle, dec_angle], [0.0, 1.72], color=CORAL, ls="--", lw=1.3)
    ax.set_ylim(0, 1.85)
    ax.set_yticks([])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0\u00b0", "90\u00b0", "180\u00b0", "270\u00b0"],
                       fontsize=6.5)
    ax.grid(False)
    sdeg = (np.degrees(dec_angle) + 180.0) % 360.0 - 180.0
    ax.set_title(f"{name} ring  {sdeg:+6.1f}\u00b0", fontsize=9.5, pad=8)


def _donut(ax, A):
    """The bump on the actual 2-torus: activity as surface colour."""
    import numpy as np
    from matplotlib import cm
    Ap = np.concatenate([A, A[:1, :]], 0)
    Ap = np.concatenate([Ap, Ap[:, :1]], 1)
    n0, n1 = Ap.shape
    th = np.linspace(0, 2 * np.pi, n0)
    ph = np.linspace(0, 2 * np.pi, n1)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    R, r = 1.55, 0.72
    X = (R + r * np.cos(PH)) * np.cos(TH)
    Y = (R + r * np.cos(PH)) * np.sin(TH)
    Z = r * np.sin(PH)
    C = cm.viridis(Ap / (Ap.max() + 1e-9))
    ax.plot_surface(X, Y, Z, facecolors=C, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((1, 1, 0.42))
    ax.view_init(elev=26, azim=-52)
    ax.set_axis_off()
    p = ax.get_position()
    ax.set_position([p.x0 - 0.28 * p.width, p.y0 - 0.30 * p.height,
                     p.width * 1.55, p.height * 1.55])
    ax.set_title("module 0 \u2014 the bump on the torus", fontsize=9, pad=0)


def draw_master(rgb_smr, rgb_bare_tgt, title_smr, snaps_i, xi, traj, tgt_xy,
                K1pos, periods=PERIODS, ring_names=("yaw", "pitch", "roll")):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from smr.viz import TEAL, CORAL, GRAY, fig_to_frame
    fig = plt.figure(figsize=(13.8, 9.4))
    gs = gridspec.GridSpec(3, 12, height_ratios=[1.55, 0.95, 1.0],
                           hspace=0.42, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0:4]); ax.imshow(np.clip(rgb_smr, 0, 1))
    ax.axis("off"); ax.set_title(title_smr, fontsize=11)
    ax = fig.add_subplot(gs[0, 4:8]); ax.imshow(np.clip(rgb_bare_tgt, 0, 1))
    ax.axis("off")
    ax.set_title("backbone reprojection\n@ the GIVEN target pose (no path)",
                 fontsize=9.5)
    ax = fig.add_subplot(gs[0, 8:12])
    ax.plot([K1pos[0]], [K1pos[1]], "o", color=GRAY, ms=7, label="reference")
    if len(traj) > 1:
        tr = np.array(traj)
        ax.plot(tr[:, 0], tr[:, 1], "-", color=CORAL, lw=2.2,
                label="mental path")
    ax.plot([tgt_xy[0]], [tgt_xy[1]], "*", color=TEAL, ms=15, label="target")
    ax.set_title("top-down: path integration", fontsize=11)
    ax.axis("equal"); ax.legend(fontsize=7.5, loc="best", framealpha=0.85)
    for i, nm in enumerate(list(ring_names)[:3]):
        ax = fig.add_subplot(gs[1, 4 * i:4 * i + 4], projection="polar")
        _polar_ring(ax, snaps_i[f"ring_{nm}"], float(xi[i]) % (2 * np.pi), nm)
    ax = fig.add_subplot(gs[2, 0:3], projection="3d")
    _donut(ax, snaps_i["tor_0"])
    for m in range(3):
        ax = fig.add_subplot(gs[2, 3 + 3 * m: 6 + 3 * m])
        ax.imshow(snaps_i[f"tor_{m}"].T, origin="lower", cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        ph = xi[3 + 3 * m: 6 + 3 * m]
        ax.set_title(f"module {m + 1}/3  \u03bb={periods[m]}  "
                     f"\u03c6=({np.degrees(ph[0]):.0f}\u00b0,"
                     f"{np.degrees(ph[1]):.0f}\u00b0)", fontsize=8.4)
    return fig_to_frame(fig)


def draw_pair(rgb_smr, rgb_bare_tgt, title_smr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smr.viz import fig_to_frame
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.9))
    axs[0].imshow(np.clip(rgb_smr, 0, 1)); axs[0].set_title(title_smr,
                                                            fontsize=10)
    axs[1].imshow(np.clip(rgb_bare_tgt, 0, 1))
    axs[1].set_title("backbone @ given target (static)", fontsize=10)
    for ax in axs: ax.axis("off")
    fig.tight_layout()
    return fig_to_frame(fig)


# --------------------------------------------------------------- pipeline ---
def load_t3():
    if "t3" not in _CACHE:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "t3", ROOT / "experiments" / "run_tier3_vggt.py")
        t3 = importlib.util.module_from_spec(spec)
        argv = sys.argv; sys.argv = ["t3"]; spec.loader.exec_module(t3)
        sys.argv = argv
        _CACHE["t3"] = t3
    return _CACHE["t3"]


def load_net():
    if "net" not in _CACHE and CKPT.exists():
        import torch
        from smr.completion import CompletionUNet
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(CKPT, map_location=dev, weights_only=False)
        net = CompletionUNet(base=ck.get("args", {}).get("base", 48))
        net.load_state_dict(ck["model"]); net.to(dev).eval()
        _CACHE["net"] = (net, dev, ck.get("step", "?"))
    return _CACHE.get("net")


def complete_frame(rgb_s, dep_s, msk_s):
    got = load_net()
    if got is None:
        return None
    import torch
    net, dev, _ = got
    x = np.concatenate([rgb_s, np.clip(dep_s, 0, 8)[..., None] / 4.0,
                        msk_s[..., None]], -1).astype(np.float32)
    xt = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(dev)
    Hh, Ww = xt.shape[-2:]
    ph, pw = (-Hh) % 16, (-Ww) % 16
    xt = torch.nn.functional.pad(xt, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        p_rgb, _ = net(xt)
    p_rgb = p_rgb[..., :Hh, :Ww].cpu().numpy()[0].transpose(1, 2, 0)
    return np.where(msk_s[..., None], rgb_s, p_rgb)


def run_pipeline(image, backbone, az, el, dist, speed, use_completion,
                 path, sess, progress=None):
    from smr.dynamics import ScaffoldState
    from smr.pipeline import bind
    from smr.scene import Camera, splat, transform
    from smr.utils.geometry import euler_zyx_to_R, make_T, pose_errors
    from smr.viz import write_gif
    t3 = load_t3()

    def report(p, s):
        if progress is not None:
            progress(p, desc=s)

    sess = pathlib.Path(sess); fr = sess / "frames"
    if fr.exists():
        shutil.rmtree(fr)
    fr.mkdir(parents=True)
    image.convert("RGB").save(fr / "view_000.jpg", quality=95)

    report(0.05, f"{backbone}: lifting your photo to geometry")
    out, _, _ = t3.backbone_output(backbone, str(fr), "cuda")
    out = t3.fit_decode_window(out, periods=PERIODS)
    cam = Camera(H=out.depth.shape[1], W=out.depth.shape[2],
                 f=float(out.intrinsics[0, 0]))

    ss = ScaffoldState(periods=PERIODS, ring_N=128, torus_N=32, seed=0,
                       omega_max=0.16)
    ss.calibrate()

    # The imagined target must sit inside the residue-decode envelope
    # (+-0.8 * max(period)/2 = +-1.6): fit_decode_window guards only the
    # BOUND poses, so pre-scale the scene for the worst pose we'll visit.
    T0 = out.poses[0]
    T_probe, _ = target_pose_from_sliders(T0, az, el, dist)
    m2 = max(float(np.abs(T_probe[:3, 3]).max()),
             float(np.abs(T0[:3, 3]).max()))
    env = 0.8 * (max(PERIODS) / 2.0)
    s3 = max(1.0, m2 / env)
    if s3 > 1.0:
        out.poses = out.poses.copy(); out.poses[:, :3, 3] /= s3
        out.depth = out.depth / s3
        T0 = out.poses[0]
    subj = 1.0 / s3
    T_tgt, centre = target_pose_from_sliders(T0, az, el, dist,
                                             subj_dist=subj)
    report(0.25, "binding the view into scaffold memory")
    bound = bind(t3.subset(out, [0]), ss, cam, formation_extent=1.2,
                 formation_spacing=0.4, torus_N=32, N_h=1024, k=64, seed=0)
    ss.place_pose(T0)
    every, max_frames, fps = SPEED[speed]
    snaps = []

    ring_names = list(ss.rings.keys())          # introspected, not assumed

    def snap(s):
        d = dict(xi=s.state().copy())
        for nm in ring_names:
            d[f"ring_{nm}"] = s.rings[nm].u.copy()
        for m in range(len(s.modules)):
            d[f"tor_{m}"] = s.modules[m].torus.f(s.modules[m].torus.u).copy()
        snaps.append(d)

    report(0.35, f"driving the bumps to your target pose ({path} path)")
    snap(ss)
    if path == "orbital":
        n_way = max(1, int(abs(az) // 18))
        n_steps = 0
        for w in range(1, n_way + 1):
            f = w / n_way
            T_w, _ = target_pose_from_sliders(
                T0, az * f, el * f, 1.0 + (dist - 1.0) * f,
                subj_dist=subj)
            n_w, _ = ss.drive_to(T_w, snapshot_fn=snap,
                                 snapshot_every=every)
            n_steps += n_w
    else:
        n_steps, _ = ss.drive_to(T_tgt, snapshot_fn=snap,
                                 snapshot_every=every)
    xi_end = ss.state()
    T_hat = make_T(euler_zyx_to_R(*xi_end[:3]),
                   decode_pos_from_phases(xi_end[3:].reshape(3, 3), PERIODS))
    re_, pe_ = pose_errors(T_hat, T_tgt)

    stride = max(1, len(snaps) // max_frames)
    picks = snaps[::stride]
    pay = bound.store.content[bound.store.nearest(xi_end, k=1)[0]]
    pts_t = transform(pay["pts"], pay["T"], T_tgt)
    rgb_bt, _, _ = splat(pts_t, pay["cols"], T_tgt, cam)   # GIVEN pose, once
    title_smr = ("SMR: driven + recalled + completed"
                 if (use_completion and load_net()) else
                 "SMR: driven + recalled")
    master, pair, traj = [], [], []
    t_start = time.time()
    for i, sn in enumerate(picks):
        report(0.40 + 0.5 * i / len(picks),
               f"rendering sweep frame {i + 1}/{len(picks)}")
        xf = sn["xi"]
        T_f = make_T(euler_zyx_to_R(*xf[:3]),
                     decode_pos_from_phases(xf[3:].reshape(3, 3), PERIODS))
        pts = transform(pay["pts"], pay["T"], T_f)
        rgb_s, dep_s, msk_s = splat(pts, pay["cols"], T_f, cam)
        smr = rgb_s
        if use_completion:
            comp = complete_frame(rgb_s, dep_s, msk_s)
            if comp is not None:
                smr = comp
        traj.append(T_f[[0, 2], 3] if False else T_f[:3, 3][[0, 2]])
        master.append(draw_master(smr, rgb_bt, title_smr, sn, xf, traj,
                                  T_tgt[:3, 3][[0, 2]], T0[:3, 3][[0, 2]],
                                  ring_names=ring_names))
        pair.append(draw_pair(smr, rgb_bt, title_smr))

    master = [master[0]] * 6 + master + [master[-1]] * 10
    pair = [pair[0]] * 6 + pair + [pair[-1]] * 10
    gdir = sess / "out"; gdir.mkdir(exist_ok=True)
    write_gif(master, gdir / "mental_rotation_full.gif", fps=fps)
    write_gif(pair, gdir / "view_comparison.gif", fps=fps)

    rep = dict(backbone=backbone, az_deg=az, el_deg=el, dist=dist, path=path,
               envelope_rescale=round(s3, 3),
               n_steps=int(n_steps), snapshots=len(snaps),
               arrival_rot_rad=float(re_), arrival_pos=float(pe_),
               completion_ckpt=str(CKPT) if (use_completion and load_net())
               else None,
               render_seconds=round(time.time() - t_start, 1))
    (gdir / "report.json").write_text(json.dumps(rep, indent=2))
    zp = sess / "smr_demo_results.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in gdir.iterdir():
            z.write(f, f.name)

    summary = (f"**{n_steps} integration steps** to a target "
               f"{az:+.0f}° azimuth / {el:+.0f}° elevation away — arrival "
               f"error **{np.degrees(re_):.2f}° / {pe_:.4f}** against the "
               f"commanded pose. Every intermediate frame is rendered at the "
               f"pose *decoded from the bumps*, never interpolated. "
               + (f"Disocclusions filled by the completion head "
                  f"(CO3D-trained, step {load_net()[2]}) — zero-shot on "
                  f"your photo." if (use_completion and load_net()) else
                  "Completion off: black regions are honest disocclusions."))
    return (str(gdir / "mental_rotation_full.gif"),
            str(gdir / "view_comparison.gif"), summary, str(zp))


# ------------------------------------------------------------------- UI -----
CSS = """
.gradio-container {font-family: 'Source Sans 3', system-ui, sans-serif;}
#hdr h1 {font-size: 26px; margin-bottom: 2px;}
#hdr p {color: #4A5E74; margin-top: 0;}
footer {display: none !important;}
"""


def build_ui():
    import gradio as gr
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
    with gr.Blocks(theme=theme, css=CSS, title="SMR — mental rotation") as demo:
        gr.HTML('<div id="hdr"><h1>SMR — single-image mental rotation</h1>'
                '<p>One photo in. A scaffold of attractor networks binds it, '
                'physically drives its bumps to the viewpoint you choose, and '
                'renders the journey — path integration, live.</p></div>')
        sess = gr.State(lambda: tempfile.mkdtemp(prefix="smrdemo_"))
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="your photo (one image)")
                backbone = gr.Radio(["vggt", "vggt_omega", "pi3", "fast3r"],
                                    value="vggt",
                                    label="geometry backbone (the eyes)  "
                                          "[vggt ~9 GB / vggt_omega ~6 GB / "
                                          "pi3 ~10 GB / fast3r ~9 GB]")
                az = gr.Slider(-150, 150, 60, step=5,
                               label="azimuth — orbit around the subject (°)")
                el = gr.Slider(-40, 40, 8, step=2, label="elevation (°)")
                dist = gr.Slider(0.6, 1.6, 1.0, step=0.05,
                                 label="distance (× reference)")
                path = gr.Radio(["orbital", "direct"], value="orbital",
                                label="mental path \u2014 orbital keeps the "
                                      "subject in view; direct is the chord")
                speed = gr.Radio(["slow", "normal", "fast"], value="slow",
                                 label="traversal speed (slow shows the "
                                       "integration best)")
                comp = gr.Checkbox(True, label="fill disocclusions "
                                               "(completion head, CO3D-trained)")
                with gr.Row():
                    run = gr.Button("Drive the bumps", variant="primary")
                    wipe = gr.Button("Start afresh")
                with gr.Accordion("what am I looking at?", open=False):
                    gr.Markdown(
                        "The backbone lifts your photo to surfels + a pose. "
                        "The scaffold (3 head-direction rings + 3 grid "
                        "modules) binds it, then **physically drives its "
                        "activity bumps** to your target — the bottom row "
                        "shows every ring and module during the drive. "
                        "Rendering happens at the pose **decoded from the "
                        "bumps**. The right panel is what the backbone can "
                        "do alone: reproject once, with holes, no notion of "
                        "a path.")
            with gr.Column(scale=1):
                preview = gr.Plot(label="target selector — 3-D")
                summary = gr.Markdown()
        with gr.Row():
            master = gr.Image(label="mental rotation + path integration "
                                    "(rings & modules in sync)", type="filepath")
        with gr.Row():
            pairg = gr.Image(label="view comparison (SMR vs backbone-only)",
                             type="filepath")
            dl = gr.File(label="download everything (gifs + report)")

        for c in (az, el, dist, backbone):
            c.change(pose_preview, [az, el, dist, backbone], preview,
                     show_progress="hidden")
        demo.load(pose_preview, [az, el, dist, backbone], preview)

        def go(image, backbone, az, el, dist, speed, comp, path, sess,
               progress=gr.Progress()):
            if image is None:
                raise gr.Error("Upload a photo first.")
            try:
                return run_pipeline(image, backbone, az, el, dist, speed,
                                    comp, path, sess, progress)
            except Exception as e:                       # surface real errors
                raise gr.Error(f"{type(e).__name__}: {e}")

        run.click(go, [img, backbone, az, el, dist, speed, comp, path, sess],
                  [master, pairg, summary, dl])

        def fresh(sess):
            shutil.rmtree(sess, ignore_errors=True)
            return (tempfile.mkdtemp(prefix="smrdemo_"), None, None, None,
                    "", None)
        wipe.click(fresh, [sess], [sess, img, master, pairg, summary, dl])
    return demo


def selftest():
    T0 = np.eye(4)
    T, c = target_pose_from_sliders(T0, 60, 10, 1.0)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8), "R not orthonormal"
    assert abs(np.linalg.det(R) - 1) < 1e-8, "det != 1"
    fwd = R[:, 2]; to_c = c - T[:3, 3]; to_c /= np.linalg.norm(to_c)
    assert float(fwd @ to_c) > 0.999, "look-at broken"
    d = np.linalg.norm(T[:3, 3] - c)
    assert abs(d - 1.0) < 1e-6, "distance broken"
    T2, c2 = target_pose_from_sliders(T0, 120, 0, 1.2, subj_dist=0.5)
    assert abs(np.linalg.norm(T2[:3, 3] - c2) - 0.6) < 1e-6, "subj_dist broken"
    fig = pose_preview(60, 10, 1.0, "vggt")
    assert fig is not None
    rng = np.random.default_rng(0)
    sn = {"xi": rng.normal(size=12)}
    for nm in ("yaw", "pitch", "roll"):
        sn[f"ring_{nm}"] = rng.normal(size=128)
    for m in range(3):
        sn[f"tor_{m}"] = rng.random((32, 32))
    fr = draw_master(rng.random((120, 160, 3)), rng.random((120, 160, 3)),
                     "SMR", sn, sn["xi"], [np.zeros(2), np.ones(2)],
                     np.ones(2), np.zeros(2),
                     ring_names=("yaw", "pitch", "roll"))
    fr2 = draw_pair(rng.random((120, 160, 3)), rng.random((120, 160, 3)), "SMR")
    print("selftest OK:", fr.shape, fr2.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        build_ui().queue(max_size=4).launch(server_name="0.0.0.0",
                                            server_port=a.port, share=a.share)
