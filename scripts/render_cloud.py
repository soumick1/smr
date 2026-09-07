#!/usr/bin/env python3
"""Reconstruction figures in the style of VGGT-SLAM's Figs. 1/7/8, without a display or a 3D library.

    python scripts/render_cloud.py --ply outputs/points/fig_S_chained/fused.ply outputs/points/fig_S_smr/fused.ply \
        --labels "raw (chained Sim(3) windows)" "+SMR" --est outputs/est/fig_S.npz --rows chained smr \
        --json outputs/reports/fig_S.json --zoom 0.55 0.40 0.18 --out outputs/figures/recon_S

Draws each cloud from the SAME camera (fitted to the first cloud): z-buffered point splats with the PLY's colours
(or height colouring when the PLY has none), camera frusta of the matching trajectory row coloured by window,
accepted closures as green links between the closing window and its remembered site, and an optional zoom inset
(--zoom X Y HALF in image fractions) shown at the same place in every panel -- ghosted geometry on the left,
aligned on the right.  --view oblique|top, --azimuth/--elev set the camera.  Supersampled 2x for anti-aliasing.
"""
import argparse, json, pathlib, re, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4", "ushort": "<u2", "short": "<i2"}


def read_ply_rgb(p):
    raw = pathlib.Path(p).read_bytes()
    head, body = raw.split(b"end_header\n", 1); hdr = head.decode("ascii", "replace")
    n = int(re.search(r"element vertex (\d+)", hdr).group(1))
    vert = hdr.split("element vertex")[1].split("element")[0]
    props = [(nm, _TYPES[t]) for t, nm in re.findall(r"property (\w+) (\w+)", vert)]
    if "format ascii" in hdr:
        arr = np.loadtxt(body.splitlines()[:n]); xyz = arr[:, :3]; rgb = arr[:, 3:6].astype(np.uint8) if arr.shape[1] >= 6 else None
        return xyz, rgb
    arr = np.frombuffer(body, dtype=np.dtype(props), count=n)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], -1).astype(float)
    names = [nm for nm, _ in props]
    rgb = np.stack([arr[c] for c in ("red", "green", "blue")], -1) if all(c in names for c in ("red", "green", "blue")) else None
    return xyz, rgb


def look_at(cam, target, up):
    z = target - cam; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, cam
    return M


def fit_camera(xyz, centres, view, azimuth, elev, fov, W, H, margin=0.95, fit="trajectory"):
    """Camera looking at the scene from an oblique or top viewpoint.  fit='trajectory': frame the camera path and
    what it looks at (the path's box grown by 60 %, plus the points within it) -- a floor that extends far beyond
    the orbit does not dictate the framing; fit='cloud': the 2nd-98th percentile box of all points.  'up' = the axis
    along which the camera centres (or points) vary least."""
    if fit == "trajectory" and centres is not None and len(centres) > 3:
        clo, chi = centres.min(0), centres.max(0); span = max(np.max(chi - clo), 1e-6)
        clo, chi = clo - 0.6 * span, chi + 0.6 * span
        inside = ((xyz >= clo) & (xyz <= chi)).all(1)
        pts = xyz[inside] if inside.sum() > 1000 else xyz
        lo, hi = np.minimum(np.percentile(pts, 2, 0), clo), np.maximum(np.percentile(pts, 98, 0), chi)
    else:
        lo, hi = np.percentile(xyz, 2, 0), np.percentile(xyz, 98, 0)
    centre = 0.5 * (lo + hi); ext = hi - lo
    src = centres if centres is not None and len(centres) > 3 else xyz
    up_axis = int(np.argmin(src.var(0))); up = np.zeros(3); up[up_axis] = 1.0
    if centres is not None and len(centres) > 3 and (centres[:, up_axis].mean() < centre[up_axis]):
        up = -up                                                # cameras below the cloud centre: flip
    horiz = [i for i in range(3) if i != up_axis]
    az, el = np.deg2rad(azimuth), np.deg2rad(90.0 if view == "top" else elev)
    d = np.zeros(3); d[horiz[0]] = np.cos(el) * np.cos(az); d[horiz[1]] = np.cos(el) * np.sin(az); d += np.sin(el) * up
    f = 0.5 * H / np.tan(np.deg2rad(fov) / 2)
    dist = margin * 0.5 * np.linalg.norm(ext) / np.tan(np.deg2rad(fov) / 2)
    cam = centre + dist * d
    c2w = look_at(cam, centre, up if view != "top" else np.array([1.0 if h == horiz[0] else 0.0 for h in range(3)]) * np.eye(3)[horiz[1]])
    if view == "top":
        c2w = look_at(cam, centre, np.eye(3)[horiz[1]])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return K, c2w


def project(K, c2w, X):
    w2c = np.linalg.inv(c2w); Xc = (w2c[:3, :3] @ X.T).T + w2c[:3, 3]
    z = Xc[:, 2]; uv = (K[:2, :2] @ (Xc[:, :2] / np.where(z[:, None] > 1e-6, z[:, None], np.nan)).T).T + K[:2, 2]
    return uv, z


def splat(xyz, rgb, K, c2w, W, H, radius, bg=255):
    uv, z = project(K, c2w, xyz)
    ok = np.isfinite(uv).all(1) & (z > 1e-3) & (uv[:, 0] >= -radius) & (uv[:, 0] < W + radius) & (uv[:, 1] >= -radius) & (uv[:, 1] < H + radius)
    uv, z, col = uv[ok], z[ok], rgb[ok]
    zbuf = np.full(W * H, np.inf); img = np.full((W * H, 3), bg, np.uint8)
    offs = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1) if dx * dx + dy * dy <= radius * radius + 0.1]
    px = np.round(uv[:, 0]).astype(int); py = np.round(uv[:, 1]).astype(int)
    for dx, dy in offs:
        x = px + dx; y = py + dy; m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        idx = y[m] * W + x[m]
        np.minimum.at(zbuf, idx, z[m])
    for dx, dy in offs:
        x = px + dx; y = py + dy; m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        idx = y[m] * W + x[m]; win = z[m] <= zbuf[idx] * (1 + 1e-4)
        img[idx[win]] = col[m][win]
    return img.reshape(H, W, 3)


def height_colours(xyz, up_axis):
    import matplotlib; matplotlib.use("Agg")
    h = xyz[:, up_axis]; lo, hi = np.percentile(h, 2), np.percentile(h, 98)
    return (matplotlib.colormaps["viridis"]((h - lo) / max(hi - lo, 1e-9))[:, :3] * 255).astype(np.uint8)


def window_palette(n):
    import matplotlib; matplotlib.use("Agg")
    return [tuple(int(255 * c) for c in matplotlib.colormaps["tab20"](i % 20)[:3]) for i in range(n)]


def draw_frusta(draw, poses, K, c2w, W, H, colours, size, width):
    # pyramid: centre + 4 corners of a virtual image plane at depth `size`
    corners = np.array([[-0.5, -0.35, 1], [0.5, -0.35, 1], [0.5, 0.35, 1], [-0.5, 0.35, 1]]) * size
    for P, colr in zip(poses, colours):
        pts = np.concatenate([np.zeros((1, 3)), corners])
        Xw = (P[:3, :3] @ pts.T).T + P[:3, 3]
        uv, z = project(K, c2w, Xw)
        if not (np.isfinite(uv).all() and (z > 0).all()):
            continue
        c = tuple(uv[0]); cs = [tuple(u) for u in uv[1:]]
        for q in cs:
            draw.line([c, q], fill=colr, width=width)
        draw.line(cs + [cs[0]], fill=colr, width=width)


def closure_links(report, chunks, row):
    ev_list = (report.get("events", {}) or {}).get(row) or []
    out = []
    for ev in ev_list:
        loop = ev.get("loop") if isinstance(ev, dict) else None
        if not loop or not loop.get("accepted") or "revoked_at" in loop:
            continue
        k = int(ev.get("chunk", -1)); sites = [s for s in (ev.get("sites") or []) if isinstance(s, dict) and s.get("ok", True)]
        if k < 0 or k >= len(chunks) or not sites:
            continue
        anchor = int(sites[0].get("view", sites[0].get("frame", -1)))
        cur = int(chunks[k][len(chunks[k]) // 2])
        if anchor >= 0:
            out.append((anchor, cur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", nargs="+", required=True); ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--est", default=""); ap.add_argument("--rows", nargs="+", default=None); ap.add_argument("--json", default="")
    ap.add_argument("--view", choices=["oblique", "top"], default="oblique"); ap.add_argument("--azimuth", type=float, default=35.0)
    ap.add_argument("--elev", type=float, default=38.0); ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--res", nargs=2, type=int, default=[1400, 900]); ap.add_argument("--radius", type=int, default=2, help="splat radius at 2x supersampling")
    ap.add_argument("--max-points", type=int, default=1500000); ap.add_argument("--frustum-size", type=float, default=0.0, help="0 = auto (3%% of the scene)")
    ap.add_argument("--zoom", nargs=3, type=float, default=None, metavar=("X", "Y", "HALF"), help="inset window in image fractions")
    ap.add_argument("--frustum-stride", type=int, default=0, help="draw every k-th keyframe frustum (0 = auto, ~60 frusta)")
    ap.add_argument("--fit", choices=["trajectory", "cloud"], default="trajectory", help="what the camera is framed on")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = 2; W, H = a.res[0] * S, a.res[1] * S
    clouds = [read_ply_rgb(p) for p in a.ply]
    rng = np.random.default_rng(0)
    clouds = [(x[i], (c[i] if c is not None else None)) for (x, c) in clouds for i in [rng.choice(len(x), min(len(x), a.max_points), replace=False)]]
    est = np.load(a.est, allow_pickle=True) if a.est else None
    rows = a.rows or ([str(r) for r in est["rows"]] if est is not None else [])
    chunks = [np.asarray(c, int) for c in est["chunks"]] if est is not None else []
    report = json.load(open(a.json)) if a.json and pathlib.Path(a.json).exists() else {}
    poses0 = np.asarray(est[f"est_{rows[0]}"], float) if est is not None and rows else None
    K, c2w = fit_camera(clouds[0][0], poses0[:, :3, 3] if poses0 is not None else None, a.view, a.azimuth, a.elev, a.fov, W, H, fit=a.fit)
    up_axis = int(np.argmin((poses0[:, :3, 3] if poses0 is not None else clouds[0][0]).var(0)))
    if poses0 is not None:
        ext = float(np.max(poses0[:, :3, 3].max(0) - poses0[:, :3, 3].min(0)))          # size of the camera path
    else:
        ext = np.linalg.norm(np.percentile(clouds[0][0], 98, 0) - np.percentile(clouds[0][0], 2, 0))
    fsize = a.frustum_size or 0.06 * ext
    fstride = a.frustum_stride or (max(1, len(poses0) // 60) if poses0 is not None else 1)
    win_of = np.zeros(len(poses0), int) if poses0 is not None else None
    if chunks:
        for k, c in enumerate(chunks):
            win_of[c] = k
    pal = window_palette(len(chunks) or 1)
    panels = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22 * S)
    except Exception:
        font = ImageFont.load_default()
    for i, (xyz, rgb) in enumerate(clouds):
        col = rgb if rgb is not None else height_colours(xyz, up_axis)
        img = Image.fromarray(splat(xyz, col, K, c2w, W, H, a.radius))
        draw = ImageDraw.Draw(img)
        if est is not None and i < len(rows):
            P = np.asarray(est[f"est_{rows[i]}"], float)
            sel = slice(None, None, fstride)
            draw_frusta(draw, P[sel], K, c2w, W, H, [pal[w] for w in win_of[sel]], fsize, 2 * S)
            for anchor, cur in closure_links(report, chunks, rows[i]):
                if anchor < len(P) and cur < len(P):
                    uv, z = project(K, c2w, np.stack([P[anchor, :3, 3], P[cur, :3, 3]]))
                    if np.isfinite(uv).all() and (z > 0).all():
                        draw.line([tuple(uv[0]), tuple(uv[1])], fill=(20, 170, 60), width=3 * S)
        if a.zoom:
            x, y, h = a.zoom; x0, y0 = int((x - h) * W), int((y - h) * H); x1, y1 = int((x + h) * W), int((y + h) * H)
            crop = img.crop((x0, y0, x1, y1)); zw = int(0.36 * W); crop = crop.resize((zw, int(zw * (y1 - y0) / max(1, x1 - x0))), Image.LANCZOS)
            draw.rectangle([x0, y0, x1, y1], outline=(255, 140, 0), width=3 * S)
            px, py = W - zw - 12 * S, H - crop.size[1] - 12 * S
            img.paste(crop, (px, py)); draw.rectangle([px, py, px + zw, py + crop.size[1]], outline=(255, 140, 0), width=4 * S)
            draw.line([(x1, y1), (px, py + crop.size[1])], fill=(255, 140, 0), width=2 * S)
        label = (a.labels[i] if a.labels and i < len(a.labels) else pathlib.Path(a.ply[i]).parent.name)
        draw.rectangle([10 * S, 10 * S, 10 * S + int(font.getlength(label)) + 20 * S, 10 * S + 34 * S], fill=(255, 255, 255))
        draw.text((20 * S, 14 * S), label, fill=(0, 0, 0), font=font)
        panels.append(img.resize((W // S, H // S), Image.LANCZOS))
    out = Image.new("RGB", (sum(p.size[0] for p in panels) + 10 * (len(panels) - 1), panels[0].size[1]), (255, 255, 255))
    xo = 0
    for p in panels:
        out.paste(p, (xo, 0)); xo += p.size[0] + 10
    o = pathlib.Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(o) + ".png"); out.save(str(o) + ".pdf", resolution=200)
    print(f"wrote {o}.png/.pdf ({', '.join(f'{len(x):,} pts' for x, _ in clouds)}; camera {a.view}, az {a.azimuth}, elev {a.elev})")


if __name__ == "__main__":
    main()
