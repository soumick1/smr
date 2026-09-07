#!/usr/bin/env python3
"""Reconstruction figures with colours, image frusta and a sensible viewpoint (VGGT-SLAM Fig. 8 style).

    python scripts/fig_recon.py --est outputs/est/fig_S.npz --json outputs/reports/fig_S.json \
        --ply outputs/points/fig_S_chained/fused.ply outputs/points/fig_S_smr/fused.ply --rows chained smr \
        --labels "raw (chained Sim(3) windows)" "+SMR" --K 585 585 320 240 --out outputs/figures/recon_S

Colours: each point is projected into the keyframes of its row (poses from the est npz, intrinsics --K for the
original image size) and takes the colour of the NEAREST camera that sees it -- the point was produced from one of
these cameras' depth maps, so this recovers its source colour without any dependency on the fusion code.  If the
PLY already carries rgb it is used directly.
Viewpoint: 'up' is the cameras' own up (mean -y axis), the eye sits behind and above the trajectory looking at the
volume the cameras observed; --azimuth rotates around 'up', --elev tilts.  Points farther than --crop x the path
extent from the observed centre are dropped (distant wall fragments otherwise dictate the framing).
Frusta: every --frustum-stride-th keyframe (auto: ~24), coloured by window, with the keyframe image warped onto the
far plane; accepted closures as green links between the closing window's frame and the remembered site.
--thumbs DIR: use small thumbnails (fig_pack.py) instead of the full images named in the est npz.
"""
import argparse, json, pathlib, re, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_T = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4", "ushort": "<u2", "short": "<i2"}


def read_ply(p):
    raw = pathlib.Path(p).read_bytes(); head, body = raw.split(b"end_header\n", 1); hdr = head.decode("ascii", "replace")
    n = int(re.search(r"element vertex (\d+)", hdr).group(1)); vert = hdr.split("element vertex")[1].split("element")[0]
    props = [(nm, _T[t]) for t, nm in re.findall(r"property (\w+) (\w+)", vert)]
    arr = np.frombuffer(body, dtype=np.dtype(props), count=n); names = [nm for nm, _ in props]
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], -1).astype(float)
    rgb = np.stack([arr[c] for c in ("red", "green", "blue")], -1) if all(c in names for c in ("red", "green", "blue")) else None
    return xyz, rgb


def load_images(paths, thumbs, n):
    ims = []
    for i in range(n):
        p = pathlib.Path(thumbs) / f"{i:04d}.jpg" if thumbs else pathlib.Path(paths[i])
        ims.append(np.asarray(Image.open(p).convert("RGB")))
    return ims


def colourize(xyz, poses, K0, size0, ims, chunk=25000):
    """Colour of the nearest camera that sees each point (image coordinates from K0 scaled to the thumbnail size)."""
    H, W = ims[0].shape[:2]; sx, sy = W / size0[0], H / size0[1]
    K = np.array([[K0[0] * sx, 0, K0[2] * sx], [0, K0[1] * sy, K0[3] * sy], [0, 0, 1]])
    w2c = np.linalg.inv(poses).astype(np.float32)                # (C,4,4)
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    stack = np.stack(ims)                                        # (C,H,W,3)
    out = np.zeros((len(xyz), 3), np.uint8); seen = np.zeros(len(xyz), bool)
    for s in range(0, len(xyz), chunk):
        X = xyz[s:s + chunk].astype(np.float32)                  # (P,3)
        Xc = np.einsum("cij,pj->cpi", R, X) + t[:, None]         # (C,P,3)
        z = Xc[..., 2]
        u = K[0, 0] * Xc[..., 0] / np.where(z > 1e-6, z, np.nan) + K[0, 2]
        v = K[1, 1] * Xc[..., 1] / np.where(z > 1e-6, z, np.nan) + K[1, 2]
        ok = (z > 0.05) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        zz = np.where(ok, z, np.inf); best = zz.argmin(0)        # nearest camera that sees the point
        p_idx = np.arange(len(X)); good = np.isfinite(zz[best, p_idx])
        ui = np.clip(np.nan_to_num(np.round(u[best, p_idx])).astype(int), 0, W - 1); vi = np.clip(np.nan_to_num(np.round(v[best, p_idx])).astype(int), 0, H - 1)
        col = stack[best, vi, ui]
        out[s:s + chunk][good] = col[good]; seen[s:s + chunk] = good
    return out, seen


def look_at(eye, target, up):
    z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, eye
    return M


def scene_frame(poses, xyz, keep_pct=95.0):
    """up = cameras' mean up; centre and radius from the points themselves: the centre is the median of the points
    within 8x the path extent of the cameras (throws away only far-flung fragments), the radius is the keep_pct-th
    percentile of the distance to that centre.  Returns up, fwd, centre, scene_radius, path_extent."""
    up = -poses[:, :3, 1].mean(0); up /= np.linalg.norm(up)
    fwd = poses[:, :3, 2].mean(0); fwd -= up * (fwd @ up); fwd /= np.linalg.norm(fwd)
    c = poses[:, :3, 3]; lo, hi = c.min(0), c.max(0); pext = max(float(np.max(hi - lo)), 0.3)
    near = np.linalg.norm(xyz - c.mean(0), axis=1) < 8 * pext
    P = xyz[near] if near.sum() > 1000 else xyz
    centre = np.median(P, 0)
    radius = float(np.percentile(np.linalg.norm(P - centre, axis=1), keep_pct))
    return up, fwd, centre, radius, pext


def project(K, c2w, X):
    w2c = np.linalg.inv(c2w); Xc = (w2c[:3, :3] @ X.T).T + w2c[:3, 3]; z = Xc[:, 2]
    uv = (K[:2, :2] @ (Xc[:, :2] / np.where(z[:, None] > 1e-6, z[:, None], np.nan)).T).T + K[:2, 2]
    return uv, z


def splat(xyz, rgb, K, c2w, W, H, radius, bg=255):
    uv, z = project(K, c2w, xyz)
    ok = np.isfinite(uv).all(1) & (z > 1e-3) & (uv[:, 0] > -radius) & (uv[:, 0] < W + radius) & (uv[:, 1] > -radius) & (uv[:, 1] < H + radius)
    uv, z, col = uv[ok], z[ok], rgb[ok]
    zbuf = np.full(W * H, np.inf); img = np.full((W * H, 3), bg, np.uint8)
    px = np.round(uv[:, 0]).astype(int); py = np.round(uv[:, 1]).astype(int)
    offs = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1) if dx * dx + dy * dy <= radius * radius + 0.1]
    for dx, dy in offs:
        x = px + dx; y = py + dy; m = (x >= 0) & (x < W) & (y >= 0) & (y < H); idx = y[m] * W + x[m]
        np.minimum.at(zbuf, idx, z[m])
    for dx, dy in offs:
        x = px + dx; y = py + dy; m = (x >= 0) & (x < W) & (y >= 0) & (y < H); idx = y[m] * W + x[m]
        win = z[m] <= zbuf[idx] * (1 + 2e-3); img[idx[win]] = col[m][win]
    return img.reshape(H, W, 3), zbuf.reshape(H, W)


def persp_coeffs(src, dst):
    """PIL PERSPECTIVE coefficients mapping output quad `dst` -> source quad `src` (both 4x2)."""
    A = []; b = []
    for (x, y), (X, Y) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.append(Y)
    return np.linalg.solve(np.array(A, float), np.array(b, float))


def draw_frusta(img, poses, ims, K, c2w, zbuf, colours, size, aspect, width, alpha=0.9):
    """Wireframe pyramids with the keyframe image on the far plane; drawn back to front, occluded by the cloud only
    where the cloud is clearly in front of the frustum's far plane."""
    W, H = img.size
    depth_of = lambda P: project(K, c2w, P[None, :3, 3])[1][0]
    order = np.argsort([-depth_of(P) for P in poses])
    for i in order:
        P = poses[i]; colr = colours[i]
        corners = np.array([[-0.5, -0.5 / aspect, 1], [0.5, -0.5 / aspect, 1], [0.5, 0.5 / aspect, 1], [-0.5, 0.5 / aspect, 1]]) * size
        Xw = np.concatenate([np.zeros((1, 3)), corners]) @ P[:3, :3].T + P[:3, 3]
        uv, z = project(K, c2w, Xw)
        if not (np.isfinite(uv).all() and (z > 0).all()):
            continue
        quad = uv[1:]
        # image on the far plane (perspective warp of the thumbnail into the projected quad)
        if ims is not None:
            th = Image.fromarray(ims[i]); tw, thh = th.size
            x0, y0 = np.floor(quad.min(0)).astype(int); x1, y1 = np.ceil(quad.max(0)).astype(int)
            if x1 - x0 > 4 and y1 - y0 > 4 and x1 > 0 and y1 > 0 and x0 < W and y0 < H:
                coeffs = persp_coeffs(np.array([[0, 0], [tw, 0], [tw, thh], [0, thh]], float), quad - [x0, y0])
                warped = th.transform((int(x1 - x0), int(y1 - y0)), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
                mask = Image.new("L", warped.size, 0); ImageDraw.Draw(mask).polygon([tuple(q) for q in (quad - [x0, y0])], fill=int(255 * alpha))
                img.paste(warped, (int(x0), int(y0)), mask)
        d = ImageDraw.Draw(img)
        c = tuple(uv[0]); q = [tuple(p) for p in quad]
        for p in q:
            d.line([c, p], fill=colr, width=width)
        d.line(q + [q[0]], fill=colr, width=width)


def closure_links(report, chunks, row):
    ev_list = (report.get("events", {}) or {}).get(row) or []; out = []
    for ev in ev_list:
        loop = ev.get("loop") if isinstance(ev, dict) else None
        if not loop or not loop.get("accepted") or "revoked_at" in loop:
            continue
        k = int(ev.get("chunk", -1))
        if k < 0 or k >= len(chunks):
            continue
        anchor = None
        for s in (ev.get("sites") or []):
            if isinstance(s, dict) and s.get("ok", True) and "view" in s:
                anchor = int(s["view"]); break
        if anchor is None and "anchor_chunk" in loop and 0 <= int(loop["anchor_chunk"]) < len(chunks):
            ac = chunks[int(loop["anchor_chunk"])]; anchor = int(ac[len(ac) // 2])
        if anchor is not None:
            out.append((anchor, int(chunks[k][len(chunks[k]) // 2])))
    return out


def window_palette(n):
    import matplotlib; matplotlib.use("Agg")
    return [tuple(int(255 * c) for c in matplotlib.colormaps["tab10"](i % 10)[:3]) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True); ap.add_argument("--json", default=""); ap.add_argument("--ply", nargs="+", required=True)
    ap.add_argument("--rows", nargs="+", required=True); ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--K", nargs=4, type=float, default=None, metavar=("FX", "FY", "CX", "CY"), help="intrinsics of the original images (default: from the GT npz, else 7-Scenes' 585/585/320/240)")
    ap.add_argument("--size0", nargs=2, type=int, default=None, help="original image size for --K (default: the loaded image size, or 640x480 with --thumbs)")
    ap.add_argument("--thumbs", default="", help="directory of NNNN.jpg thumbnails (fig_pack.py); default: images named in the est npz")
    ap.add_argument("--azimuth", type=float, default=-12.0); ap.add_argument("--elev", type=float, default=34.0); ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--dist", type=float, default=1.9, help="eye distance in units of the scene radius")
    ap.add_argument("--crop", type=float, default=1.25, help="keep points within this x scene-radius of the centre")
    ap.add_argument("--keep-pct", type=float, default=95.0, help="percentile of point distances defining the scene radius")
    ap.add_argument("--res", nargs=2, type=int, default=[2600, 1800], help="pixels per panel (2x supersampled internally)")
    ap.add_argument("--radius", type=int, default=0, help="splat radius in internal pixels (0 = auto, scales with --res)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--max-points", type=int, default=8000000); ap.add_argument("--frustum-stride", type=int, default=0)
    ap.add_argument("--frustum-size", type=float, default=0.0, help="far-plane width in scene units (0 = 7%% of the scene radius)")
    ap.add_argument("--zoom", nargs=3, type=float, default=None, metavar=("X", "Y", "HALF")); ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = 2; W, H = a.res[0] * S, a.res[1] * S
    radius = a.radius or max(2, int(round(3 * a.res[0] / 1300)))
    E = np.load(a.est, allow_pickle=True); chunks = [np.asarray(c, int) for c in E["chunks"]]
    kpaths = [str(p) for p in E["kpaths"]]; report = json.load(open(a.json)) if a.json and pathlib.Path(a.json).exists() else {}
    rows = {r: np.asarray(E[f"est_{r}"], float) for r in a.rows}
    n_key = len(next(iter(rows.values())))
    if a.K is None:
        K = None
        try:
            G = np.load(str(E["gt"]), allow_pickle=True)
            if "K" in G.files:
                Kg = np.asarray(G["K"], float); Kg = Kg[int(E["key"][0])] if Kg.ndim == 3 else Kg
                K = [Kg[0, 0], Kg[1, 1], Kg[0, 2], Kg[1, 2]]; print(f"intrinsics from GT npz: {np.round(K, 1)}")
        except Exception as ex:
            print(f"GT npz not readable for K ({ex})")
        a.K = K or [585.0, 585.0, 320.0, 240.0]
    ims = load_images(kpaths, a.thumbs, n_key) if (a.thumbs or all(pathlib.Path(p).exists() for p in kpaths[:3])) else None
    aspect = ims[0].shape[1] / ims[0].shape[0] if ims is not None else 4 / 3
    if a.size0 is None:
        a.size0 = [640, 480] if a.thumbs else ([ims[0].shape[1], ims[0].shape[0]] if ims is not None else [640, 480])
    win_of = np.zeros(n_key, int)
    for k, c in enumerate(chunks):
        win_of[c] = k
    pal = window_palette(len(chunks))
    rng = np.random.default_rng(0)
    # camera from the FIRST row (shared by all panels)
    P0 = rows[a.rows[0]]; xyz0, _ = read_ply(a.ply[0])
    up, fwd, centre, ext, pext = scene_frame(P0, xyz0, a.keep_pct)
    side = np.cross(fwd, up); az, el = np.deg2rad(a.azimuth), np.deg2rad(a.elev)
    back = -(np.cos(az) * fwd + np.sin(az) * side)
    eye = centre + a.dist * ext * (np.cos(el) * back + np.sin(el) * up)
    c2w = look_at(eye, centre, up); f = 0.5 * H / np.tan(np.deg2rad(a.fov) / 2); Kv = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    fsize = a.frustum_size or 0.07 * ext; fstride = a.frustum_stride or max(1, n_key // 24)
    print(f"scene radius {ext:.2f} (path extent {pext:.2f}); eye at {a.dist * ext:.2f} from the centre")
    fs = max(12, W // 70)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", fs)
    except Exception:
        font = ImageFont.load_default()
    panels = []
    for i, (ply, row) in enumerate(zip(a.ply, a.rows)):
        xyz, rgb = read_ply(ply); P = rows[row]
        keep = np.linalg.norm(xyz - centre, axis=1) < a.crop * ext
        xyz = xyz[keep]; rgb = rgb[keep] if rgb is not None else None
        if len(xyz) > a.max_points:
            sel = rng.choice(len(xyz), a.max_points, replace=False); xyz = xyz[sel]; rgb = rgb[sel] if rgb is not None else None
        if rgb is None:
            if ims is None:
                raise SystemExit("PLY has no colours and no images are available (--thumbs or est kpaths)")
            rgb, seen = colourize(xyz, P, a.K, a.size0, ims); xyz, rgb = xyz[seen], rgb[seen]
            print(f"{row}: coloured {seen.sum():,}/{len(seen):,} points by reprojection")
        img_arr, zbuf = splat(xyz, rgb, Kv, c2w, W, H, radius)
        img = Image.fromarray(img_arr)
        sel = np.arange(0, n_key, fstride)
        draw_frusta(img, P[sel], None if a.no_images or ims is None else [ims[j] for j in sel], Kv, c2w, zbuf, [pal[win_of[j]] for j in sel], fsize, aspect, max(2, W // 1300))
        d = ImageDraw.Draw(img)
        for anchor, cur in closure_links(report, chunks, row):
            uv, z = project(Kv, c2w, np.stack([P[anchor, :3, 3], P[cur, :3, 3]]))
            if np.isfinite(uv).all() and (z > 0).all():
                d.line([tuple(uv[0]), tuple(uv[1])], fill=(30, 180, 60), width=max(4, W // 650))
        if a.zoom:
            x, y, h = a.zoom; x0, y0, x1, y1 = int((x - h) * W), int((y - h) * H), int((x + h) * W), int((y + h) * H)
            crop = img.crop((x0, y0, x1, y1)); zw = int(0.34 * W); crop = crop.resize((zw, int(zw * (y1 - y0) / max(1, x1 - x0))), Image.LANCZOS)
            lw = max(3, W // 900)
            d.rectangle([x0, y0, x1, y1], outline=(255, 140, 0), width=lw)
            px, py = W - zw - 10 * S, H - crop.size[1] - 10 * S; img.paste(crop, (px, py)); d.rectangle([px, py, px + zw, py + crop.size[1]], outline=(255, 140, 0), width=lw + 1)
        label = a.labels[i] if a.labels and i < len(a.labels) else row
        d.rectangle([10 * S, 10 * S, 10 * S + int(font.getlength(label)) + 20 * S, 10 * S + int(1.6 * fs)], fill=(255, 255, 255)); d.text((20 * S, 10 * S + int(0.15 * fs)), label, fill=(0, 0, 0), font=font)
        panels.append(img.resize((W // S, H // S), Image.LANCZOS))
    out = Image.new("RGB", (sum(p.size[0] for p in panels) + 12 * (len(panels) - 1), panels[0].size[1]), (255, 255, 255)); xo = 0
    for p in panels:
        out.paste(p, (xo, 0)); xo += p.size[0] + 12
    o = pathlib.Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(o) + ".png", dpi=(a.dpi, a.dpi)); out.save(str(o) + ".pdf", resolution=a.dpi)
    print(f"wrote {o}.png/.pdf {out.size[0]}x{out.size[1]} px at {a.dpi} dpi  (view: azimuth {a.azimuth}, elev {a.elev}, dist {a.dist}x, crop {a.crop}x, {len(sel)} frusta, splat radius {radius})")


if __name__ == "__main__":
    main()
