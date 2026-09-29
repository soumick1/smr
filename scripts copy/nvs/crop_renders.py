#!/usr/bin/env python3
"""Centre-crop a render root and resize back to the original resolution (= a narrower camera).

    python scripts/nvs/crop_renders.py --root ~/data/nvs/objaverse --out ~/data/nvs/objaverse_c70 --frac 0.7
    python scripts/nvs/crop_renders.py --root ~/data/nvs/gso --out ~/data/nvs/gso_c70 --frac 0.7

Every camera looks at the origin, so a centred crop of fraction f followed by a resize to the original
size is exactly a camera with focal length f'/f = 1/f and the same principal point (res/2); cams.json is
rewritten accordingly (K scaled, c2w unchanged, "crop" recorded).  Alpha is cropped with the colour.
Object coverage rises from ~12 % to ~24 % of the frame at f = 0.7, i.e. the backbones see the object at
~1.4x the pixel size.  exclude.txt and FAILED markers are copied.  Multiprocess; resume-safe.
"""
import argparse, json, pathlib, shutil, sys
from multiprocessing import Pool

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--frac", type=float, default=0.7); ap.add_argument("--procs", type=int, default=16)
a = ap.parse_args()
root = pathlib.Path(a.root).expanduser(); out = pathlib.Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)


def one(obj_dir):
    from PIL import Image
    src = pathlib.Path(obj_dir); dst = out / src.name
    if (dst / "cams.json").exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    cams = json.load(open(src / "cams.json"))
    res = int(cams["res"]); c = int(round(res * (1 - a.frac) / 2)); n = 0
    for p in sorted(src.glob("*.png")):
        im = Image.open(p).convert("RGBA").crop((c, c, res - c, res - c)).resize((res, res), Image.LANCZOS)
        im.save(dst / p.name); n += 1
    K = np.asarray(cams["K"], float); s = res / (res - 2 * c)
    K2 = K.copy(); K2[0, 0] *= s; K2[1, 1] *= s; K2[0, 2] = (K[0, 2] - c) * s; K2[1, 2] = (K[1, 2] - c) * s
    cams["K"] = K2.tolist(); cams["crop"] = dict(frac=a.frac, pixels=c, source=str(src)); cams["fov_deg"] = float(2 * np.degrees(np.arctan(res / (2 * K2[1, 1]))))
    json.dump(cams, open(dst / "cams.json", "w"))
    if (src / "meta.json").exists():
        shutil.copy(src / "meta.json", dst / "meta.json")
    if (src / "FAILED").exists():
        shutil.copy(src / "FAILED", dst / "FAILED")
    return n


objs = sorted(str(p.parent) for p in root.glob("*/cams.json"))
print(f"{len(objs)} objects -> {out} (crop {a.frac})", flush=True)
with Pool(a.procs) as pool:
    done = 0
    for i, n in enumerate(pool.imap_unordered(one, objs, chunksize=8)):
        done += n
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(objs)} objects, {done:,} images", flush=True)
if (root / "exclude.txt").exists():
    shutil.copy(root / "exclude.txt", out / "exclude.txt")
print(f"done: {done:,} images written; exclude.txt copied" if (root / "exclude.txt").exists() else f"done: {done:,} images written")
