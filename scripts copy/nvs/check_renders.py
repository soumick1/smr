#!/usr/bin/env python3
"""Integrity check of a render root (objaverse or gso): completeness, format, object coverage.

    python scripts/nvs/check_renders.py --root ~/data/nvs/objaverse --views 16 [--sample 2000]
    python scripts/nvs/check_renders.py --root ~/data/nvs/gso --views 14

Reports: objects with cams.json, FAILED markers, objects missing views, non-RGBA images, and the
alpha coverage (fraction of object pixels) per object — flags EMPTY objects (< 0.5 % coverage: the
object is not visible, e.g. a degenerate or off-centre asset) and SATURATED ones (> 85 %: the frame
is filled, i.e. the object was not normalised as intended).  Writes <root>/exclude.txt with the ids
to skip; the trainer and the GSO evaluator honour it.
"""
import argparse, json, pathlib, random, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--views", type=int, required=True)
ap.add_argument("--sample", type=int, default=0, help="check alpha on a random sample of this many objects (0 = all)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
from PIL import Image
root = pathlib.Path(a.root).expanduser()
objs = sorted(p.parent for p in root.glob("*/cams.json"))
failed = sorted(p.parent.name for p in root.glob("*/FAILED"))
print(f"{root}: {len(objs)} objects with cams.json, {len(failed)} FAILED markers")
missing, non_rgba, empty, saturated, cover = [], [], [], [], []
check = objs if not a.sample or a.sample >= len(objs) else random.Random(a.seed).sample(objs, a.sample)
for k, o in enumerate(check):
    pngs = sorted(o.glob("*.png"))
    if len(pngs) < a.views:
        missing.append(o.name); continue
    covs = []
    for p in pngs[: a.views]:
        im = Image.open(p)
        if im.mode != "RGBA":
            non_rgba.append(o.name); break
        alpha = np.asarray(im.getchannel("A"), dtype=np.float32) / 255.0
        covs.append(float((alpha > 0.5).mean()))
    if not covs:
        continue
    c = float(np.mean(covs)); cover.append(c)
    if c < 0.005:
        empty.append(o.name)
    elif c > 0.85:
        saturated.append(o.name)
    if (k + 1) % 500 == 0:
        print(f"  checked {k+1}/{len(check)}", flush=True)
cover = np.array(cover)
print(f"checked {len(check)} objects: missing views {len(missing)}, non-RGBA {len(non_rgba)}, "
      f"EMPTY (<0.5%) {len(empty)}, SATURATED (>85%) {len(saturated)}")
if len(cover):
    print(f"object coverage per image: median {np.median(cover)*100:.1f}%, 10-90% {np.percentile(cover,10)*100:.1f}-{np.percentile(cover,90)*100:.1f}%")
excl = sorted(set(failed + missing + non_rgba + empty + saturated))
(root / "exclude.txt").write_text("\n".join(excl) + ("\n" if excl else ""))
print(f"wrote {root/'exclude.txt'} ({len(excl)} ids to skip); usable objects ~ {len(objs) - len(excl)}")
if empty[:5]: print("  e.g. EMPTY:", empty[:5])
if saturated[:5]: print("  e.g. SATURATED:", saturated[:5])
