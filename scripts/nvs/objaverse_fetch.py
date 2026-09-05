#!/usr/bin/env python3
"""Select and download an Objaverse-LVIS subset (GLBs) and write the render index.

    pip install objaverse
    python scripts/nvs/objaverse_fetch.py --n 20000 --out ~/data/nvs/objaverse_glb --index data/nvs/objaverse_index.json

Selection: the LVIS-annotated subset (~46K curated objects, 1,156 categories), sampled
round-robin across categories with a fixed seed so every category is represented before any
is repeated (stratified, reproducible).  Downloads run through the `objaverse` package from
Hugging Face (parallel); `objaverse.BASE_PATH` is redirected to --out.  Resume-safe: already
downloaded uids are reused.  Writes [{id, path, category, seed}] for scripts/nvs/render_bpy.py.
"""
import argparse, json, pathlib, random, time, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="~/data/nvs/objaverse_glb")
ap.add_argument("--index", default="data/nvs/objaverse_index.json")
ap.add_argument("--processes", type=int, default=8)
ap.add_argument("--dry-run", type=int, default=0, help="download only this many (smoke test)")
a = ap.parse_args()

import objaverse  # noqa: E402  (pip install objaverse)
out = pathlib.Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
objaverse.BASE_PATH = str(out)
objaverse._VERSIONED_PATH = str(out / "hf-objaverse-v1")   # the package derives its paths from BASE_PATH at import; set both

t0 = time.time()
lvis = objaverse.load_lvis_annotations()                  # {category: [uid, ...]}
cats = sorted(lvis)
rng = random.Random(a.seed)
per_cat = {c: rng.sample(lvis[c], len(lvis[c])) for c in cats}
uids, cat_of = [], {}
while len(uids) < a.n and any(per_cat.values()):
    for c in cats:                                        # round-robin: one per category per round
        if per_cat[c] and len(uids) < a.n:
            u = per_cat[c].pop(); uids.append(u); cat_of[u] = c
if a.dry_run:
    uids = uids[: a.dry_run]
print(f"selected {len(uids)} uids from {len(cats)} LVIS categories ({time.time()-t0:.0f} s)", flush=True)

paths = objaverse.load_objects(uids=uids, download_processes=a.processes)   # {uid: local path}
index = [dict(id=u, path=str(paths[u]), category=cat_of[u], seed=zlib.crc32(u.encode()) % (2 ** 31)) for u in uids if u in paths]
pathlib.Path(a.index).parent.mkdir(parents=True, exist_ok=True)
json.dump(index, open(a.index, "w"), indent=0)
sizes = [pathlib.Path(p).stat().st_size for p in paths.values() if pathlib.Path(p).exists()]
print(f"downloaded {len(index)}/{len(uids)} GLBs, {sum(sizes)/1e9:.1f} GB total, median {sorted(sizes)[len(sizes)//2]/1e6:.1f} MB -> {a.index} ({time.time()-t0:.0f} s)")
