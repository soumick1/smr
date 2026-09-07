#!/usr/bin/env python3
"""Download Google Scanned Objects (GSO, 1,030 models, CC-BY 4.0) from Gazebo Fuel and write the render index.

    python scripts/nvs/gso_fetch.py --out ~/data/nvs/gso_models --index data/nvs/gso_index.json [--limit 5]
    python scripts/nvs/gso_fetch.py --names-file gso_names.txt ...      # if every listing endpoint fails

Listing: the collection "Scanned Objects by Google Research" (owner GoogleResearch).  Fuel's REST routes have
moved over the years, so the script tries, in order: the model search filtered by collection
(/1.0/models?q=collections:<name>), the collection route (/1.0/<owner>/collections/<name>/models), and the owner
route (/1.0/<owner>/models), printing the HTTP status of each attempt.  Download: "<model URL>.zip" (Fuel docs:
"add .zip to the end of the URL"), extracted to <out>/<name>/ (meshes/model.obj + materials/textures/*.png).
GSO meshes are z-up (Gazebo) -> render with --obj-up Z.  Resume-safe.  Writes [{id, path, seed}].
"""
import argparse, io, json, pathlib, shutil, time, urllib.error, urllib.parse, urllib.request, zipfile, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="~/data/nvs/gso_models")
ap.add_argument("--index", default="data/nvs/gso_index.json")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--names-file", default="", help="one model name per line; skips the listing step")
ap.add_argument("--owner", default="GoogleResearch")
ap.add_argument("--collection", default="Scanned Objects by Google Research")
a = ap.parse_args()
out = pathlib.Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
BASE = "https://fuel.gazebosim.org/1.0"
Q = urllib.parse.quote


def get(url, binary=False, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "smr-nvs/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read() if binary else json.loads(r.read().decode())


def list_models():
    """Union of every listing route that answers (the collection search returned 299 of GSO's 1,030 models on
    the first try; the owner route may hold the rest).  Prints per-route counts."""
    routes = [
        ("collection search", lambda p: f"{BASE}/models?page={p}&per_page=100&q={Q('collections:' + a.collection)}"),
        ("collection route", lambda p: f"{BASE}/{a.owner}/collections/{Q(a.collection)}/models?page={p}&per_page=100"),
        ("owner route", lambda p: f"{BASE}/{a.owner}/models?page={p}&per_page=100"),
        ("owner route (lower)", lambda p: f"{BASE}/{a.owner.lower()}/models?page={p}&per_page=100"),
        ("owner search", lambda p: f"{BASE}/models?page={p}&per_page=100&q={Q('owner:' + a.owner)}"),
    ]
    all_names = set()
    for label, route in routes:
        names, page = [], 1
        try:
            while True:
                batch = get(route(page))
                if not isinstance(batch, list) or not batch:
                    break
                names += [m["name"] for m in batch if str(m.get("owner", a.owner)).lower() == a.owner.lower()]
                page += 1
                if len(batch) < 100 or page > 60:
                    break
        except urllib.error.HTTPError as e:
            print(f"  {label}: HTTP {e.code}"); continue
        except Exception as e:
            print(f"  {label}: {type(e).__name__}: {e}"); continue
        print(f"  {label}: {len(set(names))} models (+{len(set(names) - all_names)} new)")
        all_names |= set(names)
    if not all_names:
        raise SystemExit("all listing routes failed; pass --names-file (one model name per line, see the collection page "
                         "https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research)")
    print(f"  union: {len(all_names)} models")
    return sorted(all_names)


names = [l.strip() for l in open(a.names_file) if l.strip()] if a.names_file else list_models()
if a.limit:
    names = names[: a.limit]
index, t0, failed = [], time.time(), []
for i, name in enumerate(names):
    d = out / name
    objs = sorted(d.glob("**/*.obj"))
    if not objs:
        data, err = None, None
        for url in (f"{BASE}/{a.owner}/models/{Q(name)}.zip", f"{BASE}/{a.owner}/models/{Q(name)}/1/{Q(name)}.zip"):
            try:
                data = get(url, binary=True); break
            except Exception as ex:
                err = f"{url} -> {ex}"
        if data is None:
            print(f"  {name}: download failed ({err})"); failed.append(name); continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            zipfile.ZipFile(io.BytesIO(data)).extractall(d)
        except zipfile.BadZipFile:
            print(f"  {name}: not a zip ({len(data)} bytes)"); failed.append(name); continue
        objs = sorted(d.glob("**/*.obj"))
    # GSO's model.mtl references textures by bare filename ("map_Kd texture.png") while the archive
    # stores them under materials/textures/; Blender resolves MTL paths next to the .obj -> link them there.
    for obj in objs:
        for tex in list(d.glob("**/textures/*")):
            dst = obj.parent / tex.name
            if not dst.exists():
                try:
                    dst.symlink_to(tex.resolve())
                except OSError:
                    shutil.copy(tex, dst)
    if objs:
        index.append(dict(id=name, path=str(objs[0]), seed=zlib.crc32(name.encode()) % (2 ** 31)))
    else:
        print(f"  {name}: no .obj in archive"); failed.append(name)
    if (i + 1) % 25 == 0 or i + 1 == len(names):
        print(f"  {i+1}/{len(names)} ({time.time()-t0:.0f} s)", flush=True)
pathlib.Path(a.index).parent.mkdir(parents=True, exist_ok=True)
json.dump(index, open(a.index, "w"), indent=0)
print(f"{len(index)} GSO models with meshes -> {a.index}" + (f"; failed: {failed[:10]}{'...' if len(failed) > 10 else ''}" if failed else ""))
