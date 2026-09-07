#!/usr/bin/env python3
"""Pack one figure scene for offline rendering work: est npz, json report, both fused PLYs subsampled, and small
keyframe thumbnails (for frusta with images).  ~30-60 MB per scene.

    python scripts/fig_pack.py --stem fig_7scenes_pumpkin_seq01 --out /tmp/pack_pumpkin.tgz [--points 1500000]
"""
import argparse, io, json, pathlib, re, tarfile, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--stem", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--points", type=int, default=1500000); ap.add_argument("--thumb", type=int, default=192)
a = ap.parse_args()
from PIL import Image
root = pathlib.Path(".")
est = root / "outputs/est" / f"{a.stem}.npz"; rep = root / "outputs/reports" / f"{a.stem}.json"
plys = {row: root / "outputs/points" / f"{a.stem}_{row}" / "fused.ply" for row in ("chained", "smr")}
for p in [est, rep] + list(plys.values()):
    assert p.exists(), f"missing {p}"
E = np.load(est, allow_pickle=True); kpaths = [str(p) for p in E["kpaths"]]
gt = pathlib.Path(str(E["gt"]))
rng = np.random.default_rng(0)
_T = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
def sub_ply(p):
    raw = p.read_bytes(); head, body = raw.split(b"end_header\n", 1); hdr = head.decode()
    n = int(re.search(r"element vertex (\d+)", hdr).group(1)); props = [(nm, _T[t]) for t, nm in re.findall(r"property (\w+) (\w+)", hdr)]
    arr = np.frombuffer(body, dtype=np.dtype(props), count=n)
    sel = np.sort(rng.choice(n, min(n, a.points), replace=False)); arr = arr[sel]
    hdr2 = hdr.replace(f"element vertex {n}", f"element vertex {len(arr)}") + "end_header\n"
    return hdr2.encode() + arr.tobytes(), n, [nm for nm, _ in props]
with tarfile.open(a.out, "w:gz") as tf:
    def add_bytes(name, data):
        ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
    tf.add(est, arcname=f"{a.stem}/est.npz"); tf.add(rep, arcname=f"{a.stem}/report.json")
    if gt.exists(): tf.add(gt, arcname=f"{a.stem}/gt.npz")
    for row, p in plys.items():
        data, n, names = sub_ply(p); add_bytes(f"{a.stem}/{row}.ply", data)
        print(f"{row}: {n:,} points -> {min(n, a.points):,}; properties {names}")
    for i, kp in enumerate(kpaths):
        try:
            im = Image.open(kp).convert("RGB"); im.thumbnail((a.thumb, a.thumb)); b = io.BytesIO(); im.save(b, "JPEG", quality=85)
            add_bytes(f"{a.stem}/thumbs/{i:04d}.jpg", b.getvalue())
        except Exception as ex:
            print(f"thumb {i} failed: {ex}")
    add_bytes(f"{a.stem}/kpaths.json", json.dumps(kpaths).encode())
print(f"wrote {a.out} ({pathlib.Path(a.out).stat().st_size/1e6:.1f} MB)")
