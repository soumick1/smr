#!/usr/bin/env python3
"""Bank growth, retrieval capacity and pruning for continuous mapping (reviewer question, 2026-09-18).

    python scripts/bank_scaling.py --gt-dir data/gt --cache-dir outputs/cache --dataset 7scenes --md outputs/bank_scaling.md
    python scripts/bank_scaling.py --synthetic --md /tmp/bank_scaling.md          # no data needed

What it does, with the production memory classes (smr.stitch.memory_index.ScaffoldIndex / DescriptorIndex), no backbone:
  1. Pairs every cached DINO descriptor file (`<cache>/*.desc_<kind>.npy`) with its GT npz (scene tokens + keyframe count).
  2. Builds ONE bank for a continuous-mapping scenario: all sequences of all rooms are bound one after another, rooms
     placed on a grid in scaffold units (7-Scenes sequences of one room share the room's world frame, so later
     sequences revisit places seen in earlier ones). Poses are GT (metres / --unit, default 1.5 m per scaffold unit).
     The last sequence of every room is held out as the query set; the others are bound in order.
  3. After every bound sequence: retrieval recall@1 / recall@5 of each query against GT revisits (strict: within 10 %
     of the room's bounding-box diagonal and 30 deg; covis: 30 % and 60 deg), the fraction of top-1 hits from the
     wrong room (perceptual aliasing), query latency, and bytes per entry. Scaffold index (RLS cue -> address, address
     overlap) and flat cosine index side by side.
  4. Pruning at bind time, each evaluated on the full bank: pose spacing (skip a view within d_min metres and 15 deg
     of a stored view of the same room), scaffold-cell deduplication (skip when the new address overlaps a stored
     address by >= tau AND their cues agree, cosine >= 0.8), cue deduplication (cosine >= c). Reports entries kept,
     recall, and the resulting index and dense-content bytes.
Dense content per entry is not stored by the stitcher; it is estimated from the backbone resolution:
--pixels x --conf-keep x --bytes-per-point (default 518x392 x 0.8 x 19 B: xyz float32, rgb uint8, footprint float32).
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smr.eval.trajectory import rotation_angle_deg  # noqa: E402
from smr.stitch.chunks import keyframe_indices  # noqa: E402
from smr.stitch.memory_index import DescriptorIndex, ScaffoldIndex  # noqa: E402

ROOMS = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")


def _norm(s):
    s = re.sub(r"seq[-_ ]?0*(\d+)", r"seq\1", str(s).lower())
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


# ------------------------------------------------------------------ data
def load_desc(path):
    """(N, D) float array from a descriptor cache saved as an array, an object array, or a dict."""
    D = np.load(path, allow_pickle=True)
    if isinstance(D, np.ndarray) and D.dtype == object:
        D = D.item() if D.ndim == 0 else list(D)
    if isinstance(D, dict):
        for key in ("desc", "descriptors", "D", "s"):
            if key in D:
                D = D[key]; break
        else:
            D = next(v for v in D.values() if isinstance(v, np.ndarray) and v.ndim == 2)
    if isinstance(D, list):
        D = np.stack([np.asarray(x, float) for x in D])
    D = np.asarray(D, float)
    if D.ndim != 2:
        raise ValueError(f"descriptor array has shape {D.shape}")
    return D


def discover(gt_dir, cache_dir, kind, stride, dataset, debug=False):
    """[(room, seq_id, gt_poses[key], descriptors)] for every cached descriptor file with a matching GT npz."""
    descs = {pathlib.Path(p): None for p in glob.glob(str(pathlib.Path(cache_dir) / f"*.desc_{kind}.npy"))}
    gts = [pathlib.Path(p) for p in glob.glob(str(pathlib.Path(gt_dir) / "**" / "*.npz"), recursive=True)]
    if debug:
        print(f"[discover] {len(descs)} descriptor caches, {len(gts)} GT files")
    out = []
    for g in gts:
        gs = _norm(g.stem)
        if dataset == "7scenes" and not any(r in gs for r in ROOMS):
            continue
        if any(t in gs.split("_") for t in ("all", "multi", "concat")) or re.search(r"_s\d{4,}", gs):
            continue                                                   # multi-session composites
        room = next((r for r in ROOMS if r in gs), gs)
        seq = re.search(r"seq(\d+)", gs)
        seq_id = int(seq.group(1)) if seq else 1
        try:
            z = np.load(g, allow_pickle=True)
            poses = np.asarray(z["poses"], float)
        except Exception:  # noqa: BLE001
            continue
        key = keyframe_indices(len(poses), stride)
        best, why = None, []
        for d in descs:
            ds = _norm(d.stem)
            if room not in ds:
                continue
            dseq = re.search(r"seq(\d+)", ds)
            if dseq and int(dseq.group(1)) != seq_id:
                why.append(f"{d.name}: seq {dseq.group(1)} != {seq_id}"); continue
            if not dseq and seq_id != 1:
                why.append(f"{d.name}: no seq token, GT is seq {seq_id}"); continue
            try:
                D = load_desc(d)
            except Exception as ex:  # noqa: BLE001
                why.append(f"{d.name}: load failed ({str(ex)[:60]})"); continue
            if len(D) != len(key):
                why.append(f"{d.name}: {len(D)} rows vs {len(key)} keyframes"); continue
            best = (d, D)
            break
        if best is None:
            if debug:
                print(f"[discover] {g.name} ({len(poses)} frames, {len(key)} keyframes, room {room}, seq {seq_id}): no match; "
                      + ("; ".join(why[:6]) if why else "no cache mentions this room"))
            continue
        out.append(dict(room=room, seq=seq_id, gt=poses[key], desc=np.asarray(best[1], float), src=(str(g.name), str(best[0].name))))
    return out


def synthetic(n_rooms=4, n_seqs=4, n_kf=200, D=384, seed=0):
    """Rooms of 3x3 m with random-walk sequences; descriptors depend on position and viewing direction plus noise,
    so nearby views look alike and different rooms look different (place-like cue)."""
    rng = np.random.default_rng(seed)
    out = []
    proj = rng.standard_normal((6 + n_rooms, D))
    for r in range(n_rooms):
        for s in range(1, n_seqs + 1):
            p = np.array([1.5, 1.5, 1.2]) + np.cumsum(rng.normal(0, 0.08, (n_kf, 3)), 0)
            p = np.clip(p, 0.2, 2.8)
            yaw = np.cumsum(rng.normal(0, 0.08, n_kf))
            gt = np.tile(np.eye(4), (n_kf, 1, 1))
            for i in range(n_kf):
                c, sn = np.cos(yaw[i]), np.sin(yaw[i])
                gt[i, :3, :3] = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
                gt[i, :3, 3] = p[i]
            feat = np.concatenate([p / 3.0, np.stack([np.cos(yaw), np.sin(yaw), 0 * yaw], 1), np.eye(n_rooms)[r][None].repeat(n_kf, 0)], 1)
            desc = feat @ proj + 0.6 * rng.standard_normal((n_kf, D))
            desc /= np.linalg.norm(desc, axis=1, keepdims=True)
            out.append(dict(room=f"room{r}", seq=s, gt=gt, desc=desc, src=("synthetic", "synthetic")))
    return out


# ------------------------------------------------------------------ truth
def room_truth(entries_by_room, queries, dist_frac=0.10, angle=30.0, covis_frac=0.30, covis_angle=60.0):
    """For each query: sets of stored global ids that are strict / covis GT revisits (same room only)."""
    truth = []
    for q in queries:
        room = q["room"]
        ids, poses = entries_by_room.get(room, ([], np.zeros((0, 4, 4))))
        if len(ids) == 0:
            truth.append((set(), set()))
            continue
        c = poses[:, :3, 3]
        diag = q["diag"]
        d = np.linalg.norm(c - q["T"][:3, 3], axis=1)
        s1, s3 = set(), set()
        for j in np.where(d < covis_frac * diag)[0]:
            ra = rotation_angle_deg(q["T"][:3, :3].T @ poses[j, :3, :3])
            if d[j] < dist_frac * diag and ra < angle:
                s1.add(ids[j])
            va = float(np.degrees(np.arccos(np.clip(q["T"][:3, 2] @ poses[j, :3, 2], -1, 1))))
            if va < covis_angle:
                s3.add(ids[j])
        truth.append((s1, s3))
    return truth


def evaluate(index, queries, truth, room_of, top=5):
    """recall@1 / @5 (strict), covis recall@5, wrong-room top-1 rate, query latency."""
    r1 = r5 = c5 = wrong = n = 0
    t0 = time.perf_counter()
    for q, (s1, s3) in zip(queries, truth):
        if not s1:
            continue
        n += 1
        sc = index._scores(q["s"])
        order = np.argsort(-sc)[:top]
        ids = [index.ids[i] for i in order]
        r1 += ids[0] in s1
        r5 += any(i in s1 for i in ids)
        c5 += any(i in s3 for i in ids)
        wrong += room_of[ids[0]] != q["room"]
    dt = (time.perf_counter() - t0) / max(1, len(queries)) * 1e3
    return dict(n=n, recall1=r1 / max(1, n), recall5=r5 / max(1, n), covis5=c5 / max(1, n), wrong_room1=wrong / max(1, n), ms=dt)


def index_bytes(index):
    """Bytes as implemented (float64 arrays) and with the address stored as k active indices (int16)."""
    n = len(index.ids)
    S = sum(a.nbytes for a in index.S)
    T = n * 16 * 8
    if isinstance(index, ScaffoldIndex):
        H = sum(a.nbytes for a in index.H)
        XI = sum(a.nbytes for a in index.XI)
        k = index.block.k
        const = index.mem.W_hs.nbytes + index.mem.W_sh.nbytes + index.mem.C.nbytes + index.mem.Cs.nbytes
        return dict(per_entry=(S + T + H + XI) / max(1, n), per_entry_packed=(S + T + XI + k * 2) / max(1, n), const=const)
    return dict(per_entry=(S + T) / max(1, n), per_entry_packed=(S + T) / max(1, n), const=0)


# ------------------------------------------------------------------ main
def build(seqs, a, prune=None, record_curve=False):
    """Bind bank sequences; return indices, per-room stored poses, curve rows, kept count."""
    rooms = sorted({e["room"] for e in seqs})
    grid = int(np.ceil(np.sqrt(len(rooms))))
    offset = {r: np.array([a.spacing * (i % grid), a.spacing * (i // grid), 0.0]) for i, r in enumerate(rooms)}
    diag = {}
    for r in rooms:
        c = np.concatenate([e["gt"][:, :3, 3] for e in seqs if e["room"] == r]) / a.unit
        diag[r] = float(np.linalg.norm(np.ptp(c, axis=0)))
    by_room = {r: sorted([e for e in seqs if e["room"] == r], key=lambda e: e["seq"]) for r in rooms}
    held = {r: by_room[r][-1] for r in rooms if len(by_room[r]) > 1}
    bank_seqs = [e for r in rooms for e in by_room[r] if not (r in held and e is held[r])]
    D = seqs[0]["desc"].shape[1]
    scaffold = ScaffoldIndex(N_h=a.N_h, torus_N=a.torus_N, seed=a.seed, desc_dim=D)
    flat = DescriptorIndex()
    room_of, stored_by_room, kept, gid = {}, {}, 0, 0
    # queries: every --query-stride-th keyframe of each held-out sequence
    queries = []
    for r, e in held.items():
        for i in range(0, len(e["gt"]), a.query_stride):
            T = e["gt"][i].copy(); T[:3, 3] = T[:3, 3] / a.unit + offset[r]
            queries.append(dict(room=r, T=T, s=e["desc"][i], diag=diag[r]))
    curve = []
    for e in bank_seqs:
        r = e["room"]
        for i in range(len(e["gt"])):
            T = e["gt"][i].copy(); T[:3, 3] = T[:3, 3] / a.unit + offset[r]
            s = e["desc"][i]
            skip = False
            if prune and stored_by_room.get(r):
                ids_r, P_r = stored_by_room[r]
                if prune[0] == "pose":
                    d = np.linalg.norm(P_r[:, :3, 3] - T[:3, 3], axis=1) * a.unit
                    near = np.where(d < prune[1])[0]
                    skip = any(rotation_angle_deg(T[:3, :3].T @ P_r[j, :3, :3]) < 15.0 for j in near)
                elif prune[0] == "cell":
                    h, _ = scaffold.address(T)
                    ov = np.stack(scaffold.H) @ h
                    j = int(np.argmax(ov))
                    skip = ov[j] >= prune[1] and float(scaffold.S[j] @ s) >= 0.8
                elif prune[0] == "cue":
                    skip = float(np.max(flat._scores(s))) >= prune[1]
            if not skip:
                scaffold.add(gid, T, s); flat.add(gid, T, s)
                room_of[gid] = r
                ids_r, P_r = stored_by_room.get(r, ([], np.zeros((0, 4, 4))))
                stored_by_room[r] = (ids_r + [gid], np.concatenate([P_r, T[None]]))
                kept += 1
            gid += 1
        if record_curve:
            truth = room_truth(stored_by_room, queries)
            row = dict(bank=len(scaffold.ids), rooms=len(stored_by_room),
                       scaffold=evaluate(scaffold, queries, truth, room_of), flat=evaluate(flat, queries, truth, room_of),
                       bytes=index_bytes(scaffold))
            curve.append(row)
    truth = room_truth(stored_by_room, queries)
    final = dict(kept=kept, total=gid, bank=len(scaffold.ids),
                 scaffold=evaluate(scaffold, queries, truth, room_of), flat=evaluate(flat, queries, truth, room_of),
                 bytes=index_bytes(scaffold), n_queries=len(queries))
    return curve, final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", default="data/gt"); ap.add_argument("--cache-dir", default="outputs/cache")
    ap.add_argument("--descriptor", default="dino"); ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--dataset", default="7scenes"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--unit", type=float, default=1.5, help="metres per scaffold unit (median depth)")
    ap.add_argument("--spacing", type=float, default=8.0, help="room spacing on the scaffold grid (units)")
    ap.add_argument("--query-stride", type=int, default=4)
    ap.add_argument("--N-h", type=int, default=2048); ap.add_argument("--torus-N", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pixels", type=int, default=518 * 392); ap.add_argument("--conf-keep", type=float, default=0.8); ap.add_argument("--bytes-per-point", type=int, default=19)
    ap.add_argument("--md", default=None); ap.add_argument("--json", default=None)
    ap.add_argument("--debug", action="store_true", help="print why each GT file did or did not get a descriptor cache")
    a = ap.parse_args()
    seqs = synthetic() if a.synthetic else discover(a.gt_dir, a.cache_dir, a.descriptor, a.stride, a.dataset, a.debug)
    if not seqs:
        sys.exit("no (descriptor, GT) pairs found; check --cache-dir / --gt-dir / --descriptor")
    print(f"{len(seqs)} sequences in {len({e['room'] for e in seqs})} rooms:")
    for e in seqs:
        print(f"  {e['room']:<12} seq {e['seq']:>2}  {len(e['gt']):>4} keyframes   {e['src'][0]}  <-  {e['src'][1]}")
    dense = a.pixels * a.conf_keep * a.bytes_per_point
    curve, final = build(seqs, a, record_curve=True)
    lines = ["## Retrieval as the bank grows (queries: last sequence of every room; strict GT revisit = 10 % diag, 30 deg)",
             "| bank entries | rooms | scaffold R@1 | scaffold R@5 | scaffold covis@5 | wrong-room top-1 | flat R@1 | flat R@5 | flat covis@5 | wrong-room top-1 | scaffold ms/query | index B/entry | packed B/entry |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in curve:
        s, f, b = r["scaffold"], r["flat"], r["bytes"]
        lines.append(f"| {r['bank']} | {r['rooms']} | {s['recall1']:.2f} | {s['recall5']:.2f} | {s['covis5']:.2f} | {s['wrong_room1']:.2f} | "
                     f"{f['recall1']:.2f} | {f['recall5']:.2f} | {f['covis5']:.2f} | {f['wrong_room1']:.2f} | {s['ms']:.2f} | {b['per_entry']:.0f} | {b['per_entry_packed']:.0f} |")
    b = final["bytes"]
    lines += ["", f"Constant memory (RLS maps and covariances, N_h={a.N_h}, D={seqs[0]['desc'].shape[1]}): {b['const'] / 1e6:.1f} MB. "
              f"Per-entry index record as implemented: {b['per_entry'] / 1e3:.1f} kB (address as {a.N_h} float64); with the address packed as "
              f"{ScaffoldIndex(N_h=a.N_h, torus_N=a.torus_N).block.k} int16 indices: {b['per_entry_packed'] / 1e3:.1f} kB. "
              f"Dense content per entry if kept: {dense / 1e6:.2f} MB ({a.pixels} px x {a.conf_keep} x {a.bytes_per_point} B). "
              f"Full bank of {final['bank']} entries: index {final['bank'] * b['per_entry'] / 1e6:.0f} MB, dense {final['bank'] * dense / 1e9:.1f} GB. "
              f"Queries: {final['n_queries']} ({final['scaffold']['n']} with a strict GT revisit)."]
    # pruning
    lines += ["", "## Pruning at bind time (full bank order; recall on the same queries)",
              "| rule | entries kept | fraction | scaffold R@1 | scaffold R@5 | flat R@1 | flat R@5 | index MB | dense GB |",
              "|---|---|---|---|---|---|---|---|---|"]
    rules = [("none", None), ("pose spacing 0.10 m", ("pose", 0.10)), ("pose spacing 0.25 m", ("pose", 0.25)), ("pose spacing 0.50 m", ("pose", 0.50)),
             ("scaffold cell overlap >= 0.5 and cue >= 0.8", ("cell", 0.5)), ("scaffold cell overlap >= 0.7 and cue >= 0.8", ("cell", 0.7)),
             ("cue cosine >= 0.95", ("cue", 0.95)), ("cue cosine >= 0.90", ("cue", 0.90))]
    prune_rows = []
    for name, rule in rules:
        _, fin = build(seqs, a, prune=rule)
        s, f = fin["scaffold"], fin["flat"]
        lines.append(f"| {name} | {fin['bank']} | {fin['bank'] / max(1, fin['total']):.2f} | {s['recall1']:.2f} | {s['recall5']:.2f} | {f['recall1']:.2f} | {f['recall5']:.2f} | "
                     f"{fin['bank'] * fin['bytes']['per_entry'] / 1e6:.0f} | {fin['bank'] * dense / 1e9:.2f} |")
        prune_rows.append(dict(rule=name, **{k: v for k, v in fin.items() if k != "bytes"}))
        print(lines[-1])
    out = "\n".join(lines)
    print("\n" + out)
    if a.md:
        pathlib.Path(a.md).write_text(out + "\n"); print("->", a.md)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(dict(curve=curve, final=final, pruning=prune_rows), indent=1, default=float))


if __name__ == "__main__":
    main()
