#!/usr/bin/env python3
"""Relocalisation into memory (Tables T2 / T3).

Sessions of one 7-Scenes scene are stitched into a memory (mapping); every
keyframe of the held-out session(s) is then localised from ONE image.

    python experiments/relocalise.py --gt data/gt/7scenes_chess_s0106.npz --backbone vggt_omega \
        --map-seqs 1,2,4,6 --query-seqs 3,5 --rows lastk,plain,smr,oracle \
        --corrupt none,gauss:0.08,occlude:0.3,blur:4,dark:0.5

Rows: lastk (no memory: query passed with the last K map keyframes), plain
(descriptor nearest neighbour), smr (memory address cue), oracle (sites by
ground-truth proximity; diagnostic).  Corruptions degrade the QUERY image
before both the descriptor and the backbone pass.  Map errors are metric
because the map is aligned to ground truth by one Sim(3); queries are not.
"""
import argparse, json, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smr.stitch import (AnchoredStitcher, BackboneRunner, DescriptorIndex,     # noqa: E402
                        PassCache, ScaffoldIndex, image_descriptors)
from smr.stitch.chunks import make_session_chunks                              # noqa: E402
from smr.stitch.relocalise import MemoryMap, corrupt, localise, pose_error, summarise   # noqa: E402


def descriptors_for(paths, kind, device, cache_path):
    cache_path = pathlib.Path(cache_path)
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True).item()
        if d.get("paths") == list(paths):
            return d["desc"]
    if kind == "dino":
        from smr.stitch.passes import dino_descriptors
        desc = dino_descriptors(list(paths), device=device)
    else:
        desc = image_descriptors(list(paths))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, dict(paths=list(paths), desc=desc), allow_pickle=True)
    return desc


def corrupted_paths(paths, spec, out_dir, seed=0):
    """Write degraded copies of the query images once; return their paths."""
    if spec in ("", "none"):
        return list(paths)
    from PIL import Image
    out_dir = pathlib.Path(out_dir) / spec.replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = []
    for i, p in enumerate(paths):
        q = out_dir / f"q{i:05d}.png"
        if not q.exists():
            img = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
            Image.fromarray((corrupt(img, spec, rng) * 255).astype(np.uint8)).save(q)
        out.append(str(q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="multi-session npz (indoor_gt_poses.py --seq a,b,c)")
    ap.add_argument("--backbone", default="vggt_omega")
    ap.add_argument("--map-seqs", required=True, help="sessions used to build the memory, e.g. 1,2,4,6")
    ap.add_argument("--query-seqs", required=True, help="held-out sessions, e.g. 3,5")
    ap.add_argument("--keyframe-stride", type=int, default=10)
    ap.add_argument("--query-stride", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--sites", type=int, default=2)
    ap.add_argument("--K", type=int, default=15, help="frames for the last-K baseline")
    ap.add_argument("--rows", default="lastk,plain,smr,oracle")
    ap.add_argument("--corrupt", default="none",
                    help="comma list of query degradations: none, gauss:s, occlude:f, blur:px, dark:f")
    ap.add_argument("--map-solver", default="batch", choices=["stream", "batch"],
                    help="mapping is offline: the batch solve (with outlier-edge rejection) "
                         "gives the better map, and a query inherits its anchors' map error")
    ap.add_argument("--descriptor", default="dino", choices=["dino", "rgb"])
    ap.add_argument("--desc-thresh", type=float, default=0.5)
    ap.add_argument("--N-h", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    npz = np.load(a.gt, allow_pickle=True)
    paths = [str(s) for s in npz["image_paths"]]
    gt = np.asarray(npz["poses"], float)
    fid = np.asarray(npz["frame_ids"])
    sess = fid // 100000
    scene = str(npz["scene"]) if "scene" in npz else pathlib.Path(a.gt).stem
    map_seqs = [int(x) for x in a.map_seqs.split(",")]
    q_seqs = [int(x) for x in a.query_seqs.split(",")]
    map_idx = [i for i in range(len(paths)) if sess[i] in map_seqs][::a.keyframe_stride]
    q_idx = [i for i in range(len(paths)) if sess[i] in q_seqs][::a.query_stride]
    if a.max_queries:
        q_idx = q_idx[:a.max_queries]
    map_paths = [paths[i] for i in map_idx]
    print(f"scene {scene} | map sessions {map_seqs}: {len(map_idx)} keyframes | "
          f"query sessions {q_seqs}: {len(q_idx)} queries | backbone {a.backbone}")

    cache_dir = ROOT / "outputs" / "cache"
    tag = f"reloc_{scene}_{a.backbone}_m{''.join(map(str, map_seqs))}_s{a.keyframe_stride}"
    # the query pass cache does not depend on the map solver; the map file name does not either
    # ---------------------------------------------------------- mapping
    map_desc = descriptors_for(map_paths, a.descriptor, a.device, cache_dir / f"{tag}.desc_{a.descriptor}.npy")
    runner = BackboneRunner(a.backbone, map_paths, device=a.device)
    cache = PassCache(cache_dir / f"{tag}.npy")
    chunks = make_session_chunks([int(sess[i]) for i in map_idx], a.chunk, a.overlap)
    t0 = time.time()
    index = ScaffoldIndex(N_h=a.N_h, seed=a.seed, desc_dim=map_desc.shape[1])
    res = AnchoredStitcher(index, n_sites=a.sites, desc_thresh=a.desc_thresh).run(
        chunks, cache, runner, map_desc)
    from smr.eval.trajectory import ate_rmse
    stream_ate = float(ate_rmse(res["est"], gt[map_idx]))
    if a.map_solver == "batch":
        from smr.stitch import posegraph
        pg, info = posegraph.solve(res, chunks)
        res = dict(res); res["est"] = pg
    mmap = MemoryMap(res, gt[map_idx], index, map_desc)
    print(f"  map: {len(chunks)} chunks, {res['n_loops']} closures, ATE {mmap.map_ate:.3f} m "
          f"({a.map_solver}; streaming {stream_ate:.3f}) ({time.time() - t0:.0f}s)")
    plain = DescriptorIndex()
    for g in mmap.frames:
        plain.add(g, mmap.pose[g], map_desc[g])
    mmap_plain = MemoryMap.__new__(MemoryMap)
    mmap_plain.__dict__.update(mmap.__dict__); mmap_plain.index = plain

    # ---------------------------------------------------------- queries
    report = dict(scene=scene, backbone=a.backbone, map_seqs=map_seqs, query_seqs=q_seqs,
                  n_map=len(map_idx), n_queries=len(q_idx), map_ate=mmap.map_ate,
                  map_ate_stream=stream_ate, map_solver=a.map_solver,
                  map_loops=res["n_loops"], results={})
    rows = [r for r in a.rows.split(",") if r]
    for spec in [s.strip() for s in a.corrupt.split(",") if s.strip()]:
        qpaths = corrupted_paths([paths[i] for i in q_idx], spec,
                                 ROOT / "outputs" / "reloc_queries" / scene, seed=a.seed)
        q_desc = descriptors_for(qpaths, a.descriptor, a.device,
                                 cache_dir / f"{tag}_q{''.join(map(str, q_seqs))}_{spec.replace(':', '_')}.desc_{a.descriptor}.npy")
        all_paths = map_paths + qpaths
        qrunner = BackboneRunner(a.backbone, all_paths, device=a.device)
        qcache = PassCache(cache_dir / f"{tag}_q{''.join(map(str, q_seqs))}_{spec.replace(':', '_')}.npy")
        report["results"][spec] = {}
        for row in rows:
            m = mmap_plain if row == "plain" else mmap
            errs, reasons, t1 = [], {}, time.time()
            for qi, gi in enumerate(q_idx):
                q_global = len(map_paths) + qi

                def run_pass(anchors, q_global=q_global):
                    return qcache.get([q_global] + [int(g) for g in anchors], qrunner)["poses"]

                out = localise(m, q_desc[qi], run_pass, mode=row, n_sites=a.sites,
                               gt_pose=gt[gi], K=a.K, desc_thresh=a.desc_thresh)
                errs.append(pose_error(out["T"], gt[gi]))
                reasons[out["reason"]] = reasons.get(out["reason"], 0) + 1
            s = summarise(errs)
            s.update(reasons=reasons, sec_per_query=round((time.time() - t1) / max(1, len(q_idx)), 3),
                     errors=[[float(e[0]), float(e[1])] for e in errs])
            report["results"][spec][row] = s
            print(f"  [{spec:>12}] {row:<7} median {s['median_t_cm']:6.1f} cm / {s['median_r_deg']:5.2f} deg | "
                  f"recall@5cm5deg {s['recall_5cm_5deg']:.2f}  @10/10 {s['recall_10cm_10deg']:.2f}  "
                  f"@25/25 {s['recall_25cm_25deg']:.2f} | fail {s['fail_rate']:.2f} | {s['sec_per_query']:.2f} s/q | {reasons}")
    out = pathlib.Path(a.json or (ROOT / "outputs" / "reports" / f"reloc_{scene}_{a.backbone}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
