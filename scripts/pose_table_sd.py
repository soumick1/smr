#!/usr/bin/env python3
"""Across-sequence mean and standard deviation of AUC@30 for the pose-length table (Table 1).

    python scripts/pose_table_sd.py --glob 'outputs/**/*.json' --datasets co3d re10k
    python scripts/pose_table_sd.py --glob 'outputs/pose/**/*.json' --metric auc30 --rows chained smr native

Every JSON with a `rows` list is a run report for one sequence. Reports are grouped by (backbone, dataset, N, row name),
where N is taken from the first of: report n_keyframes / n_frames / len(keyframes) / args.max_frames / len(est row).
For each group the script prints n, mean and SD (ddof = 1) of the metric, then a grid (backbone x N) of "mean (sd)" per
dataset and row, and a LaTeX grid of the SDs alone in the order of the manuscript's Table 1. Check that the printed means
reproduce the table; if a cell has the wrong n or mean, narrow --glob or use --exclude to drop directories (ablations,
seeds, gate sweeps) that also contain reports for the same backbone.
"""
import argparse, glob, json, pathlib, re
import numpy as np

ORDER_BB = ["dust3r", "fast3r", "mast3r", "vggt", "vggt_omega", "vggt_omega_v99", "pi3", "streamvggt", "stream3r"]
NS = [10, 50, 100, 200]


STRIDE_TO_N = {20: 10, 4: 50, 2: 100, 1: 200}


def n_of(r):
    for k in ("n_keyframes", "n_frames", "num_frames", "n_views", "N"):
        if isinstance(r.get(k), (int, float)):
            return int(r[k])
    for k in ("keyframes", "frame_ids", "frames", "key"):
        if isinstance(r.get(k), list) and r[k]:
            return len(r[k])
    a = r.get("args") or (r.get("provenance") or {}).get("args") or {}
    for k in ("max_frames", "n_frames", "N"):
        if isinstance(a.get(k), (int, float)):
            return int(a[k])
    for row in r.get("rows", []):
        for k in ("n", "n_frames", "n_keyframes", "frames"):
            if isinstance(row.get(k), (int, float)):
                return int(row[k])
    stem = pathlib.Path(r.get("_path", "")).name
    if stem.startswith("t1b_"):
        return 200                                                     # the 37-orbit N=200 runs
    m = re.search(r"_s(\d+)\.json$", r.get("_path", ""))
    if m:
        return STRIDE_TO_N.get(int(m.group(1)), -int(m.group(1)))      # negative = unmapped stride, reported as is
    m = re.search(r"[_/-]N?(10|50|100|200)(?:[_/.-]|$)", r.get("_path", ""))
    return int(m.group(1)) if m else None


def metric_of(row, metric):
    """row[metric] if present, else the first key whose lower-cased letters/digits contain the metric's (e.g. 'auc@30', 'AUC_30')."""
    if isinstance(row.get(metric), (int, float)):
        return float(row[metric])
    want = re.sub(r"[^a-z0-9]", "", metric.lower())
    for k, v in row.items():
        if isinstance(v, (int, float)) and want in re.sub(r"[^a-z0-9]", "", str(k).lower()):
            return float(v)
    m = row.get("metrics")
    if isinstance(m, dict):
        return metric_of(m, metric)
    return None


def ds_of(r):
    d = str(r.get("dataset", "")).lower()
    p = r.get("_path", "").lower()
    stem = pathlib.Path(p).name
    if stem.startswith("re10kaxis") or stem.startswith("suite_pose_re10k"):
        return "re10k"
    if stem.startswith(("t1axis", "t1b_", "suite_pose_vggt", "suite_pose_co3d")):
        return "co3d"
    for name in ("co3d", "re10k", "realestate", "7scenes", "sevenscenes", "tum", "kitti", "scannet"):
        if name in d or name in p:
            return {"realestate": "re10k", "sevenscenes": "7scenes"}.get(name, name)
    return d or "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True); ap.add_argument("--metric", default="auc_all", help="auc_all = AUC@30 over all pairs (Table 1); auc_within / auc_cross / ate_rmse also valid")
    ap.add_argument("--datasets", nargs="+", default=["re10k", "co3d"]); ap.add_argument("--rows", nargs="+", default=["chained", "smr", "smr_pgo", "native"])
    ap.add_argument("--exclude", nargs="*", default=["ablate", "seeds", "gate", "localfrom", "vpr", "gated", "nvs", "mech_", "fig_", "_omega.json"], help="path substrings to skip (default drops ablations, the mech/fig duplicates and the dead t1b _omega run)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    groups, strides, row_names = {}, {}, set()
    suite_sample_shown = [False]; stream_sample_shown = [False]
    files = [f for f in glob.glob(a.glob, recursive=True) if not any(x in f for x in a.exclude)]
    for f in files:
        try:
            r = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(r, dict):
            continue
        r["_path"] = f
        ds = ds_of(r)
        if ds not in a.datasets:
            continue
        # --- aggregate suite files (vggt_suite.py): per_seq records at N = 10; single pass = chained, consensus = smr
        if isinstance(r.get("per_seq"), (list, dict)) and pathlib.Path(f).name.startswith("suite_pose_re10k"):
            bb = str(r.get("backbone", pathlib.Path(f).stem.split("_")[-1])).lower()
            recs = r["per_seq"] if isinstance(r["per_seq"], list) else list(r["per_seq"].values())
            if recs and not suite_sample_shown[0]:
                print(f"suite sample {f}: per_seq[0] = {json.dumps(recs[0])[:400]}"); suite_sample_shown[0] = True
            for i, rec in enumerate(recs):
                if not isinstance(rec, dict):
                    continue
                sid = str(rec.get("seq", rec.get("id", rec.get("scene", i))))
                for role, keys in (("chained", ("single", "raw", "baseline", "auc30_single", "auc_single")),
                                   ("smr", ("consensus", "smr", "auc30_consensus", "auc_consensus"))):
                    val = None
                    for k in keys:
                        v = rec.get(k)
                        if isinstance(v, dict):
                            v = metric_of(v, a.metric) or metric_of(v, "auc30")
                        if isinstance(v, (int, float)):
                            val = float(v); break
                    if val is None:
                        for k, v in rec.items():                        # e.g. "auc30_single", "single_auc30"
                            if isinstance(v, (int, float)) and "auc" in k.lower() and any(t in k.lower() for t in keys):
                                val = float(v); break
                    if val is not None and role in a.rows:
                        if val <= 1.5:
                            val *= 100.0                                   # suite files store AUC as a fraction
                        groups.setdefault((bb, ds, "10s", role), {})[sid] = val    # "10s": the 1,653 / 300-clip N=10 set
                        strides.setdefault((bb, ds, "10s", role), set()).add("suite")
            continue
        if pathlib.Path(f).name.startswith("suite_pose") or not isinstance(r.get("rows"), list):
            continue
        bb = str(r.get("backbone", "?")).lower(); N = n_of(r); seq = r.get("scene") or pathlib.Path(f).stem
        if pathlib.Path(f).name.startswith("t1b_"):
            tail = pathlib.Path(f).stem.split("_")[-1]
            bb = {"omega": "vggt_omega", "v99": "vggt_omega_v99"}.get(tail, bb)
        # --- ceiling = one pass over all N frames = native causal inference for streaming backbones
        if bb in ("streamvggt", "stream3r") and not stream_sample_shown[0]:
            print(f"streaming sample {f}: ceiling={json.dumps(r.get('ceiling'))[:200]} reference={json.dumps(r.get('reference'))[:200]} probe={json.dumps(r.get('probe'))[:200]}")
            stream_sample_shown[0] = True
        c = r.get("ceiling") if isinstance(r.get("ceiling"), dict) else (r.get("reference") if isinstance(r.get("reference"), dict) else None)
        if isinstance(c, dict) and "native" in a.rows:
            val = metric_of(c, a.metric) or metric_of(c, "auc30")
            if val is not None:
                key = (bb, ds, N, "native")
                groups.setdefault(key, {})[re.sub(r"_s\d+$", "", pathlib.Path(f).stem)] = val
                strides.setdefault(key, set()).add(n_of(r) and int(re.search(r"_s(\d+)\.json$", f).group(1)) if re.search(r"_s(\d+)\.json$", f) else 1)
                row_names.add("native(ceiling)")
        ms = re.search(r"_s(\d+)\.json$", f); stride = int(ms.group(1)) if ms else None
        seq = re.sub(r"_s\d+$", "", pathlib.Path(f).stem)          # one key per sequence across strides
        for row in r["rows"]:
            nm = next((row[k] for k in ("name", "row", "variant", "label", "method") if isinstance(row.get(k), str)), None)
            row_names.add(nm)
            name = next((row[k] for k in ("name", "row", "variant", "label", "method") if isinstance(row.get(k), str)), None)
            val = metric_of(row, a.metric)
            if name in a.rows and val is not None:
                groups.setdefault((bb, ds, N, name), {})[seq] = val   # dict keyed by sequence: duplicates collapse
                strides.setdefault((bb, ds, N, name), set()).add(stride)
    sample = next((f for f in files if pathlib.Path(f).name.startswith(("t1axis", "re10kaxis"))), files[0] if files else None)
    if sample:
        rr = json.load(open(sample)); print(f"sample {sample}:\n  top-level keys {sorted(rr.keys())}\n  row[0] {json.dumps(rr['rows'][0])[:600]}")
    if not groups:
        raise SystemExit("no reports grouped: the row-name or metric keys above are not among the expected ones; send this output")
    print(f"{len(files)} files scanned; row names seen: {sorted(x for x in row_names if x)}; groups (N from stride: s20->10, s4->50, s2->100, s1->200):")
    for key in sorted(groups, key=lambda k: (k[1], ORDER_BB.index(k[0]) if k[0] in ORDER_BB else 99, str(k[2]), k[3])):
        v = np.array(list(groups[key].values()))
        print(f"  {key[1]:<6} {key[0]:<11} N={str(key[2]):<5} stride={sorted(map(str, strides[key]))} {key[3]:<8} n={len(v):<5} mean {v.mean():.2f}  sd {v.std(ddof=1) if len(v) > 1 else float('nan'):.2f}")
    lines = []
    for ds in a.datasets:
        for row in a.rows:
            cells = []
            for bb in ORDER_BB:
                vals = []
                for N in NS:
                    v = groups.get((bb, ds, N, row))
                    if N == 10 and groups.get((bb, ds, "10s", row)):
                        v = groups[(bb, ds, "10s", row)]           # the table's N=10 column is the large set
                    if v and len(v) > 1:
                        arr = np.array(list(v.values())); vals.append(f"{arr.std(ddof=1):.1f}")
                    else:
                        vals.append("--")
                if any(x != "--" for x in vals):
                    cells.append(f"{bb:<11} & " + " & ".join(vals) + r" \\")
            if cells:
                lines.append(f"% {ds} {row}: SD of {a.metric} across sequences at N = 10 / 50 / 100 / 200")
                lines += cells
    out = "\n".join(lines)
    print("\n" + out)
    if a.out:
        pathlib.Path(a.out).write_text(out + "\n"); print("->", a.out)


if __name__ == "__main__":
    main()