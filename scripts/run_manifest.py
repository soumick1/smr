#!/usr/bin/env python3
"""Run manifest: every report, its configuration and its numbers (v177, plan Task 1).

    python scripts/run_manifest.py --glob 'outputs/**/*.json' 'outputs2/**/*.json' --csv outputs/manifest.csv --md outputs/manifest.md
    python scripts/run_manifest.py --glob 'outputs/ablate/**/*.json' --diff default index_flat   # what differs between two blocks

One row per (report, method row).  Columns: file, mtime, dataset, scene,
backbone, keyframes, stride, chunk, overlap, sites, n_chunks, method,
ate_rmse, auc30, n_loops, n_rejected, n_passes, frames_per_pass, peak_mem_gb,
backbone_secs, elapsed, and the provenance fields (commit, dirty, host,
checkpoint sha) when the report carries them (v177+; older reports show
"none").  --diff prints, for two report directories, the configuration keys
whose values differ per scene, so a "default" block that disagrees with
another block can be explained (different backbone, subset, stride, code
commit) or, when nothing differs, flagged for a rerun.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import pathlib
import re

CFG = ["dataset", "scene", "backbone", "keyframes", "keyframe_stride", "chunk", "overlap", "sites", "n_chunks", "n_revisit_pairs"]
MET = ["ate_rmse", "auc30", "auc_within", "auc_cross", "n_loops", "n_rejected", "n_passes", "frames_per_pass", "peak_mem_gb",
       "backbone_secs", "stitch_secs"]


def load(path):
    try:
        r = json.load(open(path))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(r, dict) or "rows" not in r:
        return None
    return r


def rows_of(path, r):
    p = pathlib.Path(path)
    prov = r.get("provenance") or {}
    git = prov.get("git") or {}
    ck = prov.get("checkpoints") or []
    base = dict(file=str(p), mtime=dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes"),
                simulated=bool(r.get("SIMULATED")), elapsed=r.get("elapsed_secs"),
                commit=(git.get("commit") or "none")[:10], dirty=git.get("dirty", "none"),
                host=(prov.get("env") or {}).get("host", "none"),
                checkpoint=";".join(f"{pathlib.Path(c['path']).name}:{c['sha256']}" for c in ck) or "none",
                argv=(" ".join(prov.get("argv", [])) if prov else "none"))
    for k in CFG:
        base[k] = r.get(k)
    out = []
    for row in r["rows"]:
        d = dict(base, method=row.get("method"))
        for k in MET:
            d[k] = row.get(k)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", nargs="+", default=["outputs/**/*.json", "outputs2/**/*.json"])
    ap.add_argument("--csv", default=None); ap.add_argument("--md", default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="restrict to these row methods")
    ap.add_argument("--diff", nargs=2, metavar=("DIR_A", "DIR_B"), default=None,
                    help="two directories of reports (matched by scene): print differing config/provenance keys")
    a = ap.parse_args()
    paths = sorted({p for g in a.glob for p in glob.glob(g, recursive=True)})
    recs = []
    for p in paths:
        r = load(p)
        if r is None:
            continue
        recs += [x for x in rows_of(p, r) if not a.methods or x["method"] in a.methods]
    print(f"{len(paths)} json files scanned, {len({x['file'] for x in recs})} reports, {len(recs)} rows; "
          f"{sum(1 for x in recs if x['commit'] == 'none')} rows without provenance")
    cols = ["file", "mtime", "simulated"] + CFG + ["method"] + MET + ["elapsed", "commit", "dirty", "host", "checkpoint", "argv"]
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(recs)
        print("csv ->", a.csv)
    if a.md:
        short = ["file", "scene", "backbone", "keyframes", "keyframe_stride", "chunk", "overlap", "sites", "method", "ate_rmse", "auc30",
                 "n_loops", "commit", "checkpoint"]
        lines = ["| " + " | ".join(short) + " |", "|" + "---|" * len(short)]
        for x in recs:
            lines.append("| " + " | ".join("" if x.get(k) is None else (f"{x[k]:.3f}" if isinstance(x[k], float) else str(x[k])) for k in short) + " |")
        pathlib.Path(a.md).write_text("\n".join(lines) + "\n"); print("markdown ->", a.md)
    if a.diff:
        A = {load(p)["scene"]: (p, load(p)) for p in glob.glob(str(pathlib.Path(a.diff[0]) / "*.json")) if load(p)}
        B = {load(p)["scene"]: (p, load(p)) for p in glob.glob(str(pathlib.Path(a.diff[1]) / "*.json")) if load(p)}
        for sc in sorted(set(A) & set(B)):
            ra, rb = A[sc][1], B[sc][1]
            diffs = {k: (ra.get(k), rb.get(k)) for k in CFG if ra.get(k) != rb.get(k)}
            pa, pb = ra.get("provenance") or {}, rb.get("provenance") or {}
            for k in ("commit",):
                va, vb = (pa.get("git") or {}).get(k), (pb.get("git") or {}).get(k)
                if va != vb:
                    diffs["git." + k] = (va, vb)
            ca = {c["path"].split("/")[-1]: c["sha256"] for c in pa.get("checkpoints", [])}
            cb = {c["path"].split("/")[-1]: c["sha256"] for c in pb.get("checkpoints", [])}
            if ca != cb:
                diffs["checkpoints"] = (ca, cb)
            aa = {k: v for k, v in (pa.get("args") or {}).items()}
            ab = {k: v for k, v in (pb.get("args") or {}).items()}
            for k in sorted(set(aa) | set(ab)):
                if k in ("json", "save_est", "verbose") or aa.get(k) == ab.get(k):
                    continue
                diffs["args." + k] = (aa.get(k), ab.get(k))
            ma = {x["method"]: x.get("ate_rmse", x.get("auc30")) for x in ra["rows"]}
            mb = {x["method"]: x.get("ate_rmse", x.get("auc30")) for x in rb["rows"]}
            print(f"{sc}: metric A {ma}  B {mb}")
            if diffs:
                print("   differs:", diffs)
            else:
                hint = []
                for tag, (pth, rr) in (("A", A[sc]), ("B", B[sc])):
                    lg = pathlib.Path(pth).with_suffix(".log")
                    if lg.exists():
                        for line in lg.read_text(errors="ignore").splitlines()[:40]:
                            if re.search(r"index|scaffold|flat|--seed|budget|site", line, re.I):
                                hint.append(f"{tag} log: {line.strip()[:120]}")
                                break
                print("   differs: nothing among the RECORDED fields" +
                      (" (pre-v177 reports do not record --index/--seed/gate flags; the run directory name is the only record)"
                       if not (ra.get("provenance") and rb.get("provenance")) else " nor in provenance -> unexplained; rerun") +
                      (("; " + "; ".join(hint)) if hint else ""))
        for sc in sorted(set(A) ^ set(B)):
            print(f"{sc}: only in {'A' if sc in A else 'B'}")


if __name__ == "__main__":
    main()
