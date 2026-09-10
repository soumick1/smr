#!/usr/bin/env python3
"""Paired statistics for every comparison in the paper (v177, plan Task 8).

    # two variants, matched by scene (e.g. scaffold vs flat index):
    python scripts/paired_stats.py reports --a 'outputs/ablate/7scenes/default/*.json' \
        --b 'outputs/ablate/7scenes/index_flat/*.json' --row smr --metric ate_rmse --per-item
    # two rows of the same reports (raw vs +SMR):
    python scripts/paired_stats.py rows --reports 'outputs2/reports/pilotA_7scenes_*_seq01_vggt_omega_s5_c32.json' \
        --row-a chained --row-b smr --metric ate_rmse
    # NVS per-object jsonl (raw vs read):
    python scripts/paired_stats.py nvs --a outputs/nvs/vggt/gso_reads1.jsonl --b outputs/nvs/vggt/gso_reads4.jsonl --metric psnr
    # any CSV with an id column and two value columns:
    python scripts/paired_stats.py csv --file pairs.csv --id-col scene --a-col raw --b-col smr --metric ate_rmse

Reports: mean of A and B, mean/median paired delta (B - A), a 95 % paired
bootstrap interval of the aggregate difference (resampling unit = the paired
item: sequence for pose, object for NVS; the aggregate rule is recomputed in
every resample), proportion of items improved, exact sign-test p-value and
the Wilcoxon signed-rank p-value.  Direction is inferred from the metric
name (ate/lpips/chamfer lower is better; auc/psnr/ssim higher) or forced
with --higher-better / --lower-better.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pathlib
import re
import sys

import numpy as np

LOWER = ("ate", "lpips", "chamfer", "rpe", "err", "loop_rot", "loop_trans", "acc", "comp")
HIGHER = ("auc", "psnr", "ssim", "recall", "precision")


def direction(metric, a):
    if a.higher_better:
        return +1
    if a.lower_better:
        return -1
    m = metric.lower()
    if any(m.startswith(k) or k in m for k in HIGHER):
        return +1
    if any(m.startswith(k) or k in m for k in LOWER):
        return -1
    sys.exit(f"cannot infer direction for {metric}; pass --higher-better or --lower-better")


def sign_test_p(n_pos, n_neg):
    """Two-sided exact binomial test on the non-tied items."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def wilcoxon_p(d):
    try:
        from scipy.stats import wilcoxon
        d = np.asarray(d, float)
        d = d[d != 0]
        if len(d) < 6:
            return None
        return float(wilcoxon(d).pvalue)
    except Exception:  # noqa: BLE001
        return None


def scene_of(name):
    """chess_seq03_vggt_omega -> chess; used by --macro."""
    m = re.match(r"^(.*?)_seq\d+", str(name))
    return m.group(1) if m else str(name)


def paired(names, va, vb, sign, agg="mean", n_boot=10000, seed=0, per_item=False, label_a="A", label_b="B", macro=False):
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    d = vb - va
    groups = np.array([scene_of(x) for x in names])
    if macro:
        # per-scene mean, then mean over scenes (the handover / sevenscenes_table convention)
        def f(v, idx=None):
            g = groups if idx is None else groups[idx]
            return float(np.mean([np.mean(v[g == sc]) for sc in np.unique(g)]))
        print(f"aggregate: macro mean over {len(np.unique(groups))} scenes " + str({str(k): int(v) for k, v in zip(*np.unique(groups, return_counts=True))}))
    else:
        f0 = np.mean if agg == "mean" else np.median
        f = lambda v, idx=None: float(f0(v))  # noqa: E731
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        boots[b] = f(vb[i], i) - f(va[i], i)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    improved = int(np.sum(sign * d > 0))
    worsened = int(np.sum(sign * d < 0))
    out = dict(n=n, agg_a=float(f(va)), agg_b=float(f(vb)), delta=float(f(vb) - f(va)), median_delta=float(np.median(d)),
               ci_lo=float(lo), ci_hi=float(hi), improved=improved, worsened=worsened, ties=n - improved - worsened,
               sign_p=sign_test_p(improved, worsened), wilcoxon_p=wilcoxon_p(d),
               mean_gain_when_improved=(float(np.mean(np.abs(d[sign * d > 0]))) if improved else 0.0),
               mean_loss_when_worsened=(float(np.mean(np.abs(d[sign * d < 0]))) if worsened else 0.0))
    print(f"n={n}  {label_a} {agg} {out['agg_a']:.4f}  {label_b} {agg} {out['agg_b']:.4f}  delta {out['delta']:+.4f} "
          f"[95% paired bootstrap {lo:+.4f}, {hi:+.4f}]  median delta {out['median_delta']:+.4f}")
    print(f"improved {improved}/{n}  worsened {worsened}/{n}  ties {out['ties']}  "
          f"(gain when improved {out['mean_gain_when_improved']:.4f}, loss when worsened {out['mean_loss_when_worsened']:.4f})  "
          f"sign p={out['sign_p']:.3g}" + (f"  Wilcoxon p={out['wilcoxon_p']:.3g}" if out["wilcoxon_p"] is not None else ""))
    if per_item:
        order = np.argsort(-sign * d)
        print(f"{'item':<40} {label_a:>10} {label_b:>10} {'delta':>10}")
        for i in order:
            tag = "+" if sign * d[i] > 0 else ("-" if sign * d[i] < 0 else "=")
            print(f"{str(names[i]):<40} {va[i]:>10.4f} {vb[i]:>10.4f} {d[i]:>+10.4f} {tag}")
    return out


STRIP = re.compile(r"(_(template|flat|scaffold|default|seed\d+))+$")


def load_reports(pattern, key="stem"):
    """Pair reports by file stem (variant suffixes such as _template/_flat
    stripped) or, with key="scene", by the report's scene field.  Stems are
    the default because several sequences of one scene share a scene name
    (chess_seq03 and chess_seq05 both say "chess") and would overwrite each
    other."""
    out = {}
    for p in sorted(glob.glob(pattern)):
        r = json.load(open(p))
        k = str(r.get("scene") or pathlib.Path(p).stem) if key == "scene" else STRIP.sub("", pathlib.Path(p).stem)
        if k in out:
            print(f"WARNING: pairing key {k!r} appears twice ({out[k][0]} and {p}); the later file wins")
        out[k] = (p, r)
    return out


def row_metric(r, row, metric):
    for x in r.get("rows", []):
        if x.get("method") == row:
            v = x.get(metric)
            return None if v is None else float(v)
    return None


def load_jsonl(path, metric):
    out, reads, errors = {}, {}, 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        reads[rec.get("reads")] = reads.get(rec.get("reads"), 0) + 1
        if "error" in rec or metric not in rec:
            errors += int("error" in rec)
            continue
        out[str(rec["id"])] = float(rec[metric])
    print(f"{pathlib.Path(path).name}: {len(out)} objects, reads field {reads}, {errors} error records")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["reports", "rows", "nvs", "csv"])
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--reports")
    ap.add_argument("--row", default="smr"); ap.add_argument("--row-a", default="chained"); ap.add_argument("--row-b", default="smr")
    ap.add_argument("--metric", default="ate_rmse")
    ap.add_argument("--file"); ap.add_argument("--id-col", default="id"); ap.add_argument("--a-col"); ap.add_argument("--b-col")
    ap.add_argument("--agg", default="mean", choices=["mean", "median"])
    ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-item", action="store_true")
    ap.add_argument("--higher-better", action="store_true"); ap.add_argument("--lower-better", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--key", default="stem", choices=["stem", "scene"], help="pairing key for report modes")
    ap.add_argument("--macro", action="store_true", help="aggregate = mean over scenes of the per-scene mean (items grouped by the name before _seqNN)")
    a = ap.parse_args()
    sign = direction(a.metric, a)
    names, va, vb = [], [], []
    if a.mode == "reports":
        A, B = load_reports(a.a, a.key), load_reports(a.b, a.key)
        for k in sorted(set(A) & set(B)):
            x, y = row_metric(A[k][1], a.row, a.metric), row_metric(B[k][1], a.row, a.metric)
            if x is not None and y is not None:
                names.append(k); va.append(x); vb.append(y)
        missing = sorted(set(A) ^ set(B))
        if missing:
            print(f"unmatched items ignored: {missing}")
        la, lb = "A", "B"
    elif a.mode == "rows":
        R = load_reports(a.reports, a.key)
        for k in sorted(R):
            x, y = row_metric(R[k][1], a.row_a, a.metric), row_metric(R[k][1], a.row_b, a.metric)
            if x is not None and y is not None:
                names.append(k); va.append(x); vb.append(y)
        la, lb = a.row_a, a.row_b
    elif a.mode == "nvs":
        A, B = load_jsonl(a.a, a.metric), load_jsonl(a.b, a.metric)
        for k in sorted(set(A) & set(B)):
            names.append(k); va.append(A[k]); vb.append(B[k])
        print(f"objects: {len(A)} in A, {len(B)} in B, {len(names)} paired")
        la, lb = pathlib.Path(a.a).stem, pathlib.Path(a.b).stem
    else:
        with open(a.file) as f:
            for rec in csv.DictReader(f):
                try:
                    names.append(rec[a.id_col]); va.append(float(rec[a.a_col])); vb.append(float(rec[a.b_col]))
                except (KeyError, ValueError):
                    continue
        la, lb = a.a_col, a.b_col
    if len(names) < 2:
        sys.exit(f"only {len(names)} paired items")
    print(f"metric {a.metric} ({'higher' if sign > 0 else 'lower'} is better), aggregate {a.agg}")
    out = paired(names, va, vb, sign, a.agg, a.n_boot, a.seed, a.per_item, la, lb, a.macro)
    if a.json_out:
        out.update(metric=a.metric, mode=a.mode, items=names, a=va, b=vb)
        pathlib.Path(a.json_out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
