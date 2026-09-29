#!/usr/bin/env python3
"""Compare the candidate sites and accepted closures of two pilot_a reports of the SAME sequence
(v177, plan Task 3: does the index change WHICH revisits are found, or only whether they are accepted?).

    python scripts/compare_closures.py A.json B.json [--row smr]

Prints, per window: sites proposed by A only / B only / both (by historical view id), the accepted closure
(anchor window) of each, and the Jaccard overlap of the accepted-closure sets; then the paired trajectory rows.
"""
import argparse, json, sys

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--row", default="smr")
a = ap.parse_args()
RA, RB = json.load(open(a.a)), json.load(open(a.b))
ea = {e["chunk"]: e for e in (RA["events"].get(a.row) or next(iter(RA["events"].values())))}
eb = {e["chunk"]: e for e in (RB["events"].get(a.row) or next(iter(RB["events"].values())))}
if RA.get("scene") != RB.get("scene") or RA.get("keyframes") != RB.get("keyframes"):
    print(f"WARNING: different sequences? {RA.get('scene')}/{RA.get('keyframes')} vs {RB.get('scene')}/{RB.get('keyframes')}")
acc_a, acc_b = set(), set()
same_sites = 0
print(f"{'win':>4} {'A sites (view:ok)':<34} {'B sites (view:ok)':<34} {'A closure':<12} {'B closure':<12}")
for k in sorted(set(ea) | set(eb)):
    A, B = ea.get(k, {}), eb.get(k, {})
    sa = {s["view"]: s["ok"] for s in A.get("sites", [])}; sb = {s["view"]: s["ok"] for s in B.get("sites", [])}
    la, lb = A.get("loop") or {}, B.get("loop") or {}
    ca = f"->{la['anchor_chunk']} {'ACC' if la.get('accepted') else la.get('reason', 'rej')}" if la else ""
    cb = f"->{lb['anchor_chunk']} {'ACC' if lb.get('accepted') else lb.get('reason', 'rej')}" if lb else ""
    if la.get("accepted"): acc_a.add((k, la["anchor_chunk"]))
    if lb.get("accepted"): acc_b.add((k, lb["anchor_chunk"]))
    same_sites += int(set(sa) == set(sb))
    if sa or sb or la or lb:
        fa = " ".join(f"{v}:{int(o)}" for v, o in sorted(sa.items())); fb = " ".join(f"{v}:{int(o)}" for v, o in sorted(sb.items()))
        mark = "" if set(sa) == set(sb) else "  <- different candidates"
        print(f"{k:>4} {fa:<34} {fb:<34} {ca:<12} {cb:<12}{mark}")
u = acc_a | acc_b
print(f"\nwindows with identical candidate sets: {same_sites}/{len(set(ea) | set(eb))}")
print(f"accepted closures: A {len(acc_a)}  B {len(acc_b)}  both {len(acc_a & acc_b)}  Jaccard {len(acc_a & acc_b) / max(1, len(u)):.2f}")
for R, name in ((RA, "A"), (RB, "B")):
    for r in R["rows"]:
        if r["method"] in ("chained", a.row, a.row + "_pgo"):
            print(f"  {name} {r['method']:<8} ATE {r.get('ate_rmse', float('nan')):.4f}  AUC30 {r.get('auc30', float('nan')):.1f}  "
                  f"loops {r.get('n_loops')}  rejected {r.get('n_rejected')}")
