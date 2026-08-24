#!/usr/bin/env python3
"""ATE / loop error / cross-chunk AUC versus sequence length, from the
length-sweep reports.  Writes <out>.pdf and <out>.png (matplotlib)."""
import argparse, glob, json, pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reps = [json.load(open(p)) for p in sorted(glob.glob(a.reports))]
    reps = [r for r in reps if not r.get("SIMULATED")]
    reps.sort(key=lambda r: r["keyframes"])
    if not reps:
        raise SystemExit("no reports")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["chained", "smr", "smr_pgo"]
    keys = [("ate_rmse", "ATE (m)"), ("loop_trans", "loop translation error (m)"),
            ("auc_cross", "AUC@30, cross-chunk pairs")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    N = [r["keyframes"] for r in reps]
    for ax, (k, lab) in zip(axes, keys):
        for m in methods:
            ys = [next((row[k] for row in r["rows"] if row["method"] == m), np.nan) for r in reps]
            ax.plot(N, ys, marker="o", label=m)
        if k == "ate_rmse":
            ce = [(r["keyframes"], r["ceiling"]["ate"]) for r in reps
                  if r.get("ceiling") and r["ceiling"].get("n") == r["keyframes"]]
            if ce:
                ax.plot([c[0] for c in ce], [c[1] for c in ce], "k--", marker="s",
                        label="single pass (fits)")
        ax.set_xlabel("keyframes"); ax.set_ylabel(lab); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{reps[0]['scene']} -- {reps[0]['backbone']}")
    fig.tight_layout()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"-> {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
