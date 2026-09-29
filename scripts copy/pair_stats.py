#!/usr/bin/env python3
"""Report what the completion corpus actually contains: coverage and
view-novelty histograms per context class.

    python scripts/pair_stats.py data/completion_co3d data/completion
"""
import pathlib, re, sys

import numpy as np

CTX = re.compile(r"_c(all|\d{2})\.npz$")


def main():
    files = sorted(sum([list(pathlib.Path(d).glob("*.npz"))
                        for d in sys.argv[1:]], []))
    assert files, "no pairs given"
    by = {}
    for f in files:
        m = CTX.search(f.name)
        c = m.group(1) if m else "all(pre-tag)"
        d = np.load(f)
        by.setdefault(c, []).append((
            float(d["splat_mask"].mean()),
            float(np.degrees(d["novelty_rot"])) if "novelty_rot" in d
            else float("nan"),
            float(d["novelty_pos"]) if "novelty_pos" in d else float("nan")))
    print(f"{len(files)} pairs")
    print(f"{'ctx':>12} {'n':>5} {'coverage':>18} {'novelty rot(deg)':>18} "
          f"{'novelty pos':>14}")
    for c in sorted(by):
        a = np.array(by[c])
        cov, rot, pos = a[:, 0], a[:, 1], a[:, 2]
        def s(x):
            x = x[~np.isnan(x)]
            return (f"{np.median(x):.2f} [{np.percentile(x,10):.2f},"
                    f"{np.percentile(x,90):.2f}]") if len(x) else "n/a (pre-novelty pairs)"
        print(f"{c:>12} {len(a):>5} {s(cov):>18} {s(rot):>18} {s(pos):>14}")


if __name__ == "__main__":
    main()
