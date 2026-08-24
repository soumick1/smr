#!/usr/bin/env python3
"""Aggregate Pilot A JSON reports into one table (markdown to stdout, LaTeX
to --tex).  Groups by (scene, backbone); rows in the canonical order; the
ceiling appears as the first row of each group with its frame count.

    python scripts/pilot_a_table.py outputs/reports/pilotA_*_v67.json --tex outputs/reports/table_pilotA.tex
"""
import argparse, glob, json, pathlib, sys

ORDER = ["ceiling", "chained", "smr", "smr_pgo", "classical", "smr_jump", "plain"]
COLS = [("ate_rmse", "ATE", "{:.3f}"), ("rpe_trans", "RPE-t1", "{:.3f}"),
        ("auc_within", "AUCin", "{:.1f}"), ("auc_cross", "AUCx", "{:.1f}"),
        ("loop_rot_deg", "loopR", "{:.2f}"), ("loop_trans", "loopT", "{:.3f}"),
        ("n_loops", "loops", "{:d}"), ("n_rejected", "rej", "{:d}"),
        ("sec_per_frame", "s/kf", "{:.2f}"), ("peak_mem_gb", "GB", "{:.1f}")]


def load(paths):
    reps = []
    for p in paths:
        r = json.load(open(p))
        if r.get("SIMULATED"):
            print(f"skipping SIMULATED report {p}", file=sys.stderr)
            continue
        reps.append((p, r))
    return reps


def fmt(row, key, f):
    v = row.get(key)
    if v is None:
        return "---"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--tex", default="")
    a = ap.parse_args()
    paths = sorted(set(sum([glob.glob(p) for p in a.reports], [])))
    reps = load(paths)
    if not reps:
        raise SystemExit("no (non-simulated) reports")
    md, tex = [], []
    hdr = "| method | " + " | ".join(c[1] for c in COLS) + " |"
    tex.append("\\begin{tabular}{@{}ll" + "r" * len(COLS) + "@{}}\n\\toprule")
    tex.append("& & " + " & ".join(c[1] for c in COLS) + " \\\\")
    for p, r in reps:
        title = (f"{r['scene']} ({r['dataset']}), {r['keyframes']} keyframes, "
                 f"{r['n_chunks']} chunks, {r['n_revisit_pairs']} revisit pairs --- {r['backbone']}")
        md += [f"\n**{title}**  gate {'PASSED' if r['gate']['passed'] else 'FAILED'}", "", hdr,
               "|" + "---|" * (len(COLS) + 1)]
        tex.append("\\midrule\n\\multicolumn{%d}{@{}l}{\\textit{%s}} \\\\" % (len(COLS) + 2, title.replace("_", "\\_")))
        rows = {row["method"]: row for row in r["rows"]}
        ce = r.get("ceiling")
        if ce and ce.get("n") and "ceiling" not in rows:
            rows["ceiling"] = dict(method=f"ceiling [{ce['n']}]", ate_rmse=ce["ate"],
                                   sec_per_frame=ce["secs"] / max(1, ce["n"]),
                                   peak_mem_gb=ce["peak_gb"])
        elif "ceiling" in rows and ce:
            rows["ceiling"]["method"] = f"ceiling [{ce['n']}]"
        for m in ORDER:
            if m not in rows:
                continue
            row = rows[m]
            cells = [fmt(row, k, f) for k, f, _ in [(c[0], c[2], 0) for c in COLS]]
            md.append(f"| {row['method']} | " + " | ".join(cells) + " |")
            tex.append("& " + row["method"].replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    tex.append("\\bottomrule\n\\end{tabular}")
    # standard-protocol summary: ATE per sequence, methods as rows, one
    # block per backbone (the layout of VGGT-SLAM Table 1 / MASt3R-SLAM)
    by_bb = {}
    for p, r in reps:
        by_bb.setdefault(r["backbone"], {})[r["scene"]] = {row["method"]: row["ate_rmse"] for row in r["rows"]}
    for bb, scenes in by_bb.items():
        names = sorted(scenes)
        md += [f"\n**ATE (m) per sequence --- {bb}**", "", "| method | " + " | ".join(names) + " | avg |",
               "|" + "---|" * (len(names) + 2)]
        for m in ORDER[1:]:
            vals = [scenes[n].get(m) for n in names]
            if all(v is None for v in vals):
                continue
            got = [v for v in vals if v is not None]
            md.append(f"| {m} | " + " | ".join("---" if v is None else f"{v:.3f}" for v in vals)
                      + f" | {sum(got) / len(got):.3f} |")
    print("\n".join(md))
    if a.tex:
        out = pathlib.Path(a.tex)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(tex) + "\n")
        print(f"\nLaTeX -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
