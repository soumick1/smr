#!/usr/bin/env python3
"""v177 patch for experiments/pilot_a.py, v179 rewrite: structural anchors (regex + parenthesis matching), so it lands
on trees that changed after v167. Idempotent. Refuses, with the reason and the nearby lines, if a target is absent.

    python scripts/apply_v177_patch.py --check      # report where each edit would go
    python scripts/apply_v177_patch.py              # patch (backup pilot_a.py.pre_v177.bak)

Edits: (1) `from smr.utils import provenance`; (2) provenance=... inside `report = dict(...)`; (3) gate flags in argparse;
(4) the flags passed inside `common = dict(...)`. Defaults reproduce every existing run.
"""
from __future__ import annotations
import argparse, pathlib, re, shutil, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]

ARGS = '''    # -- v177: operative verification constants as flags (defaults = every existing run)
    ap.add_argument("--site-rot", type=float, default=10.0, help="per-site gate: max rotation residual of the site's internal pose (deg)")
    ap.add_argument("--site-dir", type=float, default=25.0, help="per-site gate: max translation-direction residual (deg)")
    ap.add_argument("--extent-factor", type=float, default=3.0, help="per-site gate: max distance from the window centroid (spreads)")
    ap.add_argument("--budget-rot", default="10,3,45", help="two-site drift budget: r0,r1,rmax -> min(rmax, r0 + r1*n_stretch) deg")
    ap.add_argument("--budget-pos", default="1.0,0.5", help="two-site drift budget: p0,p1 -> p0 + p1*n_stretch spreads")
    ap.add_argument("--budget-logscale", type=float, default=0.5)
    ap.add_argument("--tight-rot", default="3,1,15", help="single-site budget (a lone site may only nudge)")
    ap.add_argument("--tight-pos", default="0.5,0.15")
    ap.add_argument("--no-smooth-junctions", action="store_true", help="no per-frame blend of the previous window's overlap at a closure")
    ap.add_argument("--top-proposals", type=int, default=5)
'''
COMMON = '''site_rot_deg=a.site_rot, site_dir_deg=a.site_dir, extent_factor=a.extent_factor,
                  budget_rot=tuple(float(x) for x in a.budget_rot.split(",")),
                  budget_pos=tuple(float(x) for x in a.budget_pos.split(",")),
                  budget_logscale=a.budget_logscale,
                  tight_rot=tuple(float(x) for x in a.tight_rot.split(",")),
                  tight_pos=tuple(float(x) for x in a.tight_pos.split(",")),
                  smooth_junctions=not a.no_smooth_junctions, top_proposals=a.top_proposals'''
REPORT = "provenance=provenance.run_record(a, backbone=a.backbone)"


def call_span(src, name):
    """(open_idx, close_idx) of the parentheses of the first `<name> = dict(` statement."""
    m = re.search(r"^[ \t]*" + re.escape(name) + r"\s*=\s*dict\(", src, re.M)
    if not m:
        return None
    i = m.end() - 1
    depth, j, n = 0, i, len(src)
    while j < n:
        c = src[j]
        if c in "\"'":                                   # skip string literals
            q = c; j += 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i, j
        j += 1
    return None


def insert_kwarg(src, name, text):
    span = call_span(src, name)
    if span is None:
        return None, f"no `{name} = dict(` statement"
    i, j = span
    body = src[i + 1:j]
    tail = body.rstrip()
    sep = "" if tail.endswith(",") else ","
    indent = " " * 18
    new = src[:i + 1] + tail + sep + "\n" + indent + text + src[j:]
    line = src[:i].count("\n") + 1
    return new, f"`{name} = dict(` at line {line}, {j - i} chars"


def context(src, pat, before=1, after=1):
    m = re.search(pat, src, re.M)
    if not m:
        return "not found"
    ln = src[:m.start()].count("\n")
    lines = src.splitlines()
    return "\n".join(f"{k + 1:>5}: {lines[k]}" for k in range(max(0, ln - before), min(len(lines), ln + after + 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--target", default=str(ROOT / "experiments" / "pilot_a.py"))
    a = ap.parse_args()
    p = pathlib.Path(a.target)
    src = p.read_text()
    if "provenance.run_record(a" in src and "--budget-rot" in src:
        print("already patched (v177 markers present)"); return 0
    plan, problems = [], []
    # 1. import
    if "from smr.utils import provenance" not in src:
        m = re.search(r"^from smr\.stitch import \(", src, re.M) or re.search(r'^sys\.path\.insert\(0, str\(ROOT / "src"\)\)\n', src, re.M)
        if not m:
            problems.append("import anchor: neither `from smr.stitch import (` nor `sys.path.insert(0, str(ROOT / \"src\"))` found")
        else:
            pos = m.start() if m.group(0).startswith("from") else m.end()
            plan.append(("import", pos, "from smr.utils import provenance  # noqa: E402  (v177)\n"))
    # 2. argparse block
    m = (re.search(r'^[ \t]*ap\.add_argument\("--no-robust-batch"', src, re.M)
         or re.search(r'^[ \t]*ap\.add_argument\("--seed"', src, re.M)
         or re.search(r"^[ \t]*a = ap\.parse_args\(\)", src, re.M))
    if not m:
        problems.append("argparse anchor: none of --no-robust-batch / --seed / `a = ap.parse_args()` found")
    else:
        plan.append(("argparse", m.start(), ARGS))
    if a.check or problems:
        print("import  ->", context(src, r"^from smr\.stitch import \(") if not problems or "import" not in problems[0] else "?")
        print("argparse->", context(src, m.re.pattern) if m else "?")
        for name in ("common", "report"):
            sp = call_span(src, name)
            print(f"{name:<8}->", (f"line {src[:sp[0]].count(chr(10)) + 1}" if sp else "NOT FOUND"))
        if problems:
            print("REFUSING:\n  " + "\n  ".join(problems))
            print("Send `grep -n 'ap.add_argument\\|common = dict\\|report = dict\\|^from smr' experiments/pilot_a.py` and I will re-base.")
            return 2
        if a.check:
            print("all targets found; safe to patch"); return 0
    # apply text insertions from the back so offsets stay valid
    for _, pos, text in sorted(plan, key=lambda t: -t[1]):
        src = src[:pos] + text + src[pos:]
    # 3./4. kwargs (structural)
    src2, msg = insert_kwarg(src, "common", COMMON)
    if src2 is None:
        print("REFUSING:", msg); return 2
    src, msg2 = insert_kwarg(src2, "report", REPORT)
    if src is None:
        print("REFUSING:", msg2); return 2
    shutil.copy2(p, p.with_suffix(".py.pre_v177.bak"))
    p.write_text(src)
    import py_compile; py_compile.compile(str(p), doraise=True)
    print(f"patched {p}: {msg}; {msg2}; compiles (backup {p.with_suffix('.py.pre_v177.bak').name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
