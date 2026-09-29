#!/usr/bin/env python3
"""v181: `local_from` for the anchored stitcher WITHOUT touching the constructor (supersedes apply_v180_patch.py).

    python scripts/apply_v181_patch.py --check
    python scripts/apply_v181_patch.py

anchored.py: (1) class attribute `local_from = "anchored"` before `def __init__` (pilot_a sets the class attribute, tests the instance);
             (2) the plain-pass block right after `pos = {gi: li for li, gi in enumerate(pass_idx)}`;
             (3) `anchored_distortion=...` inside `ev = dict(chunk=k, ...)`;
             (4) the junction blend is skipped when correction == "none".
pilot_a.py:  `--local-from {anchored,plain}` and `AnchoredStitcher.local_from = a.local_from` right after argument
             parsing (a class attribute, so no constructor kwarg is needed). Requires the v177 patch for provenance/flags
             but does not depend on it.
Anchors are single lines matched by regex; if one is missing the script prints the reason and the nearby lines.
"""
from __future__ import annotations
import argparse, pathlib, py_compile, re, shutil, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]

LOOP_ADD = '''            anchored_distortion = None
            if getattr(self, "local_from", "anchored") == "plain" and anchors:
                # v181: the window's own pass (cached from the raw chain) supplies its geometry; the enlarged pass is
                # only a measurement of the revisit, carried into the plain pass's coordinates through the one
                # similarity that maps the window frames of the anchored pass onto the plain pass
                _cp = [pos[g] for g in idx]
                P0 = cache.get(list(idx), runner)["poses"]
                T_ap, _, info_ap = self._fit(P[_cp], P0)
                anchored_distortion = dict(rot_deg=float(info_ap["rot_res_deg"].max()),
                                           pos_rel=float(info_ap["pos_res"].max() / max(float(info_ap["spread"]), 1e-9)))
                P = P.copy()
                P[_cp] = P0
                for g in anchors:
                    P[pos[g]] = sim3.apply(T_ap, P[pos[g]][None])[0]
'''
PILOT_ARG = ('    ap.add_argument("--local-from", default="anchored", choices=["anchored", "plain"],\n'
             '                    help="window geometry from the enlarged anchored pass (default, all runs so far) or from the '
             'window\'s own pass (v181)")\n')


def ctx(src, m, n=2):
    lines = src.splitlines(); ln = src[:m.start()].count("\n")
    return "\n".join(f"{k + 1:>5}: {lines[k]}" for k in range(max(0, ln - n), min(len(lines), ln + n + 1)))


def patch_anchored(src):
    if 'self.local_from = "anchored"' in src or "local_from" in src:
        return src, "already patched"
    problems = []
    m_attr = re.search(r"^([ \t]+)def __init__\(self,", src, re.M)
    if not m_attr:
        problems.append("`def __init__(self,` not found")
    m_pos = re.search(r"^[ \t]+pos = \{gi: li for li, gi in enumerate\(pass_idx\)\}[^\n]*\n", src, re.M)
    if not m_pos:
        problems.append("`pos = {gi: li for li, gi in enumerate(pass_idx)}` not found")
    if not re.search(r'P = cache\.get\(pass_idx, runner[^\n]*\)\["poses"\]', src):
        problems.append('`P = cache.get(pass_idx, runner...)["poses"]` not found')
    m_ev = re.search(r"^[ \t]+ev = dict\(chunk=k,", src, re.M)
    if not m_ev:
        problems.append("`ev = dict(chunk=k,` not found")
    m_bl = re.search(r'^([ \t]+)if self\.smooth_junctions and loop and loop\["accepted"\]:', src, re.M)
    if not m_bl:
        problems.append('`if self.smooth_junctions and loop and loop["accepted"]:` not found')
    if problems:
        return None, "; ".join(problems)
    # apply from the bottom of the file upwards so earlier offsets stay valid
    edits = []
    edits.append((m_bl.start(), m_bl.end(), m_bl.group(0)[:-1] + ' and self.correction != "none":'))
    edits.append((m_ev.end(), m_ev.end(), " anchored_distortion=anchored_distortion,"))
    edits.append((m_pos.end(), m_pos.end(), LOOP_ADD))
    edits.append((m_attr.start(), m_attr.start(), m_attr.group(1) + 'local_from = "anchored"   # v181 class attribute: pilot_a sets AnchoredStitcher.local_from; tests set it per instance\n\n'))
    for s0, s1, text in sorted(edits, key=lambda t: -t[0]):
        src = src[:s0] + text + src[s1:]
    return src, "ok"


def patch_pilot(src):
    if "AnchoredStitcher.local_from = a.local_from" in src:
        return src, "already patched"
    # undo the v180 constructor-kwarg variant if it was applied
    src = src.replace(", local_from=a.local_from", "")
    if "--local-from" not in src:
        m = (re.search(r'^[ \t]*ap\.add_argument\("--top-proposals"', src, re.M)
             or re.search(r'^[ \t]*ap\.add_argument\("--no-robust-batch"', src, re.M)
             or re.search(r'^[ \t]*ap\.add_argument\("--seed"', src, re.M))
        if not m:
            return None, "argparse anchor (--top-proposals / --no-robust-batch / --seed) not found"
        src = src[:m.start()] + PILOT_ARG + src[m.start():]
    m = re.search(r"^([ \t]*)a = ap\.parse_args\(\)[^\n]*\n", src, re.M)
    if not m:
        return None, "`a = ap.parse_args()` not found"
    src = src[:m.end()] + m.group(1) + "AnchoredStitcher.local_from = a.local_from   # v181 (class attribute; every stitcher instance reads it)\n" + src[m.end():]
    if not re.search(r"^from smr\.stitch import \(?[^\n]*AnchoredStitcher", src, re.M) and "AnchoredStitcher" not in src.split("a = ap.parse_args()")[0]:
        return None, "AnchoredStitcher is not imported in pilot_a.py"
    return src, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--anchored", default=str(ROOT / "src" / "smr" / "stitch" / "anchored.py"))
    ap.add_argument("--pilot", default=str(ROOT / "experiments" / "pilot_a.py"))
    a = ap.parse_args()
    rc = 0
    for path, fn in ((pathlib.Path(a.anchored), patch_anchored), (pathlib.Path(a.pilot), patch_pilot)):
        src = path.read_text()
        new, msg = fn(src)
        if new is None:
            print(f"REFUSING {path.name}: {msg}"); rc = 2; continue
        if msg == "already patched":
            print(f"{path.name}: already patched"); continue
        if a.check:
            print(f"{path.name}: all targets found"); continue
        shutil.copy2(path, path.with_suffix(".py.pre_v181.bak"))
        path.write_text(new)
        py_compile.compile(str(path), doraise=True)
        print(f"patched {path.name}; compiles")
    return rc


if __name__ == "__main__":
    sys.exit(main())
