#!/usr/bin/env python3
"""v180 patch: `local_from` for the anchored stitcher (plan Task 2B made a fix, motivated by the seed-1 chess blow-up).

    python scripts/apply_v180_patch.py --check
    python scripts/apply_v180_patch.py            # patches src/smr/stitch/anchored.py and experiments/pilot_a.py

AnchoredStitcher(local_from="anchored"|"plain"), pilot_a --local-from {anchored,plain} (default anchored = every run so far).
  anchored: a window with proposed sites takes its own geometry from the enlarged pass (window + anchor frames). A bad
            proposal, accepted or not, can distort the window (chess seed 1: 0 closures accepted, ATE 0.025 -> 0.459,
            AUC_within 93 -> 83).
  plain:    the window's geometry comes from its own 32-frame pass (already cached from the raw chain; no new backbone
            work). The anchored pass is used only to measure the revisit: its window frames are fitted to the plain pass
            (one Sim(3)) and the anchor frames are carried through that similarity, so all downstream code (site
            verification, S_A, S_B, edges, budget, write-back) runs unchanged in the plain pass's coordinates. Rejected
            proposals then leave the geometry untouched and correction="none" becomes an exact identity control
            (tests/test_guarantees.py::test_local_from_plain_*).
  correction="none" also skips the junction blend (there is no correction to smooth), so anchored+none isolates the
  anchored-pass effect against raw, and plain+none equals raw exactly.
  The event of every window with anchors records `anchored_distortion` = {rot_deg, pos_rel}: how far the enlarged pass
  moved the window's own frames (the diagnostic that explains seed sensitivity).
Structural anchors; refuses with the reason if a target is missing. Idempotent.
"""
from __future__ import annotations
import argparse, pathlib, py_compile, re, shutil, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]

A_SIG = re.compile(r"(\n[ \t]+)(site_agree_pos=1e9,[^\n]*?revoke=True,)")           # constructor signature line
A_ATTR = "        self.smooth_junctions = smooth_junctions\n"
A_LOOP = ('            P = cache.get(pass_idx, runner)["poses"]\n'
          "            passes.append(pass_idx)\n"
          "            pos = {gi: li for li, gi in enumerate(pass_idx)}\n"
          "            chunk_pos = [pos[g] for g in idx]\n")
A_EV = re.compile(r"(\n([ \t]+)ev = dict\(chunk=k, n_new=len\(new\),)")

LOOP_ADD = '''            anchored_distortion = None
            if getattr(self, "local_from", "anchored") == "plain" and anchors:
                # the window's own pass (cached from the raw chain) supplies its geometry; the enlarged pass is
                # only a measurement of the revisit, carried into the plain pass's coordinates through the one
                # similarity that maps the window frames of the anchored pass onto the plain pass
                P0 = cache.get(list(idx), runner)["poses"]
                T_ap, _, info_ap = self._fit(P[chunk_pos], P0)
                anchored_distortion = dict(rot_deg=float(info_ap["rot_res_deg"].max()),
                                           pos_rel=float(info_ap["pos_res"].max() / max(float(info_ap["spread"]), 1e-9)))
                P = P.copy()
                P[chunk_pos] = P0
                for g in anchors:
                    P[pos[g]] = sim3.apply(T_ap, P[pos[g]][None])[0]
'''
PILOT_ARG = '    ap.add_argument("--local-from", default="anchored", choices=["anchored", "plain"],\n' \
            '                    help="window geometry from the enlarged anchored pass (default, all runs so far) or from the window\'s own pass (v180)")\n'


def patch_anchored(src):
    if "local_from" in src:
        return src, "already patched"
    m = A_SIG.search(src)
    if not m:
        return None, "constructor signature line `site_agree_pos=1e9, revoke=True,` not found"
    src = src[:m.start()] + m.group(1) + m.group(2).replace("revoke=True,", 'revoke=True, local_from="anchored",') + src[m.end():]
    if src.count(A_ATTR) != 1:
        return None, "`self.smooth_junctions = smooth_junctions` not found once"
    src = src.replace(A_ATTR, A_ATTR + '        self.local_from = str(local_from)\n')
    if src.count(A_LOOP) != 1:
        return None, "the 4-line pass block (P = cache.get(pass_idx ...) / passes.append / pos / chunk_pos) not found once"
    src = src.replace(A_LOOP, A_LOOP + LOOP_ADD)
    m = A_EV.search(src)
    if not m:
        return None, "`ev = dict(chunk=k, n_new=len(new),` not found"
    src = src[:m.end()] + " anchored_distortion=anchored_distortion," + src[m.end():]
    # the junction blend smooths a CORRECTION; with correction="none" there is nothing to smooth, and blending the
    # overlap toward the chain-placed window would break the identity control by the chain fit's residual
    blend = 'if self.smooth_junctions and loop and loop["accepted"]:'
    if src.count(blend) != 1:
        return None, "junction-blend condition not found once"
    src = src.replace(blend, 'if self.smooth_junctions and loop and loop["accepted"] and self.correction != "none":')
    return src, "ok"


def patch_pilot(src):
    if "--local-from" in src:
        return src, "already patched"
    m = re.search(r'^[ \t]*ap\.add_argument\("--top-proposals"', src, re.M) or re.search(r'^[ \t]*ap\.add_argument\("--no-robust-batch"', src, re.M)
    if not m:
        return None, "argparse anchor (--top-proposals / --no-robust-batch) not found; apply the v177 patch first"
    src = src[:m.start()] + PILOT_ARG + src[m.start():]
    key = "smooth_junctions=not a.no_smooth_junctions, top_proposals=a.top_proposals"
    if src.count(key) != 1:
        return None, "v177 `common` kwargs not found; apply the v177 patch first"
    src = src.replace(key, key + ", local_from=a.local_from")
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
        shutil.copy2(path, path.with_suffix(".py.pre_v180.bak"))
        path.write_text(new)
        py_compile.compile(str(path), doraise=True)
        print(f"patched {path.name}; compiles (backup {path.with_suffix('.py.pre_v180.bak').name})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
