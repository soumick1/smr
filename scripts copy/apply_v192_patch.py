#!/usr/bin/env python3
"""v192: distortion-gated window geometry (`--local-from gated --distortion-gate DEG`) and distortion logging in every mode.

    python scripts/apply_v192_patch.py          # patches anchored.py (v181 block) and pilot_a.py; idempotent

The v181 block computed the anchored-pass distortion of the window's own frames only in `plain` mode. Now, whenever a
window has anchor frames, the plain pass (cached from the raw chain) is fetched and the Sim(3) residual between the
window's frames in the anchored pass and in its own pass is logged as `anchored_distortion` {rot_deg, pos_rel, used}.
Modes: anchored (default; never replaces), plain (always replaces), gated (replaces only when rot_deg exceeds
AnchoredStitcher.distortion_gate, default 1.0 deg, set by pilot_a --distortion-gate). Requires the v181 patch.
"""
import pathlib, py_compile, re, shutil, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]

OLD_COND = '            if getattr(self, "local_from", "anchored") == "plain" and anchors:\n'
NEW_BLOCK = '''            if anchors:
                # v181/v192: the window's own pass (cached from the raw chain) is the reference geometry; the enlarged
                # pass is a measurement of the revisit.  The Sim(3) residual between the window's frames in the two
                # passes is the distortion the anchors caused.  plain: always keep the own pass; gated: keep it only
                # when the distortion exceeds the gate; anchored: never (log only).
                _mode = getattr(self, "local_from", "anchored")
                _cp = [pos[g] for g in idx]
                P0 = cache.get(list(idx), runner)["poses"]
                T_ap, _, info_ap = self._fit(P[_cp], P0)
                _rot = float(info_ap["rot_res_deg"].max())
                _replace = (_mode == "plain") or (_mode == "gated" and _rot > float(getattr(self, "distortion_gate", 1.0)))
                anchored_distortion = dict(rot_deg=_rot,
                                           pos_rel=float(info_ap["pos_res"].max() / max(float(info_ap["spread"]), 1e-9)),
                                           used=("plain" if _replace else "anchored"))
            if anchors and _replace:
'''
# lines of the old block that must be removed because NEW_BLOCK recomputes them
OLD_TAIL = ('''                _cp = [pos[g] for g in idx]
                P0 = cache.get(list(idx), runner)["poses"]
                T_ap, _, info_ap = self._fit(P[_cp], P0)
                anchored_distortion = dict(rot_deg=float(info_ap["rot_res_deg"].max()),
                                           pos_rel=float(info_ap["pos_res"].max() / max(float(info_ap["spread"]), 1e-9)))
''')


def patch_anchored(p):
    s = p.read_text()
    if "distortion_gate" in s:
        return "already patched"
    if s.count(OLD_COND) != 1 or s.count(OLD_TAIL) != 1:
        return "REFUSING: v181 block not found in its expected form (apply v181 first, or the file was edited)"
    # remove the v181 comment lines between the condition and the tail (any '# v181' comment block)
    s = s.replace(OLD_COND, NEW_BLOCK)
    s = s.replace(OLD_TAIL, "")
    s = re.sub(r"\n[ \t]+# v181: the window's own pass[^\n]*\n([ \t]+# [^\n]*\n){0,3}", "\n", s)
    m = re.search(r'^([ \t]+)local_from = "anchored"[^\n]*\n', s, re.M)
    if not m:
        return "REFUSING: class attribute local_from not found"
    s = s[:m.end()] + m.group(1) + 'distortion_gate = 1.0       # v192: degrees; used by local_from="gated"\n' + s[m.end():]
    shutil.copy2(p, p.with_suffix(".py.pre_v192.bak")); p.write_text(s); py_compile.compile(str(p), doraise=True)
    return "patched"


def patch_pilot(p):
    s = p.read_text()
    if "--distortion-gate" in s:
        return "already patched"
    s2 = s.replace('choices=["anchored", "plain"],', 'choices=["anchored", "plain", "gated"],', 1)
    if s2 == s:
        return "REFUSING: --local-from choices not found (apply v181 first)"
    key = "AnchoredStitcher.local_from = a.local_from"
    if key not in s2:
        return "REFUSING: v181 class-attribute line not found"
    ind = re.search(r"^([ \t]*)" + re.escape(key), s2, re.M).group(1)
    s2 = s2.replace(key, key + "\n" + ind + "AnchoredStitcher.distortion_gate = a.distortion_gate   # v192", 1)
    m = re.search(r'^[ \t]*ap\.add_argument\("--local-from"', s2, re.M)
    s2 = s2[:m.start()] + ('    ap.add_argument("--distortion-gate", type=float, default=1.0,\n'
                           '                    help="local-from gated: keep the anchored-pass window geometry only if it moved the window\'s own frames by less than this (deg)")\n') + s2[m.start():]
    shutil.copy2(p, p.with_suffix(".py.pre_v192.bak")); p.write_text(s2); py_compile.compile(str(p), doraise=True)
    return "patched"


if __name__ == "__main__":
    rc = 0
    for path, fn in ((ROOT / "src" / "smr" / "stitch" / "anchored.py", patch_anchored), (ROOT / "experiments" / "pilot_a.py", patch_pilot)):
        msg = fn(path); print(f"{path.name}: {msg}")
        rc |= 2 if msg.startswith("REFUSING") else 0
    sys.exit(rc)
