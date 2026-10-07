#!/usr/bin/env python3
"""v201: distortion gate 5 deg as the paper default; sequence NVS with per-target source selection, confidence and
flying-pixel filtering, and a revisited-target subset.  Run from the repo root:  python scripts/apply_v201_patch.py"""
import pathlib, py_compile, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ok = True

p = ROOT / "experiments" / "pilot_a.py"; s = p.read_text()
if "distortion_gate=5.0" not in s:
    s = s.replace('local_from="gated", distortion_gate=2.0,', 'local_from="gated", distortion_gate=5.0,', 1)
    p.write_text(s)
print("pilot_a: gate 5 deg" if "distortion_gate=5.0" in p.read_text() else "pilot_a: preset line not found (gate unchanged)")

t = ROOT / "tests" / "test_guarantees.py"; u = t.read_text()
t.write_text(u.replace('"distortion_gate=2.0"', '"distortion_gate=5.0"'))

n = ROOT / "experiments" / "nvs_sequence.py"; s = n.read_text()
s = s.replace("AnchoredStitcher.distortion_gate = 2.0", "AnchoredStitcher.distortion_gate = 5.0")
if '"--sources"' not in s:
    s = s.replace('''    ap.add_argument("--methods"''', '''    ap.add_argument("--sources", type=int, default=12, help="input frames rendered per target (nearest by GT camera, same for every method)")
    ap.add_argument("--conf-keep", type=float, default=0.7, help="fraction of each frame's points kept, by backbone confidence")
    ap.add_argument("--edge-thresh", type=float, default=0.05, help="drop pixels whose relative depth jump to a neighbour exceeds this (flying pixels)")
    ap.add_argument("--methods"''', 1)
if "src_of = []" not in s:
    a0 = s.index("    results = {}\n"); a1 = s.index("    summary = dict(")
    block = (ROOT / "scripts" / "nvs_v201_block.py.txt").read_text()
    s = s[:a0] + block + s[a1:]
n.write_text(s)
try:
    py_compile.compile(str(n), doraise=True); py_compile.compile(str(p), doraise=True)
    print("nvs_sequence: source selection, confidence / edge filtering, revisit subset; both files compile")
except Exception as ex:  # noqa: BLE001
    print("COMPILE ERROR:", ex); ok = False
sys.exit(0 if ok else 1)
