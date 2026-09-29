#!/usr/bin/env python3
"""v182: place-recognition descriptors as retrieval cues in pilot_a (structural anchors; idempotent).

    python scripts/apply_v182_patch.py            # patches experiments/pilot_a.py
Edits: `--descriptor` accepts any kind understood by smr.stitch.vpr_descriptors (eigenplaces, cosplace, salad, boq,
netvlad, mixvpr, hybrids a+b, projection kind@N) in addition to rgb / dino / feat; desc_fn dispatches unknown kinds to
vpr_descriptors. The DINO gate default (0.5) is unchanged; other kinds keep the adaptive rule unless --desc-thresh is given.
"""
import pathlib, py_compile, re, shutil, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
p = ROOT / "experiments" / "pilot_a.py"
s = p.read_text()
if "vpr_descriptors" in s:
    print("already patched"); sys.exit(0)
m = re.search(r'ap\.add_argument\("--descriptor", default=None, choices=\[[^\]]*\],', s)
if not m:
    print("REFUSING: `ap.add_argument(\"--descriptor\", default=None, choices=[...]` not found"); sys.exit(2)
s = s[:m.start()] + 'ap.add_argument("--descriptor", default=None,' + s[m.end():]
m = re.search(r"^([ \t]+)else:\n[ \t]+desc = image_descriptors\(sel\)\n", s, re.M)
if not m:
    print("REFUSING: `else: desc = image_descriptors(sel)` block not found in desc_fn"); sys.exit(2)
ind = m.group(1)
add = (f'{ind}elif a.descriptor not in ("rgb", "dino", "feat"):\n'
       f'{ind}    from smr.stitch.vpr_descriptors import vpr_descriptors   # v182: eigenplaces, cosplace, salad, boq, netvlad, mixvpr, a+b, kind@N\n'
       f'{ind}    desc = vpr_descriptors(sel, a.descriptor, device=a.device)\n')
s = s[:m.start()] + add + s[m.start():]
# help text: mention the new kinds once
s = s.replace('help="place descriptor: DINOv2 ViT-S/14 CLS (default for "',
              'help="place descriptor: rgb | dino | feat | eigenplaces | cosplace | salad | boq | netvlad | mixvpr | a+b | kind@N; DINOv2 ViT-S/14 CLS (default for "', 1)
shutil.copy2(p, p.with_suffix(".py.pre_v182.bak")); p.write_text(s); py_compile.compile(str(p), doraise=True)
print(f"patched {p}; compiles")
