#!/usr/bin/env python3
"""v178 patch (v179 rewrite): register the vggt_omega_repro backbone. Idempotent; never refuses: if the usual import line
is not found, the import is appended to src/smr/backbones/__init__.py (harmless for a registry module).

    python scripts/apply_v178_patch.py
"""
import pathlib, py_compile, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
p = ROOT / "src" / "smr" / "backbones" / "__init__.py"
s = p.read_text()
if "vggt_omega_repro" in s:
    print("already patched"); sys.exit(0)
if not (ROOT / "src" / "smr" / "backbones" / "vggt_omega_repro.py").exists():
    print("REFUSING: src/smr/backbones/vggt_omega_repro.py missing -- unzip smr_updates179.zip first"); sys.exit(2)
line = "from . import vggt_omega_repro                                      # noqa: F401  (v178)\n"
m = re.search(r"^from \. import [^\n]*vggt_omega[^\n]*\n", s, re.M)
s = (s[:m.end()] + line + s[m.end():]) if m else (s.rstrip("\n") + "\n" + line)
p.write_text(s); py_compile.compile(str(p), doraise=True)
print(f"patched {p} ({'after the vggt_omega import' if m else 'appended'}); compiles")
