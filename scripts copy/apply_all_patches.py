#!/usr/bin/env python3
"""Apply every pending patch in order and print one status line each (v177 flags+provenance, v178 Omega-repro backbone,
v181 local_from). Exit code is non-zero if any patch REFUSED; the reason is printed with it.

    python scripts/apply_all_patches.py
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
ok = True
for name in ("apply_v177_patch.py", "apply_v178_patch.py", "apply_v181_patch.py", "apply_v182_patch.py", "apply_v192_patch.py"):
    p = ROOT / "scripts" / name
    if not p.exists():
        print(f"{name:<24} MISSING (unzip the update first)"); ok = False; continue
    r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else f"REFUSED (exit {r.returncode})"
    print(f"{name:<24} {status}: {' | '.join(l for l in (r.stdout + r.stderr).strip().splitlines() if l.strip())[:400]}")
    ok &= r.returncode == 0
# verification
checks = {
    "pilot_a has --budget-rot": "--budget-rot" in (ROOT / "experiments" / "pilot_a.py").read_text(),
    "pilot_a has --local-from": "--local-from" in (ROOT / "experiments" / "pilot_a.py").read_text(),
    "anchored.py has local_from": "local_from" in (ROOT / "src" / "smr" / "stitch" / "anchored.py").read_text(),
    "backbones import vggt_omega_repro": "vggt_omega_repro" in (ROOT / "src" / "smr" / "backbones" / "__init__.py").read_text(),
    "pilot_a dispatches VPR descriptors": "vpr_descriptors" in (ROOT / "experiments" / "pilot_a.py").read_text(),
    "pilot_a has --distortion-gate": "--distortion-gate" in (ROOT / "experiments" / "pilot_a.py").read_text(),
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
sys.exit(0 if ok and all(checks.values()) else 1)
