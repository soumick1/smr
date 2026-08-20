#!/usr/bin/env python3
"""Run the full ladder (tiers 1-3) and print a consolidated verdict."""
import argparse, json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "outputs" / "reports"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    flags = ["--fast"] if a.fast else []
    scripts = ["run_tier1_dynamics.py", "run_tier2_memory.py",
               "run_tier3_integration.py"]
    for s in scripts:
        print(f"\n=== {s} ===")
        subprocess.run([sys.executable, str(HERE / s), *flags], check=False)
    print("\n================ consolidated ================")
    total = passed = 0
    for name in ("tier1", "tier2", "tier3"):
        p = OUT / f"{name}.json"
        if not p.exists():
            print(f"{name}: MISSING"); continue
        r = json.loads(p.read_text())
        n = len(r["checks"]); k = sum(c["passed"] for c in r["checks"])
        total += n; passed += k
        print(f"{name}: {k}/{n} passed  (all_passed={r['all_passed']})")
    print(f"TOTAL: {passed}/{total}")
    sys.exit(0 if passed == total and total > 0 else 1)

if __name__ == "__main__":
    main()
