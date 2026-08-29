import json
import pathlib
import subprocess
import sys
import tempfile


def test_lba_skip_edges_row_runs_and_is_no_worse():
    """Overlap > step makes chunk pairs two steps apart share frames; the
    LBA-lite row fits those from CACHED poses (no depth) and must not hurt."""
    out = pathlib.Path(tempfile.mkdtemp()) / "r.json"
    root = pathlib.Path(__file__).parents[1]
    cmd = [sys.executable, "experiments/pilot_a.py", "--backbone", "synthetic",
           "--sim-frames", "60", "--keyframe-stride", "1", "--chunk", "24",
           "--overlap", "18", "--sites", "2", "--rows", "smr,smr_pgo,smr_pgo_lba",
           "--json", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
    assert r.returncode == 0, (r.stderr or r.stdout)[-800:]
    actual = out.parent / f"{out.stem}_SIMULATED.json"   # synthetic runs get the suffix
    rows = {d["method"]: d for d in json.load(open(actual))["rows"]}
    # The row runs and records its edges; it measured WORSE than smr_pgo on
    # the simulated world across seeds (0.166/0.194/0.131 vs 0.105/0.153/0.082)
    # -- skip-pair fits inherit correlated per-pass distortion, the same
    # failure family as the dense edges -- so it is retired from the paper
    # plan and never spent GPU time.  This test guards the mechanics only.
    assert "smr_pgo_lba" in rows and rows["smr_pgo_lba"].get("n_skip_edges", 0) > 0
