#!/bin/bash
# Control-flow test of scripts/run_dense_matrix.sh with stubbed suite/evaluators (no GPU, no data).
# Scenario: backbone 'fakebb' (4 reads); scan 10 runs out of memory; ETH3D scene 'meadow' fails otherwise.
# Checks: markers, OOM/FAILED lines in the logs, combined evaluation, resume (no re-work), RETRY_FAILED.
set -e
T=$(mktemp -d); cd $T
mkdir -p scripts experiments data/gt/dtu data/gt/eth3d
for s in 1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118; do : > data/gt/dtu/scan$s.npz; done
for sc in courtyard meadow; do : > data/gt/eth3d/$sc.npz; done
cat > experiments/points_suite.py <<'PY'
import sys, pathlib, numpy as np
a = sys.argv; out = pathlib.Path(a[a.index("--out-dir")+1]); out.mkdir(parents=True, exist_ok=True)
gt = a[a.index("--gt")+1]; bb = a[a.index("--backbone")+1]
if bb == "dust3r" and "--backbone-kw" not in a and "dtu" in gt: print("torch.OutOfMemoryError: CUDA out of memory"); sys.exit(1)
if "scan10" in gt: print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20 GB"); sys.exit(1)
if "meadow" in gt: print("RuntimeError: something else"); sys.exit(1)
for n in ("single", "fused"):
    with open(out/f"{n}.ply", "wb") as f: f.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
if "--save-maps" in a: (out/"maps.npz").write_bytes(b"x")
print("ok")
PY
cat > scripts/dtu_eval.py <<'PY'
import sys
a = sys.argv; preds = a[a.index("--pred")+1:a.index("--tags")]; tags = a[a.index("--tags")+1:a.index("--scan")]; s = a[a.index("--scan")+1]
for t, p in zip(tags, preds):
    print(f"== {t} scan{s} =="); print(f"scan{s}: Acc 0.500  Comp 0.300  Overall 0.400  (pred kept 70.0%, MaxDist 20.0)")
PY
cat > scripts/eth3d_pointmap_eval.py <<'PY'
import sys, json
a = sys.argv; m = a[a.index("--maps")+1]; src = a[a.index("--source")+1]; j = a[a.index("--json")+1]
open(j, "a").write(json.dumps({"maps": m, "source": src, "umeyama": {"acc": 0.2, "comp": 0.3, "overall": 0.25}, "n_corr": 1}) + "\n")
PY
cp $OLDPWD/scripts/run_dense_matrix.sh scripts/
mkdir -p ~/data/dtu/SampleSet ~/data/eth3d 2>/dev/null || true
bash scripts/run_dense_matrix.sh 0 fakebb > run1.out 2>&1
L=outputs/reports/matrix/fakebb_dtu.log; J=outputs/reports/matrix/fakebb_eth.jsonl
test -f outputs/points/matrix/fakebb_dtu_s10/OOM || { echo "FAIL: no OOM marker"; exit 1; }
test -f outputs/points/matrix/fakebb_dtu_s1/DONE || { echo "FAIL: no DONE marker"; exit 1; }
test -f outputs/points/matrix/fakebb_eth_meadow/FAILED || { echo "FAIL: no FAILED marker"; exit 1; }
grep -q "== fakebb_raw scan10 ==" $L && grep -A1 "== fakebb_smr scan10 ==" $L | grep -q "scan10: OOM" || { echo "FAIL: OOM not logged"; cat $L; exit 1; }
[ $(grep -c "^scan.*Overall 0.400" $L) -eq 42 ] || { echo "FAIL: expected 42 results, got $(grep -c 'Overall 0.400' $L)"; exit 1; }   # 21 scans x 2 tags
[ $(grep -c '"status": "FAILED"' $J) -eq 2 ] && [ $(grep -c '"overall"' $J) -eq 2 ] || { echo "FAIL: eth jsonl"; cat $J; exit 1; }
# resume: nothing new happens, nothing duplicated
cp $L L1; cp $J J1; bash scripts/run_dense_matrix.sh 0 fakebb > run2.out 2>&1
cmp -s $L L1 && cmp -s $J J1 || { echo "FAIL: resume changed logs"; diff $L L1; exit 1; }
# retry: with RETRY_FAILED=1 the OOM unit is attempted again (still OOM here) but not duplicated in the log
RETRY_FAILED=1 bash scripts/run_dense_matrix.sh 0 fakebb > run3.out 2>&1
[ $(grep -c "== fakebb_raw scan10 ==" $L) -eq 1 ] || { echo "FAIL: duplicated OOM header on retry"; exit 1; }
grep -q "_s10: OOM (see" run3.out || { echo "FAIL: retry did not re-run the OOM unit"; cat run3.out | tail -3; exit 1; }
# a retry that SUCCEEDS: the stub now succeeds on scan10 when this flag exists; the result must be
# recorded (new block) and the builder must prefer it over the earlier OOM mark
: > succeed10
sed -i 's/if "scan10" in gt:/if "scan10" in gt and not pathlib.Path("succeed10").exists():/' experiments/points_suite.py
RETRY_FAILED=1 bash scripts/run_dense_matrix.sh 0 fakebb > run4.out 2>&1
grep -A6 "== fakebb_raw scan10 ==" $L | grep -q "scan10: Acc" || { echo "FAIL: successful retry not evaluated"; cat $L | tail -8; exit 1; }
test -f outputs/points/matrix/fakebb_dtu_s10/DONE || { echo "FAIL: no DONE after successful retry"; exit 1; }
mkdir -p mx && sed 's/fakebb_/dust3r_/g' $L > mx/dust3r_dtu.log && cp $J mx/dust3r_eth.jsonl
python3 $OLDPWD/scripts/build_main_table.py --main $OLDPWD/paper/tab_pose_camera.tex --matrix mx --out mx/t.tex > build.out
grep -q "OOM units" build.out && { echo "FAIL: builder still reports the OOM after a successful retry"; cat build.out; exit 1; }
grep "DUSt3R (raw)" mx/t.tex | grep -q "0.400 &" || { echo "FAIL: DTU row not complete (22 scans)"; grep "DUSt3R (raw)" mx/t.tex; exit 1; }
# memory ladder: 'dust3r' OOMs on the complete graph and succeeds with the first swin rung
bash scripts/run_dense_matrix.sh 0 dust3r > run5.out 2>&1
test "$(cat outputs/points/matrix/dust3r_dtu_s1/GRAPH)" = "scene_graph=swin-5" || { echo "FAIL: ladder rung not recorded"; cat outputs/points/matrix/dust3r_dtu_s1/GRAPH 2>/dev/null; exit 1; }
test -f outputs/points/matrix/dust3r_dtu_s1/DONE && ! test -f outputs/points/matrix/dust3r_dtu_s1/OOM || { echo "FAIL: ladder markers"; exit 1; }
grep -A6 "== dust3r_raw scan1 ==" outputs/reports/matrix/dust3r_dtu.log | grep -q "scan1: Acc" || { echo "FAIL: ladder result missing"; exit 1; }
grep -q "scan1 (scene_graph=swin-5)" run5.out || { echo "FAIL: ladder not announced"; tail -3 run5.out; exit 1; }
echo "launcher control-flow test passed ($T)"
