#!/bin/bash
# The CALIBRATED row for Table 2: +SMR+PGO with dense-edge re-measurement using the known 7-Scenes intrinsics, next to its
# uncalibrated twin (dense edges with the backbone's predicted intrinsics).  seq-01 of each scene, Table-2 configuration.
# The pose-based rows (chained / smr / smr_pgo) are re-emitted unchanged: they never use intrinsics.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_7scenes_calibrated.sh          (~2-4 min per scene per variant; passes cached)
cd "$(dirname "$0")/.."; source .venv/bin/activate; OUT=outputs/7scenes_calibrated; mkdir -p $OUT
step() { echo "[$(date +%H:%M:%S)] $*"; }
for sc in ${SCENES:-chess fire heads office pumpkin redkitchen stairs}; do GT=data/gt/7scenes_${sc}_seq01.npz
  for v in dense dense_cal; do J=$OUT/${sc}_$v.json; [ -f $J ] && continue
    extra=""; [ $v = dense_cal ] && extra="--known-K 585 585 320 240 --known-K-res 640 480"
    step "$sc $v"; python experiments/pilot_a.py --gt $GT --backbone ${BB:-vggt_omega} --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 $extra \
       --rows chained,smr,smr_pgo,smr_pgo_dense --json $J > ${J%.json}.log 2>&1 || step "  FAILED: $(tail -n 1 ${J%.json}.log)"
    grep -h "\[calibrated\]" ${J%.json}.log | head -1
  done
done
python - <<'PY'
import json, glob, pathlib, numpy as np
rows = {}
for f in glob.glob("outputs/7scenes_calibrated/*.json"):
    sc, v = pathlib.Path(f).stem.rsplit("_", 1) if not pathlib.Path(f).stem.endswith("dense_cal") else (pathlib.Path(f).stem[:-10], "dense_cal")
    rows.setdefault(sc, {})[v] = {r["method"]: r["ate_rmse"] for r in json.load(open(f))["rows"]}
print(f"{'scene':<11} raw    +SMR   +PGO   +PGO dense (pred K)  +PGO dense (known K)")
acc = []
for sc in ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]:
    if sc not in rows or "dense" not in rows[sc] or "dense_cal" not in rows[sc]: print(f"{sc:<11} --"); continue
    d, c = rows[sc]["dense"], rows[sc]["dense_cal"]
    acc.append([d["chained"], d["smr"], d["smr_pgo"], d["smr_pgo_dense"], c["smr_pgo_dense"]])
    print(f"{sc:<11} {d['chained']:.3f}  {d['smr']:.3f}  {d['smr_pgo']:.3f}  {d['smr_pgo_dense']:.3f}               {c['smr_pgo_dense']:.3f}")
if acc: m = np.mean(acc, 0); print(f"{'MEAN':<11} {m[0]:.3f}  {m[1]:.3f}  {m[2]:.3f}  {m[3]:.3f}               {m[4]:.3f}")
PY
