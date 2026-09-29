#!/usr/bin/env bash
# Smoke test: every backbone through the REAL pilot on 48 keyframes (5 chunks,
# ~10 backbone passes), one line per backbone.  Catches a crash in minutes and
# measures seconds-per-pass so the sweep budget can be planned.
#   bash scripts/smoke_pilot_a.sh                 # all nine
#   bash scripts/smoke_pilot_a.sh dust3r mast3r   # a subset
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
GT=${GT:-data/gt/7scenes_chess_seq01.npz}
BACKBONES=${*:-"vggt pi3 dust3r mast3r fast3r stream3r streamvggt monst3r vggt_omega"}
mkdir -p outputs/smoke
printf "%-11s %-6s %8s %8s %8s %6s %7s  %s\n" backbone status ATE_ch ATE_smr loops s/pass GB note
for bb in $BACKBONES; do
  log=outputs/smoke/$bb.log; js=outputs/smoke/$bb.json; rm -f "$js"
  t0=$(date +%s)
  timeout 2400 python experiments/pilot_a.py --gt "$GT" --backbone "$bb" --keyframe-stride 5 \
      --max-frames 48 --chunk 16 --overlap 8 --sites 2 --rows chained,smr \
      --cache outputs/smoke/${bb}_cache.npy --json "$js" > "$log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  if [ -f "$js" ]; then
    python - "$js" "$bb" "$dt" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); rows = {x["method"]: x for x in r["rows"]}
ch, sm = rows.get("chained", {}), rows.get("smr", {})
npass = sm.get("n_passes", 0) + 1
print(f"{sys.argv[2]:<11} {'OK':<6} {ch.get('ate_rmse', float('nan')):8.4f} {sm.get('ate_rmse', float('nan')):8.4f} "
      f"{sm.get('n_loops', 0):8d} {int(sys.argv[3]) / max(1, npass):6.1f} {sm.get('peak_mem_gb', 0):7.1f}  "
      f"gate={'ok' if r['gate']['passed'] else 'FAILED'} ({int(sys.argv[3])}s total)")
PY
  else
    note=$(grep -E "Error|error|Killed|Traceback" "$log" | tail -1 | cut -c1-90)
    printf "%-11s %-6s %8s %8s %8s %6s %7s  rc=%s after %ss: %s\n" "$bb" FAIL - - - - - "$rc" "$dt" "${note:-see $log}"
  fi
done
echo "logs: outputs/smoke/<backbone>.log"
