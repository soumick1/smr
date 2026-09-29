#!/bin/bash
# VGGT rows of the matrix from clouds already on disk (no GPU): re-scores them with the CURRENT
# evaluator so every backbone shares one gauge; copies the ETH3D read results (same evaluator).
#   bash scripts/seed_matrix_from_existing.sh
set +e
SS=~/data/dtu/SampleSet/MVS\ Data; PD=~/data/dtu/Points/stl
STD22="1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118"
R=outputs/reports/matrix; mkdir -p $R; L=$R/vggt_dtu.log; touch $L
for s in $STD22; do tags=() plys=()
  grep -q "== vggt_raw scan$s ==" $L || { tags+=(vggt_raw); plys+=(outputs/points/dtu_base_ph_s$s/single.ply); }    # v123 native point head, C>2
  grep -q "== vggt_smr scan$s ==" $L || { tags+=(vggt_smr); plys+=(outputs/points/dtu_t2_reads_s$s/fused.ply); }  # Q: 4 reads + re-measure
  [ ${#tags[@]} -gt 0 ] && python scripts/dtu_eval.py --pred "${plys[@]}" --tags "${tags[@]}" --scan $s --sampleset "$SS" --points-dir $PD --icp 50 >> $L 2>>$L.err
  echo "[$(date +%H:%M:%S)] [vggt seed] scan$s" >&2
done
[ -s $R/vggt_eth.jsonl ] || cp outputs/reports/eth_table3d.jsonl $R/vggt_eth.jsonl
echo "vggt seeded -> $L, $R/vggt_eth.jsonl" >&2
