#!/bin/bash
# Re-score EVERY DTU cloud on disk with the v126 evaluator (region-restricted,
# point-to-plane, converged Sim(3) ICP with rejection). CPU only; the four
# groups run in parallel. Output: one log per group + a combined summary.
#   bash scripts/dtu_rescore_all.sh
set +e
SS=~/data/dtu/SampleSet/MVS\ Data; PD=~/data/dtu/Points/stl
mkdir -p outputs/reports/rescore
ev() { # $1=tag $2=scan $3=ply $4=log
  echo "== $1 scan$2 ==" >> "$4"
  python scripts/dtu_eval.py --pred "$3" --scan "$2" --sampleset "$SS" --points-dir "$PD" --icp 50 >> "$4" 2>&1 || echo "scan$2: FAILED" >> "$4"
}
grp() { # $1=group name; reads "tag scan ply" lines from stdin
  L=outputs/reports/rescore/$1.log; : > "$L"; n=0
  while read -r tag scan ply; do
    n=$((n+1))
    if [ -f "$ply" ]; then ev "$tag" "$scan" "$ply" "$L"; else echo "missing $ply" >> "$L"; fi
    echo "[$1] $n done: $tag scan$scan" >&2
  done
  echo "group $1 done -> $L" >&2
}
scan_of() { echo "$1" | sed -E 's/.*_s([0-9]+)(_|\/|$).*/\1/'; }
# group A: v123 23-scan baselines (point head C>2 with predicted cams; depth branch GT cams + geo3)
( for d in outputs/points/dtu_base_ph_s*; do echo "ph_c2_single $(scan_of "$d") $d/single.ply"; done
  for d in outputs/points/dtu_base_dp_s*; do echo "dp_geo3_single $(scan_of "$d") $d/single.ply"; done ) | grp baselines_v123 &
# group B: v125 K run, point head no filter, single pass
( for d in outputs/points/dtu_s*_v125_vggtp_single; do echo "ph_nofilter_single $(scan_of "$d") $d/fused.ply"; done ) | grp nofilter_single &
# group C: v125 K run, point head no filter, 4 reads + re-measure
( for d in outputs/points/dtu_s*_v125_vggtp_reads4; do echo "ph_nofilter_reads4 $(scan_of "$d") $d/fused.ply"; done ) | grp nofilter_reads4 &
# group D: scan1 variants (native, reads, content, matrix rows)
( echo "ph_c2_native 1 outputs/points/dtu_s1_v123_ph_c2/single.ply"
  echo "ph_c2_reads4 1 outputs/points/dtu_s1_v124_ph_reads4/fused.ply"
  echo "ph_c2_reads4_ca_first 1 outputs/points/dtu_s1_v124_ph_reads4_ca/fused.ply"
  echo "ph_c2_reads4_ca_mean 1 outputs/points/dtu_s1_v125_ph_c2_reads4_ca_mean/fused.ply"
  echo "dp_geo3_native 1 outputs/points/dtu_s1_v123_d_c2_geo3/single.ply"
  echo "dp_geo3_reads4_ca 1 outputs/points/dtu_s1_v124_dp_reads4_ca/fused.ply"
  for row in chained smr smr_pgo; do echo "matrix_$row 1 outputs/points/dtu_s1_matrix_$row/fused.ply"; echo "matrix_${row}_content 1 outputs/points/dtu_s1_matrix_${row}_ca/fused.ply"; done
  echo "ph_nofilter_s29 29 outputs/points/dtu_s29_v124_ph_nofilter/fused.ply"
  echo "ph_c2_gtcams_s29 29 outputs/points/dtu_s29_v124_ph_c2_gtcams/fused.ply" ) | grp scan1_variants &
wait
python scripts/dtu_summary.py outputs/reports/rescore/*.log
