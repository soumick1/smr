#!/bin/bash
# Re-score every DTU cloud of the matrix with the current evaluator (CPU only; 4 parallel groups).
# Previous logs are kept as <bb>_dtu.prev<stamp>.log; fresh <bb>_dtu.log files are written from the clouds and the
# unit markers (OOM/FAILED units without a cloud are recorded as such).
#   bash scripts/rescore_matrix_dtu.sh            # ~1.5 h wall (VGGT included via its seed directories)
set +e
SS=~/data/dtu/SampleSet/MVS\ Data; PD=~/data/dtu/Points/stl
STD22="1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118"
R=outputs/reports/matrix; O=outputs/points/matrix
ts() { date +%H:%M:%S; }
STAMP=$(date +%m%d%H%M)
one_bb() { local bb=$1; local L=$R/${bb}_dtu.log
  [ -s $L ] && mv $L $R/${bb}_dtu.prev$STAMP.log; : > $L
  for s in $STD22; do tags=() plys=()
    add() { # tag unitdir ply
      if [ -f $2/$3 ]; then tags+=($1); plys+=($2/$3)
      elif [ -f $2/OOM ]; then printf "== %s scan%s ==\nscan%s: OOM\n" $1 $s $s >> $L
      elif [ -d $2 ]; then printf "== %s scan%s ==\nscan%s: FAILED\n" $1 $s $s >> $L; fi; }
    if [ $bb = vggt ]; then   # VGGT's clouds are the seeded ones (v123 native point head; Q reads)
      add vggt_raw outputs/points/dtu_base_ph_s$s single.ply; add vggt_smr outputs/points/dtu_t2_reads_s$s fused.ply
    else
      add ${bb}_raw $O/${bb}_dtu_s$s single.ply; add ${bb}_smr $O/${bb}_dtu_s$s fused.ply
    fi
    case $bb in streamvggt|stream3r) add ${bb}_win_raw $O/${bb}_dtuwin_raw_s$s fused.ply; add ${bb}_win_ca $O/${bb}_dtuwin_ca_s$s fused.ply;; esac
    [ ${#tags[@]} -gt 0 ] && python scripts/dtu_eval.py --pred "${plys[@]}" --tags "${tags[@]}" --scan $s --sampleset "$SS" --points-dir $PD --icp 50 >> $L 2>>$L.err
    echo "[$(ts)] [$bb rescore] scan$s (${#tags[@]} clouds)" >&2
  done; echo "[$(ts)] ==== $bb rescored ====" >&2; }
( one_bb streamvggt ) & ( one_bb stream3r ) & ( one_bb vggt; one_bb vggt_omega; one_bb pi3 ) & ( one_bb mast3r; one_bb fast3r; one_bb dust3r ) &
wait
python scripts/dtu_summary.py $R/*_dtu.log
