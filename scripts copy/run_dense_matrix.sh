#!/bin/bash
# Dense-benchmark matrix for the merged main table (Table-1 rows x {DTU, ETH3D}).
# One GPU per invocation, backbones sequential, resume-safe:
#   finished units are skipped (marker files DONE / OOM / FAILED next to each output);
#   a unit that ran out of memory is recorded as OOM in the logs and NOT retried
#   (set RETRY_FAILED=1 to retry OOM/FAILED units).
#   screen -dmS mx0 bash -c 'bash scripts/run_dense_matrix.sh 0 vggt_omega streamvggt 2>&1 | tee outputs/reports/matrix_gpu0.log'
# Rows: raw = native pass (49 views / 10 frames; read 0); +SMR = 4-ordering read (symmetric re-measure,
# consensus: median fallback on DTU, corroborated-only on ETH3D). Order-invariant backbones
# (dust3r, mast3r: pairwise + global alignment; pi3: permutation-equivariant) run ONE read: the read
# is the identity for them and their +SMR row equals raw by construction.
# Streaming backbones additionally get the windowed rows (16-view windows, median fusion, raw / +re-measure).
set +e
GPU=$1; shift
SS=~/data/dtu/SampleSet/MVS\ Data; PD=~/data/dtu/Points/stl
STD22="1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118"
R=outputs/reports/matrix; O=outputs/points/matrix; mkdir -p $R $O
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date +%H:%M:%S; }
classify() { grep -qiE "out of memory|OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED" "$1" && echo OOM || echo FAILED; }
# run_unit <outdir> <suite args...>: runs points_suite unless DONE/OOM/FAILED marker exists; returns 0 on success
run_unit() { local out=$1; shift
  [ -f $out/DONE ] && return 0
  if [ -f $out/OOM ] || [ -f $out/FAILED ]; then [ "${RETRY_FAILED:-0}" = 1 ] && rm -f $out/OOM $out/FAILED || return 1; fi
  mkdir -p $out
  if python experiments/points_suite.py "$@" --out-dir $out > $out.log 2>&1; then touch $out/DONE; return 0; fi
  touch $out/$(classify $out.log); echo "[$(ts)] $out: $(classify $out.log) (see $out.log)" >&2; return 1; }
# has_result <log> <tag> <scan>: a "scanN: Acc" line follows the tag's header (an OOM/FAILED line or a
# crashed evaluation does not count, so those units are re-evaluated on resume / retry)
has_result() { grep -A6 "== $2 scan$3 ==" "$1" 2>/dev/null | grep -q "^scan$3: Acc"; }
has_mark() { grep -A1 "== $2 scan$3 ==" "$1" 2>/dev/null | grep -qE "^scan$3: (OOM|FAILED)"; }
# dtu_eval_units <log> <scan> <tag1> <ply1> [<tag2> <ply2>]: one evaluator call for all clouds of a scan without a result yet
dtu_eval_units() { local L=$1 s=$2; shift 2; local tags=() plys=()
  while [ $# -ge 2 ]; do has_result $L $1 $s || { tags+=("$1"); plys+=("$2"); }; shift 2; done
  [ ${#tags[@]} -eq 0 ] && return 0
  python scripts/dtu_eval.py --pred "${plys[@]}" --tags "${tags[@]}" --scan $s --sampleset "$SS" --points-dir $PD --icp 50 >> $L 2>>$L.err; }
dtu_mark() { local L=$1 s=$2 st=$3; shift 3; for tag in "$@"; do has_result $L $tag $s || has_mark $L $tag $s || printf "== %s scan%s ==\nscan%s: %s\n" $tag $s $s $st >> $L; done; }
eth_done() { grep "\"maps\": \"$1\", \"source\": \"$2\"" "$3" 2>/dev/null | grep -q umeyama; }
eth_eval() { local maps=$1 sc=$2 src=$3 J=$4
  eth_done $maps $src $J && return 0
  python scripts/eth3d_pointmap_eval.py --maps "$maps" --scene-dir ~/data/eth3d/$sc --gt official --source $src --json "$J" >> ${J%.jsonl}.log 2>&1; }
eth_mark() { local maps=$1 st=$2 J=$3; shift 3; for src in "$@"; do
  grep -q "\"maps\": \"$maps\", \"source\": \"$src\"" "$J" 2>/dev/null || echo "{\"maps\": \"$maps\", \"source\": \"$src\", \"status\": \"$st\"}" >> "$J"; done; }
status_of() { [ -f $1/OOM ] && echo OOM || echo FAILED; }

# run_ladder <outdir> <suite args...>: like run_unit, but on OOM retries with the backbone's memory
# ladder (dust3r: complete pair graph -> swin-5 -> swin-3; the pair graph is the documented DUSt3R
# knob for larger image sets). The graph actually used is recorded in <outdir>/GRAPH.
run_ladder() { local out=$1; shift
  local ladder=("")
  case $BB in dust3r) ladder=("" "scene_graph=swin-5" "scene_graph=swin-3");; esac
  [ -n "${LADDER_SKIP_COMPLETE:-}" ] && [ ${#ladder[@]} -gt 1 ] && ladder=("${ladder[@]:1}")   # known-OOM rerun: start at swin-5
  for kw in "${ladder[@]}"; do
    if [ -n "$kw" ]; then rm -f $out/OOM; run_unit $out "$@" --backbone-kw $kw && { echo $kw > $out/GRAPH; return 0; }
    else run_unit $out "$@" && { echo complete > $out/GRAPH; return 0; }; fi
    [ -f $out/OOM ] || return 1                 # a non-OOM failure is not retried
  done; return 1; }

for bb in "$@"; do BB=$bb
  case $bb in dust3r|mast3r|pi3) READS=1;; *) READS=4;; esac
  L=$R/${bb}_dtu.log; J=$R/${bb}_eth.jsonl; touch $L $J
  echo "[$(ts)] ==== $bb: DTU (reads=$READS) ====" >&2
  for s in $STD22; do out=$O/${bb}_dtu_s$s
    if run_ladder $out --gt data/gt/dtu/scan$s.npz --backbone $bb --w 49 --overlap 0 --k-ctx 1 --stride 1 \
         --points-from auto --conf-abs 2.0 --reads $READS --content-align --fallback median; then
      dtu_eval_units $L $s ${bb}_raw $out/single.ply ${bb}_smr $out/fused.ply
    else dtu_mark $L $s $(status_of $out) ${bb}_raw ${bb}_smr; fi
    echo "[$(ts)] [$bb dtu] scan$s$( [ -f $out/GRAPH ] && [ "$(cat $out/GRAPH)" != complete ] && echo " ($(cat $out/GRAPH))")" >&2
  done
  echo "[$(ts)] ==== $bb: ETH3D ====" >&2
  for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz); out=$O/${bb}_eth_$sc
    if run_unit $out --gt $f --backbone $bb --n-views 10 --view-seed 0 --w 10 --overlap 0 --k-ctx 1 --stride 1 \
         --points-from auto --conf-abs 2.0 --reads $READS --content-align --fallback none --save-maps; then
      eth_eval $out/maps.npz $sc single $J; eth_eval $out/maps.npz $sc fused $J
    else eth_mark $out/maps.npz $(status_of $out) $J single fused; fi
    echo "[$(ts)] [$bb eth] $sc" >&2
  done
  case $bb in streamvggt|stream3r)
    echo "[$(ts)] ==== $bb: windowed rows ====" >&2
    for s in $STD22; do args=()
      for v in raw ca; do out=$O/${bb}_dtuwin_${v}_s$s; [ $v = ca ] && CA="--content-align" || CA=""
        if run_unit $out --gt data/gt/dtu/scan$s.npz --backbone $bb --w 16 --overlap 8 --k-ctx 0 --stride 1 \
             --points-from auto --conf-abs 2.0 $CA --fallback median --abstain-rel 0; then args+=(${bb}_win_$v $out/fused.ply)
        else dtu_mark $L $s $(status_of $out) ${bb}_win_$v; fi
      done
      [ ${#args[@]} -gt 0 ] && dtu_eval_units $L $s "${args[@]}"
      echo "[$(ts)] [$bb dtu-win] scan$s" >&2
    done
    for f in data/gt/eth3d/*.npz; do sc=$(basename $f .npz)
      for v in raw ca; do out=$O/${bb}_ethwin_${v}_$sc; JW=$R/${bb}_ethwin_$v.jsonl; touch $JW; [ $v = ca ] && CA="--content-align" || CA=""
        if run_unit $out --gt $f --backbone $bb --w 16 --overlap 8 --k-ctx 0 --stride 1 \
             --points-from auto --conf-abs 2.0 $CA --fallback median --abstain-rel 0 --save-maps; then eth_eval $out/maps.npz $sc fused $JW
        else eth_mark $out/maps.npz $(status_of $out) $JW fused; fi
      done
      echo "[$(ts)] [$bb eth-win] $sc" >&2
    done ;;
  esac
  echo "[$(ts)] ==== $bb complete ====" >&2
done
