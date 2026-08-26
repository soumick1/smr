#!/usr/bin/env bash
# Everything for Tables 1/2/5 on ONE GPU, in value order, resumable (finished
# reports are skipped).  Launch once in a screen and leave it:
#   CUDA_VISIBLE_DEVICES=0 screen -S all -dm bash -c 'bash scripts/run_all_gpu0.sh 2>&1 | tee -a outputs/reports/all_gpu0.log'
# Budget (measured s/pass on 16-frame chunks: vggt 0.8, pi3 0.5, vggt_omega 3.5,
# stream3r 4, streamvggt 6, fast3r 6, monst3r 51, mast3r 64, dust3r 74):
#   G1 standard protocol, vggt/pi3/vggt_omega         ~3 h
#   G2 multi-session (fire, office), same three        ~3 h
#   G3 standard protocol, fast3r/stream3r/streamvggt   ~4 h
#   G4 multi-session, those three                      ~3 h
#   G5 chess x6 only, dust3r/mast3r/monst3r, rows chained+smr, no ceiling  ~9 h
#   G6 length sweep, vggt (cached passes)              ~0.5 h
set -u
cd "$(dirname "$0")/.." && source .venv/bin/activate
mkdir -p outputs/reports outputs/figures
run() {  # run <gt> <backbone> <stride> <chunk> <rows> <extra...>
  local gt=$1 bb=$2 stride=$3 chunk=$4 rows=$5; shift 5
  local scene; scene=$(basename "$gt" .npz)
  local tag=outputs/reports/pilotA_${scene}_${bb}_s${stride}_c${chunk}
  [ -f "$tag.json" ] && { echo "skip $tag"; return; }
  local free; free=$(df -BG --output=avail "$PWD" | tail -1 | tr -d ' G')
  [ "$free" -lt 5 ] && { echo "!!! under 5 GB free -- stopping"; exit 2; }
  echo "=== $(date '+%m-%d %H:%M') $scene / $bb (stride $stride, chunk $chunk, rows $rows) ==="
  python experiments/pilot_a.py --gt "$gt" --backbone "$bb" --keyframe-stride "$stride" \
      --chunk "$chunk" --overlap $((chunk/2)) --sites 2 --revisit-gap 32 --rows "$rows" "$@" \
      --json "$tag.json" 2>&1 | tee "$tag.log" \
      | grep -E "^(scene|  gate|  \*\*\* GATE|    fits|method|ceiling|chained|smr|classical|  [a-z_]+ +vs chained|report|Traceback|[A-Za-z]*Error|  File )"
}
STD7=$(ls data/gt/7scenes_*_seq01.npz 2>/dev/null); TUM=$(ls data/gt/tum_fr1_*.npz 2>/dev/null)
MULTI=$(ls data/gt/7scenes_*_s0*.npz 2>/dev/null)
FAST3="vggt pi3 vggt_omega"; FAST6="fast3r stream3r streamvggt"; SLOW="dust3r mast3r monst3r"
echo "### G1 standard protocol: $FAST3"
for bb in $FAST3; do for gt in $STD7; do run "$gt" "$bb" 5 32 chained,smr,smr_pgo,classical --ceiling; done
                     for gt in $TUM;  do run "$gt" "$bb" 3 32 chained,smr,smr_pgo,classical --ceiling; done; done
echo "### G2 multi-session: $FAST3"
for bb in $FAST3; do for gt in $MULTI; do run "$gt" "$bb" 10 16 chained,smr,smr_pgo,classical --ceiling; done; done
echo "### G3 standard protocol: $FAST6"
for bb in $FAST6; do for gt in $STD7; do run "$gt" "$bb" 5 32 chained,smr,smr_pgo,classical --ceiling; done
                     for gt in $TUM;  do run "$gt" "$bb" 3 32 chained,smr,smr_pgo,classical --ceiling; done; done
echo "### G4 multi-session: $FAST6"
for bb in $FAST6; do for gt in $MULTI; do run "$gt" "$bb" 10 16 chained,smr,smr_pgo,classical --ceiling; done; done
echo "### G5 chess x6 only, slow backbones: $SLOW"
for bb in $SLOW; do run data/gt/7scenes_chess_s0106.npz "$bb" 10 16 chained,smr; done
echo "### G6 length sweep, vggt"
bash scripts/run_length_sweep.sh data/gt/7scenes_chess_s0106.npz vggt
python scripts/pilot_a_table.py "outputs/reports/pilotA_*_c*.json" --tex outputs/reports/table_pilotA.tex > outputs/reports/table_pilotA.md
echo "### DONE $(date)"
