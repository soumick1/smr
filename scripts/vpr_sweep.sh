#!/bin/bash
# v182: established place-recognition descriptors as the retrieval cue (reviewer request), Table 7 protocol.
# Same sequences and cached window passes as ablate_all.sh; anchored passes are recomputed where a descriptor proposes
# different sites (GPU, a few seconds each). Every arm uses the SAME candidate budget (top-5) with the cosine gate
# DISABLED (--desc-thresh -1), so the comparison is about ranking quality and geometry decides; `dino05` is the
# production setting (DINO, gate 0.5) for reference. Runs both window-geometry modes (anchored = paper default; plain =
# v181), because a weaker descriptor's wrong proposals can poison windows in the anchored mode.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/vpr_sweep.sh
#   ARMS="dino eigenplaces salad@4096" MODES=plain DATASETS=7scenes bash scripts/vpr_sweep.sh
# Then: for m in anchored plain; do python scripts/gate_table.py --root outputs/ablate/vpr/$m --gt-dir data/gt \
#          --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --md outputs/vpr_$m.md; done
# Optional arms need extra installs (see src/smr/stitch/vpr_descriptors.py): netvlad (hloc), mixvpr (repo + checkpoint),
# boq (torch.hub). A failing arm is reported and skipped; the others continue.
cd "$(dirname "$0")/.."; source .venv/bin/activate; ROOT=${ROOT:-outputs/ablate/vpr}; mkdir -p $ROOT
python experiments/pilot_a.py --help 2>/dev/null | grep -q "eigenplaces" || { echo "pilot_a lacks the VPR descriptors: run  python scripts/apply_all_patches.py  first"; exit 2; }
step() { echo "[$(date +%H:%M:%S)] $*"; }
ARMS=${ARMS:-"dino05 dino eigenplaces cosplace salad@4096 boq@4096 netvlad mixvpr dino+eigenplaces dino+salad@4096"}
MODES=${MODES:-"anchored plain"}; DATASETS=${DATASETS:-"7scenes co3d"}
SCENES7=${SCENES7:-"chess fire heads office pumpkin redkitchen stairs"}
# REVERSE=1 iterates scenes back to front, so two screens sharing a dataset rarely touch the same scene's cache at once
[ -n "${REVERSE:-}" ] && SCENES7=$(echo $SCENES7 | tr " " "\n" | tac | tr "\n" " ")
run() { # $1 out json prefix, rest = pilot_a args
  local out=$1; shift
  python experiments/pilot_a.py "$@" --rows chained,smr,smr_pgo --json $out.json --save-est $out.npz > $out.log 2>&1 \
    || { step "  FAILED: $(grep -m1 -E 'Error|needs|REFUS' $out.log | cut -c1-160)"; return 1; }
}
for mode in $MODES; do
  for arm in $ARMS; do
    if [ "$arm" = dino05 ]; then desc=dino; gate="--desc-thresh 0.5"; else desc=$arm; gate="--desc-thresh -1"; fi
    for ds in $DATASETS; do
      out=$ROOT/$mode/$ds/$arm; mkdir -p $out; fails=0
      if [ $ds = 7scenes ]; then
        for s in $SCENES7; do GT=data/gt/7scenes_${s}_seq01.npz; [ -f $GT ] || continue; [ -f $out/$s.json ] && continue
          step "$mode / $arm / 7scenes $s"
          run $out/$s --gt $GT --backbone ${SLAMBB:-vggt_omega} --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 \
              --descriptor $desc $gate --local-from $mode --index template --seed 0 || { fails=$((fails+1)); [ $fails -ge 2 ] && { step "  skipping arm $arm (descriptor unavailable?)"; break; }; }
        done
      else
        list=$(ls data/gt/co3d_full/*.npz | { [ -n "${ALL:-}" ] && cat || head -${NSEQ:-12}; } | { [ -n "${REVERSE:-}" ] && tac || cat; })
        for GT in $list; do s=$(basename $GT .npz); [ -f $out/$s.json ] && continue
          step "$mode / $arm / co3d $s"
          run $out/$s --gt $GT --backbone ${BB:-vggt} --keyframe-stride 1 --max-frames 200 --chunk 32 --overlap 16 --sites 2 \
              --descriptor $desc $gate --local-from $mode --index template --seed 0 || { fails=$((fails+1)); [ $fails -ge 2 ] && { step "  skipping arm $arm (descriptor unavailable?)"; break; }; }
        done
      fi
    done
  done
done
step "done -> python scripts/gate_table.py --root $ROOT/anchored --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --md outputs/vpr_anchored.md (and plain)"
