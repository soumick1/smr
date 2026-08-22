# SMR completion-head RUNBOOK (single source of truth)

## State right now
  DONE     vggt subset harvest (1,848 pairs, 88 scenes)
  DONE     co3d_subset checkpoint: psnr_hole 29.2 dB, no overfitting
           through 95 epochs, perceptual live, pass-through 0.0389
  RUNNING  pi3 subset harvest (GPU 1, screen harvest_pi3)

## GPU layout
  GPU 0: training | GPU 1: pi3 subset harvest | GPU 2: category cycle

## [A] OPTIONAL, 2 min, do anytime: first zero-shot number
CUDA_VISIBLE_DEVICES=0 python experiments/train_completion.py \
  --eval-only outputs/logs/completion/co3d_subset/ckpt_best.pt \
  --data data/completion --all-val --name zshot_llff_subset
# Goal: CO3D-trained head evaluated on LLFF+orbit it never saw.

## [B] Category cycle -- CORRECTED BUDGET (measured 4.8 min/scene/
##     backbone from the 88-scene subset harvest; the old 1-2 h/category
##     estimate was wrong).  Run TWO parallel streams, LIMIT=40:

# stream A (screen co3dcycA):
CUDA_VISIBLE_DEVICES=0 DATA=~/co3d_full_a BACKBONES="vggt pi3" LIMIT=40 \
  bash scripts/co3d_cycle.sh hydrant vase teddybear bench plant
# stream B (screen co3dcycB):
CUDA_VISIBLE_DEVICES=2 DATA=~/co3d_full_b BACKBONES="vggt pi3" LIMIT=40 \
  bash scripts/co3d_cycle.sh toytruck cup bowl book backpack

# Per category: download 0.5-2 h + 2 x 40 x 4.8 min ~ 6.5 h harvest
# => ~7-8 h/category, 5 categories/stream => ~1.5 days both streams.
# Yield: ~17k pairs (+ ~3.7k subset) ~ 20k total -- ample for a 10.6M
# head.  Depth-per-category (LIMIT) trades against category diversity;
# 40 favours diversity, which is what generalisation needs.
# When co3d_v1 training starts on GPU 0 later, stream A just runs
# slower; no conflict.

## [C] Main checkpoint -- when [B] lands (corpus is mixed-backbone
##     by construction, so this IS the robustness model)
CUDA_VISIBLE_DEVICES=0 python experiments/train_completion.py \
  --data data/completion_co3d --name co3d_v1 \
  --val-frac 0.1 --steps 60000 --perc-weight 0.05
# ~3 h at 37 img/s.  Goal: the paper's primary completion head.

## [D] Zero-shot, the paper number
CUDA_VISIBLE_DEVICES=0 python experiments/train_completion.py \
  --eval-only outputs/logs/completion/co3d_v1/ckpt_best.pt \
  --data data/completion --all-val --name zshot_llff_v1

## [E] Next session with Claude: --complete into run_orbit_demo
# (before/after disocclusion on the condo orbit; the completion
#  section's opening figure)
