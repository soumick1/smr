# Update v2.0 -- zero-shot eval mode + the full training ladder

Code: train_completion.py gains --all-val (every pair becomes
validation; valid only with --eval-only or --dry-run).  This is the
zero-shot switch: evaluate a CO3D-trained checkpoint on the LLFF+orbit
pairs it never saw.

Doc (already in smr_overleaf.zip): appendix "Every parameter, with its
origin" -- ~45 parameters in six origin-tagged tables ([D]erived /
[C]alibrated / [S]tandard / [A]blatable), each with its one-line
defense and citation; five new bib entries (unet, adamw, sgdr,
perceptual, llff); risk-register status section; ridge-form clauses;
corpus citations.  configs/*.yaml declared canonical for numerals.

## THE LADDER (each step: command + what it achieves)

# [1] First established-corpus checkpoint -- once the subset harvest
#     finishes.  Goal: corpus-scale pipeline proof + first checkpoint
#     capable of zero-shot claims.  ~1 h.
screen -S train1
python experiments/train_completion.py --data data/completion_co3d \
    --name co3d_subset --val-frac 0.1 --steps 20000 --perc-weight 0.05

# [2] pi3 harvest of the same subset -> SAME out dir (filenames carry
#     the backbone tag).  Goal: backbone-diverse pairs so one head
#     completes anyone's geometry.  ~6-9 h, run after [1]'s harvest
#     screen is free.
for cat in ~/co3d_data/*/; do
  python scripts/harvest_root.py --root "$cat" --images-sub images \
     --backbone pi3 --out data/completion_co3d \
     --holdouts every:5 --context 1,4,all --max-views 32
done
# (Other backbones: only verified pose+depth multi-view models qualify.
#  DUSt3R/MASt3R family are candidates pending repo verification;
#  VGGT-Omega still unverified; mono-depth models don't qualify.)

# [3] Main head on the enlarged corpus -- after co3d_cycle.sh lands its
#     categories.  Goal: the paper's primary checkpoint.  ~2.5 h.
python experiments/train_completion.py --data data/completion_co3d \
    --name co3d_v1 --val-frac 0.1 --steps 60000 --perc-weight 0.05

# [4] Mixed-backbone retrain -- after [2] completes.  Goal: the
#     backbone-robustness checkpoint (corpus now contains vggt+pi3).
python experiments/train_completion.py --data data/completion_co3d \
    --name co3d_v2_mixbb --val-frac 0.1 --steps 60000 --perc-weight 0.05

# [5] ZERO-SHOT evaluation on LLFF+orbit (never in any corpus run).
#     Goal: the generalisation number for the paper.
python experiments/train_completion.py \
    --eval-only outputs/logs/completion/co3d_v1/ckpt_best.pt \
    --data data/completion --all-val --name zshot_llff_v1

# [6] Next session, once a checkpoint exists: --complete wired into
#     run_orbit_demo -> the before/after disocclusion figure on your
#     condo orbit.
