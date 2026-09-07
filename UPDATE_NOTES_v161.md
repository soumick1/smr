# smr_updates161 — loop-closure overlay figures (two windows, two colours) + viewpoint sweep

## Why office looks smeared from every angle
Office's recovery is small (0.047 -> 0.026 m): twelve windows overlap the same desk with residual misalignments of
a few cm in BOTH rows, which at close range renders as smear, not as the distinct duplicated layers pumpkin shows
(0.161 m). No viewpoint fixes that. Pumpkin is the main SLAM figure (full-cloud, inset on the cabinets).

## The readable version of a small closure: only the two windows the closure joins
`points_suite --per-ctx` now also writes ctx_XX.ply (each window's points in the row frame); `fig_recon --windows A B`
draws only those two windows, each tinted with its window colour (texture kept, --tint 0.55), with their frusta, raw
and +SMR side by side: misaligned on the left, aligned on the right. This is the classic loop-closure figure and is
what the memory literally does.

## Office (and any scene) recipe
```bash
cd ~/smr && unzip -o smr_updates161.zip && source .venv/bin/activate
n=fig_7scenes_office_seq01; GT=data/gt/7scenes_office_seq01.npz
# 1. which windows does a closure join?  -> pairs (closing window, remembered window)
python -c "import json; d=json.load(open('outputs/reports/$n.json')); print([(e['chunk'], e['loop'].get('anchor_chunk')) for e in d['events']['smr'] if e.get('loop') and e['loop'].get('accepted')])"
# 2. per-window clouds for both rows (~2.5 min each, GPU 0)
for row in chained smr; do CUDA_VISIBLE_DEVICES=0 python experiments/points_suite.py --gt $GT --pilot outputs/est/$n.npz --row $row --backbone vggt_omega --w 32 --overlap 16 --k-ctx 0 --stride 1 \
   --points-from auto --conf-abs 2.0 --fallback median --abstain-rel 0 --per-ctx --out-dir outputs/points/${n}_$row > outputs/points/${n}_$row.log 2>&1; done
# 3. the overlay (CPU): A = remembered window, B = closing window from step 1; viewpoint from the sweep
python scripts/fig_recon.py --est outputs/est/$n.npz --json outputs/reports/$n.json --ply outputs/points/${n}_chained outputs/points/${n}_smr --rows chained smr \
   --windows A B --labels "raw: windows A and B" "+SMR: windows A and B" --azimuth 0 --elev 30 --out outputs/figures/closure_office
```
For pumpkin the accepted closures are (5,0), (6,0), (11,6): `--windows 0 6` or `--windows 6 11`.
The full-cloud figure and the sweep are unchanged (`--sweep` for angles; `--zoom X Y HALF` for the inset).
Everything is 600 dpi; renders 2600x1800 px per panel.
