# smr_updates171 — point 1 scored; point 2 invalid because of a cache-key bug (fixed); re-run

## Point 1 (does the scaffold matter?) — P68 HIT, honest outcome
Same passes, same everything but the memory that ranks candidates (7-Scenes seq-01 VGGT-Omega; 12 CO3D orbits VGGT):
  template (as run) 0.045 -> 0.030 / 0.029, 4.1 closures | CO3D 88.8 -> 93.4 / 93.1
  flat key-value    0.045 -> 0.033 / 0.033, 3.4 closures | CO3D 88.8 -> 93.3 / 92.9   (template better on 6/12 orbits: a tie)
  dynamics          0.045 -> 0.096 / 0.094, 3.6 closures | CO3D 88.8 -> 93.1 / 93.0
Per scene the difference is where a wrong proposal is costly: redkitchen flat 0.046 vs template 0.028 (a wrong closure
passes with cosine ranking), dynamics 0.488 (settled-bump addresses alias in that room; the template/dynamics
equivalence held on synthetic poses, not here). Conclusion for the paper: the stitching gains come from
recognise -> re-measure -> verify -> revise; the scaffold's ranking buys fewer wrong proposals in look-alike scenes.
Its necessity is shown where it acts (memory experiments), not by Tables 1-2. Will be written exactly so.

## Point 2 (7-Scenes test split) — the run was INVALID: pass-cache collision
pilot_a's default pass cache was outputs/cache/{scene}_{backbone}_s{stride}.npy, keyed by keyframe position inside.
chess seq-03 therefore reused chess seq-01's cached passes and was scored against seq-03's GT -> reference AUC ~0 on
every non-seq-01 sequence. GT files are correct (script reproduces seq-01 bit-for-bit; c2w scores 94.1 on seq-03).
Fixed: sequences other than seq-01 get their own cache file ({scene}_seq03_...); seq-01 caches remain valid; CO3D and
multi-session runs were never affected (unique scene names). The split runner now detects a stale report (reference
AUC < 20) and redoes it.
```bash
cd ~/smr && unzip -o smr_updates171.zip
screen -dmS split bash -c "$ENV && SEVEN=$HOME/7scenes CUDA_VISIBLE_DEVICES=1 bash scripts/run_7scenes_testsplit.sh > outputs/reports/7scenes_testsplit.log 2>&1"
# ~1 h (15 new trajectories x ~3 min template + seconds flat). Then:
python scripts/sevenscenes_table.py --root outputs/7scenes_test --bb vggt_omega | tee outputs/7scenes_test/summary.txt
tar czf outputs/7scenes_test.tgz outputs/7scenes_test/*.json outputs/7scenes_test/summary.txt
```
Sanity while it runs: `grep -h '"reference"' -A0 outputs/7scenes_test/chess_seq03_vggt_omega_template.json | head -c 0; python -c "import json; d=json.load(open('outputs/7scenes_test/chess_seq03_vggt_omega_template.json')); print(d['reference']['auc30'])"` must print ~90, not ~1.
Still needed from you: which sequences MASt3R-SLAM / VGGT-SLAM use for 7-Scenes ATE.
