# smr_updates170 — point 1 (does the scaffold matter?) and point 2 (7-Scenes protocol)

## Point 1: the flat-memory baseline and the decomposition  (~20 min, cached passes)
`pilot_a --index {flat,template,dynamics}`: everything downstream identical (same cue, same top-5 proposals, same
two-site verification, same re-measurement, same correction, same PGO); only the memory that ranks candidates changes.
  flat      = DescriptorIndex: cosine over the cues, no scaffold (the reviewer's key-value baseline)
  template  = ScaffoldIndex as run in every table: cue -> RLS -> grid-code address, ranked by address overlap
  dynamics  = the same, with addresses read from the SETTLED attractor bumps instead of the analytic template
The report records `index` and `descriptor`.  Both new arms are ablation variants (Table 8 rows):
```bash
cd ~/smr && unzip -o smr_updates170.zip && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 VARIANTS="index_flat index_dynamics" bash scripts/ablate_all.sh      # 7-Scenes seq-01 (VGGT-Omega) + 12 CO3D orbits (VGGT)
python scripts/ablate_table.py --variants default index_flat index_dynamics
```
Prediction P68 (recorded): flat within 0.005 m / 1 AUC of template on the means (near-tie), template accepting fewer
wrong proposals on pumpkin/office; dynamics == template within noise (the equivalence test).  If the tie holds, the
paper's positioning changes (the scaffold's necessity is shown in the memory experiments, the stitching gains come from
recognise / re-measure / verify / revise around it) -- honest either way.

## Point 2: 7-Scenes on the standard test split  (~1-1.5 h for VGGT-Omega, GPU 0)
`scripts/run_7scenes_testsplit.sh` reads each scene's TestSplit.txt, extracts the missing seq-XX.zip, writes the GT
npz, and runs the Table-2 configuration on EVERY test trajectory, for --index template AND flat (same passes).
`scripts/sevenscenes_table.py` then prints per-scene means both ways: seq-01 only (the current Table 2) and the full
test split, for each index kind.
```bash
SEVEN=~/data/7scenes CUDA_VISIBLE_DEVICES=0 bash scripts/run_7scenes_testsplit.sh
python scripts/sevenscenes_table.py --root outputs/7scenes_test --bb vggt_omega | tee outputs/7scenes_test/summary.txt
tar czf outputs/7scenes_test.tgz outputs/7scenes_test/*.json outputs/7scenes_test/summary.txt outputs/ablate/*/index_*/*.json
```
YOU verify from the PDFs which protocol MASt3R-SLAM / VGGT-SLAM / VGGT-SLAM++ use for their 7-Scenes ATE (one sequence
per scene or the full test split).  Whichever it is, Table 2 must use the same one; if the full split, the runs above
are the table.  Send outputs/7scenes_test.tgz.
Prediction P69: full-split mean ATE within 20 % of the seq-01 mean for +SMR; pumpkin's share of the improvement falls.
