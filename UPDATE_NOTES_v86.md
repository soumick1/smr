# smr_updates86 — dense 3D-3D placement (the ACE/Marepo fight)

    scp/unzip as usual; python -m pytest -q     # expect 165 passed, 23 skipped
    git add -A && git commit -m "v86: dense 3D-3D query placement from the surfel bank" && git push

## What it is
The GT-map ablation showed our residual loss to ACE/Marepo is PLACEMENT:
they solve pose from thousands of per-pixel correspondences, we fitted four
anchor camera poses.  v86 places each query by robust point-set Umeyama over
the anchors' pass depth unprojected at the exact pixels the surfel bank
stored in metric world coordinates -- same proposals, same verification,
same consensus rules, ~10^3-10^4 constraints instead of 4.
`--placement dense` (bank built once per map, cached as npz).
Rejected earlier levers stay rejected (sites-3 / refine notes stand).

## Run, in this order (GT-map probes first: they show the new ceiling in minutes)
    for sc in chess office pumpkin stairs; do
      train=$(grep -o '[0-9]*' ~/7scenes/$sc/TrainSplit.txt | paste -sd,); test=$(grep -o '[0-9]*' ~/7scenes/$sc/TestSplit.txt | paste -sd,)
      python experiments/relocalise.py --gt data/gt/7scenes_${sc}_all.npz --backbone vggt_omega \
          --map-seqs $train --query-seqs $test --map-poses gt --placement dense --rows plain \
          --json outputs/reports/relocGTd_${sc}_vggt_omega.json 2>&1 | tee outputs/reports/relocGTd_${sc}_vggt_omega.log
    done
    # then the real maps, full protocol, all seven (kitchen uses its registered map):
    for sc in chess fire heads office pumpkin stairs; do
      train=$(grep -o '[0-9]*' ~/7scenes/$sc/TrainSplit.txt | paste -sd,); test=$(grep -o '[0-9]*' ~/7scenes/$sc/TestSplit.txt | paste -sd,)
      python experiments/relocalise.py --gt data/gt/7scenes_${sc}_all.npz --backbone vggt_omega \
          --map-seqs $train --query-seqs $test --placement dense --query-stride 1 --rows plain,smr \
          --json outputs/reports/reloc7d_${sc}_vggt_omega.json 2>&1 | tee outputs/reports/reloc7d_${sc}_vggt_omega.log
    done
    train=$(grep -o '[0-9]*' ~/7scenes/redkitchen/TrainSplit.txt | paste -sd,); test=$(grep -o '[0-9]*' ~/7scenes/redkitchen/TestSplit.txt | paste -sd,)
    python experiments/relocalise.py --gt data/gt/7scenes_redkitchen_all.npz --backbone vggt_omega \
        --map-seqs $train --query-seqs $test --map-mode register --placement dense --query-stride 1 --rows plain,smr \
        --json outputs/reports/reloc7d_redkitchen_vggt_omega.json 2>&1 | tee outputs/reports/reloc7d_redkitchen_vggt_omega.log

Cost: dense passes are not pose-cached (they need depth), so the stride-1
loop is ~1-2 s/query -> plan an overnight for the seven scenes.

## Predictions, recorded before the run
* GT-map probes: chess median 2.4 -> 1.3-1.8 cm and rotation 1.2 -> 0.7-1.0 deg;
  office 0.74 -> 0.82-0.90 @5/5; pumpkin 0.57 -> 0.68-0.78; stairs 0.44 -> 0.50-0.58
  (its gross-error fraction is retrieval aliasing and will not move).
* Real maps: chess/fire/heads -> 0.96-0.99 @5/5 (DSAC*/ACE band), office -> 0.70+,
  pumpkin -> 0.65+, kitchen -> 0.45-0.55, stairs unchanged-ish; 7-scene average
  0.72-0.78 (from 0.66), medians ~2.5-3 cm / ~1.3 deg -- Marepo's median band,
  with the matched-budget reruns still to come for the recall fight.
