# smr_updates84 — session registration by consensus (the kitchen/stairs map fix) + placement refine

    scp/unzip as usual; python -m pytest -q   # expect 163 passed, 23 skipped
    git add -A && git commit -m "v84: per-session map registration by consensus; reloc refine pass" && git push

## Why (measured this morning)
GT-map ablation: the relocalisation machinery reaches 0.96/0.97/1.00/0.74/0.57/**0.73**/0.44
@5cm/5deg on the seven scenes.  redkitchen's 0.00 with our map is ENTIRELY the map
(machinery ceiling 0.73; every intra-session kitchen map is fine at ~0.026 ATE);
stairs is half map, half true aliasing (12% gross errors even with a GT map).

## What v84 adds
* `--map-mode register`: stitch each session ALONE (works everywhere), then
  register sessions by CONSENSUS over ~24 per-pair small-pass Sim(3)
  estimates -- wrong-counter pairs form minority clusters and are discarded,
  which per-pair verification cannot do.  Unregistered sessions are reported,
  never silently chained.  Sandbox: recovers a scrambled second session with
  a third of the pairs poisoned (cluster 16-18/24, ATE 0.07).
* `--refine`: one extra placement pass against the map keyframes nearest the
  first estimate (spatial verification); replaces the estimate only if
  consistent (0.5 m / 15 deg).
* `--sites` now flows into relocalisation placement (try 3).

## Run (order; kitchen first -- it decides the average)
    for sc in redkitchen stairs; do
      train=$(grep -o '[0-9]*' ~/7scenes/$sc/TrainSplit.txt | paste -sd,); test=$(grep -o '[0-9]*' ~/7scenes/$sc/TestSplit.txt | paste -sd,)
      python experiments/relocalise.py --gt data/gt/7scenes_${sc}_all.npz --backbone vggt_omega \
          --map-seqs $train --query-seqs $test --map-mode register --query-stride 1 --rows plain,smr \
          --json outputs/reports/reloc7reg_${sc}_vggt_omega.json 2>&1 | tee outputs/reports/reloc7reg_${sc}_vggt_omega.log
    done
    # then the placement levers on the three placement-limited scenes (keyframe queries, fast):
    for sc in office pumpkin stairs; do
      train=$(grep -o '[0-9]*' ~/7scenes/$sc/TrainSplit.txt | paste -sd,); test=$(grep -o '[0-9]*' ~/7scenes/$sc/TestSplit.txt | paste -sd,)
      python experiments/relocalise.py --gt data/gt/7scenes_${sc}_all.npz --backbone vggt_omega \
          --map-seqs $train --query-seqs $test --map-poses gt --sites 3 --refine --rows plain \
          --json outputs/reports/relocGT3r_${sc}_vggt_omega.json 2>&1 | tee outputs/reports/relocGT3r_${sc}_vggt_omega.log
    done

## Predictions, recorded
* kitchen registered map ATE 0.03-0.08 -> plain @5/5 0.55-0.73 (ceiling 0.73), median 3-5 cm.
* stairs registered map: either registers (ATE ~0.05-0.15 -> @5/5 0.30-0.44) or some
  sessions report UNREGISTERED (also a valid, reported outcome on the most aliased scene).
* sites 3 + refine on GT maps: office 0.74 -> 0.80+/-, pumpkin 0.57 -> 0.65+/-,
  stairs ceiling +0.05; if these hold they become the defaults and the full loop reruns.
