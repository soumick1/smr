# smr_updates81 — full-protocol relocalisation, T4 imagination experiment, gradio backbones

    scp -P 22 /mnt/c/MyFiles/smr_updates/smr_updates81.zip soumick@10.97.144.63:~/
    ssh -p 22 soumick@10.97.144.63 "cd ~/smr && unzip -o ~/smr_updates81.zip && rm ~/smr_updates81.zip"
    cd ~/smr && source .venv/bin/activate && python -m pytest -q     # expect 161 passed, 23 skipped
    git add -A && git commit -m "v81: full-protocol reloc, imagine.py (T4), gradio fast3r/omega" && git push

--------------------------------------------------------------
## 1. Relocalisation at the FULL protocol (every test frame)
--------------------------------------------------------------
relocalise.py now keys the query caches by query stride (a stride-1 rerun
can never collide with the stride-10 caches) and prints an ETA per row.
The map stays at keyframe stride 10 -- that is our system's design and
only handicaps us -- so the one remaining deviation vanishes: same query
set, same metric, same GT as DSAC*/ACE/Marepo/Reloc3r.

    # headline rows at stride 1 (~2.5-3 h total for vggt_omega on GPU 0)
    for sc in chess fire heads office pumpkin redkitchen stairs; do
      train=$(grep -o '[0-9]*' ~/7scenes/$sc/TrainSplit.txt | paste -sd,)
      test=$(grep -o '[0-9]*' ~/7scenes/$sc/TestSplit.txt | paste -sd,)
      python experiments/relocalise.py --gt data/gt/7scenes_${sc}_all.npz --backbone vggt_omega \
          --map-seqs $train --query-seqs $test --query-stride 1 --rows plain,smr \
          --json outputs/reports/reloc7full_${sc}_vggt_omega.json 2>&1 | tee outputs/reports/reloc7full_${sc}_vggt_omega.log
    done
    # optional: denser maps for the two failing scenes (prediction: stairs may
    # improve; kitchen's aliasing likely persists)
    for sc in redkitchen stairs; do
      train=$(grep -o '[0-9]*' ~/7scenes/$sc/TrainSplit.txt | paste -sd,)
      test=$(grep -o '[0-9]*' ~/7scenes/$sc/TestSplit.txt | paste -sd,)
      python experiments/relocalise.py --gt data/gt/7scenes_${sc}_all.npz --backbone vggt_omega \
          --map-seqs $train --query-seqs $test --keyframe-stride 5 --query-stride 1 --rows plain,smr \
          --json outputs/reports/reloc7full_${sc}_vggt_omega_m5.json 2>&1 | tee -a outputs/reports/reloc7full_${sc}_vggt_omega_m5.log
    done

Prediction, recorded: stride-1 medians/recalls within ~1 cm / 0.03 of the
keyframe-level numbers per scene (same distribution, 10x the samples);
the caveat line under the table is deleted, the conclusions do not move.

--------------------------------------------------------------
## 2. T4 -- imagination (experiments/imagine.py + src/smr/stitch/imagine.py)
--------------------------------------------------------------
One localised image, k integrated motion commands, no further images; the
memory is read out at the integrated pose and scored against the real frame
and the dataset depth.  Rows: nearest stored view / single-view reprojection /
SMR bank readout / fresh backbone pass on the recalled views (upper bound) /
--score-external DIR for a generative NVS model given the same views.
Metrics: PSNR, SSIM, LPIPS (auto-off if lpips missing: pip install lpips),
depth abs-rel, coverage.  Tests +2 (splat roundtrip exact on a plane;
metric sanity).

    pip install lpips
    python experiments/imagine.py --gt data/gt/7scenes_office_all.npz --backbone vggt_omega \
        --map-seqs 1,3,4,5,8,10 --target-seqs 2,6,7,9 --n-paths 25 --steps 1,2,4,8,16 \
        2>&1 | tee outputs/reports/imagine_office_vggt_omega.log
    python experiments/imagine.py --gt data/gt/7scenes_chess_s0106.npz --backbone vggt_omega \
        --map-seqs 1,2,4,6 --target-seqs 3,5 --n-paths 25 --steps 1,2,4,8,16 \
        2>&1 | tee outputs/reports/imagine_chess_vggt_omega.log

Budget: bank build ~4 min/scene; per path-step: smr/nearest/reproject are
CPU-cheap, rerun costs one backbone pass (~3 s) -> 25 paths x 5 steps
~ 20-30 min/scene with all four rows.

Predictions, recorded before the run:
* coverage: smr >> reproject at k >= 4 (the single view runs out of pixels;
  the memory does not), nearest always 1.0 but PSNR collapses with k.
* PSNR: rerun >= smr > reproject at every k; smr within ~1 dB of rerun
  (same geometry, lifted once vs fresh) at ZERO backbone cost per query.
* absrel: smr < nearest at k >= 2 (a copied image has wrong parallax);
  smr ~ rerun.
* degradation with k is gentle for smr (the bank does not care how far you
  integrated; only the start localisation error and map error matter).

--------------------------------------------------------------
## 3. Gradio
--------------------------------------------------------------
app/gradio_demo.py now offers vggt / vggt_omega / pi3 / fast3r (registry
dispatch; DUSt3R-family excluded: 50-75 s a pass is not a demo).
app/README_GRADIO.md is the standalone gradio-only runbook (env, launch,
ssh tunnel vs --share, per-backbone GPU costs, 30-second demo script,
troubleshooting).
