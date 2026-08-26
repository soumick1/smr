# smr_updates76 — session-aware chunking + relocalisation at session starts, robust batch solve

    scp -P 22 /mnt/c/MyFiles/smr_updates/smr_updates76.zip soumick@10.97.144.63:~/
    ssh -p 22 soumick@10.97.144.63 "cd ~/smr && unzip -o ~/smr_updates76.zip && rm ~/smr_updates76.zip && git status --short"
    cd ~/smr && source .venv/bin/activate && python -m pytest -q      # expect 154 passed, 23 skipped
    git add -A && git commit -m "v76: session-aware chunks, relocalisation at session start, robust batch" && git push

## What the sweep + A/B said
* 8 of 9 backbones have valid multi-session rows (MASt3R 0.148 -> 0.059!); MonST3R
  fails the probe (reference AUC 5.4) and is reported as a backbone failure.
* Drift curve (VGGT, chess x6 prefixes 100..600 kf): chained 0.041 -> 0.083,
  SMR flat at 0.045-0.047, single pass 0.039 -> 0.049 until it OOMs past 450.
* office x6 was a PROTOCOL bug, not a closure problem: the chunk straddling a
  session boundary mixes two disjoint places, its pass is garbage, the chain
  breaks (VGGT-Omega chained 0.551 with every chunk fine).  Fixed here.
* floor / room (true aliasing): no verification variant wins uniformly (v75
  fixed pi3-floor and broke pi3-room; remeasure helped Omega-floor and
  VGGT-room but not to chained level).  Single wrong closures dominate, so
  the fix is validation after the fact: robust batch (below).  The v75 rules
  are OFF by default again (the sweep's rules), available as --mutual-nn and
  --site-agree for the ablation table.

## Changed
* chunks.make_session_chunks: chunks never straddle a session (frame_ids //
  100000 from the loader); pilot_a.py uses it automatically for multi-session npz.
* AnchoredStitcher: a session-start chunk (no overlap) is placed by
  RELOCALISATION -- verified sites only, scale carried from the previous pass
  -- and the stretch is in RECOVERY (no drift budget) until a closure with two
  agreeing sites lands.  Events carry session_start / recovering / relocalisation.
* posegraph.solve(robust_reject=True): after the first solve, loop edges with
  residual > max(3 x median, 0.3) are dropped and the graph re-solved; the
  smr_pgo row reports n_dropped_edges (--no-robust-batch for the ablation).
* Tests +3 (session chunks; relocalisation at a session start; robust batch
  drops an injected bad edge).

## Run (cached passes: minutes for the fast backbones)
    for bb in vggt_omega vggt pi3 fast3r streamvggt stream3r; do
      python experiments/pilot_a.py --gt data/gt/7scenes_office_s0106.npz --backbone $bb --keyframe-stride 10 \
          --chunk 16 --overlap 8 --sites 2 --revisit-gap 32 --rows chained,smr,smr_pgo --ceiling \
          --json outputs/reports/pilotA_7scenes_office_s0106_${bb}_s10_c16_v76.json 2>&1 | grep -E "sessions|^(chained|smr)"
      for s in floor room; do
        python experiments/pilot_a.py --gt data/gt/tum_fr1_$s.npz --backbone $bb --keyframe-stride 3 \
            --chunk 32 --overlap 16 --sites 2 --revisit-gap 32 --rows chained,smr,smr_pgo \
            --json outputs/reports/pilotA_tum_fr1_${s}_${bb}_s3_c32_v76.json 2>&1 | grep -E "^(chained|smr)" | sed "s/^/$s $bb /"
      done
    done
    python scripts/pilot_a_table.py "outputs/reports/pilotA_*_v76.json" > outputs/reports/table_v76.md

## Predictions
* office x6: chained itself improves (no straddling chunk); Omega and VGGT
  smr fall from 0.55 / 0.76 to <= 0.15 (single pass 0.040 / 0.109 on 300
  frames); if the relocalisation at a session start finds no verified site
  the session is reported as unplaced rather than silently chained.
* floor / room: smr_pgo (robust) <= chained on every backbone, with 1-3
  dropped edges; the streaming smr row unchanged (its fix is the next step:
  revoke a closure the following chunk contradicts).
