# smr_updates70 — disk-full fixes: MASt3R cache leak, atomic pass cache, tolerant table, sweep guard

    scp -P 22 /mnt/c/MyFiles/smr_updates/smr_updates70.zip soumick@10.97.144.63:~/
    ssh -p 22 soumick@10.97.144.63 "cd ~/smr && unzip -o ~/smr_updates70.zip && rm ~/smr_updates70.zip && git status --short"

## Cause
The MASt3R adapter created a fresh mkdtemp cache per pass and never removed
it; MASt3R's sparse alignment writes per-image features and per-pair
correspondences there (hundreds of MB per 20-image pass).  The sweep ran
MASt3R on ~500 passes -> tens of GB in /tmp.  Everything after that wrote
truncated logs/JSON/cache files.

## Fixed
* mast3r.py: cache lives in a TemporaryDirectory for the duration of the
  call (pi3 / cut3r already did this).
* passes.py: PassCache writes atomically (tmp + rename) and, if a cache
  file is unreadable, moves it aside and starts fresh instead of crashing.
* pilot_a_table.py: skips unreadable reports.
* run_pilot_a_sweep.sh: stops if the repo or /tmp has < 5 GB free.

## Recover and resume (in this order)
    # 1. see where the space went
    df -h ~ /tmp; du -sh /tmp/mast3r_cache_* 2>/dev/null | tail -3; du -sh ~/7scenes ~/tum ~/smr/outputs ~/.cache/torch 2>/dev/null
    # 2. free it: the leaked caches, the download archives (already extracted), unused seq zips
    rm -rf /tmp/mast3r_cache_* /tmp/pi3_views_*
    rm -f ~/7scenes/*.zip ~/tum/*.tgz ~/7scenes/*/seq-0[2-9].zip     # keep the extracted seq-XX folders
    # 3. drop anything truncated by the full disk
    cd ~/smr && find outputs/reports -type f -size 0 -delete
    python - <<'PY'
import json, glob, os
for p in glob.glob("outputs/reports/pilotA_*.json"):
    try: json.load(open(p))
    except Exception: print("removing truncated", p); os.remove(p)
PY
    for f in outputs/cache/*.npy; do python -c "import numpy as np,sys; np.load(sys.argv[1], allow_pickle=True)" "$f" 2>/dev/null || { echo "corrupt cache $f -> removed"; rm -f "$f"; }; done
    # 4. apply v70, then resume (finished reports are skipped; cached passes are reused)
    python -m pytest -q tests/test_backbones.py tests/test_stitch.py
    git add -A && git commit -m "v70: MASt3R cache leak, atomic pass cache, disk guards" && git push
    bash scripts/prepare_standard_sets.sh
    CUDA_VISIBLE_DEVICES=0 screen -S sweep0 -dm bash -c 'bash scripts/run_pilot_a_sweep.sh vggt pi3 dust3r 2>&1 | tee -a outputs/reports/sweep0.log'
    CUDA_VISIBLE_DEVICES=1 screen -S sweep1 -dm bash -c 'bash scripts/run_pilot_a_sweep.sh mast3r fast3r stream3r 2>&1 | tee -a outputs/reports/sweep1.log'
    CUDA_VISIBLE_DEVICES=2 screen -S sweep2 -dm bash -c 'bash scripts/run_pilot_a_sweep.sh streamvggt monst3r vggt_omega 2>&1 | tee -a outputs/reports/sweep2.log'

Note the reports that did complete before the disk filled (chess x6 for
vggt/pi3, room for vggt, ...) are valid and are kept; the v69 sweep
names carry the stride, so old-style names in outputs/reports are simply
ignored by the skip logic and re-run at the protocol strides.
