# Update v4.8 -- MonST3R weights: Google Drive, now automated

`wget exited 8` = HTTP error: my HuggingFace URL was a guess and 404'd.
Their repo exists, but the filename inside it is not the repo name.

Their OWN data/download_ckpt.sh settles it -- MonST3R ships via Google
Drive:
    gdown --fuzzy https://drive.google.com/file/d/1Z1jO_JmfZj0z3bgMvCwqfUhyZ1bIbc9E/view

Rather than hand you manual commands for the second time (CUT3R is the
same), fetch_weights.py now has a gdown transport: adapters declare
`weights_gdrive`, and the fetcher picks the right transport per source --
https via wget (with HF_TOKEN for gated repos) or Drive via gdown, with the
same size sanity check and skip-if-present behaviour.

    pip install gdown
    python scripts/fetch_weights.py --backbones monst3r cut3r

Also fixed: the summary table ran long filenames into the status column.

Note streamvggt (4.7 GB) and vggt_omega (4.3 GB) downloaded fine -- and
vggt_omega means your HF_TOKEN and the gated-download path both work.

    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r \
                    streamvggt monst3r vggt_omega --max-views 4

Tests: 107 passed, 10 GPU-tier skipped.
