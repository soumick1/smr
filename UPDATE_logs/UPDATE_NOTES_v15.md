# Update v1.5 -- results banked; completion-head data pipeline

Doc: measured-status gains the server-scale paragraph (room floor
6.5x down at full scale; 25/25 ladder at scale) and the orbit-path
figure with the honest 2-deg drift accounting (matches the T6 intrinsic
rate -- the documented long-integration property, deliberately left
uncorrected in the demo).

NEW scripts/harvest_completion_pairs.py -- generates (partial splat ->
actual frame) supervision for the completion head.  Start the dataset
tonight:
    for s in fern flower fortress horns leaves orchids room trex; do \
      python scripts/harvest_completion_pairs.py \
             --frames lab_photos/$s/images_8 --backbone vggt \
             --out data/completion; done
    python scripts/harvest_completion_pairs.py --frames orbit_frames \
           --backbone vggt --out data/completion
Expect ~330 pairs (~one per view).  The head itself (small UNet on
splat_rgb/depth/mask -> tgt_rgb/depth, trained only where tgt_mask)
comes next session.
