# Update v4.5 -- fast3r device type (last backbone fix)

    AttributeError: 'str' object has no attribute 'type'
    at fast3r/dust3r/inference_multiview.py:41, in loss_of_one_batch

Their code does `autocast_dict = dict(device_type=device.type)`, so the
plain "cuda" string that dust3r and mast3r accept happily fails here.  The
adapter now passes `torch.device(self.device)`.  Checked the other
adapters: dust3r and mast3r take strings fine, so nothing else changes.

Note the weights downloaded cleanly (2.59 GB from the hub) and the
stub mechanism reported exactly what it faked -- open3d and the pl_bolts
chain, both training/visualisation only.

    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r stream3r fast3r --max-views 4

Tests: 93 passed, 8 GPU-tier skipped.

## Backbone work: done
Six models, one socket, conventions declared and tested:
    vggt(w2c) pi3(c2w) dust3r(c2w) mast3r(c2w) stream3r(w2c) fast3r(c2w)
spanning feed-forward multi-view, pairwise+global-alignment, sparse-GA,
causal/streaming and fast many-view.  That is a stronger plug-in claim than
the paper needs.

## Next session: Pilot A -- and nothing else
    * chunk CO3D sequences (already on the server) into overlapping
      windows of 16
    * stitch two ways: Umeyama/Sim(3) chaining on the overlaps (what
      everyone does) vs binding each chunk into the scaffold and reading
      the trajectory out
    * hierarchical: scaffold supplies the coarse drift-free global frame,
      the backbone's within-chunk poses supply local precision
    * report ATE-RMSE, RPE-trans, RPE-rot, peak GPU memory, time/frame
    * pass condition: +SMR beats chained Sim(3) on ATE at >=100 frames on
      >=60% of sequences

28 days to the ICLR abstract.  That experiment is the paper.
