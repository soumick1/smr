# Update v4.7 -- group 3: StreamVGGT, MonST3R, VGGT-Omega (9 backbones)

6/6 confirmed working before this.  Three more adapters, APIs read from
each repo.

## New adapters
* **streamvggt** (w2c) -- wzzheng/streamvggt.  Package lives at
  `src/streamvggt`, so the third_party path is "streamvggt/src".  Note it
  uses `model.inference(frames)`, NOT `model(...)`, with frames as
  [{"img": img}] dicts; per-view results come back on `output.ress` with
  pts3d_in_other_view / conf / depth / depth_conf / camera_pose.  Weights:
  direct URL (lch01/StreamVGGT).  Saved locally as
  `streamvggt_checkpoints.pth` -- their remote name is the dangerously
  generic "checkpoints.pth", and a shared checkpoint dir should not contain
  a file by that name.  The rename rule is encoded in the tests.
* **monst3r** (c2w) -- junyi42/monst3r, the dynamic-scene member of the
  DUSt3R family.  Keeps DUSt3R's API, vendors its own dust3r fork (so the
  adapter isolates sys.modules first, same trap as mvdust3r).  Weights:
  direct URL (Junyi42/MonST3R_PO-TA-S-W...).  Useful to us specifically as
  the backbone for sequences with moving content.
* **vggt_omega** (w2c) -- facebookresearch/vggt-omega.  VGGTOmega() +
  encoding_to_camera; pip-installable (pyproject).  **Weights are
  ACCESS-GATED**: request at huggingface.co/facebook/VGGT-Omega, then
  `export HF_TOKEN=hf_...`; fetch_weights.py now sends it as a bearer
  token for huggingface.co URLs.

  PAPER CAVEAT, from their own README: an ancestor checkpoint of the
  released 1B model may have caused benchmark contamination, so its
  reported Table 1/2 numbers "may be inflated".  If VGGT-Omega appears in
  our tables, that has to be cited beside its row.

## lagernvs: dropped, and it was never a backbone
Their own minimal_inference.py says it "creates a target camera trajectory
(using VGGT for pose estimation)" -- LaGeR-NVS *consumes* geometry and
renders novel views.  It emits images, not per-view pose+depth, so there is
nothing for SMR to plug into.  It belongs in Table B as an NVS baseline to
compare against, not in the socket.  (It is also HF-gated.)

## Also dropped
mvdust3r (pytorch3d, build-from-source) and must3r (needed a newer
torchvision -- now moot after the environment fix, so it is a ~15-minute
restore from git history if you ever want a 10th row).

## Commands
    bash server/setup_backbones.sh group3
    export HF_TOKEN=hf_...            # only needed for vggt_omega
    python scripts/fetch_weights.py --backbones streamvggt monst3r vggt_omega
    python scripts/check_backbones.py --frames lab_photos/room/images_8 \
        --backbones vggt pi3 dust3r mast3r fast3r stream3r \
                    streamvggt monst3r vggt_omega --max-views 4

## The socket, complete
    vggt(w2c)  pi3(c2w)  dust3r(c2w)  mast3r(c2w)  fast3r(c2w)
    stream3r(w2c)  streamvggt(w2c)  monst3r(c2w)  cut3r(c2w)  vggt_omega(w2c)
spanning feed-forward multi-view, pairwise+global-alignment, sparse-GA,
two independent causal/streaming designs, dynamic scenes, and the newest
Omega model.  Nine adapters plus cut3r; 107 tests.

Mengmi asked for a plug-in that works on ALL existing models.  This is that
table's left column.

## Next session: Pilot A.  Nothing else.
