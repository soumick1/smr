# smr_updates182 — established place-recognition descriptors as the retrieval cue (reviewer request)

Reviewer: "Retrieval choices are primarily DINOv2-based; comparisons to established place-recognition methods (NetVLAD,
EigenPlaces, MixVPR) or hybrid retrieval strategies are absent." The cue is a per-keyframe unit vector consumed by the same
RLS cue memory at any dimension, so this is one new module, one `--descriptor` dispatch line, and the Table 7 protocol.

## What is in the zip
* `src/smr/stitch/vpr_descriptors.py`: kinds `eigenplaces` (ResNet50, 2048-D, torch.hub gmberton/eigenplaces), `cosplace`
  (2048-D, torch.hub), `salad` (DINOv2-B + SALAD, 8448-D, torch.hub serizba/salad), `boq` (12288-D, torch.hub
  amaralibey/bag-of-queries), `netvlad` (VGG16 Pitts30k, 4096-D, through hloc), `mixvpr` (official repo + checkpoint,
  4096-D), hybrids `a+b` (concatenated unit vectors, renormalised: cosine = mean of the two), projection `kind@N` (seeded
  Gaussian; keeps the O(D²) cue memory bounded for the 8k/12k-D methods). Each method runs at its authors' input size.
* `scripts/apply_v182_patch.py` (also run by `apply_all_patches.py`): `--descriptor` accepts these kinds; `desc_fn`
  dispatches them; DINO's gate default is unchanged.
* `scripts/vpr_sweep.sh`: Table 7 protocol on the same sequences and cached window passes, both window-geometry modes.
  Arms: `dino05` (production: DINO, gate 0.5), then `dino`, `eigenplaces`, `cosplace`, `salad@4096`, `boq@4096`, `netvlad`,
  `mixvpr`, `dino+eigenplaces`, `dino+salad@4096`, all with the cosine gate DISABLED (`--desc-thresh -1`) and the same
  top-5 budget, so the comparison is about ranking quality and geometry decides; no per-descriptor threshold tuning.
  An arm whose model cannot be loaded is reported and skipped.
* `scripts/gate_table.py`: recall@1 / recall@5 columns (retrieval quality against GT revisits) next to closures,
  precision, usefulness and ATE/AUC.
* `scripts/closure_reliability.py`: the dataset alias bug (`7scenes` vs the report's `sevenscenes`) that invalidated the
  `plain_seed0/1` reliability outputs is fixed; re-run those two calls.

## Install (once)
```bash
pip install torchvision                                   # if missing (EigenPlaces/CosPlace/MixVPR are ResNet50)
python -m smr.stitch.vpr_descriptors eigenplaces data/7scenes/chess/seq-01/frame-000000.color.png data/7scenes/chess/seq-01/frame-000500.color.png   # smoke: shape (2, 2048)
# optional arms
pip install git+https://github.com/cvg/Hierarchical-Localization.git          # netvlad
git clone https://github.com/amaralibey/MixVPR third_party/MixVPR && pip install pytorch_lightning   # mixvpr; put the authors' resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt in third_party/checkpoints/
```
(adjust the smoke-test image paths to any two keyframes of one scene; a revisit pair should score higher than a
random pair.)

## Run
```bash
unzip -o smr_updates182.zip && python scripts/apply_all_patches.py          # five PASS lines
PYTHONPATH=src python -m pytest -q tests/test_guarantees.py                   # 11 pass
CUDA_VISIBLE_DEVICES=0 bash scripts/vpr_sweep.sh                              # ~10 min per arm and mode; 10 arms x 2 modes ~ 3 h
for m in anchored plain; do python scripts/gate_table.py --root outputs/ablate/vpr/$m --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --md outputs/vpr_$m.md; done
for arm in eigenplaces cosplace salad@4096 dino+eigenplaces; do echo == $arm; python scripts/paired_stats.py reports --a 'outputs/ablate/vpr/plain/7scenes/dino/*.json' --b "outputs/ablate/vpr/plain/7scenes/$arm/*.json" --row smr --metric ate_rmse --per-item | tail -9; done
# redo the two invalid reliability outputs
for v in plain_seed0 plain_seed1; do python scripts/closure_reliability.py --glob "outputs/ablate/localfrom/7scenes/$v/*.json" --gt-dir data/gt --gt-pattern '7scenes=7scenes_{stem}_seq01.npz' --est-dir outputs/ablate/localfrom/7scenes/$v --md outputs/rel_$v.md | tail -45; done
```

## What the result answers
Table 7 gains rows for each descriptor with recall@1/@5 of the proposals against GT revisits, accepted closures per
sequence, co-visibility precision, useful/harmful closures and ATE / AUC, in both modes. The claim the paper can then
make is one of: (a) DINO CLS is as good a cue as dedicated VPR descriptors for this task (likely on 7-Scenes, where the
extent gate and the backbone re-measurement do the work), (b) a VPR descriptor or a hybrid proposes better candidates
(higher recall@1) and that converts, or does not convert, into accuracy. Either is a fair answer to the reviewer; the
hybrid arms cover "hybrid retrieval strategies".

Predictions: P89 EigenPlaces/CosPlace recall@1 within ±0.1 of DINO on 7-Scenes and ATE within ±0.002 m in plain mode;
P90 SALAD has the highest recall@1 (≥ DINO + 0.05) but the same ATE within ±0.002 (geometry is the bottleneck);
P91 in anchored mode at least one weaker descriptor (cosplace or netvlad) produces a poisoned-window outlier
(ATE > 0.1 on some scene) that plain mode does not.
