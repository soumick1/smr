# smr_updates167 — the scaffold-key ablation (engineered cue / DINOv2 / backbone features) + paper corrected to the code

## What the code says (code > Overleaf), now reflected in Sec. 3 and App. cue
In the stitcher the scaffold is CONTENT-ADDRESSABLE: one appearance descriptor both keys it (bound at add time) and cues
it (propose = cue -> address -> stored views, then a cosine gate). `pilot_a --descriptor` selects that descriptor, and
its default for real backbones is DINOv2 CLS. So in Tables 1-2 the scaffold is keyed AND cued by DINOv2; the 448-d
engineered cue keys the rendering-recall memory (bind/rotate/read, relocalisation probe 97.5 %). The paper had said
"the 448-d cue keys the scaffold, DINO proposes revisits" -- corrected. The existing `desc_rgb` ablation row IS
"engineered cue as scaffold key+cue": 0.7 closures/sequence vs 4.1 -> the engineered cue cannot recognise a place
across viewpoints. App. cue also corrected: only the histogram block is L2-normalised (the whole vector once more);
per-block normalisation AND a depth channel were the two rejected enrichments (97.5 % -> 64.5 %).

## The missing arm: pooled backbone features as key+cue (`--descriptor feat`, new)
`feat_descriptors`: each keyframe passed ALONE through the VGGT aggregator (so the descriptor does not depend on the
window), last-layer patch tokens mean-pooled, L2, fixed random projection to 448, L2. VGGT-family wrappers only
(needs `_model.aggregator`; VGGT-Omega's wrapper may not expose it -- the run then fails loud; use BB=vggt).
Run the three arms on ONE backbone (VGGT) so only the descriptor changes:
```bash
cd ~/smr && unzip -o smr_updates167.zip && source .venv/bin/activate
ROOT=outputs/ablate_key SLAMBB=vggt BB=vggt VARIANTS="desc_dino desc_rgb desc_feat" CUDA_VISIBLE_DEVICES=0 bash scripts/ablate_all.sh   # ~30 min (new VGGT passes on 7-Scenes)
python scripts/ablate_table.py --root outputs/ablate_key --variants desc_dino desc_rgb desc_feat --label tab:ablation-key \
   --caption "\textbf{Which descriptor keys and cues the scaffold.} Same backbone (VGGT) and passes in every row; only the appearance descriptor bound into the scaffold and used to cue it changes. 7-Scenes: ATE RMSE (m) over the seven seq-01 sequences and accepted closures per sequence; CO3D: AUC@30 at $N{=}200$ over twelve orbits." \
   --out paper/tab_ablation_key.tex
tar czf outputs/ablation_key.tgz outputs/ablate_key paper/tab_ablation_key.tex
```
If `desc_feat` fails on the first scene, paste the tail of outputs/ablate_key/7scenes/desc_feat/chess.log (the feature
path was never executed before; the failure will be an attribute/shape at the aggregator call and is a one-line fix).

## Predictions
P66: desc_feat accepts fewer closures than DINO (<= 60 %) and its ATE is between rgb and dino: geometry-trained
     tokens are viewpoint-sensitive, not place-discriminative. P67: the three arms have IDENTICAL raw rows (same passes).
Then App. L gets Table 10 with the three arms and one paragraph explaining why each descriptor sits where it does.
