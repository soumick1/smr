# smr_updates199 — sequence NVS from the memory map (stage 1, frozen decoders)

Requires v198. `experiments/nvs_sequence.py` and `scripts/nvs_sequence_all.sh`.

## Smoke test of the aligned core on real data (one sequence each, ~2 min)
```bash
cd ~/smr && unzip -o ~/smr_updates199.zip && git status --short
CUDA_VISIBLE_DEVICES=0 python experiments/pilot_a.py --gt data/gt/7scenes_pumpkin_seq01.npz --backbone vggt_omega --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 \
    --rows chained,smr,smr_pgo --json /tmp/paper_pumpkin.json 2>&1 | grep -E "scene unit|^chained |^smr |^smr_pgo |Traceback|Error"
CUDA_VISIBLE_DEVICES=0 python experiments/pilot_a.py --gt $(ls data/gt/co3d_full/*.npz | head -1) --backbone vggt --keyframe-stride 1 --max-frames 200 --chunk 32 --overlap 16 --sites 2 \
    --rows chained,smr,smr_pgo --json /tmp/paper_co3d.json 2>&1 | grep -E "scene unit|^chained |^smr |^smr_pgo |Traceback|Error"
```
Expected: a "scene unit: median depth of window 0 = ..." line and three rows; the loops column tells whether the agreement
test accepts closures on real data (on the heavy-noise synthetic world it accepts none).

## Sequence NVS
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/nvs_sequence_all.sh            # 7 x 7-Scenes (VGGT-Omega) + 12 CO3D (VGGT), ~3-4 min per sequence
cat outputs/nvs_seq/7scenes_pumpkin/summary.json
```
Per sequence: every 8th keyframe is a held-out target; inputs run through the 32/16 windows with the paper configuration
(`--legacy` for the old rules); points of each input frame come from the pass the stitcher used and are placed by the
per-frame placement of each method (raw chain, +SMR online, +SMR+PGO, oracle = GT placement of the local windows);
the frozen decoder renders the targets; PSNR / SSIM / LPIPS per target with its window index (`<out>/<method>.jsonl`),
means and per-window PSNR in `summary.json`, a few renders under `images/` with `--save-images`.
Options: `--footprint pixel` (default: geometric per-pixel footprint, the same for every method) or `head` (the decoder's
own scales, trained at object scale); `--stride 2` pixel subsampling (about 16k Gaussians per frame); `--holdout 8`;
`--methods raw smr`; `--K fx,fy,cx,cy` for the GT intrinsics (7-Scenes 585,585,320,240; CO3D from the npz if stored,
otherwise the backbone's prediction).
Send back the sweep's final block (means per method per dataset) and `outputs/nvs_seq/7scenes_pumpkin/summary.json`.
Expected: raw and smr identical on sequences without an accepted closure; smr ahead of raw where closures exist, most on
targets in late windows; oracle well above both (the zero-shot decoder sets a low absolute level; the retrained
sequence decoder is stage 2).
