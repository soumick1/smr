# smr_updates203 — figures for the sequence-NVS results (Fig. 2B, Fig. S2, Fig. S3)

`scripts/fig_nvs_seq.py` (unpack at the repo root). Inputs: `outputs/nvs_seq_v2/<bb>/{raw,smr,smr_pgo,gt}.jsonl`,
`train_{raw,smr}.log`, `images/<seq>/*.png`; the test caches for the input-keyframe strip; the current Fig. 2 PNG for
the composite.

```bash
cd ~/smr && unzip -o ~/smr_updates203.zip && source .venv/bin/activate
# 1. save every held-out target of VGGT-Omega (same numbers, more renders to choose from; ~3 min)
CUDA_VISIBLE_DEVICES=0 python experiments/nvs_seq_decoder.py eval --caches 'cache/nvsseq/vggt_omega/test/*' \
    --ckpt-raw ckpt/nvsseq/vggt_omega_raw.pt --ckpt-smr ckpt/nvsseq/vggt_omega_smr.pt --out outputs/nvs_seq_v2/vggt_omega --save-images 24 2>&1 | grep -vi warn | tail -6
# 2. the current Fig. 2 (from the arXiv source) for the composite:  scp -P 22 /mnt/c/MyFiles/.../images/Fig3_mod.png soumick@10.97.144.63:~/smr/images/
# 3. figures
python scripts/fig_nvs_seq.py --root outputs/nvs_seq_v2 --caches cache/nvsseq --out outputs/figures/nvs_seq --old-fig2 images/Fig3_mod.png \
    --bb vggt_omega --fig2-seq office_seq01 --fig2-target 128
```
Outputs in `outputs/figures/nvs_seq/`: `fig2B_nvs.png/.pdf`, `Fig2_new.png`, `figS2_nvs_backbones.png/.pdf`,
`figS2_nvs_scenes.png/.pdf`, `figS3_nvs_training.png/.pdf`. The script prints the zoom box it chose and the PSNR of the
chosen target; try other targets with `--fig2-target 64|192|...` (after step 1 any multiple of 8 works) or fix the box
with `--fig2-box x,y,w,h` (pixels of the 384 x 288 render). `figS2_nvs_scenes` picks, per sequence, the saved revisited
target with the largest +SMoRe gain (say so in the caption, or pass nothing and it falls back to the largest gain).
Pull: `scp -P 22 -r soumick@10.97.144.63:~/smr/outputs/figures/nvs_seq /mnt/c/MyFiles/smr_results/`
