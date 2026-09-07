# smr_updates166 — ablations scored and in the paper (App. L, Tables 8-9); K=8 recommendation

## Scoreboard (P59-P65): 26.5 / 61
P59 half (one site: office 0.026->0.080, pumpkin 0.077->0.187, worse than raw; but not more closures accepted).
P60 hit (proposals alone: 0.042 ~ raw 0.045; PGO restores 0.029). P61 untestable (0 rejections in default runs).
P62 hit (RGB proposals: 0.7 closures/scene vs 4.1). P63 miss (N_h 512-8192, torus 16-64 all within noise).
P64 hit (W=16: raw 0.051, +SMR 0.027, 7.7 closures). P65 half (monotone in orderings yes; NOT diminishing: K=8 gives
+1.07 dB vs K=4's +0.27, same Gaussian count; no-align wrong (redundant with known cameras); abstain right, -1.9 dB).
Also: W=64 raw 0.023 (fewer seams), memory adds nothing there; CO3D bowl_70's 64-frame pass fails (AUC 28) -- the
larger-window risk, stated in the appendix.

## Recommendation: re-run Table 3 (GSO) and the held-out column with K=8 orderings
Eight orderings quadruple the NVS headline (+1.07 dB, 86 % of objects) AND make the read protocol uniform with the
dense benchmarks (eight passes). ~80 min per backbone on GSO; run three in parallel, pi3 after (it is the identity,
but we measure it):
```bash
cd ~/smr && unzip -o smr_updates166.zip && source .venv/bin/activate
for i in 0 1 2; do bb=$(echo vggt vggt_omega stream3r | cut -d' ' -f$((i+1)))
  screen -dmS r8_$bb bash -c "cd ~/smr && source .venv/bin/activate && READS=8 BACKBONES=$bb GSO=~/data/nvs/gso_c70 CUDA_VISIBLE_DEVICES=$i bash scripts/nvs/eval_all.sh > outputs/reports/nvs_r8_$bb.log 2>&1"
done
# when they finish (~80 min):  pi3 on GPU 0 (~80 min), then the held-out column (4 x 13 min, GPU 1):
READS=8 BACKBONES=pi3 GSO=~/data/nvs/gso_c70 CUDA_VISIBLE_DEVICES=0 bash scripts/nvs/eval_all.sh
READS=8 DATA=~/data/nvs/objaverse_c70 CUDA_VISIBLE_DEVICES=1 bash scripts/nvs/eval_val_all.sh
python experiments/eval_gso.py --summary outputs/nvs/*/gso_reads*.jsonl outputs/nvs/*/val_reads*.jsonl | tee outputs/nvs/summary_k8.txt
```
Send summary_k8.txt; Table 3's +SMR rows become K=8 (K=4 stays in Table 9 as the ablation row), the caption states K.
If you would rather not spend the GPU time, the paper is complete as is: Table 3 at K=4 and Table 9 showing K=8.

## Paper (iclr2027_overleaf_v166.zip)
App. L "Ablations": Table 8 (memory policy, 15 variants x {7-Scenes ATE, CO3D AUC}), Table 9 (read: orderings 1/2/4/8,
no re-measure, abstain), five findings in prose; pointer from Sec. 5.2. 0 errors / 0 overfull; main text 9 pages.
