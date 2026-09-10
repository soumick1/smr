# smr_updates175 — NVS heads for the remaining four backbones (DUSt3R, MASt3R, Fast3R, StreamVGGT)

Same protocol as the first four (train_nvs.py / eval_gso.py unchanged): 1 object per step, 4 inputs -> 4 targets, AdamW
2e-4, 500-step warm-up + cosine; evaluation 4 inputs / 10 targets, reads 1 and 4, GSO 1,033 + held-out 174.
Step budgets: Fast3R, StreamVGGT 30K (0.3-0.5 s/step); DUSt3R, MASt3R 10K (global alignment inside every pass,
1.5-3 s/step; the first four heads were within 0.3 dB of final by 10K). Reads=4 also for the order-invariant ones (measured).

## 1. Smoke test first (GPU 0, ~10 min) -- the geometry path has never run these wrappers on 4-view object renders
```bash
cd ~/smr && unzip -o smr_updates175.zip && source .venv/bin/activate
SMOKE_ONLY=1 bash scripts/nvs/train_rest.sh
```
Expect for each: "step 20: loss ... gaussians ..." and "val@20". If one prints an Error, paste it -- I fix the wrapper
path while the others train (launch the others with the queue below anyway; edit the `queue` lines to drop the broken one).

## 2. Launch (3 screens: GPU0 fast3r -> streamvggt; GPU1 dust3r; GPU2 mast3r; each: train -> GSO -> held-out)
```bash
bash scripts/nvs/train_rest.sh
tail -n 2 outputs/nvs/{fast3r,streamvggt,dust3r,mast3r}/train.log
```
Wall time ~12-14 h (MASt3R the longest: 10K x ~3 s + ~4 h of reads=4 evaluation).

## 3. When `screen -ls` shows no nvs_gpu*:
```bash
python scripts/nvs/fill_tab_nvs_full.py           # paper/tab_nvs_full.tex: all 8 backbones, both column groups
python experiments/eval_gso.py --summary outputs/nvs/{fast3r,streamvggt,dust3r,mast3r}/*_reads*.jsonl | tee outputs/nvs/summary_rest.txt
tar czf outputs/nvs_rest.tgz outputs/nvs/{fast3r,streamvggt,dust3r,mast3r}/{gso,val}_reads*.jsonl outputs/nvs/{fast3r,streamvggt,dust3r,mast3r}/{metrics,val}.csv outputs/nvs/{fast3r,streamvggt,dust3r,mast3r}/train.log paper/tab_nvs_full.tex outputs/nvs/summary_rest.txt
```
Send nvs_rest.tgz. Table 3 (main) keeps its four rows; the eight-backbone table goes to the appendix (tab_nvs_full.tex),
and fig_nvs_training.py extends to eight curves with `--backbones vggt_omega vggt pi3 stream3r fast3r streamvggt dust3r mast3r`.

## Predictions (recorded)
P70: raw GSO PSNR ordering follows point-map quality on DTU/ETH3D: pi3 > VGGT > MASt3R ~ DUSt3R > StreamVGGT ~ Fast3R;
     Fast3R lowest (worst DTU 4.4 mm). P71: read gain: DUSt3R/MASt3R identity within 0.05 dB (order-invariant alignment),
     Fast3R negative (non-commensurable orderings, as on ETH3D), StreamVGGT negative. P72: the 10K-step heads land within
     0.3 dB of where 30K would (validation curves flatten by 10K for every head so far).
