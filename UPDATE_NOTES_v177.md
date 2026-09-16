# smr_updates177 — the second training variant: the head trained WITH the read in the loop

## What the professor asked, translated to the protocol
The shipped heads were trained on RAW single-pass geometry (reads=1) and only evaluated with the read, so the existing
training figure IS the without-SMR training. The requested comparison is a second head per backbone trained on the
memory's geometry (the four-ordering read applied inside every training step), plotted against the raw-trained head at
EQUAL step budget, plus the 2x2 matrix {trained raw, trained +SMR} x {evaluated raw, evaluated +SMR}.

## Code
- train_nvs.py: `--reads 4` trains on the read; new `--val-reads` (default = training reads) so each head is validated on
  the geometry it trains on; new `--save-every N` keeps ckpt_step<N>.pt.
- scripts/nvs/train_with_smr.sh: trains the +SMR head (STEPS, default 10K; ~1.5 s/step for VGGT -> ~4 h), evaluates it on
  GSO and held-out with reads 1 and 4 (~2.5 h), prints the 2x2 summary next to the shipped head's numbers.
- fig_nvs_training.py --compare <dir>: (a) loss, (b) training PSNR, (c) held-out PSNR during training, (d) the 2x2 table.
  Curves cut at the shorter run so budgets match. A4 width, 600 dpi.
- scripts/nvs/val_curve.sh (optional): re-trains the raw head with checkpoints every 2K and evaluates each with reads 1/4
  -> "does the read's gain depend on how trained the head is" (a pure geometry effect should give a constant gap).

## Run (two free GPUs; DUSt3R on GPU 1 should be done -- check `screen -ls`)
```bash
cd ~/smr && unzip -o smr_updates177.zip
screen -dmS trsmr0 bash -c "$ENV && BB=vggt STEPS=10000 CUDA_VISIBLE_DEVICES=0 bash scripts/nvs/train_with_smr.sh > outputs/reports/train_with_smr_vggt.log 2>&1"
screen -dmS trsmr2 bash -c "$ENV && BB=vggt_omega STEPS=10000 CUDA_VISIBLE_DEVICES=2 bash scripts/nvs/train_with_smr.sh > outputs/reports/train_with_smr_vggt_omega.log 2>&1"
# ~6.5 h each. Then:
python scripts/fig_nvs_training.py --nvs outputs/nvs --backbones vggt --compare outputs/nvs/vggt_trainsmr --out outputs/figures/nvs_compare_vggt
python scripts/fig_nvs_training.py --nvs outputs/nvs --backbones vggt_omega --compare outputs/nvs/vggt_omega_trainsmr --out outputs/figures/nvs_compare_vggt_omega
tar czf outputs/train_with_smr.tgz outputs/nvs/*_trainsmr/{metrics,val}.csv outputs/nvs/*_trainsmr/*.jsonl outputs/nvs/*_trainsmr/summary_2x2.txt outputs/figures/nvs_compare_*
```
Caption note: the raw head's final-evaluation cells in (d) are the shipped 30K head unless val_curve.sh has produced a
10K checkpoint; the curves in (a)-(c) are at equal budget regardless.

## Predictions (recorded)
P73: at equal steps the +SMR-trained head's validation PSNR (on +SMR geometry) exceeds the raw head's (on raw) by about
the read's evaluation gain (+0.2-0.3 dB), i.e. training on the read does not compound: {train +SMR, eval +SMR} ~
{train raw, eval +SMR} within 0.15 dB. P74: {train +SMR, eval raw} is the worst cell (the head over-fits the smoother
consensus geometry and meets rawer inputs at test).
