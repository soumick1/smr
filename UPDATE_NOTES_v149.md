# smr_updates149 — GSO read rows: abstention bug + crash-proof evaluation; results so far

## What the first pass showed (paired objects, raw vs read, same head)
| backbone | raw PSNR/SSIM/LPIPS | read (n paired) | dPSNR |
|---|---|---|---|
| π³ | 21.91 / 0.844 / 0.132 | 21.91 / 0.844 / 0.132 (1033) | +0.00 — **P54 hit** (identity measured) |
| VGGT | 20.30 / 0.816 / 0.158 (on the 232 paired) | 20.54 / 0.819 / 0.148 (232) | **+0.25**, better on 68 % of objects, LPIPS −6 % |
| VGGT-Ω | 18.90 / 0.794 / 0.206 (816 paired) | 18.37 / 0.795 / 0.211 (816) | −0.53 and **−18 % Gaussians** |
| STream3R | 18.13 (1 paired) | 12.99 (1) | n=1, meaningless |
Full-GSO raw rows (n=1033): π³ 21.91, VGGT 20.71, VGGT-Ω 18.89, STream3R 18.79.
Scored: P48 hit (π³ raw ≥ VGGT raw); P52 miss (VGGT-Ω raw < VGGT raw on these renders); P47 miss (VGGT raw 20.7,
not 24-27: the head is LGM-class, see v148 discussion); P43 pending (on 232 objects it is +0.25, the low edge);
P53 pending.

## Two defects, one causal (fixed)
1. The NVS read called `consensus_fuse` with the DTU default `abstain_rel=0.10`: pixels whose four witnesses
   disagree grossly are set to NaN instead of taking the median. Right for DTU (a failing pass must not drag the
   cloud), wrong for rendering (a hole is worse than a compromise). VGGT-Ω's orderings disagree more than VGGT's,
   hence the 18 % Gaussian loss and the −0.53 dB. Now `abstain_rel=0` for NVS (as our windowed-fusion policy).
2. The read rows died at deterministic objects (VGGT-Ω #817, VGGT #233, STream3R #2) with no Python exception:
   when abstention emptied the Gaussian set, the evaluator launched gsplat with N=0 (the trainer guarded this, the
   evaluator did not) — a zero-element CUDA launch aborts the process. `render()` now returns background for empty or
   non-finite Gaussian sets, the evaluator records "no valid Gaussians" instead of crashing, and `eval_all.sh` is
   crash-proof: a process death marks the in-flight object in `<json>.crashed` and resumes (≤ 40 restarts per row).

## Re-run the read rows (raw rows and π³ are final; ~45 min with the three GPUs in parallel)
```bash
cd ~/smr && unzip -o smr_updates149.zip && source .venv/bin/activate
rm -f outputs/nvs/{vggt_omega,vggt,stream3r}/gso_reads4.jsonl outputs/nvs/*/gso_reads4.{inflight,crashed}
for i in 0 1 2; do bb=$(echo vggt_omega vggt stream3r | cut -d' ' -f$((i+1)))
  screen -dmS ev_$bb bash -c "cd ~/smr && source .venv/bin/activate && export CUDA_HOME=\$HOME/miniconda3 PATH=\$PATH:\$HOME/miniconda3/bin CPLUS_INCLUDE_PATH=\$HOME/miniconda3/targets/x86_64-linux/include LD_LIBRARY_PATH=\$HOME/miniconda3/targets/x86_64-linux/lib; CUDA_VISIBLE_DEVICES=$i READS=4 BACKBONES=$bb GSO=~/data/nvs/gso_c70 bash scripts/nvs/eval_all.sh > outputs/reports/nvs_eval_r4_$bb.log 2>&1"
done
# when the three screens are gone:
python experiments/eval_gso.py --summary outputs/nvs/*/gso_reads*.jsonl | tee outputs/nvs/summary.txt
cat outputs/nvs/*/gso_reads4.crashed 2>/dev/null | wc -l          # objects that still killed a process (expect 0)
tar czf outputs/nvs/nvs_results_v149.tgz outputs/nvs/*/gso_reads*.jsonl outputs/nvs/summary.txt outputs/nvs/*/gso_reads4.crashed
```
**Predictions:** P55: VGGT read gain on all 1033 objects between +0.15 and +0.5 dB; P56: VGGT-Ω read ≥ raw −0.1 dB
once abstention is off (the loss was the holes); P57: STream3R read ≤ raw (P53) by ≤ 1 dB; P58: zero crashes.
