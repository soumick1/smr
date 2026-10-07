# smr_updates205 — Table 2 generator (7-Scenes + CO3Dv2 blocks, mean ± s.e.m.)

```bash
cd ~/smr && unzip -o ~/smr_updates205.zip && source .venv/bin/activate
python scripts/nvs_table.py --roots outputs/nvs_seq_v2 outputs/nvs_seq_v2_co3d --names 7-Scenes CO3Dv2 --out outputs/tab_nvs_seq.tex | tee outputs/tab_nvs_seq_stats.txt
```
Then the figures (v203 steps 1-2 first if not done yet: eval with --save-images 24, and images/Fig3_mod.png in ~/smr/images/):
```bash
python scripts/fig_nvs_seq.py --root outputs/nvs_seq_v2 --root-co3d outputs/nvs_seq_v2_co3d --caches cache/nvsseq --out outputs/figures/nvs_seq \
    --old-fig2 images/Fig3_mod.png --bb vggt_omega --fig2-seq office_seq01 --fig2-target 128
cd ~/smr && tar czf /tmp/nvs_final.tgz outputs/tab_nvs_seq.tex outputs/tab_nvs_seq_stats.txt outputs/figures/nvs_seq
```
Pull `/tmp/nvs_final.tgz` (scp) and upload it; the text changes follow with those numbers in them.
