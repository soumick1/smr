# smr_updates200 — acceptance-rule sweep (choose the rule, then fix the manuscript text)

Paper configuration throughout (v198); only the acceptance test varies. 21 arms on the controlled blocks (7 x 7-Scenes
seq-01 with VGGT-Omega, 12 CO3D orbits with VGGT): the agreement test alone at rotation {5, 10, 20, off} deg x position
{0.25, 0.5, 1.0, off} E_k (16), the legacy pair-validity gates + drift budget combined with the agreement test at four
points (4), and the legacy rule (1). The first arm per scene computes the new anchored passes; every later arm reuses them.

```bash
cd ~/smr && unzip -o ~/smr_updates200.zip
CUDA_VISIBLE_DEVICES=0 SCENES7="chess fire heads" DATASETS=7scenes bash scripts/acceptance_sweep.sh > outputs/acc_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 SCENES7="office pumpkin redkitchen stairs" DATASETS=7scenes bash scripts/acceptance_sweep.sh > outputs/acc_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 DATASETS=co3d bash scripts/acceptance_sweep.sh > outputs/acc_gpu2.log 2>&1 &
wait; tail -n 1 outputs/acc_gpu*.log
python scripts/acceptance_pick.py --root outputs/ablate/acceptance | tee outputs/acceptance_pick.txt
```
About 40 min (GPU 2 is the long one: 12 orbits x 21 arms). `acceptance_pick.py` prints every arm with 7-Scenes ATE
(online / PGO), CO3D AUC (online / PGO), closures per sequence, co-visibility precision and useful / harmful counts,
ranked by the sum of the two accuracy ranks with harmful closures as tie-break. Send `outputs/acceptance_pick.txt`.
Decision rule agreed in advance: adopt the best-ranked arm unless a simpler arm is within 1 mm (7-Scenes) and 0.3 AUC
(CO3D) of it; the manuscript then states that rule with its numbers, and the NVS sweep reruns under it.
