# smr_updates173 — the nine mechanism figures (review item 10)

`scripts/fig_mechanism.py <cmd>`: seams · revisits · settling · funnel · flatbar · curves · timeline · nvshist · gpumem.
Each writes <out>.pdf (vector) + <out>.png (600 dpi). Inputs: pilot reports (rows, events, ceiling, peak_mem_gb),
`--save-est` trajectories, NVS jsonl. New in the stitcher: every window's raw candidate list (view, address score,
cosine, owner) is logged in the event (`candidates`) for the funnel.

Tested here: settling (real scaffold code, CPU), nvshist (real GSO jsonl), flatbar (real point-1 data + full split);
the others on a synthetic test bed (trajectories with drifting chains, pilot-shaped reports). Layout is final; numbers
come from your run.

## Run (GPU 0, ~45 min; most passes are cached)
```bash
cd ~/smr && unzip -o smr_updates173.zip && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/mech_data.sh     # saves est npz for 37 CO3D orbits, 7-Scenes seq-01, multi-session; renders all
ls outputs/reports | grep -i "len_\|_n200" | head        # find the Table-1 length-sweep reports, then:
python scripts/fig_mechanism.py curves --reports 'outputs/reports/<glob>' --out outputs/figures/mech_curves
python scripts/fig_mechanism.py gpumem --reports 'outputs/reports/<glob>' --out outputs/figures/mech_gpumem
tar czf outputs/mech_figures.tgz outputs/figures/mech_*
```
Send outputs/mech_figures.tgz (PNGs are enough to choose and caption). The `revisits` plot should also take the RE10K
reports once their glob is known (the low-revisit control on the same axes is the point of that figure).

## What the settling figure already shows (honest wording for App. C)
The loop DETECTS incoherent states (novelty 1-cos drops from 0.19 to 0.001) and partly corrects them (decoded error
0.20 -> 0.08 m after one iteration, then drifts to 0.24); coherent displacements are valid states elsewhere: novelty at
the floor, error unchanged at the displacement. Correction proper is the landmark re-anchor in both cases -- as the
code's docstring says ("the loop is a detector, not a large-amplitude corrector"). App. C's "transverse perturbations
contract" should read "are detected" (the residual contracts; the position only partly).
