# SMR — HANDOVER (written 2026-09-10; v175, addendum v177 at the end)

Read this first. Then `REVIEW_RESPONSE.md` (point-by-point status against the third-party review) and, only if a
detail is needed, the `UPDATE_NOTES_v1xx.md` files (one per shipped update, newest = v175). Transcripts of earlier
sessions are catalogued in `/mnt/transcripts/journal.txt`; the last full one is
`/mnt/transcripts/2026-09-07-05-27-25-smr-iclr2027-nvs-pipeline.txt`. The previous handover is `HANDOVER_old_v166.md`.

---

## 0. Project in three sentences
**SMR (Scaffold Memory for Reconstruction)** is an external, revisable memory wrapped around frozen 3D geometry
foundation models (VGGT, VGGT-Ω, π³, MASt3R, DUSt3R, Fast3R, StreamVGGT, STream3R). A grid-cell scaffold (Vector-HaSH
style) holds pose-addressed cards; appearance retrieves revisit hypotheses, the backbone itself re-measures them, a
two-site geometric consensus verifies, and accepted evidence revises the past (online correction, optional PGO); an
in-window "read" fuses repeated orderings of the same views into a consensus geometry. Target: **ICLR 2027**
(abstract 2026-09-18, paper 2026-09-25). Soumick (NTU, PI Mengmi Zhang) is the author; Claude is the engineer/co-writer.

## 1. Infrastructure
- **Server**: `ssh soumick@zlab-ws3`, repo `~/smr`, branch `memory_test`, venv `~/smr/.venv` (Python 3.13, torch 2.11+cu128),
  3× 46 GB GPUs (`CUDA_VISIBLE_DEVICES=0/1/2`). CUDA toolkit for JIT extensions (gsplat) is conda's 12.8. The env block
  `ENV='cd ~/smr && source .venv/bin/activate && export CUDA_HOME=$HOME/miniconda3 PATH=$PATH:$HOME/miniconda3/bin CPLUS_INCLUDE_PATH=$HOME/miniconda3/targets/x86_64-linux/include LD_LIBRARY_PATH=$HOME/miniconda3/targets/x86_64-linux/lib'`
  must be defined in every new shell before `screen -dmS name bash -c "$ENV && ..."` (a screen with an empty $ENV dies silently; happened three times).
- **GitHub**: `https://github.com/soumick1/smr.git` (branch `memory_test`). The user commits; Claude ships zips.
- **Data on server**: `~/data/dtu`, `~/data/eth3d`, `~/7scenes` (7-Scenes root — NOT `~/data/7scenes`), `data/gt/` (GT npz:
  `7scenes_<scene>_seq01.npz`; `7scenes_<scene>_seqNN.npz` for the test split, made 09-08; `7scenes_<scene>_s0106.npz`
  multi-session; `co3d_full/` 37 orbits ≈200 frames; `co3d/`=N10, `co3d_N50/`, `co3d_N100/`), NVS renders
  `~/data/nvs/{objaverse,gso}_c70` (70 % centre crop; 19,209 / 1,033 objects; `exclude.txt` per root), GLBs
  `~/data/nvs/objaverse_glb` (155 GB, deletable).
- **Results**: `outputs/` (pose, dense, NVS, ablations, figures, `cache/` pass caches); **`outputs2/`** holds the Table-2 SLAM
  reports (`outputs2/reports/pilotA_7scenes_<scene>_seq01_<bb>_s5_c32.json`).
- **Sandbox (this machine; persists within one chat, NOT across chats)**: repo copy `/home/claude/smr`, paper
  `/home/claude/iclr_paper/iclr2027/` (pdflatex compiles; 0 errors / 0 overfull at v172). No torch here: code is tested on
  numpy paths, synthetic worlds and `pilot_a --backbone synthetic --simulate-chunks`. A new chat starts with an empty sandbox:
  re-create it from the latest `smr_updatesNNN.zip` chain / GitHub and the latest `iclr2027_overleaf_vNNN.zip`.
- **Overleafs**: (a) the Claude-maintained copy, shipped as `iclr2027_overleaf_vNNN.zip` (latest **v172**); (b) the user's own
  Overleaf (uploaded as `ICLR_2027_Soumick__N_.zip`; v4 current) — different prose, same tables. The user merges by hand.

## 2. Update logic (how work is shipped)
- Every change = `smr_updatesNNN.zip` (code) and/or `iclr2027_overleaf_vNNN.zip` (paper) + `UPDATE_NOTES_vNNN.md` (what
  changed, exact commands to run, predictions). Numbering continues from **v175**.
- Laws: **code > Overleaf > memory** (the code is ground truth; where text disagrees with code, fix the text);
  **verify-before-package** (every patch asserts its landing; every script test-runs here on synthetic data);
  **predictions before runs** (scoreboard **29.5 / 65** through P69; P70–P72 open on the NVS heads);
  **never two runs of the same backbone + sequence concurrently** (pass cache); **honest reporting** (negative cells stay,
  "best average" not "outperforms", per-trajectory counts stated, protocol choices fixed a priori and said so).
- Workflow: the user runs commands in screens on the server and pastes logs / uploads tgz files; Claude analyses (paired,
  per-sequence, with CIs), updates the paper, ships the next zip. `REVIEW_RESPONSE.md` tracks the external review
  (DONE / IN PROGRESS / PLANNED / NOT DOING / USER).

## 3. What exists and what it says (final unless marked)

### 3.1 Camera pose vs sequence length (Table 1, `paper/tab_main.tex`) — 8 backbones, W=32/O=16, 2 sites
CO3D N=200 VGGT: raw 90.0 → +SMR 93.7 AUC@30 (headline). Native full-N "ceiling" is in the `len_*` reports where it fits.

### 3.2 Dense (DTU mm / ETH3D m) — evaluator v137 (truncated Chamfer, robust Sim(3) init)
DTU VGGT 0.441 → 0.396 (22 scans, better on every scan); VGGT-Ω 0.749 → 0.655; π³ / MASt3R / DUSt3R order-invariant (=);
Fast3R 4.40 → 4.08; STream3R 1.71 → 1.22; StreamVGGT 2.89 → 2.25. ETH3D VGGT 0.490 → 0.437; the read hurts Fast3R and the
causal models (incommensurable orderings) — stated, not removed.

### 3.3 SLAM, 7-Scenes (Table 2) — VGGT-Ω, stride 5, W=32/O=16, 2 sites
- seq-01 per scene: raw 0.045 → +SMR 0.030 → +PGO 0.029 (reports in `outputs2/`).
- **Full standard test split (18 trajectories; valid run 09-08 after the cache fix)**: 0.049 → 0.042 → 0.038. Per trajectory
  +SMR improves 9/18 and worsens 9/18 (large errors fall a lot: pumpkin seq-01 0.160→0.077, office seq-07 0.144→0.105; good
  chains lose 1–3 cm; PGO recovers most). Best average of the table under both protocols; +SMR alone ties SLAM-Former (0.042).
  Reports: `outputs/7scenes_test/<scene>_seqNN_vggt_omega_{template,flat}.json`.
- **OPEN (USER)**: which protocol MASt3R-SLAM / VGGT-SLAM report (seq-01 or full split). Table 2 shows both blocks with a
  `% TODO(protocol)`; the baselines' block goes first.
- Multi-session persistence (streaming backbones, `*_s0106`/`*_s0104` files; stride 10, chunk 16/8, revisit-gap 32):
  StreamVGGT 0.293 → 0.064, STream3R 0.337 → 0.086. Needs a proper write-up + timeline plot (`mech_data.sh` step 3).

### 3.4 Ablations (App. L, Tables 8–10) — done
- **Table 8** (memory policy, 17 variants incl. the index arms): two-site consensus is the gate (1 site: office 0.026→0.080,
  pumpkin 0.077→0.187, worse than raw); proposals-without-correction ≈ raw (0.042), PGO restores 0.029; no_robust = default
  (the gate rejected nothing); desc_rgb 0.7 closures vs 4.1; N_h 512–8192 and torus 16–64 within noise; W=16 raw 0.051 /
  +SMR 0.027 (7.7 closures); W=64 raw 0.023, memory adds nothing, one CO3D 64-frame pass fails (AUC 28).
- **Flat key–value vs scaffold (the review's central question): a TIE** — seq-01 0.033 vs 0.030; CO3D 93.3 vs 93.4 (scaffold
  better on 6/12); full split online 0.040 vs 0.042, PGO 0.039 vs 0.038 (scaffold better on 8/18). Dynamics-encoded
  addresses worse on one real room (redkitchen 0.488). App. L says: the stitching gains come from
  recognise→re-measure→verify→revise; the scaffold's contribution is measured where it acts (capacity/aliasing, coherent–
  incoherent division, cross-session persistence, rendering recall), not by the ranking of a closure proposal.
- **Table 10** (descriptor keying+cueing the scaffold, same VGGT passes): DINOv2 4.0 closures / 0.060 m / CO3D 93.4; pooled
  backbone features 2.9 / 0.062 / 91.2; engineered cue 0.7 / 0.061 / 91.4. **The paper now states the truth of the code: the
  stitcher's scaffold is keyed AND cued by the 384-d DINOv2 ViT-S/14 CLS token (224², ℓ2-normalised; no PCA, no padding);
  the 448-d engineered cue (192 colour thumbnail + 144 grey thumbnail + 48 histogram, zero-padded) keys only the
  rendering-recall memory (bind/rotate/read; 97.5 % probe).** The user's Overleaf v4 adopted a "pooled features + PCA → 448"
  cue description that no reported result used; the ablation says keep DINOv2; the 14-location audit for the alternative is
  in the 09-08 chat.
- **Table 9** (in-window read): reads 1/2/4/8 on 299 GSO objects (VGGT): +0.26/+0.27/**+1.07**; K=8 on all objects: VGGT
  +1.10, VGGT-Ω −0.04, π³ identity, STream3R −1.52 → Table 3 keeps the a-priori K=4 and reports the per-backbone optimum;
  no re-measure = same as with (known cameras); abstain −1.9 dB.

### 3.5 NVS (Table 3 + appendix `tab_nvs_full.tex`) — status only, see UPDATE_NOTES_v175
Heads (30K steps): VGGT, VGGT-Ω, π³, STream3R, Fast3R, StreamVGGT done. **DUSt3R** restarted at 5K steps (9.5 s/step),
running on GPU 1 (screen `nvs_gpu1`), done ≈ 09-11 morning. **MASt3R**: training died at the first validation with
`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` (its aligner path inside a
`no_grad`/autograd context) — UNFIXED; its `gso_reads*.jsonl` / `val_reads*.jsonl` hold only error records and must be deleted
before re-running. Results GSO raw→read: VGGT 20.71→20.95, VGGT-Ω 18.89→19.07, π³ 21.91=, STream3R 18.79→18.60,
Fast3R 14.07→14.20, StreamVGGT 18.30→18.59; held-out (174): 22.79→22.88, 20.84→20.89, 22.20=, 21.15→21.07, 16.99→16.87,
21.21→21.62 (StreamVGGT's read helps — opposite to STream3R; P71 half-miss). Fill: `scripts/nvs/fill_tab_nvs_full.py`.

### 3.6 Figures
- Qualitative: `fig_qualitative.py` (pairwise-error matrices, trajectories, per-frame error), `fig_recon.py`
  (reconstruction renders: reprojection colours, image frusta, `--sweep`, `--windows A B` two-window overlay; 2600×1800 px
  per panel, 600 dpi), `fig_nvs_strip.py`. The user composed the pumpkin SLAM figure and the NVS strip; captions drafted.
- Mechanism (review item 10): `scripts/fig_mechanism.py` — seams, revisits, settling, funnel, flatbar, curves, timeline,
  nvshist, gpumem — plus `scripts/mech_data.sh` (GPU 0, ~45 min; **not yet run**). Settling plot from the real scaffold code:
  the loop DETECTS incoherence (novelty −2 orders) but only partly corrects it (0.20→0.08 m, then drifts); coherent shifts
  are invisible → App. C "transverse perturbations contract" must become "are detected; correction is the landmark re-anchor".
  The stitcher now logs each window's candidate list (`events[].candidates`) for the funnel.
- `fig_nvs_training.py` (A4 width, 600 dpi; 6 panels) — done and shipped.

## 4. The external review (2026-09-08) — plan and status (details in REVIEW_RESPONSE.md)
1. Flat-memory baseline — **DONE** (tie; App. L rewritten). 2. Figure 1 vs code — figure fixed by the user; formal h↔s
definition (W_hs ∈ ℝ^{384×1024}, W_sh ∈ ℝ^{1024×384}, RLS λ=1e2; s = cue only; the bank index is the card's integer row) —
PLANNED as text. 3. 7-Scenes protocol — full-split run DONE; baseline protocol USER. 4. Over-strong phrases
("bit-identical", "without training", "backbone-agnostic", "train end-to-end", PGO monotone) — PLANNED (½ day text).
5. Small corrections (abstract 22 scans and 0.441→0.396; DTU "one pass, no cross-window graph"; 9.3× gating → tier-1 table
or drop; bank "optional task-specific content"; ≈19K / 174) — PLANNED. 6. 2026 related work (TALO, AMB3R, GlueMap, EAGLE;
GaussianGPT one sentence as complementary generation; the holistic-BA paper arXiv 2609.04026 is NOT relevant) — PLANNED;
running TALO NOT DOING. 7. **GlueMap** (github.com/colmap/gluemap) as an OFFLINE reference row in Tables 1–2 (not a
loop-closure competitor; same 200 keyframes, same alignment, label "offline global SfM") — recommended, ~1 day, not started.
8. Native full-N ceilings — in the reports, PLANNED for the length curves. 9. DTU naive 4-pass ensemble without alignment —
PLANNED (~1 h). 10. Nine mechanism plots — tooling done, data run pending. 11. Multi-session write-up + timeline — PLANNED.

## 5. Protocol decisions (do not re-derive)
DTU `--icp 50`, truncated-Chamfer fit; ETH3D `--fallback none` (corroborated) / windowed `--fallback median --abstain-rel 0`;
DUSt3R DTU `scene_graph=swin-5`; VGGT-Ω ETH3D 0.047 carries the contamination caveat §; NVS geometry: bounding SPHERE
r=0.5, GT cameras for placement, scale from Sim(3), consensus `abstain_rel=0` (holes are worse than the median), K=4
orderings (fixed before the K=8 result), `align=True`; 7-Scenes SLAM: stride 5, chunk 32/16, 2 sites; CO3D N=200: stride 1
on `co3d_full`; GSO protocol: 4 inputs at el 20°, az 0/90/180/270, 10 targets; renders 70 % centre-cropped (`*_c70`).

## 6. Gotchas learned the hard way
- Pass cache was per SCENE (`outputs/cache/{scene}_{bb}_s{stride}.npy`, keyed by keyframe position) → non-seq-01 7-Scenes
  runs silently reused seq-01's passes (reference AUC ≈ 0). Fixed v171 (per-sequence files); `run_7scenes_testsplit.sh` redoes
  reports whose 10-frame reference AUC < 20. Everything before v171 used unique scene names and is unaffected.
- 7-Scenes GT: `--convention c2w` is right (diagnose scored 94 AUC); the GT script reproduces seq-01 bit-for-bit.
- `screen bash -c` does not source `.bashrc`; define `$ENV` first. `$CUDA_HOME/bin` must be APPENDED to PATH.
- gsplat: conda toolkit + `CPLUS_INCLUDE_PATH=$HOME/miniconda3/targets/x86_64-linux/include`; extension built and cached.
- `torch.load(..., weights_only=False)` for our checkpoints.
- `points_suite` fused PLYs are xyz-only when the colour sampler fails (it did, silently); `fig_recon.py` colours by
  reprojection; `--per-ctx` writes per-window clouds (`ctx_XX.ply`).
- GSO-style evaluation of Objaverse renders needs the view-role fallback (all views are `train`).
- Descriptor widths: DINO 384 (stitcher key+cue), engineered 448 (rendering memory), `feat` arm random-projected to 448.
- MASt3R/DUSt3R wrappers accept `niter` (global-alignment iterations, default 300); DUSt3R's 4-view NVS step is 9.5 s.

## 7. Immediate next actions (in order)
1. DUSt3R finishes (GPU 1) → `python scripts/nvs/fill_tab_nvs_full.py`. Fix MASt3R's `requires_grad` crash in its wrapper
   path (the aligner needs `torch.enable_grad()` inside the trainer's/evaluator's `no_grad`; smoke with
   `train_nvs.py --backbone mast3r --smoke`), delete its jsonl, retrain 5K, evaluate.
2. `CUDA_VISIBLE_DEVICES=0 bash scripts/mech_data.sh` → nine mechanism figures; find the Table-1 length-report glob for
   `curves` / `gpumem` (`ls outputs/reports | grep -i "len_\|_n200"`).
3. Text pass: review items 4, 5, 6, 2 (definitions); App. C settling wording; the K=8 and StreamVGGT sentences in §5.4;
   the multi-session paragraph. Then merge into the user's Overleaf (he merges; give him exact old→new text).
4. USER: confirm the SLAM baselines' 7-Scenes protocol; decide DINOv2 vs feature-cue wording in his Overleaf.
5. Recommended: GlueMap offline rows; DTU naive-ensemble baseline.
6. Page budget only at the end (main text spills to page 10 at v172; user said not to worry yet). Deadlines: abstract 09-18, paper 09-25.

## 8. Key commands
```bash
# pose / SLAM run (report + saved trajectories; --index and --descriptor are the ablation arms)
python experiments/pilot_a.py --gt data/gt/7scenes_pumpkin_seq01.npz --backbone vggt_omega --keyframe-stride 5 --chunk 32 --overlap 16 --sites 2 \
   --index {template|flat|dynamics} --descriptor {dino|rgb|feat} --rows chained,smr,smr_pgo --json R.json --save-est E.npz
SEVEN=$HOME/7scenes bash scripts/run_7scenes_testsplit.sh ; python scripts/sevenscenes_table.py --root outputs/7scenes_test --bb vggt_omega
VARIANTS="..." DATASETS="7scenes co3d" ROOT=outputs/ablate bash scripts/ablate_all.sh ; python scripts/ablate_table.py [--variants ...]
bash scripts/mech_data.sh ; python scripts/fig_mechanism.py <seams|revisits|settling|funnel|flatbar|curves|timeline|nvshist|gpumem> ...
python scripts/fig_recon.py --est E.npz --json R.json --ply RAW.ply SMR.ply --rows chained smr [--sweep] [--zoom X Y H] [--windows A B] --out F
python scripts/nvs/fill_tab_nvs_full.py ; python scripts/fig_nvs_training.py --nvs outputs/nvs --out outputs/figures/nvs_training
```


---

## v177 addendum (2026-09-10, new chat) — agenda changed: trust and attribution before breadth
The agenda is now the eight-task plan (audit provenance and the Omega checkpoint; executable guarantee tests;
scaffold-vs-flat attribution; dense-read attribution; retrieval/verification reliability; accuracy-compute; controlled
multi-session; paired uncertainty). The 09-08 and 09-09 external reviews are superseded by it. `UPDATE_NOTES_v177.md`
is the entry point. Shipped: `tests/test_guarantees.py` (9 tests, pass), `scripts/{closure_reliability,paired_stats,
run_manifest,gate_table,compare_closures,multisession_eval}.py`, `scripts/{gate_sweep,seed_sweep}.sh`,
`src/smr/utils/provenance.py`, `scripts/apply_v177_patch.py` (pilot_a: provenance in every report + gate flags).
Findings that change the text (details in the notes §2): the site estimator is orientation-aware (App. A wrong);
`corr_none` is not an identity control (anchored pass replaces the window pass; rejected proposals change geometry);
`smooth_junctions` blends the previous window's overlap per frame at each accepted closure (App. B exception); the
operative gate is per-site consistency + a drift budget (App. D Eqs. 5-6 are the OFF-by-default `--site-agree` arm);
DINO gate is absolute 0.5; the stitching index runs at torus 32, N_h 2048, k 128, position-only (N_g 3,168), not the
48/1024/64/7,824 of App. C; addresses are write-time. Two Overleaf rows disagree with the recorded runs (Table 5 flat
row; Table 3 pi^3 row). VGGT-Omega: the original checkpoint is superseded by `vggt_omega_1b_512_reproduce.pt`
(image_resolution 416) per the authors; our wrapper uses the original -> Table 2 rerun planned once the hash is
confirmed. GitHub is at v167; v168-v176 must be pushed before v178 touches `anchored.py`. Scoreboard: P75-P79 open.
