# SMR: Complete Methodological Walkthrough
### From pixels to mental rotation — every component, every shape, every equation
*(Shapes given for the `compact` configuration, with `full` in parentheses.
`configs/*.yaml` is canonical. K = number of input images.)*

---

## 0. One-paragraph overview

A frozen geometry backbone (VGGT or π³) turns K photographs into per-view
depth, pose, and confidence — the *eyes*. A bank of continuous-attractor
networks (3 head-direction rings + M=3 grid-cell torus modules) holds a
*pose state* as physical activity bumps — the *internal world coordinate*.
A Vector-HaSH-style associative memory binds each view's sensory content to
the scaffold state occupied at its pose — the *hippocampus*. Queries are
answered by *physically moving the bumps* (imagination), reading out which
stored views the arrived-at state indexes (recall), splatting their surfels
to the query pose (rendering), and letting a small trained UNet fill the
disocclusions (completion). Only that UNet is trained; nothing else has a
single learned parameter of ours.

```
images (K,3,H,W)
   │  frozen backbone (VGGT / π³)
   ▼
depth (K,H',W')  poses (K,4,4)  K (3,3)  conf (K,H',W')
   │  lift + normalise (median depth := 1)
   ▼
surfels {P_i (M_i,3), C_i (M_i,3)}   descriptors s_i (448,)
   │                                      │
   │   ┌──────── SCAFFOLD ────────┐       │
   └──►│ 3 HD rings  (3×128)      │◄──────┘   bind: place → g_i → h_i
        │ M=3 torus modules (3×32²)│           store (h_i ↔ s_i, payload_i)
        │ + z-axis fields          │
        └───────────┬─────────────┘
                    │ drive_to (imagination, speed-limited)
                    ▼
        decoded pose T̂ (4,4)  →  recall k=2 payloads → splat (H',W',3)+(H',W')
                                             │
                                             ▼
                              completion UNet (only trained part)
                                             │
                                             ▼
                                   composite image + depth
```

---

## 1. Stage-by-stage

### Stage 1 — Frozen backbone (perception)

**In:** K RGB images, any resolution.
**VGGT:** resized to max-side 518 → typically (K, 3, 518, 392) portrait /
(K, 3, 392, 518) landscape. **π³:** uniform (K, 3, 378, 672)-class sizing.

**Out (per scene):**
| tensor | shape | notes |
|---|---|---|
| depth | (K, H', W') | metric up to scale |
| poses | (K, 4, 4) | camera-to-world. π³ emits c2w directly; VGGT emits w2c and we invert — a per-adapter convention that must be exact |
| intrinsics | (K, 3, 3) | pinhole |
| conf | (K, H', W') | per-pixel confidence |
| rgb | (K, H', W', 3) | the resized images |

**Normalisation:** one similarity gauge per scene, median depth := 1.
Every threshold downstream is thereby scale-free (the ledger's "scene
units").

Nothing here is ours or trained by us. Portability was *measured*: the
same downstream machinery, untouched, passes the identical harness on both
backbones, eight LLFF scenes, synthetic data, and a phone video.

### Stage 2 — Surfel lift

Per view i: keep pixels with conf ≥ 80th percentile (VGGT) or sigmoid conf
> 0.1 + depth-edge filter (π³, their reference defaults), stride 2;
back-project through K⁻¹ at the pixel's depth:

  X_cam = d(u,v) · K⁻¹ (u, v, 1)ᵀ

**Out:** P_i (M_i, 3) points in *camera-i frame*, C_i (M_i, 3) colours,
with T_i retained so the surfels can be re-expressed in any frame later.
M_i ≈ 30–45k per view. Surfels are *payload*, not memory content — memory
stores the key, the payload rides along.

### Stage 3 — View descriptor

s_i ∈ R⁴⁴⁸ = colour thumbnail (8×8×3 = 192) ⊕ grey structure map
(12×12 = 144) ⊕ intensity/gradient histograms (48+64 reserve).
No per-part renormalisation: the parts' natural magnitudes act as implicit
weights (re-weighting and a depth channel were both *rejected by
measurement* — retrieval fell 97.5% → 64.5%). This is the sensory code the
associative memory cues on. (Planned upgrade: pooled backbone features,
pending verification of the feature-tap.)

### Stage 4 — The scaffold (pose as physical state)

**Head direction: 3 rings** (yaw, pitch, roll; ZYX Euler; pitch guard
excludes the gimbal region). Each ring: N = 128 (256) neurons on [0, 2π),
rate dynamics

  τ ṙ = −r + [ J ∗ r + I ]₊ ,  J(Δθ) = J₀ + J₁ cos Δθ

ReLU nonlinearity, Euler step Δt = 0.05τ (stiffness-bound-derived). The
bump width θ_c = 2.0 rad fixes J₁ through the closed form F₁(θ_c) = 1/J₁
(measured within 2%); J₀ = −κ cos θ_c / F₀ with κ = 1.35 (smallest
inhibition margin with reliable ignition; DoG kernel rejected by
measurement — peak Fourier gain 0.47 < 1, no pattern formation under
ReLU).
**State:** r ∈ R¹²⁸ per ring. **Decode:** population vector,
θ̂ = atan2(Σ r sinθ, Σ r cosθ).
**Move:** velocity input shifts the bump at commanded rate up to a
*measured* speed limit ω_max = 0.16 rad/τ — the physical origin of the
reaction-time law (T8, R² = 1.0).

**Position: M = 3 grid modules**, periods λ = (2.4, 3.2, 4.0) scene units
(ratio 3:4:5, smallest coprime triple ⇒ combined unambiguous range
λ₁·lcm ≫ scene extent, Chinese-Remainder structure; T7 shows the
characteristic CRT fragility signature, jump ≈ 30, confirming the code is
genuinely residue-based). Each module holds the phase vector
φ_m = (x mod λ_m)/λ_m · 2π, realised as toroidal attractor sheets (32²
(64²) neurons for the horizontal components plus a matched periodic field
for height, per-axis gains calibrated separately). Same dynamics family as
the rings; bump moved by conjunctive velocity cells with offset ℓ = 2
(Burak–Fiete convention).

**The gate — the design's crux.** Translation is sensed in the *body*
frame but position phases live in the *world* frame. Before driving the
modules, body velocity is rotated by **R(ŷaw, p̂itch, r̂oll)** — the
angles *decoded from the rings' own bumps*, not the commanded ones. This
is the unique choice making the phase increments exact differentials
(path-invariant integration; Prop. non-abelian). Measured: gated
closed-loop error 0.079 vs ungated 0.735 — a 9.3× gap that *is* the
theorem made visible.

**Decode position:** per-module phase popvec → residues; residues combined
by least squares on the unwrapped lattice (with the propose-verify
retrieval below when cued) → x̂ ∈ R³. Precision floor ≈ λ_min·0.5/N_torus:
predicted 0.0375, measured 0.034 (compact); full config measured 0.0053 on
real imagery — the floor is a *configuration knob with a predictive
model*, not a tuned constant.

**Scaffold state vector g:** concatenation of all field activities,
g ∈ R^{N_g} (N_g = Σ ring sizes + Σ module sizes; 14,880 at plan scale in
the ledger). This vector — the *pattern of bumps* — is the memory key.

### Stage 5 — Binding (writing memory), one view at a time

Formation (once per scene, before any view): drive the scaffold over a
pose lattice covering extent 1.2 at spacing 0.4 (0.3) — spacing ≤ λ_min/6
gives ≥ 6 phase samples per period per axis, the covering condition T14
identifies. Collect (g_j, h_j) pairs and form the return map
**W_hg = ridge regression** solving g ≈ W_hg h (Vector-HaSH's Hebbian sum
works for sparse binary patterns; for continuous g the ridge/pseudoinverse
is the faithful analogue — the Hebbian form measured at chance). Now every
lattice code is a fixed point of the g→h→g loop, and — the combinatorial
magic, T14 = 1.0 — **every unvisited combination of per-module phases is
already an attractor too**, from only a ~20% covering sample. The scaffold
is a *prebuilt, content-free address space*.

Then, per view i:
1. **place_pose(T_i):** rings clamped-and-ignited at Euler(R_i), modules
   at φ(t_i); ~150 settle steps; bumps form at the commanded state.
2. **g_i = state()** ∈ R^{N_g}.
3. **h_i = k-WTA(W_gh g_i)** ∈ {0,1}^{N_h}: fixed *random* projection
   W_gh (never learned), k winners take all. k = 64 of N_h = 1024
   (410 / 8192) — the ~5% Vector-HaSH operating point. h_i is the sparse
   *address* of pose i.
4. **Sensory heteroassociation, both directions, by RLS** (recursive
   least squares — online, one-shot, exact while patterns are linearly
   independent; capacity bounded by rank N_h, and past it T10 shows the
   pinv-family degrades *gracefully*, no Hopfield cliff):
   for the map W (either h→s or s→h), with forgetting λ = 1:

     e = target − W · cue
     K = P·cue / (1 + cueᵀ P cue)
     W ← W + e Kᵀ ,  P ← P − K (cueᵀ P)

   P is the running inverse Gram (448×448 or N_h×N_h-sparse side).
   Storage cost per view: one rank-1 update. This is why binding is
   **streaming**: views arrive one at a time, sessions can resume.
5. **Payload store:** content[i] = {P_i, C_i, T_i, s_i} indexed by h_i.
6. **Novelty gate:** residual of s_i orthogonal to the memory's current
   span, gated at τ = max(0.2, 3× the floor probed at bind time from
   lattice midpoints). T11: coherent (in-span) perturbations are
   *invisible* (< 0.02) while incoherent ones are flagged at separation
   1.4×10⁶ — selective detectability, not a generic error meter.

**What "associative memory storage" is, in one sentence for the prof:**
pose is stored as *which* sparse address the attractor lattice occupies;
appearance is stored as a *rank-1 correction* to a linear map between that
address and the view's descriptor; geometry (surfels) is a payload keyed
by the address; and the address space itself was never learned — it is the
combinatorics of the grid code.

### Stage 6 — Queries (reading memory)

**(a) Imagination / mental rotation — `drive_to(T*)`.** Command ring and
module velocities toward T* at the speed limits; the bumps *physically
traverse* every intermediate pose; steps ∝ distance (the RT law). Two
path programs: `direct` (constant world velocity — a chord through
never-visited positions: imagination ≠ replay) and `orbit` (waypointed
through experienced poses — the presentational sweep). At any instant the
decoded T̂ is a valid, renderable pose. Long unanchored integration drifts
at the *independently measured* intrinsic rate 0.0022 units/τ (the 2.0°
residual of the 804-step orbit path is this number times the elapsed
time — mechanism, not mystery); landmarks/re-anchoring exist to absorb it.

**(b) Recall + splat.** At state g: h_q = k-WTA(W_gh g); nearest k_recall
= 2 stored addresses by code overlap; their payloads re-expressed in the
query frame, X_q = T_q⁻¹ T_j X, then z-buffered through the pinhole:
**out** rgb (H', W', 3), depth (H', W'), mask (H', W') — mask false =
disocclusion. Cost: GFLOPs; no backbone forward.

**(c) Relocalisation from a corrupted glimpse.** Cue s̃ = s + unit-norm
noise (SNR stated in *signal units*: cos ≈ 0.71 to clean — the honest
convention; the old per-dimension convention was SNR 0.06, at which
neighbour discrimination is information-theoretically impossible).
Two-stage: **propose** top-5 addresses via the s→h map, **verify** by
descriptor distance, place at the winner's pose. Exact recall up to
σ* = 0.6; V3 on real scenes: 37–40/40 within 2× median view spacing
(the criterion that makes sense among near-duplicate frames; on orbits it
reduces to exact-ID, 31/31).

**(d) Novelty:** the same residual gate, as a query: "have I seen this?"
AUROC 1.0 (T13).

**(e) Completion — the only trained component.** Input (B, 5, h, w):
splat RGB (3) ⊕ splat depth/4 (1) ⊕ mask (1). UNet base 48, 4 downs,
GroupNorm+SiLU, 10.6M params; heads: RGB via sigmoid, depth via softplus.
Train on random 256² crops (hole-biased sampling); eval full-frame with
reflect-pad to multiples of 16. **Hard composite:** output = splat where
mask, prediction where hole — known pixels *bypass the network*, visible
in every log as the bit-constant known-region metric. Loss: hole-weighted
L1 (rgb) + 0.5·L1 (depth, valid pixels) + 0.05·VGG16-relu3_3 perceptual
*on the composite* (gradients therefore reach hole pixels only).
Selection: hole-PSNR.

---

## 2. K = N images vs K = 1 image

**Same machinery, different coverage.** Nothing branches on K.

| | K = N (e.g. 32) | K = 1 |
|---|---|---|
| backbone | one forward, (N,…) | one forward, (1,…) — both backbones accept K=1 natively |
| bind | N place-and-store steps (streamed) | 1 step |
| recall at novel pose | k=2 neighbours ≈ adjacent views; coverage ~0.85–0.95 | the single view reprojected; every parallax-revealed surface is a hole; coverage ~0.45–0.6 |
| burden | mostly memory/splat | mostly completion head |

This is *why* the harvest generates context ∈ {1, 4, all} per holdout:
one head is trained across the entire coverage spectrum, so "single-image
mental rotation" and "dense-capture mental rotation" are endpoints of one
continuum, not two systems. Novelty per pair is now *recorded*
(novelty_rot/pos = pose gap to nearest bound view): small-baseline NVS
(~2.5° LLFF, ~11–12° orbits) at call/c04; c01's difficulty is coverage,
not angle. The flagship's honest sentence: the 115° orbit drive is
long-range *pose traversal*; rendering at arrival is always *local recall
(+ completion)*, never 115° single-reference extrapolation.

**Streaming / multi-session (the capability bare backbones lack):** RLS
binding is online; a returning session relocalises from a glimpse (query
c), then continues binding into the same address space. Backbones need
all frames in one attention pass and retain nothing between calls.

---

## 3. Out-of-domain generalisation — the three-tier argument

**Tier 1 — the scaffold and memory are training-free.** There is nothing
for them to be out-of-domain *of*. The address space is combinatorial
(T14: unvisited phase combinations are already attractors); the dynamics
are parameter-derived or calibrated on themselves; the measured invariance
— the same decode floor on synthetic scenes, eight LLFF captures, a phone
video, and two backbones — is this tier's empirical signature ("the
scaffold machinery does not know or care where its geometry came from").

**Tier 2 — geometry generality is inherited, not learned.** The backbone
is frozen and web-scale-pretrained; our adapters only enforce its
conventions (c2w vs w2c, confidence semantics). Swapping VGGT ↔ π³
changes numbers by mask-policy margins, not behaviour.

**Tier 3 — the single domain-carrying component is the completion head,**
and it is handled the standard way: established corpora (CO3D now;
ScanNet++/RE10K for scene statistics), scene-disjoint splits, and
*measured* zero-shot transfer — CO3D-trained, LLFF-evaluated: hole-PSNR
16.1 dB vs 17.7 for an LLFF-trained reference (within 1.6 dB), with the
depth-transfer gap (0.32 vs 0.18) quantifying exactly why scene-style
corpora are queued. Multi-backbone pairs (vggt+π³ in one corpus) train
the head to complete *anyone's* geometry.

**View-level OOD (novel poses)** is Tier 1+2 by construction: the pose
manifold is the whole group, imagination traverses it continuously, and
each rendered frame is local recall completed by Tier 3.

---

## 4. Numbers to have ready (all measured, all in the doc)

- Ladder 25/25 at server scale; width 2.0% of closed form; gate gap 9.3×;
  ω_max 0.16; RT law R² = 1.0; CRT jump 30.
- Decode floor: predicted 0.0375 → measured 0.034 (compact); 0.0053
  full-scale on real imagery (6.5×).
- Retrieval: σ* = 0.6 signal-units; V3 40/40 (room), 31/31 exact (orbit);
  novelty AUROC 1.0; selectivity 1.4×10⁶.
- Flagship: 32-frame phone orbit; drive 413 steps (chord) / 804 (orbit
  path, RT-in-path-form); arrival 0.35°/0.004; recalled render 0.0067
  rel-depth @ 83% coverage vs the actual photo; orbit-path 2.0° residual
  = intrinsic drift × time, mechanistically closed.
- Drift robustness: VGGT reference-anchored drift to ~29° at the
  antipode; readout 5.1% under it; π³ drift flat 1–4°.
- Completion: shakedown 17.7 dB (overfit at ep30, diagnosed); CO3D subset
  29.2 dB, 95 epochs no overfit, perceptual live; zero-shot LLFF 16.1 dB;
  pass-through invariant 0.0389/0.0468/0.0747 (constant within each run).