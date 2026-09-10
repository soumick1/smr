# smr_updates172 — points 1 and 2 concluded; paper updated (iclr2027_overleaf_v172.zip)

## Point 2: 7-Scenes, full standard test split (18 trajectories, VGGT-Omega), valid run
Per-scene mean of trajectory ATE: raw 0.049 -> +SMR 0.042 -> +SMR+PGO 0.038 (seq-01: 0.045 -> 0.030 -> 0.029).
Per trajectory: +SMR improves 9 of 18 and worsens 9 (the mean falls because the large errors fall a lot: pumpkin
seq-01 0.160->0.077, office seq-07 0.144->0.105; several good chains lose 1-3 cm; PGO recovers most). Still the best
average of Table 2 under both protocols (+SMR ties SLAM-Former's 0.042 online). P69: half (mean +40 %, pumpkin's
share falls). Table 2 now shows BOTH protocols with a % TODO: confirm which one MASt3R-SLAM / VGGT-SLAM report and put
that block first. Sec. 5.3, abstract and conclusion softened accordingly ("best average" instead of "outperforms").

## Point 1 on 18 trajectories: a tie, and written as one
Flat key-value ranking vs scaffold: online 0.040 vs 0.042, with PGO 0.039 vs 0.038; scaffold better on 8 of 18
trajectories, worse on 9. The seq-01 redkitchen advantage did not generalise. App. L now says: the long-range gains
come from recognise -> re-measure -> verify -> revise around the frozen model and any content-addressable memory that
proposes the same candidates obtains them; the scaffold's contribution is measured where it acts (capacity/aliasing,
coherent-incoherent division, cross-session persistence, pose-consistent rendering recall). Dynamics-encoded addresses
are worse than the template on one real room (redkitchen 0.488): the synthetic equivalence does not transfer -- stated.
Table 8 (memory policy) now has 17 variants with the two index arms as its first block. Scoreboard 29.5 / 65.

## Fixed in code (v171): pass-cache file per sequence (was per scene -> seq-03 reused seq-01's passes).

## Next (points 3-6 of my list): Fig. 1 vs code (s in R^448, no 64-d index code; W_hs/W_sh definition), the four
over-strong phrases, small corrections (22 scans, DTU single pass, 9.3x, bank features, 19K/174), 2026 related work.
