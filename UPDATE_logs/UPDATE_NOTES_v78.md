# smr_updates78 — closure revocation (streaming) + relocalisation into memory (T2 / T3)

    scp -P 22 /mnt/c/MyFiles/smr_updates/smr_updates78.zip soumick@10.97.144.63:~/
    ssh -p 22 soumick@10.97.144.63 "cd ~/smr && unzip -o ~/smr_updates78.zip && rm ~/smr_updates78.zip && git status --short"
    cd ~/smr && source .venv/bin/activate && python -m pytest -q      # expect 159 passed, 23 skipped
    git add -A && git commit -m "v78: closure revocation; relocalisation experiment (T2/T3)" && git push

--------------------------------------------------------------------
## 1. Closure revocation (src/smr/stitch/anchored.py, `revoke=True`)
--------------------------------------------------------------------
Every accepted closure is provisional: a snapshot of all stored poses is
taken before it is applied.  If the NEXT chunk brings stronger evidence
(>= 2 verified sites) that disagrees with the chain by more than twice the
budget, and the previous closure rested on a single site, the previous
closure is revoked: poses restored, its loop edge dropped, the chain
re-fitted, and the new closure judged against the restored chain.  Events
carry `revoked_at`; results carry `n_revoked`.  Streaming counterpart of
the batch outlier rejection.  It cannot undo a closure that every later
chunk agrees with (Omega-floor: four wrong closures, mutually consistent);
that case is reported as measured.

    for bb in vggt_omega vggt pi3; do for s in floor room; do
      python experiments/pilot_a.py --gt data/gt/tum_fr1_$s.npz --backbone $bb --keyframe-stride 3 \
          --chunk 32 --overlap 16 --sites 2 --revisit-gap 32 --rows chained,smr,smr_pgo \
          --json outputs/reports/pilotA_tum_fr1_${s}_${bb}_s3_c32_v78.json 2>&1 | grep -E "^(chained|smr)" | sed "s/^/$s $bb /"
    done; done

Prediction: VGGT room 0.309 -> <= 0.217 (chained) with n_revoked >= 1;
Omega floor unchanged (0.66); everything else within 0.005 of v77.

--------------------------------------------------------------------
## 2. Relocalisation into memory (T2), and under corruption (T3)
--------------------------------------------------------------------
New: src/smr/stitch/relocalise.py, experiments/relocalise.py, tests (+4).
Mapping sessions are stitched into a memory (session-aware chunks, SMR);
each keyframe of the held-out session(s) is localised from ONE image.
Rows, all placing the query with the identical pass machinery:
  lastk   no memory: query + the last K=15 map keyframes (recency is all
          a backbone has)
  plain   descriptor nearest neighbour proposes the sites (a database)
  smr     memory address cue proposes the sites
  oracle  sites by ground-truth proximity (diagnostic: separates retrieval
          from placement failure)
One small pass per query (query + 2-4 site frames); sites verified inside
the pass; two sites must agree (10 deg / 0.3 m) else the query is marked
ambiguous and placed by the tighter site.  The MAP is aligned to GT by one
Sim(3) (SLAM-protocol alignment); query poses are never aligned, so errors
are metric.  Metrics: median cm / deg over all queries (misses = inf, as in
the 7-Scenes relocalisation tables), recall @5cm/5deg, @10/10, @25/25,
failure rate, s/query.  Corruptions (ImageNet-C style, applied to the
query before descriptor AND pass): gauss:s, occlude:f, blur:px, dark:f.

Sandbox (rendered two-lap world, lap 1 = map, lap 2 = queries): lastk
recall@5cm/5deg 0.17, plain 0.97, smr 1.00, oracle 0.90; median 2.2 cm.

    # official 7-Scenes split for chess: train 1,2,4,6 / test 3,5 (all six sessions are on disk)
    python experiments/relocalise.py --gt data/gt/7scenes_chess_s0106.npz --backbone vggt_omega \
        --map-seqs 1,2,4,6 --query-seqs 3,5 --keyframe-stride 10 --query-stride 10 \
        --rows lastk,plain,smr,oracle --corrupt none \
        --json outputs/reports/reloc_chess_vggt_omega.json 2>&1 | tee outputs/reports/reloc_chess_vggt_omega.log
    # T3: the same with corruptions (queries are cached per corruption)
    python experiments/relocalise.py --gt data/gt/7scenes_chess_s0106.npz --backbone vggt_omega \
        --map-seqs 1,2,4,6 --query-seqs 3,5 --rows lastk,plain,smr \
        --corrupt none,gauss:0.05,gauss:0.10,gauss:0.15,occlude:0.15,occlude:0.30,blur:3,blur:6,dark:0.5,dark:0.7 \
        --json outputs/reports/reloc_chess_vggt_omega_corrupt.json 2>&1 | tee outputs/reports/reloc_chess_vggt_omega_corrupt.log
    # then vggt and pi3 on the same split; then office (train 1,3,4,5 of the sessions on disk / test 2,6)

Budget: mapping ~8 min (400 keyframes), then ~1 s per query per row for
VGGT-class backbones: 200 queries x 4 rows ~ 15 min; the corruption sweep
x 10 specs x 3 rows ~ 1.5 h.

Predictions, recorded before the run:
* clean split, VGGT-Omega: lastk recall@5cm/5deg < 0.3 (the last 15 map
  frames cover one corner of the room); plain and smr within 0.05 of each
  other (the address is not what buys retrieval on clean queries), both
  0.6-0.8 @5cm/5deg and > 0.9 @10/10; oracle within 0.1 of smr (placement,
  not retrieval, is the floor); median 3-6 cm / 1-2 deg -- the
  scene-agnostic band (Reloc3r / Marepo), not the scene-trained band
  (ACE / DSAC* ~97%), which the paper will say.
* corruption: recall falls with severity for every row; smr degrades no
  faster than plain, and if the address cue completes a corrupted
  descriptor at all it degrades slower -- the one place the memory can
  show an advantage in retrieval.  If smr == plain at every severity, the
  paper says the RLS cue adds nothing over cosine here.



cd ~/smr && source .venv/bin/activate      # apply smr_updates79.zip
git add -A && git commit -m "v79: relocalisation map from the batch solve" && git push

# 1. chess again with the batch map (cached passes; minutes) — prediction: map ATE ~0.04, smr median 3–4 cm, @5cm/5° >= 0.5
python experiments/relocalise.py --gt data/gt/7scenes_chess_s0106.npz --backbone vggt_omega \
    --map-seqs 1,2,4,6 --query-seqs 3,5 --rows lastk,plain,smr,oracle --corrupt none \
    --json outputs/reports/reloc_chess_vggt_omega_v79.json 2>&1 | tee outputs/reports/reloc_chess_vggt_omega_v79.log

# 2. chess, the other two strong backbones (map ~8 min each, queries ~15 min)
for bb in vggt pi3; do
  python experiments/relocalise.py --gt data/gt/7scenes_chess_s0106.npz --backbone $bb \
      --map-seqs 1,2,4,6 --query-seqs 3,5 --rows lastk,plain,smr,oracle --corrupt none \
      --json outputs/reports/reloc_chess_${bb}_v79.json 2>&1 | tee outputs/reports/reloc_chess_${bb}_v79.log
done

# 3. office: larger scene, official-split sessions on disk (train 1,3,4,5 / test 2,6) — where recency should collapse
for bb in vggt_omega vggt pi3; do
  python experiments/relocalise.py --gt data/gt/7scenes_office_s0106.npz --backbone $bb \
      --map-seqs 1,3,4,5 --query-seqs 2,6 --rows lastk,plain,smr,oracle --corrupt none \
      --json outputs/reports/reloc_office_${bb}_v79.json 2>&1 | tee outputs/reports/reloc_office_${bb}_v79.log
done

# 4. T3 on office with Ω (the corruption sweep where the rows can separate), ~1.5 h
python experiments/relocalise.py --gt data/gt/7scenes_office_s0106.npz --backbone vggt_omega \
    --map-seqs 1,3,4,5 --query-seqs 2,6 --rows lastk,plain,smr \
    --corrupt none,gauss:0.10,gauss:0.20,occlude:0.30,occlude:0.50,blur:6,blur:10,dark:0.7,dark:0.85 \
    --json outputs/reports/reloc_office_vggt_omega_corrupt.json 2>&1 | tee outputs/reports/reloc_office_vggt_omega_corrupt.log