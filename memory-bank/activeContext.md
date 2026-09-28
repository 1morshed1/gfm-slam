# activeContext — current phase, gate status, next action

**As of:** 2026-09-27 (rescue Stage A pilot running on GPU-1)

## Locked decisions

| # | Decision | Value |
|---|----------|-------|
| D0 | GPU pin | **GPU-1 only** (user, 2026-09-27). GPU-2 needs explicit user permission (shared with Triton + taaha celery). GPU-1 is shared with the user's own OpenVLA LIBERO eval. |
| D1 | Deploy | No deploy for now |
| D2 | Venue | Workshop / short first. Core claim currently unsupported; paper direction depends on the rescue gate. |
| D3 | SLAM | MASt3R-SLAM primary |
| D4 | Method | PTQ-first |
| D5 | Real-time | Memory axis only under D1 |

## Evidence so far (all committed)

- **Determinism:** MASt3R-SLAM `single_thread` gives seed std ≈ 0. Seeds do not measure noise.
- **TUM fr1 (9 seqs), paired per-sequence bootstrap:** geom-K7 vs mag-K7 −1.4% [−11.1, +8.1]; geom-K7 vs uniform −4.4% [−12.8, +5.3]; no K-sweep pair significant. **H3 not supported; U-shape dropped.** (`results/paired_bootstrap_tum.json`)
- **EuRoC (5 seqs) mean vs FP16:** FP8 +15.6%; W4 mag-K7 +66.9%; uniform +78.5%; geom-K7 (TUM-fit) +96.6%. **All W4 collapses.** Geom is worst on average but mixed per sequence (beats uniform on MH_01 and MH_02).
- **Noise floor** (`ptq.py` method=`noise`, multiplicative weight noise 1e-3, 4 draws): mean-ATE std 0.76% on TUM and EuRoC. Chaotic TUM seqs: desk2 6.5%, teddy 6.1%; room and xyz about 2%; desk 0.24%. FP8's EuRoC cost is real. (`results/step1_summary.json`)
- **FP8 fused latency:** no win (trunk 0.30× with `_scaled_mm`); weight memory 2× smaller.
- Manuscript is honest: status note in the abstract, §6.1/6.2/6.4/6.6 rewritten, new §6.7 noise floor.

## Current phase — rescue, Stage A pilot (user chose rescue over negative-result paper)

Pre-registered design (user-approved, 2026-09-27):
- **Measures (both):** quantize-one (`loi`: one unit W4, rest FP16; scored vs FP16) and protect-one (`loo`: trunk W4, one unit FP16; scored vs uniform W4). 49 trunk units; heads always FP16.
- **Stage A pilot (running since 2026-09-27 11:01 UTC, about 20 h):** 5 stable seqs: TUM desk, 360, floor; EuRoC MH_01, V1_01. desk `loi` reused from 2026-09-23. `scripts/run_profile_pilot.sh` (3 streams, GPU-1); outputs `results/profiles/profile_{loi,loo}_<seq>.txt`; log `results/profile_pilot.log`.
- **Gate (pre-declared):** `scripts/profile_gate.py` — PASS if, for either measure, mean cross-dataset (TUM × EuRoC) Spearman ≥ 0.3.
- **Stage B (only if gate passes):** profile the remaining 12 stable seqs (exclude desk2/teddy from fitting, still evaluate them); leave-one-sequence-out over all 20 seqs (9 TUM + 11 EuRoC); pooled score = mean relative delta over the fit seqs; K ∈ {7, 15, 25}; baselines magnitude at the same K plus uniform.
- **Success criterion:** at least one (measure, K) variant beats BOTH uniform and magnitude on held-out seqs, with a paired bootstrap CI excluding 0 at Bonferroni 99.2% (6 variants).
- **If gate fails:** write the negative-result workshop paper (within-TUM allocation gains not significant; W4 does not transfer under any allocation; FP8 looks free on TUM but costs 15.6% on EuRoC; seeds don't measure noise, so use sequence bootstrap plus a perturbation floor).

### Stage A RESULT (2026-09-28): GATE FAIL
- Mean cross-dataset Spearman: loi +0.011, loo +0.005 (threshold 0.3). Within TUM: −0.14 / −0.03; within EuRoC: −0.04 / −0.15. Top-7 overlap ≈ chance (expected 1/7).
- Effects are mostly above each seq's 1e-3 noise floor (18–47 of 49 units > 2× floor), yet idiosyncratic: `dec_blocks2.8` is rank 1 on desk (+26%) and MH_01 (+38%) but *helps* when quantized on floor (−11%), 360 and V1_01. Protecting only it under W4 on desk is −31% (worse than uniform) → strongly non-additive.
- Reading: closed-loop per-unit ATE sensitivity is not a stable property of a unit; the trajectory responds to W4-scale perturbations chaotically. `results/profile_gate.json`, `results/profiles/`.
- Ops: capping `OMP_NUM_THREADS=12` per stream gives identical ATE (verified on TUM floor, FP16 + uniform W4) and cut host load from ~200 to ~20 (48 cores). GPU-1 went from 1% to 88% utilization.

## Next action
1. Per pre-registration: **negative-result workshop paper**. Rewrite the manuscript around the findings (not significant within TUM; W4 does not transfer under any allocation; per-unit sensitivity profiles do not agree across sequences; FP8 cost on EuRoC; methodology: seeds ≠ noise, use sequence bootstrap plus a perturbation floor). Awaiting user go-ahead.

## Gate ledger
- [x] eval_ate.py implemented
- [x] Multi-seed H3 + K-sweep (deterministic; not significant)
- [x] EuRoC 5-seq transfer (fails) + matched W4 baselines (all collapse)
- [x] Perturbation noise floor
- [x] All 11 EuRoC seqs extracted (`data/euroc/`, symlinked into `ext/MASt3R-SLAM/datasets/euroc/`)
- [x] Rescue Stage A pilot + gate — **FAIL** (Stage B not run)
- [ ] Negative-result manuscript rewrite
- [ ] Venue reframe (after gate)
