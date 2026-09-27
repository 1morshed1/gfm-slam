# activeContext — current phase, gate status, next action

**As of:** 2026-09-24 (review response: multi-seed + EuRoC in flight)

## Locked decisions

| # | Decision | Value |
|---|----------|-------|
| D0 | GPU pin | **GPU-2** |
| D1 | Deploy | No deploy for now |
| D2 | Venue | **Workshop / short first**; 3DV main only if multi-seed + transfer hold |
| D3 | SLAM | MASt3R-SLAM primary |
| D4 | Method | PTQ-first |
| D5 | Real-time | Memory axis only under D1 |

## Current phase

**Multi-seed + EuRoC shortlist DONE (2026-09-25). GPU-2 idle.** Results are sobering:

- **Seeds give ~0 std** (MASt3R-SLAM `single_thread` is deterministic): seed repeats do NOT measure noise. Need a different uncertainty estimate (per-sequence bootstrap / perturbation).
- **Per-seq paired bootstrap on TUM (9 seqs):** geom-K7 vs mag-K7 −1.4% [−11.1, +8.1] (5/9 wins); geom-K7 vs uniform −4.4% [−12.8, +5.3] (7/9); K9 vs K11 −5.8% [−12.5, +1.1]. **None significant.**
- **EuRoC 5-seq:** FP16 0.0237; FP8 **+15.6%**; W4 geom-K7 (TUM-fit) **+96.6%**. Transfer fails; uniform/mag not run on EuRoC yet.
- EXPs: `results/20260924-multiseed-*.json`, `results/20260924-euroc-shortlist-*.json`

### Manuscript
- §6.6 U-shape / over-protection **softened** pending error bars
- Honest workshop-tier framing

### Step 1 DONE (2026-09-27, GPU-1; GPU-2 needs explicit permission)
- EuRoC 5-seq mean vs FP16: geom-K7 +96.6%, uniform +78.5%, mag-K7 +66.9%. **All W4 collapses**; geom worst on average but mixed per-seq (2/5 beats uniform).
- Noise floor (`ptq.py` method=`noise`, rel 1e-3, 4 draws): mean-ATE std 0.76% on TUM and EuRoC. TUM chaotic seqs: desk2 6.5%, teddy 6.1%; desk (sensitivity fit seq) 0.24%.
- FP8 EuRoC +15.6% is real (≫ floor). Manuscript §6.4 + §6.7 updated. Summary: `results/step1_summary.json`.

## Next after runs
1. **Direction decision (user):** negative-result workshop paper (leaning) vs pooled/held-out sensitivity rescue (expensive, low prior).

## Gate ledger
- [~] Multi-seed H3 + K-sweep — **running**
- [~] EuRoC multi-seq — downloading
- [x] eval_ate.py implemented
- [~] Venue reframe
