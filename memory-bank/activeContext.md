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

## Next after runs
1. Fill tables with mean±std; drop U-shape if inside bars
2. Held-out sensitivity (Pri-3)
3. EuRoC §6.4 update

## Gate ledger
- [~] Multi-seed H3 + K-sweep — **running**
- [~] EuRoC multi-seq — downloading
- [x] eval_ate.py implemented
- [~] Venue reframe
