# activeContext — current phase, gate status, next action

**As of:** 2026-09-24 (K-budget sweep DONE; manuscript §6.6 filled)

## Locked decisions

| # | Decision | Value |
|---|----------|-------|
| D0 | GPU pin | **GPU-2** |
| D1 | Deploy | No deploy for now |
| D2 | Venue | Vision/geometry paper ~3 mo |
| D3 | SLAM | MASt3R-SLAM primary |
| D4 | Method | PTQ-first → pruning |
| D5 | Real-time | Target; Pareto = contribution |

## Current phase

Manuscript working draft + **K-sweep complete**. Best protect budget **K=9 (+5.4% vs FP16)**; K=11 regresses (+13.5%). H3 remains at matched K=7.

### K-budget sweep (greedy W4)

| K | vs FP16 |
|--:|--------:|
| 3 | +14.2% |
| 5 | +8.0% |
| 7 | +7.1% |
| **9** | **+5.4%** |
| 11 | +13.5% |

Figure: `figures/k_budget_pareto.{png,pdf}`

## Next action

1. Venue: **3DV primary** (locked in manuscript header)
2. Optional: more EuRoC / polish passes
3. Stretch: packed W4 latency if kernels appear

## Gate ledger

- [x] Phase 0–4 Track A + H3
- [x] Manuscript working draft (+ K=9 / U-shape polish)
- [x] EuRoC V1_01 shortlist
- [x] §6.5 FP8 microbench + ModelOpt/TE probe
- [x] K-budget sweep (blocker 3) — best K=9
- [x] §4.4 asymmetric arm — cut
