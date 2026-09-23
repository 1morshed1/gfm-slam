# activeContext — current phase, gate status, next action

**As of:** 2026-09-23 (EuRoC shortlist spot-check done)

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

**Track A + H3 frozen; EuRoC V1_01_easy spot-check done** (full 11-seq still deferred).

### TUM fr1 mean shortlist

| config | vs FP16 |
|--------|---------|
| FP8 e4m3 | −0.4% |
| W4 geometry-protect K=7 | +7.1% (beats mag +10.9%) |
| W4 uniform | +12.8% |

### EuRoC V1_01_easy (vs FP16 0.0395 m)

| config | RMSE | vs FP16 |
|--------|------|---------|
| **FP8** | **0.0366 m** | **−7.4%** |
| W4 geom-greedy K=7 | 0.0467 m | +18.0% |

FP8 transfers cleanly; W4-protect is softer on this seq than on TUM mean.

## Next action

1. Commit EuRoC EXPs + runner fix.
2. Manuscript draft from `figures/` + shortlist tables.
3. Optional: more EuRoC seqs / prune.

## Gate ledger

- [x] Phase 0–4 Track A + H3
- [x] Paper figures
- [x] EuRoC V1_01 shortlist (FP8 + W4-greedy)
- [ ] Manuscript draft / fuller EuRoC
