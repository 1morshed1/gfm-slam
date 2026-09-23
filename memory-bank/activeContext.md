# activeContext — current phase, gate status, next action

**As of:** 2026-09-23 (Phase 4 H3 ablation done)

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

**Phase 4 H3 claim supported.** Matched-K=7 protect: geometry-greedy beats weight-magnitude proxy on full TUM.

### Full TUM fr1 mean (W4 except protect)

| allocator | mean ATE | vs FP16 | vs uniform W4 |
|-----------|----------|---------|---------------|
| **Geometry greedy K=7** | **0.0316 m** | **+7.1%** | **−5.1%** |
| Magnitude L1 K=7 | 0.0328 m | +10.9% | −1.7% |
| Uniform W4 trunk | 0.0333 m | +12.8% | — |

Geometry wins by **3.6% abs mean ATE** vs magnitude at equal unit budget.  
EXPs: `…-tum-w4-sens-protect-d3.json`, `…-phase4-tum-w4-mag-protect-k7.json`  
Allocator: `scripts/allocate_bits.py`

### Track-A shortlist

| config | mean vs FP16 |
|--------|----------------|
| FP8 e4m3 | −0.4% |
| W8A8 | +6.8% |
| W4 geometry-protect K=7 | +7.1% |
| W4 magnitude-protect K=7 | +10.9% |
| W4 uniform | +12.8% |

## Next action

1. Commit/push magnitude TUM EXP + ates.
2. Paper tables / sensitivity heatmap; optional EuRoC on FP8 + W4-greedy.
3. Stretch: prune or W4A4.

## Gate ledger

- [x] Phase 0–2 PTQ ladder + §6.3 sensitivity
- [x] Phase 4: geometry allocator vs magnitude proxy (H3 ✓)
- [ ] Paper writeup / EuRoC shortlist
