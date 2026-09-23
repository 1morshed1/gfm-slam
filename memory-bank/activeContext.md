# activeContext — current phase, gate status, next action

**As of:** 2026-09-23 (§6.3 δ>3% protect ablation done)

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

**§6.3 allocation ablations complete (Track A).** Geometry-aware protect improves on uniform W4; stronger threshold helps further.

### Full TUM fr1 mean

| config | mean ATE | vs FP16 | vs uniform W4 |
|--------|----------|---------|---------------|
| FP8 e4m3 | 0.0294 m | −0.4% | — |
| FP16 | 0.0295 m | — | — |
| W8A8 | 0.0315 m | +6.8% | — |
| **W4 sens-protect δ>3%** | **0.0316 m** | **+7.1%** ✓ | **−5.1%** |
| W4 sens-protect (dec2.8 only) | 0.0324 m | +9.6% ✓ | −2.8% |
| W4 uniform trunk | 0.0333 m | +12.8% ✓ | — |

δ>3% also cuts teddy pain (+28% vs +46% for single-block).  
EXP: `results/20260923-phase2-tum-w4-sens-protect-d3.json`

## Next action

1. **Commit/push** §6.3 batch (sensitivity + both protect ablations).
2. Phase 4: formal greedy/ILP allocator; or shortlist FP8 + W4-d3 for paper tables.
3. Stretch: W4A4 / prune.

## Gate ledger

- [x] Phase 0–1
- [x] Phase 2 uniform PTQ ladder
- [x] Phase 2 §6.3 sensitivity + protect ablations (δ>5% and δ>3%; both beat uniform W4)
- [ ] Phase 4: fuller allocator + writeup
