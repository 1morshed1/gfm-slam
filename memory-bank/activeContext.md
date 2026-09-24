# activeContext — current phase, gate status, next action

**As of:** 2026-09-24 (ModelOpt/TE fused FP8 probe done)

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

**§6.5 fused-FP8 latency probe closed (negative for speedup).** Accuracy + memory remain the paper axes under D1.

### Real FP8 latency (GPU-2)

| path | large 4096³ | trunk-like | notes |
|------|-------------|------------|-------|
| `torch._scaled_mm` | **1.61×** | trunk stack **0.30×** | memory **2×** |
| ModelOpt real GEMM | **0.75×** | **0.12–0.17×** | cuda ext OK |
| torchao Float8Dynamic | **0.91×** | **0.13–0.14×** | secondary |
| Transformer Engine | — | — | **build failed** (no wheel) |

EXP: `results/20260924-benchfp8-modelopt-te.json`

### Track-A accuracy shortlist

| config | TUM mean vs FP16 |
|--------|------------------|
| FP8 e4m3 (fake) | −0.4% |
| W4 geom-protect K=7 | +7.1% |

## Next action

1. **Manuscript:** claim accuracy + memory Pareto; document fused-FP8 latency negative result honestly.
2. Skip further FP8 kernel hunting unless TE ships an sm_120 wheel.
3. Optional: K-budget sweep (paper blocker 3).

## Gate ledger

- [x] Phase 0–4 Track A + H3
- [x] Paper figures + manuscript skeleton
- [x] EuRoC V1_01 shortlist
- [x] §6.5 `_scaled_mm` microbench
- [x] §6.5 ModelOpt/TE fused FP8 probe (no latency win; TE unusable)
- [ ] Manuscript polish / K-sweep optional
