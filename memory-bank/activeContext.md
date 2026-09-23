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

1. **Run real FP8 speedup bench on rig GPU-2** (`bash scripts/run_bench_fp8_rig.sh kernel`
   then `trunk`) → first real §6.5 Pareto point. Track-A quant is fake-quant = no real
   speedup; `bench_fp8.py` uses `torch._scaled_mm` (native e4m3), isolated from MASt3R mp.
2. Manuscript draft (`paper/manuscript.md`) — skeleton up; novelty reframed post scoop-watch.
3. Optional: more EuRoC seqs / prune / real W4 kernel (sm_120 packing still unsolved).

## Gate ledger

- [x] Phase 0–4 Track A + H3
- [x] Paper figures
- [x] EuRoC V1_01 shortlist (FP8 + W4-greedy)
- [ ] Manuscript draft / fuller EuRoC
