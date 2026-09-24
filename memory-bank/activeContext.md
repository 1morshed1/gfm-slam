# activeContext — current phase, gate status, next action

**As of:** 2026-09-24 (§6.5 real FP8 microbench run on GPU-2)

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

**§6.5 first real-FP8 latency point (done).** `torch._scaled_mm` e4m3 on Blackwell.

### Real FP8 vs FP16 (rig GPU-2)

| mode | result |
|------|--------|
| Kernel 4096³ | **1.61×** faster (396 vs 246 TFLOP/s) |
| Kernel mean (5 shapes) | **0.87×** (skinny/small GEMMs lose) |
| Trunk 192 Linears | **0.30×** (fp8 22.5 ms vs fp16 6.9 ms) |
| Weight storage | **2.00×** smaller (944 → 472 MB) |

Interpretation: native FP8 tensor cores help **large square** GEMMs; naive per-Linear `_scaled_mm` + quant overhead **hurts** the ViT-L inventory (many mid-size mats). Memory win is clean. Accuracy Track-A FP8 still stands (−0.4% TUM).  
EXPs: `results/20260924-benchfp8-kernel.json`, `…-trunk.json`

### Track-A accuracy shortlist (unchanged)

| config | TUM mean vs FP16 |
|--------|------------------|
| FP8 e4m3 (fake) | −0.4% |
| W4 geom-protect K=7 | +7.1% |
| EuRoC V1_01 FP8 | −7.4% |

## Next action

1. **Paper honesty:** §6.5 = memory win + shape-dependent compute; don’t claim end-to-end SLAM speedup from this path yet.
2. Options for real speedup: fused/kernelized FP8 Linear (TE / ModelOpt), or only replace large mats; or report memory-Pareto as primary deploy metric under D1.
3. Continue manuscript with scoop-watch wedge + these latency caveats.

## Gate ledger

- [x] Phase 0–4 Track A + H3
- [x] Paper figures + manuscript skeleton
- [x] EuRoC V1_01 shortlist
- [x] §6.5 real FP8 microbench on GPU-2 (mixed: big GEMM ↑, trunk stack ↓)
- [ ] Manuscript polish / optional TE-fused FP8
