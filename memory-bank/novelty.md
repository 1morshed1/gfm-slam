# novelty.md — living related-work / scoop watch (§1, risk R5)

Field moves monthly. Update as you read. The scoop risk is real.

## Our defensible wedge (post-D1)

With deploy deferred, the "on real edge silicon + energy-per-frame" wedge is weaker.
Lead instead with:

1. **Full-SLAM-loop compression study** on commodity Blackwell — ATE/RPE + pointmap vs
   latency/peak-mem/energy Pareto, back-end held fixed. Still not done by the neighbours below.
2. **Downstream-aware (SLAM-aware) mixed-precision bit allocation** (§6.3) — allocate bits by
   measured ATE/pointmap sensitivity, not weight-Hessian/perplexity proxy. **Primary novelty now.**
   MUST verify not already published before leaning on it.

## Direct / near neighbours (verify dates + claims before citing)

| Paper | arXiv | What it does | Why we're still distinct |
|-------|-------|--------------|--------------------------|
| VersaQ-3D | 2601.20317 | Calibration-free W4A4 quant for VGGT, transform-coding for outliers | Custom accelerator, reconstruction only — NOT full SLAM ATE |
| Co-Me | 2511.14751 | Confidence-guided token merging, VGGT/Pi3, 21.5× | Owns token-merging sub-corner — cite as baseline, don't reinvent |
| VGGT-X | 2509.25191 | Memory-efficient VGGT (bf16, drop intermediates) | Dense NVS, not compression-for-SLAM |
| Fast-FoundationStereo | 2512.11130 | KD + NAS + pruning for stereo FM | Different task; method template only |
| GFM distillation (lunar) | 2607.01851 | Domain-reconstruction distillation | Distillation dropped (D4) |

## Scoop watch (add dated entries as you find new work)

- (none yet — start logging here)
