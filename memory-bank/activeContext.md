# activeContext — current phase, gate status, next action

**As of:** 2026-09-22 (Phase 2 W8A8 full TUM confirmed)

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

**Phase 2 — PTQ ladder (Track A).** Two configs clear the ≤15% mean-ATE gate on full TUM fr1: **W4 WO** (+12.8%) and **W8A8** (+6.8%).

### TUM fr1 mean (calib, subsample 2)

| config | mean ATE | vs our FP16 |
|--------|----------|-------------|
| FP16 | 0.0295 m | — |
| **W8A8 trunk** | **0.0315 m** | **+6.8%** ✓ |
| **W4 trunk** (fake WO) | **0.0333 m** | **+12.8%** ✓ |

W8A8 worst: teddy +23%, room +16%; desk2 only +8% (better than W4).  
EXPs: `results/20260922-phase2-tum-w8a8-trunk.json`, `…-tum-w4-trunk.json`  
Runner: `scripts/run_tum_ptq_calib.sh 8 trunk 8`

### fr1/desk ATE RMSE

| config | RMSE | vs FP16 |
|--------|------|---------|
| FP16 | 0.01613 m | — |
| W8 WO / W8A8 | ~0.016 m | ≈0% |
| W4 WO | 0.0135 m | −16% |

## Next action

1. **FP8** (Blackwell-native) on desk → full TUM if clean.
2. Or **§6.3** sensitivity / mixed-precision heads now that W8A8 + W4 give two Pareto points.
3. Stretch: W4A4 (expect head breakage per H1).

## Gate ledger

- [x] Phase 0–1
- [x] Phase 2: ≥1 compressed config ≤15% ATE (**W4** +12.8%, **W8A8** +6.8%)
- [x] Phase 2: W8A8 desk + full TUM
- [ ] Phase 2 continued: FP8 ladder + shortlist
- [ ] Phase 4: SLAM-aware bit allocation
