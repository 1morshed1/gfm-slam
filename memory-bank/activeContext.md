# activeContext — current phase, gate status, next action

**As of:** 2026-09-22 (Phase 2 FP8 full TUM confirmed)

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

**Phase 2 — PTQ ladder (Track A).** Three configs clear the ≤15% mean-ATE gate on full TUM fr1. **FP8-e4m3 is best** (slightly *better* than FP16 mean).

### TUM fr1 mean (calib, subsample 2)

| config | mean ATE | vs our FP16 |
|--------|----------|-------------|
| **FP8 e4m3 trunk** | **0.0294 m** | **−0.4%** ✓ |
| FP16 | 0.0295 m | — |
| W8A8 trunk | 0.0315 m | +6.8% ✓ |
| W4 trunk (fake WO) | 0.0333 m | +12.8% ✓ |

FP8 worst: desk2 +8.6%, teddy +6.4%; several seqs improve (room −6%, xyz −7%).  
EXP: `results/20260922-phase2-tum-fp8-trunk.json`  
Runner: `scripts/run_tum_ptq_calib.sh fp8 trunk`

## Next action

1. **§6.3** sensitivity / mixed-precision heads (method wrinkle) — enough Pareto points to allocate against.
2. Or shortlist + prune ladder.
3. Stretch: W4A4 (expect head breakage per H1).

## Gate ledger

- [x] Phase 0–1
- [x] Phase 2: ≥1 compressed config ≤15% ATE (**FP8** −0.4%, **W8A8** +6.8%, **W4** +12.8%)
- [x] Phase 2: W8A8 + FP8 desk and full TUM
- [ ] Phase 2 shortlist / §6.3
- [ ] Phase 4: SLAM-aware bit allocation
