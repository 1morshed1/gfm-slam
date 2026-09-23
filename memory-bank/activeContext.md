# activeContext — current phase, gate status, next action

**As of:** 2026-09-23 (H3 committed; paper figures generated)

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

**Track A + H3 frozen for writeup.** Paper artifacts in `figures/`.

### Full TUM fr1 shortlist

| config | mean vs FP16 |
|--------|----------------|
| FP8 e4m3 | −0.4% |
| W8A8 | +6.8% |
| **W4 geometry-protect K=7** | **+7.1%** (beats mag +10.9%, uniform +12.8%) |

### Artifacts

- `figures/sensitivity_heatmap.{png,pdf}`
- `figures/shortlist_pareto_bars.{png,pdf}`
- `figures/shortlist_table.md`
- Generator: `scripts/plot_paper_figures.py`

## Next action

1. Commit/push figures.
2. Optional: EuRoC on FP8 + W4-greedy.
3. Draft paper sections from shortlist + H3 table.

## Gate ledger

- [x] Phase 0–2 PTQ ladder + §6.3 sensitivity
- [x] Phase 4: geometry vs magnitude (H3 ✓) — commit `091825e`
- [x] Paper figures (heatmap + shortlist)
- [ ] EuRoC / manuscript draft
