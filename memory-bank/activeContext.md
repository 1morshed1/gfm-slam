# activeContext — current phase, gate status, next action

**As of:** 2026-09-24 (review response: multi-seed + EuRoC in flight)

## Locked decisions

| # | Decision | Value |
|---|----------|-------|
| D0 | GPU pin | **GPU-2** |
| D1 | Deploy | No deploy for now |
| D2 | Venue | **Workshop / short first**; 3DV main only if multi-seed + transfer hold |
| D3 | SLAM | MASt3R-SLAM primary |
| D4 | Method | PTQ-first |
| D5 | Real-time | Memory axis only under D1 |

## Current phase

Responding to review: **Pri-1 multi-seed** (H3 + K-sweep) on GPU-2; **EuRoC** download via HF continues on CPU.

### In flight
- `scripts/run_multiseed_h3_ksweep.sh` — seeds 0,1,2; configs FP16/FP8/W4u/mag-K7/geom-K{3,5,7,9,11}
- `scripts/eval_ate.py` — evo wrapper (no longer a stub)
- EuRoC HF room zips → then `run_euroc_shortlist.sh`

### Manuscript
- §6.6 U-shape / over-protection **softened** pending error bars
- Honest workshop-tier framing

## Next after runs
1. Fill tables with mean±std; drop U-shape if inside bars
2. Held-out sensitivity (Pri-3)
3. EuRoC §6.4 update

## Gate ledger
- [~] Multi-seed H3 + K-sweep — **running**
- [~] EuRoC multi-seq — downloading
- [x] eval_ate.py implemented
- [~] Venue reframe
