# activeContext — current phase, gate status, next action

**As of:** 2026-09-22 (scaffold created)

## Locked decisions

| # | Decision | Value |
|---|----------|-------|
| D1 | Deploy target | **No deploy for now.** Rig-only (Blackwell). Jetson = future Phase-3 gate. |
| D2 | Deliverable / venue | Vision/geometry venue paper (3DV/WACV class), ~3 mo, method wrinkle required. |
| D3 | Primary SLAM | **MASt3R-SLAM** primary; VGGT-SLAM large arm (deferred). |
| D4 | Method ambition | PTQ-first → pruning second. Distillation dropped. |
| D5 | Real-time | Target, not hard req. **Pareto frontier = the contribution.** |

## Scope shift from D1

Track B (Jetson/TensorRT/on-device) is DEFERRED. Study is now: Blackwell desktop
accuracy–latency–energy–memory Pareto for GFM compression inside the full SLAM loop,
+ downstream-aware (SLAM-aware) bit allocation as the novelty wrinkle (§6.3).
Energy measured on rig via NVML. `export_onnx.py`/`build_trt.py` are stubs, not on critical path.

## Current phase

**Phase 0 — bring-up & tooling gates.** Not started.

## Next action

Run `scripts/phase0_bringup.sh` on the rig (GPU-1). It:
1. verifies Blackwell `(12,0)`,
2. clones MASt3R-SLAM with `--no-deps`,
3. re-verifies torch survived,
4. probes the quant toolchain (fills techContext matrix).

## Gate ledger

- [ ] Phase 0: GFM fwd on GPU-1 at FP16 speed; ≥2 quant toolchains confirmed; datasets load.
- [ ] Phase 1: baseline FP16 ATE within tolerance of published (TUM+EuRoC).
- [ ] Phase 2: ≥1 compressed config ≤10–15% ATE increase at reduced size.
- [ ] Phase 4: method beats best off-the-shelf PTQ at matched budget (or clean negative).
