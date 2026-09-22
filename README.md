# Edge-GFM-SLAM

Compressing the geometric-foundation-model backbone of feed-forward SLAM
(MASt3R-SLAM) under an accuracy–latency–energy–memory budget on Blackwell.

**Read first:** `CLAUDE.md` (hard rules), `memory-bank/activeContext.md` (locked
decisions + current phase), `gfm-edge-slam-plan.md` (full plan).

## Status

Phase 0 (bring-up) — not started. Rig-only (D1 = no deploy). See activeContext.

## Layout

```
CLAUDE.md               hard rules (GPU-2 pin, sm_120 verify, --no-deps, SDPA, scope)
gfm-edge-slam-plan.md   full research plan
memory-bank/            techContext (rig+tooling) · activeContext (phase/gates) · novelty (scoop watch)
scripts/                phase0_bringup.sh · explog.py · eval_ate/eval_pointmap/measure_energy · export_onnx/build_trt (deferred)
configs/                one YAML per quant config
results/                per-EXP JSON (versions, seed, config hash, metrics, device)
```

## Quickstart (on the rig)

```bash
export CUDA_VISIBLE_DEVICES=2
bash scripts/phase0_bringup.sh   # verify Blackwell -> clone --no-deps -> re-verify -> probe quant libs
```

Then fill the tooling matrix in `memory-bank/techContext.md` and reproduce the FP16
baseline (Phase 1) before any compression.
