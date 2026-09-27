# techContext — rig + tooling

## Rig: vm-130-131

- GPUs 0–2: RTX PRO 6000 Blackwell Server Edition, `sm_120`, ~96 GiB VRAM each, driver 580.
- **Current pin: GPU-1 only** (shared with the user's OpenVLA eval). GPU-2 needs explicit user permission (shared with a root Triton server and taaha's celery pid 3178; never touch those). GPU-0 is someone else's.
- CUDA 12.8, torch 2.11.0+cu128. Capability must read `(12, 0)`.
- SDPA-only (no flash-attn).
- Disk: datasets under `/office/dev_workspace/morshed/projects/gfm-slam/data`.
- Conda env: **`edgeslam`** (dedicated; do not use/touch openvla).
- Build host compiler: system **g++ 13.3** (`CC`/`CXX`/`CUDAHOSTCXX=/usr/bin/g++`). Conda gxx 14 is rejected by CUDA 12.8.
- `CUDA_HOME=/office/dev_workspace/morshed/miniconda3/envs/edgeslam` (conda `cuda-nvcc=12.8.93`). Source `scripts/env_gpu2.sh` before builds.
- For curope / torch libs: put torch `lib/` on `LD_LIBRARY_PATH`.

## Edge deploy target

- **DEFERRED (D1 = no deploy for now).**

## Tooling-support matrix (Phase 0 probe, 2026-09-22, GPU-2)

| Tool | Purpose | sm_120 builds? | real speedup? | version | notes |
|------|---------|----------------|---------------|---------|-------|
| bitsandbytes | INT8, NF4 | import OK | untested | 0.50.2 | installed |
| GPTQModel | INT4 | not installed | — | — | deferred |
| AutoAWQ | INT4 | not installed | — | — | deferred |
| torchao | FP8 / low-bit | import OK | untested | 0.18.0 | mxfp8/cutlass_90a .so fail to load; Python API OK |
| nvidia-modelopt | FP8/NVFP4/INT8 PTQ+QAT | import OK | untested | 0.46.1 | most deploy-relevant |

≥3 toolchains import-clean. Real-speedup gate deferred to Phase 2.

## MASt3R-SLAM bring-up

- `ext/MASt3R-SLAM` cloned `--recursive`.
- PR#86-style patches: `at::linalg_norm`, `.scalar_type()`, `sm_120` gencode, `weights_only=False`.
- Built: `mast3r_slam_backends`, `lietorch`, `curope`.
- Checkpoints in `ext/MASt3R-SLAM/checkpoints/` (main + retrieval + codebook).
- Data: all 9 TUM fr1 seqs; all 11 EuRoC seqs in `data/euroc/<seq>/mav0` (from the HF mirror `GlowBond/EuRoC_MAV_Dataset`, since the ETHZ host times out; nested room zips cached in `data/euroc/_hf`; no `unzip`, so extraction uses Python zipfile). GT in `ext/MASt3R-SLAM/groundtruths/euroc/`.
- GFM FP16 smoke (2 TUM frames, size=512): load 6.6s, pair fwd 0.71s, peak ~3.3 GiB → **GFM_FWD_OK**.

## Run characteristics (for planning)

- Deterministic: same weights → same trajectory; seeds do not add variance.
- Per-run wall time with 2–3 streams sharing one GPU: TUM seq ≈ 4–5 min, EuRoC seq ≈ 7–15 min (MH_01 longest). Each run ≈ 7–8 GiB VRAM.
- Weights are FP32 in memory (2.75 GB), so small perturbations (noise 1e-3) are not rounded away.
- ATE: `evo_ape tum -as` (Sim3); `scripts/eval_ate.py` wraps it.

## FP8 latency probe (2026-09-24)

- `torch._scaled_mm` 1.61× only at 4096³; trunk-shaped stack 0.30×. ModelOpt 0.75/0.12–0.17×; torchao 0.91/0.13×. Transformer Engine: no sm_120 wheel, source build failed. Weight memory 2× smaller. No end-to-end speedup claim.
