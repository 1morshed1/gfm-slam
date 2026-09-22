# techContext — rig + tooling

## Rig: vm-130-131

- GPU: physical **GPU-2**, RTX PRO 6000 Blackwell, `sm_120`, ~96 GiB VRAM.
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
- Data: `data/tum/rgbd_dataset_freiburg1_desk` extracted.
- GFM FP16 smoke (2 TUM frames, size=512): load 6.6s, pair fwd 0.71s, peak ~3.3 GiB → **GFM_FWD_OK**.
