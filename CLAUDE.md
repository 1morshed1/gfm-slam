# CLAUDE.md — Edge-GFM-SLAM hard rules

Read this before running anything. These rules override defaults. They exist because
violating them breaks the environment (torch downgrade) or invalidates results.

## Hardware pin (NON-NEGOTIABLE)

- **Always pin GPU-2:** `export CUDA_VISIBLE_DEVICES=2`. (Decision 2026-09-22: use GPU-2 for this project; do not use GPU-0/1.)
- **Verify Blackwell before every experiment:**
  ```bash
  python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))"
  # expect: 2.11.0+cu128 (12, 0) NVIDIA RTX PRO 6000 Blackwell ...
  ```
  Capability must be `(12, 0)`. If not, stop — torch was downgraded.

## Install landmine

- Blind `pip install -e <slam-repo>` may pull `torch==2.2.0` and destroy `sm_120` support.
- **Always:** `pip install -e <repo> --no-deps`, then install curated deps by hand, then
  **re-run the Blackwell verify above** to confirm cu128 + a real matmul on GPU-2 survived.

## Attention

- **SDPA only. Do NOT install flash-attn.** Any attention-quant path must respect the SDPA code path.

## Disk

- Datasets and engines go under the workspace disk (rig: `/office/dev_workspace/morshed/...`),
  NOT root free space. Docker holds ~200 GiB. TartanAir/ScanNet/KITTI are large — plan before download.

## Scope (this project)

- **Compress:** the GFM forward pass only — ViT trunk, fusion/attention blocks, regression heads.
- **Do NOT touch:** the classical back-end (pose graph, Sim(3)/SL(4) opt, loop closure). Keep FP32/CPU.
  Fixed back-end = clean ablation: any metric change is attributable to GFM compression.
- **D1 = no edge deploy for now.** Track B (Jetson/TensorRT) is DEFERRED. Rig is Blackwell, so
  desktop FP8/NVFP4 is architecturally faithful; energy/frame measured on rig via NVML.

## EXP logging discipline

Every experiment logs to `results/<exp-id>.json`: driver / CUDA / torch / quant-lib versions,
`CUDA_VISIBLE_DEVICES`, GPU capability `(12,0)`, seed, config hash, dataset+sequence, all metrics,
and device (`rig` for now). Energy runs also log power-sampling method + duration.

## Locked decisions (2026-09-22)

- D0 GPU-2 pin · D1 no deploy · D2 vision/geometry venue paper · D3 MASt3R-SLAM primary
- D4 PTQ-first → pruning → (distillation dropped) · D5 real-time=target, Pareto=contribution
