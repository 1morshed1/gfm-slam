# techContext — rig + tooling

## Rig: vm-130-131

- GPU: physical **GPU-1**, RTX PRO 6000 Blackwell, `sm_120`, ~96 GiB VRAM.
- CUDA 12.8, torch 2.11.0+cu128. Capability must read `(12, 0)`.
- SDPA-only (no flash-attn).
- Disk: ~1.5 TiB free, but Docker holds ~200 GiB. Datasets under `/office/dev_workspace/morshed`.
- Compute is NOT the bottleneck — 96 GiB fits VGGT (1.2B) inference + light QAT single-GPU.

## Edge deploy target

- **DEFERRED (D1 = no deploy for now).** If revisited: Jetson AGX Thor (Blackwell, NVFP4+FP8) is
  the faithful target since rig is same arch. Orin (Ampere) would force an INT8 pivot.

## Tooling-support matrix (fill in Phase 0, on GPU-1)

| Tool | Purpose | sm_120 builds? | real speedup? | version | notes |
|------|---------|----------------|---------------|---------|-------|
| bitsandbytes | INT8, NF4 | ? | ? | | |
| GPTQModel | INT4 (Marlin/Machete) | ? | ? | | |
| AutoAWQ | INT4 | ? | ? | | |
| torchao | FP8 / low-bit | ? | ? | | |
| nvidia-modelopt | FP8/NVFP4/INT8 PTQ+QAT | ? | ? | | most deploy-relevant |

Expected: some PyTorch quant libs give accurate fake-quant but no real desktop speedup on `sm_120`
yet. Fine — accuracy from fake-quant; do NOT quote desktop wall-clock as the speedup number.
Latency/energy come from real Blackwell kernels (modelopt/FP8) where they exist.
