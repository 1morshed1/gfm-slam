# novelty.md — living related-work / scoop watch (§1, risk R5)

Field moves monthly. Update as you read. The scoop risk is real.

## Our defensible wedge (post-D1)

With deploy deferred, the "on real edge silicon + energy-per-frame" wedge is weaker.
Lead instead with:

> **2026-09-23 scoop-watch UPDATE (R5):** bare claim "downstream/task-aware mixed-precision
> for a GFM" is NO LONGER novel — QVGGT (2605.31124) and Mix-QVLA (2606.19565) got there.
> Wedge NARROWED below. Do NOT frame contribution #2 as "downstream-aware is new."

1. **Full-SLAM-loop metric AS the sensitivity signal** — allocate by measured **ATE/pointmap**
   with the **classical back-end held fixed FP32**. QVGGT stops at the forward-pass pose head
   (AUC@30); it never closes the SLAM loop. This is the surviving core. *(status: accuracy done;
   latency/energy Pareto PENDING)*
2. **Algorithmic allocator at matched budget** — greedy/ILP + **head-to-head vs magnitude proxy**
   at equal K (H3). QVGGT hand-picks blocks (heuristic); we show a proxy-vs-geometry ablation.
3. **Target = MASt3R-SLAM** (full loop) on commodity Blackwell + energy/frame via NVML (once measured).

## Direct / near neighbours (verify dates + claims before citing)

| Paper | arXiv | What it does | Why we're still distinct |
|-------|-------|--------------|--------------------------|
| VersaQ-3D | 2601.20317 | Calibration-free W4A4 quant for VGGT, transform-coding for outliers | Custom accelerator, reconstruction only — NOT full SLAM ATE |
| Co-Me | 2511.14751 | Confidence-guided token merging, VGGT/Pi3, 21.5× | Owns token-merging sub-corner — cite as baseline, don't reinvent |
| VGGT-X | 2509.25191 | Memory-efficient VGGT (bf16, drop intermediates) | Dense NVS, not compression-for-SLAM |
| Fast-FoundationStereo | 2512.11130 | KD + NAS + pruning for stereo FM | Different task; method template only |
| GFM distillation (lunar) | 2607.01851 | Domain-reconstruction distillation | Distillation dropped (D4) |

## Scoop watch (add dated entries as you find new work)

### 2026-09-23 — full arXiv/web pass (Opus)

| Paper | arXiv | What it does | Threat / action |
|-------|-------|--------------|-----------------|
| **QVGGT** | 2605.31124 | PTQ W4A16 for VGGT; **task-aware per-block sensitivity** mixed-precision (measures downstream pose-head accuracy AUC@30, keeps fragile blocks FP16) | **HIGH — near-scoop.** Owns "task-aware per-block MP for a GFM." Distinct: no full SLAM/ATE, no fixed back-end, **heuristic** block pick (not greedy/ILP), VGGT not MASt3R-SLAM. Cite prominently; reframe our #2 as *SLAM-loop metric + algorithmic allocator + proxy ablation*. |
| **Quantized VGGT** | 2509.21302 (ICLR'26) | VGGT PTQ method — heavy-tailed activation / calibration-sample instability (BRECQ/GPTQ/SmoothQuant baselines) | MED. GFM-PTQ method baseline; not sensitivity-allocation. Cite as PTQ reference. |
| **Mix-QVLA** | 2606.19565 | Task-evidence-aware mixed-precision for **VLA** models (compare FP vs quant evidence at functional boundaries) | MED. Same *philosophy* (downstream-evidence allocation), different domain (robotics VLA). Cite as prior for "task-aware > proxy." |
| **Where Bits Matter in World-Model Planning** | 2602.11882 | Planning-aware bit allocation under mem/latency constraints | LOW. Philosophical prior, different domain. Cite in intro. |
| **LeanGate** | 2604.08718 | Feed-forward frame-gating for transformer SLAM (skip >90% frames, 5× throughput) | LOW/orthogonal. Efficiency-for-SLAM but token/frame pruning, not quant. Cite as complementary baseline (like Co-Me). |
| **MXSens** | 2607.17733 | Hessian-guided fine-grained sensitivity MP for LLMs | LOW. Proxy (Hessian) — the exact thing we argue against. Cite as proxy contrast. |

**Net:** paper survives but MUST reframe. Old "downstream-aware is new" → dead. New wedge =
closed-SLAM-loop ATE sensitivity + fixed FP32 back-end + algorithmic (greedy/ILP) allocator with
matched-budget proxy ablation, on MASt3R-SLAM. Re-run this pass ~monthly and pre-submission.
