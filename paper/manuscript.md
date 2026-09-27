# SLAM-Aware Mixed-Precision Compression of Geometric Foundation Models for Feed-Forward SLAM

> **Status:** working draft (2026-09-24). Numbers from `results/` and `figures/`. Rig-only evaluation; no edge deploy (D1).

**Authors:** TBD  
**Primary target:** **3DV** (geometry/SLAM fit; ~3 mo timeline). Stretch: CVPR/ICCV workshop or main if transfer + energy land.  
**One-line claim:** Allocating quantization bits by *measured closed-loop SLAM sensitivity* (ATE), not by weight-magnitude proxies, improves the accuracy–cost trade-off when compressing the geometric foundation model (GFM) trunk of feed-forward SLAM, with the classical back-end held fixed.

---

## Abstract

Feed-forward SLAM systems such as MASt3R-SLAM spend most of their compute in a large geometric foundation model (GFM) trunk. Compressing that trunk is therefore the natural path to lower memory and energy, but standard foundation-model quantization optimizes reconstruction or language proxies that need not preserve trajectory quality. We study post-training quantization (PTQ) of the MASt3R trunk under a *closed-loop* objective: absolute trajectory error (ATE) with the classical pose-graph back-end frozen in FP32.

> **Status (2026-09-27): claims below are NOT supported by the current evidence — abstract to be rewritten.** Multi-seed repeats show the pipeline is deterministic (seed std ≈ 0), so seeds do not measure noise; per-sequence paired bootstrap on TUM finds **no significant** difference between geometry and magnitude protection (−1.4%, 95% CI [−11.1, +8.1]); on EuRoC (5 seqs) the TUM-fit W4 protect set degrades ATE by **+96.6%** and FP8 by **+15.6%**. See §6.2, §6.4, §6.6.

On TUM RGB-D fr1, uniform FP8-e4m3 matches the FP16 baseline within **−0.5%** mean ATE while cutting trunk weight storage **2×**. At W4, protecting the $K{=}7$ units with highest measured leave-one ATE sensitivity lowers mean ATE degradation from **+12.8%** (uniform W4) to **+7.0%**, versus **+10.9%** for a matched-budget magnitude-L1 proxy — but across the 9 sequences neither gap is statistically distinguishable from zero. Real fused FP8 kernels on Blackwell do **not** yield end-to-end GEMM speedups for trunk-like shapes in our probes; under a no-deploy setting we therefore treat **accuracy + memory** as the primary efficiency axes.

---

## 1. Introduction

Geometric foundation models have moved from offline reconstruction into online feed-forward SLAM. In MASt3R-SLAM, a ViT-scale trunk produces correspondence and geometry features that a classical optimizer turns into a trajectory and map. The trunk dominates parameters and activation memory; the back-end is comparatively cheap and already well studied. This paper asks a narrow question: *how should we compress the trunk so that the SLAM metric that users care about stays intact?*

Quantization for LLMs and vision transformers is mature, but its objectives—perplexity, ImageNet top-1, or even pose-head AUC—are only loosely related to closed-loop trajectory error. Recent task-aware mixed-precision work on geometry FMs (notably QVGGT) already allocates bits using downstream pose accuracy. What remains under-explored is allocating bits from **full SLAM-loop ATE** with a **fixed classical back-end**, and comparing an **algorithmic** allocator against a magnitude proxy at equal budget.

**Contributions.**

1. **Closed-loop sensitivity.** We measure per-unit ATE degradation under leave-one W4 quantization and use that profile as the allocation signal, holding the MASt3R-SLAM back-end in FP32 so metric deltas are attributable to trunk precision.
2. **Matched-budget algorithmic allocation.** A greedy (and optionally ILP) protector selects $K$ trunk units to keep in FP16; we ablate against uniform W4 and a weight-L1 magnitude proxy at the same $K$ (H3). *Current evidence:* neither the H3 gap nor any trend across $K$ is statistically significant on TUM fr1 (paired bootstrap, §6.2 and §6.6).
3. **Blackwell Track-A study.** We report a PTQ ladder and efficiency notes on commodity sm_120 hardware: accuracy and **memory** are the clean deploy-relevant axes under our constraints; fused FP8 latency is documented as a negative result for trunk-like shapes.

We do **not** claim that “task-aware mixed precision for GFMs” is new; QVGGT and Mix-QVLA already occupy that space. Our wedge is the closed SLAM loop, fixed back-end, and matched-budget proxy ablation on MASt3R-SLAM.

---

## 2. Related Work

**Task-aware mixed precision for geometry FMs.** QVGGT (arXiv:2605.31124) applies PTQ W4A16 to VGGT with per-block sensitivity from pose-head AUC@30 and keeps fragile blocks in FP16. Mix-QVLA (2606.19565) applies task-evidence-aware MP in the VLA setting. We cite both as prior that downstream evidence beats proxies; we differ by closing the full SLAM loop (ATE), freezing the classical back-end, and using a greedy/ILP allocator with a magnitude-matched ablation rather than hand-picked blocks.

**GFM PTQ baselines.** Quantized VGGT (2509.21302), VersaQ-3D (2601.20317), and VGGT-X (2509.25191) address reconstruction or memory-efficient inference without full-SLAM ATE under a fixed back-end. We use uniform W8/W8A8/W4 as ladder baselines in Track A.

**Orthogonal efficiency.** Co-Me (2511.14751) and LeanGate (2604.08718) reduce tokens/frames; complementary to quantization, not competing.

**Proxy-based MP.** MXSens (2607.17733) and HAWQ/GPTQ/AWQ-style methods allocate by Hessian or activation statistics—the objectives we argue are misaligned for SLAM.

**Distinction.** Prior GFM quant either stops at reconstruction/pose heads or lacks an algorithmic allocator with a matched-budget proxy head-to-head on full-SLAM ATE.

---

## 3. Preliminaries & Problem Setup

**Pipeline.** MASt3R-SLAM runs a shared GFM trunk (encoder, dual decoders, regression heads) then a classical pose-graph / Sim(3) optimizer with loop closure. We quantize **only** the trunk forward pass; the back-end stays FP32 so any ATE change is due to trunk precision.

**Units.** A *unit* is a named quantizable block (e.g. `enc_blocks.19`, `dec_blocks2.8`, `downstream_head1`) as enumerated by `scripts/ptq.py` / `scripts/allocate_bits.py`. Heads are always protected at FP16 in mixed-precision experiments.

**Problem.** Given a bit budget expressed as “protect $K$ trunk units at FP16, quantize the rest to W4 weight-only,” choose the protect set to minimize mean ATE on TUM fr1.

**Notation.** FP16 = baseline trunk; FP8-e4m3 = fake-quant e4m3 on Linear weights (accuracy ladder); W4 / W8 = fake int weight-only unless noted. Track-A SLAM runs use *fake* quant (quantize→dequantize into FP16 compute) so ATE isolates sensitivity without requiring sm_120 packed kernels.

---

## 4. Method

### 4.1 PTQ ladder

We evaluate a uniform ladder: FP16 → FP8-e4m3 → W8 → W8A8 → W4, implemented in `scripts/ptq.py` with SDPA attention (no flash-attn). Configs live under `configs/*.yaml`. Calibration follows the MASt3R-SLAM eval calib protocol.

### 4.2 Sensitivity profiling

For each trunk unit $u$, we run leave-one W4 (unit $u$ at W4, others FP16) on `fr1/desk` and record relative ATE delta $\delta_u$ vs FP16. The profile (Fig. sensitivity heatmap) shows heads and late encoder / `dec_blocks2.8` as high-sensitivity; many early decoder units are tolerant.

### 4.3 SLAM-aware bit allocation

Given $\{\delta_u\}$ and budget $K$:

- **Greedy:** protect the $K$ trunk units with largest $\delta_u$, plus both heads.
- **ILP (optional):** maximize $\sum_u \delta_u x_u$ subject to a parameter-count budget matched to greedy top-$K$ mass (`scripts/allocate_bits.py`).
- **Magnitude proxy:** protect top-$K$ by weight L1 mass (LLM-style), same unit count.

The rest of the trunk uses fake W4 weight-only (`fake_except`). Hypothesis **H3:** at matched $K$, geometry-greedy ATE beats magnitude and uniform W4.

### 4.4 Asymmetric precision (out of scope)

SLAM-structural asymmetric precision (plan §6.6) is **cut** from this manuscript as future work; we do not report results for it.

---

## 5. Experimental Setup

| Item | Setting |
|------|---------|
| Rig | RTX PRO 6000 Blackwell, sm_120, CUDA 12.8, torch 2.11.0+cu128, **GPU-2** |
| System | MASt3R-SLAM Track A |
| Primary data | TUM RGB-D fr1 (9 sequences), mean ATE RMSE (Sim3 align, evo) |
| Transfer | EuRoC V1_01_easy (spot-check) |
| Metrics | ATE RMSE (primary); weight memory from microbench; latency/energy only as microbench notes |
| Logging | Per-EXP JSON with driver/CUDA/torch/quant-lib versions (`scripts/explog.py`) |

**Gate:** ≤15% mean ATE degradation vs FP16 for protected W4 configs (all reported protected configs pass).

---

## 6. Results

### 6.1 PTQ ladder — TUM fr1 mean ATE

Multi-seed (seeds 0,1,2; `results/20260924-multiseed-*.json`). **Seed std is ≈0 for every config** (≤0.03 mm): MASt3R-SLAM with `single_thread` is deterministic given fixed weights, so seed repeats do not estimate run-to-run noise. W8A8 is single-run.

| config | mean ATE (m) | seed std (m) | vs FP16 |
|--------|-------------:|-------------:|--------:|
| FP16 | 0.02955 | 0.00003 | +0.0% |
| FP8 e4m3 | 0.02941 | 0.00000 | −0.5% |
| W8A8 | 0.0315 | — | +6.8% |
| W4 geom-protect $K{=}7$ | 0.03163 | 0.00000 | +7.0% |
| W4 mag-protect $K{=}7$ | 0.03276 | 0.00000 | +10.9% |
| W4 uniform | 0.03332 | 0.00000 | +12.8% |

FP8 is effectively free in accuracy **on TUM** (but not on EuRoC, §6.4).

### 6.2 H3: geometry vs magnitude at matched $K{=}7$ — not significant

Because seeds carry no variance, uncertainty is estimated across sequences: paired per-sequence bootstrap of relative ATE differences (normalized by FP16), 10k resamples over the 9 fr1 sequences (`scripts/paired_bootstrap_tum.py` → `results/paired_bootstrap_tum.json`).

| comparison | mean rel. diff | 95% CI | first better on |
|------------|---------------:|--------|----------------:|
| geom-K7 vs mag-K7 | −1.4% | [−11.1%, +8.1%] | 5/9 |
| geom-K7 vs uniform W4 | −4.4% | [−12.8%, +5.3%] | 7/9 |
| mag-K7 vs uniform W4 | −3.0% | [−10.5%, +3.5%] | 5/9 |

**Result:** the earlier “geometry beats magnitude by 3.6%” headline is driven by a few sequences; the per-sequence effect is not distinguishable from zero. H3 is **not supported** on current evidence.

### 6.3 Sensitivity profile

Leave-one W4 on desk (`figures/sensitivity_heatmap.{png,pdf}`) shows:

- Downstream heads and `dec_blocks2.8` dominate $\delta_u$.
- A small set of late encoder blocks (`enc_blocks.{19,22,9,7,0,14}`) form the rest of the $K{=}7$ protect set.
- Many decoder units are near-neutral or slightly beneficial under W4—uniform W4 wastes budget protecting the wrong places.

### 6.4 Cross-dataset transfer (EuRoC, 5 sequences) — fails

`results/20260924-euroc-shortlist-{fp16,fp8,w4-greedy}.json`. ATE RMSE (m), Sim3.

| seq | FP16 | FP8 | W4 geom-K7 (TUM-fit) |
|-----|-----:|----:|---------------------:|
| V1_01_easy | 0.0395 | 0.0366 | 0.0467 |
| MH_01_easy | 0.0227 | 0.0313 | 0.0518 |
| MH_02_easy | 0.0171 | 0.0225 | 0.0564 |
| V1_02_medium | 0.0190 | 0.0176 | 0.0312 |
| V2_01_easy | 0.0204 | 0.0292 | 0.0473 |
| **mean** | **0.0237** | **0.0274 (+15.6%)** | **0.0467 (+96.6%)** |

FP8 is no longer free (MH_01 +38%, V2_01 +43%). The TUM-desk-fit W4 protect set roughly doubles ATE on EuRoC. The single-sequence V1_01 check (−7.4% FP8, +18% W4) was unrepresentative. Uniform W4 and magnitude-K7 have **not** been run on EuRoC, so whether geometry still beats the proxies there is unknown.

### 6.5 Efficiency: memory yes, fused FP8 latency no

Track-A SLAM ATE runs use fake quant and therefore have **no** real compute speedup. Separate GPU-2 microbenches (`scripts/bench_fp8.py`):

| path | large $4096^3$ | trunk-like / stack | notes |
|------|----------------:|-------------------:|-------|
| `torch._scaled_mm` | **1.61×** | trunk stack **0.30×** | weight mem **2.00×** (944→472 MB) |
| ModelOpt real FP8 GEMM | 0.75× | 0.12–0.17× | cuda ext loads; slower than FP16 |
| torchao Float8Dynamic | 0.91× | 0.13–0.14× | same story |
| Transformer Engine 2.19 | — | — | no sm_120 wheel; source build failed |

**Paper claim under D1:** report **accuracy + memory** Pareto; do **not** claim end-to-end SLAM FLOP speedup from current FP8 paths. W4 packed-kernel latency remains open.

### 6.6 Budget sweep over $K$ — no supported trend

Greedy protect sets for $K \in \{3,5,7,9,11\}$ (heads always included), 3 seeds each (seed std ≈0, so the means equal the single-run values).

| $K$ | mean ATE (m) | vs FP16 |
|----:|-------------:|--------:|
| 3 | 0.03372 | +14.1% |
| 5 | 0.03198 | +8.2% |
| 7 | 0.03163 | +7.0% |
| 9 | 0.03113 | +5.4% |
| 11 | 0.03354 | +13.5% |

Paired per-sequence bootstrap: K9 vs K11 −5.8% [−12.5%, +1.1%]; K9 vs K7 −2.9% [−8.5%, +1.4%]; K5 vs K7 +0.3% [−4.8%, +5.1%]. **No pairwise difference is significant.** Since the pipeline is deterministic, the $K{=}11$ dip reflects the trajectory's sensitivity to small weight changes rather than seed noise, and cannot be separated from that without a perturbation-based noise estimate. **Drop the U-shape narrative.** Figure `figures/k_budget_pareto.{png,pdf}` is descriptive only.

---

## 7. Discussion & Limitations

**What the evidence supports.** On TUM, protecting ATE-sensitive units lowers mean ATE vs uniform W4 in 7/9 sequences, but neither that gap nor the geometry-vs-magnitude gap is significant across sequences. The protect set, fit on one TUM sequence, does not transfer to EuRoC.

**Limitations.** (i) Sensitivity fit on a single sequence (fr1/desk), which is also in the TUM eval mean (leakage). (ii) Seeds do not vary the deterministic pipeline, so we have no run-to-run noise estimate; only across-sequence uncertainty. (iii) Uniform/magnitude baselines not yet run on EuRoC. (iv) Fake quant isolates accuracy, not wall-clock W4. (v) Fused FP8 on Blackwell did not beat FP16 for trunk-like GEMMs; TE unavailable. (vi) Asymmetric-precision arm cut; distillation dropped (D4).

**Open questions.** Whether a sensitivity profile pooled across sequences/datasets transfers; whether FP8's EuRoC degradation is dataset-specific (motion blur, grayscale input) or a calibration artefact.

---

## 8. Conclusion

*(To be rewritten once the direction is chosen.)* Current evidence: FP8 is near-lossless on TUM but costs +15.6% ATE on EuRoC; SLAM-sensitivity-based W4 protection reduces TUM ATE relative to uniform and magnitude baselines on average, but not significantly across sequences, and a TUM-fit protect set roughly doubles ATE on EuRoC. The allocation claim is not supported as stated.

---

## Appendix

- **A.** Full per-unit sensitivity table: `results/sens_w4_full_units.txt`, EXP `20260923-phase2-sens-w4-full-fr1desk.json`.
- **B.** Allocator outputs: `results/protect_greedy_k{3,5,7,9,11}.*`, `protect_magnitude_k7.*`, `protect_ilp_match_k7.*`.
- **C.** Reproducibility: `memory-bank/techContext.md`; GPU-2 pin; `--no-deps` install landmine for quant stacks.
- **D.** Per-sequence ATE: `results/phase2_tum_*_ates.txt` and EXPs under `results/2026092{2,3,4}-*.json`.

---

## Submission checklist

1. [~] Novelty — reframed 2026-09-23; re-run scoop-watch pre-submission.
2. [~] Latency/energy — microbench done; claim memory, not FLOP speedup.
3. [x] Budget sweep over $K$ — multi-seed done; no significant trend, U-shape dropped.
4. [~] EuRoC — 5 seqs done for FP16/FP8/W4-geom; **uniform + magnitude still missing**.
5. [x] Asymmetric-precision arm — **cut** (§4.4).
6. [ ] Noise estimate — seeds don't vary the pipeline; need perturbation-based variance.
7. [ ] Held-out / pooled sensitivity (fix desk leakage, test transfer).
8. [ ] Direction decision — core claim unsupported; choose rescue vs negative-result framing.
