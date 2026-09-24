# SLAM-Aware Mixed-Precision Compression of Geometric Foundation Models for Feed-Forward SLAM

> **Status:** working draft (2026-09-24). Numbers from `results/` and `figures/`. Rig-only evaluation; no edge deploy (D1).

**Authors:** TBD  
**Primary target:** **3DV** (geometry/SLAM fit; ~3 mo timeline). Stretch: CVPR/ICCV workshop or main if transfer + energy land.  
**One-line claim:** Allocating quantization bits by *measured closed-loop SLAM sensitivity* (ATE), not by weight-magnitude proxies, improves the accuracy–cost trade-off when compressing the geometric foundation model (GFM) trunk of feed-forward SLAM, with the classical back-end held fixed.

---

## Abstract

Feed-forward SLAM systems such as MASt3R-SLAM spend most of their compute in a large geometric foundation model (GFM) trunk. Compressing that trunk is therefore the natural path to lower memory and energy, but standard foundation-model quantization optimizes reconstruction or language proxies that need not preserve trajectory quality. We study post-training quantization (PTQ) of the MASt3R trunk under a *closed-loop* objective: absolute trajectory error (ATE) with the classical pose-graph back-end frozen in FP32.

On TUM RGB-D fr1, uniform FP8-e4m3 matches the FP16 baseline within **−0.4%** mean ATE while cutting trunk weight storage **2×**. At W4, protecting the $K{=}7$ units with highest measured leave-one ATE sensitivity reduces mean ATE degradation from **+12.8%** (uniform W4) to **+7.1%**, and beats a matched-budget magnitude-L1 proxy by **3.6%** relative mean ATE; a budget sweep finds a further sweet spot at **$K{=}9$ (+5.4%)**, with over-protection at $K{=}11$ hurting mean ATE. Real fused FP8 kernels on Blackwell do **not** yield end-to-end GEMM speedups for trunk-like shapes in our probes; under a no-deploy setting we therefore treat **accuracy + memory** as the primary efficiency axes. The takeaway is methodological: for compression-for-SLAM, the right sensitivity signal is the closed SLAM loop, not a weight statistic.

---

## 1. Introduction

Geometric foundation models have moved from offline reconstruction into online feed-forward SLAM. In MASt3R-SLAM, a ViT-scale trunk produces correspondence and geometry features that a classical optimizer turns into a trajectory and map. The trunk dominates parameters and activation memory; the back-end is comparatively cheap and already well studied. This paper asks a narrow question: *how should we compress the trunk so that the SLAM metric that users care about stays intact?*

Quantization for LLMs and vision transformers is mature, but its objectives—perplexity, ImageNet top-1, or even pose-head AUC—are only loosely related to closed-loop trajectory error. Recent task-aware mixed-precision work on geometry FMs (notably QVGGT) already allocates bits using downstream pose accuracy. What remains under-explored is allocating bits from **full SLAM-loop ATE** with a **fixed classical back-end**, and comparing an **algorithmic** allocator against a magnitude proxy at equal budget.

**Contributions.**

1. **Closed-loop sensitivity.** We measure per-unit ATE degradation under leave-one W4 quantization and use that profile as the allocation signal, holding the MASt3R-SLAM back-end in FP32 so metric deltas are attributable to trunk precision.
2. **Matched-budget algorithmic allocation.** A greedy (and optionally ILP) protector selects $K$ trunk units to keep in FP16; we ablate against uniform W4 and a weight-L1 magnitude proxy at the same $K$ (H3), and show that the ATE–$K$ curve is **non-monotonic** (best at $K{=}9$ on TUM fr1; over-protection at $K{=}11$ hurts).
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

| config | mean ATE (m) | vs FP16 | role |
|--------|-------------:|--------:|------|
| FP16 | 0.0295 | +0.0% | baseline |
| FP8 e4m3 | 0.0294 | −0.4% | accuracy champ |
| W8A8 | 0.0315 | +6.8% | uniform |
| W4 geom-protect $K{=}7$ | 0.0316 | +7.1% | method |
| W4 mag-protect $K{=}7$ | 0.0328 | +10.9% | proxy |
| W4 uniform | 0.0333 | +12.8% | uniform |

FP8 is effectively free in accuracy. Uniform W4 is the costliest; geometry protection recovers most of the gap toward FP16 without restoring the full model to FP16.

### 6.2 H3: geometry vs magnitude at matched $K{=}7$

| allocator | mean ATE | vs FP16 | vs uniform W4 |
|-----------|---------:|--------:|--------------:|
| Geometry greedy | 0.0316 | +7.1% | −5.1% |
| Magnitude L1 | 0.0328 | +10.9% | −1.7% |
| Uniform W4 | 0.0333 | +12.8% | — |

**Headline:** geometry beats magnitude by **3.6%** relative mean ATE at equal unit budget. Magnitude helps vs uniform but systematically under-protects SLAM-critical units (e.g. `dec_blocks2.8` ranks first by ATE sensitivity, not by L1 mass).

### 6.3 Sensitivity profile

Leave-one W4 on desk (`figures/sensitivity_heatmap.{png,pdf}`) shows:

- Downstream heads and `dec_blocks2.8` dominate $\delta_u$.
- A small set of late encoder blocks (`enc_blocks.{19,22,9,7,0,14}`) form the rest of the $K{=}7$ protect set.
- Many decoder units are near-neutral or slightly beneficial under W4—uniform W4 wastes budget protecting the wrong places.

### 6.4 Cross-dataset transfer (EuRoC V1_01_easy)

| config | RMSE (m) | vs FP16 (0.0395) |
|--------|---------:|-----------------:|
| FP8 | 0.0366 | −7.4% |
| W4 geom-greedy $K{=}7$ | 0.0467 | +18.0% |

FP8 transfers cleanly. W4-protect is softer on this sequence than on TUM mean: allocation was fit on TUM desk sensitivity, so transfer is a remaining threat (full EuRoC deferred).

### 6.5 Efficiency: memory yes, fused FP8 latency no

Track-A SLAM ATE runs use fake quant and therefore have **no** real compute speedup. Separate GPU-2 microbenches (`scripts/bench_fp8.py`):

| path | large $4096^3$ | trunk-like / stack | notes |
|------|----------------:|-------------------:|-------|
| `torch._scaled_mm` | **1.61×** | trunk stack **0.30×** | weight mem **2.00×** (944→472 MB) |
| ModelOpt real FP8 GEMM | 0.75× | 0.12–0.17× | cuda ext loads; slower than FP16 |
| torchao Float8Dynamic | 0.91× | 0.13–0.14× | same story |
| Transformer Engine 2.19 | — | — | no sm_120 wheel; source build failed |

**Paper claim under D1:** report **accuracy + memory** Pareto; do **not** claim end-to-end SLAM FLOP speedup from current FP8 paths. W4 packed-kernel latency remains open.

### 6.6 Budget sweep over $K$

Greedy protect sets for $K \in \{3,5,7,9,11\}$ (heads always included). $K{=}7$ reuses the locked shortlist EXP. Figure: `figures/k_budget_pareto.{png,pdf}`.

| $K$ | mean ATE (m) | vs FP16 | note |
|----:|-------------:|--------:|------|
| 3 | 0.0337 | +14.2% | near 15% gate |
| 5 | 0.0319 | +8.0% | |
| 7 | 0.0316 | +7.1% | H3 operating point |
| **9** | **0.0311** | **+5.4%** | **best** |
| 11 | 0.0335 | +13.5% | over-protect; room/teddy hurt |

**Takeaway:** accuracy improves from $K{=}3\to9$, then **degrades at $K{=}11$** — protecting lower-sensitivity units can hurt mean ATE (notably `room`/`teddy`). The operating sweet spot on TUM fr1 is **$K{=}9$** (+5.4%), with $K{=}7$ still the fair H3 match to the magnitude proxy. EXPs: `results/20260924-ksweep-w4-geom-k{3,5,7,9,11}.json`.

---

## 7. Discussion & Limitations

**Why geometry wins.** Magnitude ranks large Linear slabs; SLAM ATE is dominated by a few geometry-critical blocks. Matched-$K$ protection is only as good as the ranking signal.

**Limitations.** (i) Sensitivity and allocation are fit primarily on TUM; EuRoC is one sequence. (ii) Fake quant isolates accuracy, not wall-clock W4. (iii) Fused FP8 on Blackwell did not beat FP16 for trunk-like GEMMs in ModelOpt/torchao; TE unavailable. (iv) Asymmetric-precision arm cut. (v) Distillation dropped by decision D4.

**Threats.** Overfit of protect sets to fr1/desk leave-one profile; the $K$-sweep (§6.6) shows a non-monotonic curve (best at $K{=}9$, worse at $K{=}11$), so budget choice matters and more protect is not always better.

---

## 8. Conclusion

Compressing GFMs for feed-forward SLAM should optimize the metric the system is judged on: closed-loop ATE with a fixed back-end. On MASt3R-SLAM, FP8 preserves accuracy at half the weight memory; at W4, SLAM-aware greedy protection outperforms uniform and magnitude baselines at matched $K{=}7$, and a budget sweep peaks at **$K{=}9$ (+5.4% mean ATE)** before over-protection hurts. Real fused FP8 latency on current Blackwell software stacks is not yet a win for trunk-like shapes—honesty about that keeps the contribution focused on **allocation under a SLAM objective** plus **memory**. Future work: full EuRoC transfer, packed W4 kernels, and structural asymmetric precision.

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
3. [x] Budget sweep over $K$ — done 2026-09-24; best **$K{=}9$ (+5.4%)**, U-shape at $K{=}11$.
4. [ ] More EuRoC (deferred).
5. [x] Asymmetric-precision arm — **cut** (§4.4).
