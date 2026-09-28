# Closed-Loop Quantization Sensitivity Is Sequence-Specific: A Negative Result for SLAM-Aware Bit Allocation in Feed-Forward SLAM

> **Status:** working draft (2026-09-28). Negative-result / empirical study. All numbers from committed `results/` and `figures/`. Rig-only evaluation; no edge deploy (D1).

**Authors:** TBD
**Primary target:** vision/robotics **workshop** (negative-result / "lessons learned" track), or a short empirical paper.
**One-line claim:** For W4 post-training quantization of the MASt3R-SLAM trunk under a closed-loop objective (ATE), per-unit sensitivity is *not* a stable, transferable property: it disagrees across sequences even within one dataset, so no fitted or pooled protect set transfers — and this instability, not a tuning failure, is why SLAM-aware bit allocation does not beat uniform or magnitude baselines out of distribution.

---

## Abstract

Feed-forward SLAM systems such as MASt3R-SLAM spend most of their compute in a large geometric foundation model (GFM) trunk, making that trunk the natural target for quantization. A growing line of task-aware mixed-precision work (QVGGT, Mix-QVLA) allocates bits by *downstream* sensitivity, on the premise that some units matter more than others and that this ranking is a property of the model one can measure once and reuse. We test that premise directly for **closed-loop SLAM**: we measure per-unit W4 sensitivity as the change in absolute trajectory error (ATE) with the classical pose-graph back-end frozen in FP32, and ask whether the resulting protect sets transfer.

They do not, and the reason is instructive. On TUM RGB-D fr1 (9 sequences), protecting the $K{=}7$ units with highest ATE sensitivity fit on one sequence lowers *mean* degradation from $+12.8\%$ (uniform W4) to $+7.0\%$, but a per-sequence paired bootstrap shows this gap — and the gap over a magnitude-L1 proxy — is **not statistically distinguishable from zero**. On EuRoC (5 sequences) **every** W4 variant collapses (uniform $+78.5\%$, magnitude $+66.9\%$, TUM-fit geometry $+96.6\%$): allocation does not help out of distribution. A pre-registered rank-correlation gate then explains why: per-unit sensitivity profiles are **uncorrelated across sequences** (mean cross-dataset Spearman $+0.01$; within-TUM $-0.14$; top-7 overlap at chance). This is not measurement noise — a perturbation-based noise floor (multiplicative weight noise, $20\times$ below FP8 rounding) shows 18–47 of 49 units have effects above $2\times$ their sequence's floor, yet the unit that dominates one sequence is near-inert on another (`dec_blocks2.8`: rank 1 on desk, rank 47–49 on floor/360/V1_01). We conclude that closed-loop quantization sensitivity is a property of the *sequence*, not of the *unit*, at W4-scale perturbations. Two secondary findings sharpen the picture: FP8-e4m3 is near-lossless on TUM ($-0.5\%$, $2\times$ weight memory) but costs $+15.6\%$ on EuRoC, so single-dataset compression results flatter; and because MASt3R-SLAM is deterministic (seed std $\approx 0$), seeds do not measure uncertainty — evaluation needs a sequence-level bootstrap plus a perturbation floor. We offer the instability finding, the methodology, and the negative transfer result as guidance for the compression-for-SLAM literature.

---

## 1. Introduction

Geometric foundation models have moved from offline reconstruction into online feed-forward SLAM. In MASt3R-SLAM, a ViT-scale trunk produces correspondence and geometry features that a classical optimizer turns into a trajectory and map. The trunk dominates parameters and activation memory; the back-end is comparatively cheap and well studied. Compressing the trunk is therefore the natural path to lower memory, and post-training quantization (PTQ) is the cheapest lever.

A now-standard idea is **task-aware mixed precision**: not all units are equally fragile, so measure a downstream sensitivity per unit and keep the fragile ones in higher precision. QVGGT allocates by pose-head AUC; Mix-QVLA by task evidence in a VLA. The unstated assumption behind all of it is that per-unit sensitivity is a *stable property of the model* — something you can profile on one workload and reuse on another. We set out to build a stronger version of this idea for SLAM: allocate bits by the metric users actually care about, closed-loop ATE, with the classical back-end frozen in FP32 so any change is attributable to trunk precision, and beat a magnitude proxy at matched budget.

**The idea failed, and the way it failed is the contribution.** The protect set did not transfer across datasets; before writing that off as an allocation-quality problem we ran a pre-registered test of the underlying assumption and found it false. Per-unit closed-loop sensitivity does not agree across sequences — not merely across TUM vs EuRoC, but between two sequences of the *same* dataset. The signal the entire approach depends on is sequence-specific.

We report this as a negative result because it is a real, generalizable finding about the system, and because it explains a pattern others will hit: any protect set fit on one workload will look good there and fail elsewhere, and single-dataset compression numbers (including our own near-lossless FP8 on TUM) systematically overstate robustness.

**Contributions.**

1. **Sensitivity does not transfer (main result).** Under a pre-registered rank-correlation gate, per-unit W4 ATE-sensitivity profiles are uncorrelated across sequences: mean cross-dataset Spearman $+0.01$, within-TUM $-0.14$, top-7 protect-set overlap at chance. The unit that dominates one sequence is near-inert on others. (§6.3)
2. **Allocation does not beat baselines out of distribution.** On TUM the geometry-vs-magnitude and geometry-vs-uniform gaps are not significant across sequences (paired bootstrap); on EuRoC all W4 variants collapse ($+67\%$ to $+97\%$), allocation included. (§6.2, §6.4)
3. **It is real, not noise.** A perturbation-based noise floor separates genuine per-unit effects (18–47 of 49 units above $2\times$ floor) from run-to-run variation, and confirms the transfer failure is sequence-specific sensitivity, not measurement error. (§6.5)
4. **Secondary: FP8 flatters on a single dataset.** FP8-e4m3 is near-lossless on TUM ($-0.5\%$) but costs $+15.6\%$ on EuRoC — a caution against reporting compression robustness from one dataset. (§6.1, §6.4)
5. **Methodology for deterministic SLAM.** MASt3R-SLAM is deterministic (seed std $\approx 0$), so seeds do not estimate uncertainty; we use a per-sequence paired bootstrap and a multiplicative-weight-noise floor instead, and recommend both for compression-for-SLAM evaluation. (§5, §6.5)

We do **not** claim task-aware mixed precision is a bad idea in general, nor that it fails for the reconstruction/pose objectives it was designed for; QVGGT and Mix-QVLA report gains in their settings. Our claim is scoped: for W4 closed-loop SLAM on MASt3R-SLAM across TUM and EuRoC, the sensitivity signal is not transferable, so allocation offers no reliable advantage over uniform quantization.

---

## 2. Related Work

**Task-aware mixed precision for geometry FMs.** QVGGT (arXiv:2605.31124) applies PTQ W4A16 to VGGT with per-block sensitivity from pose-head AUC@30 and keeps fragile blocks in FP16. Mix-QVLA (2606.19565) applies task-evidence-aware MP for VLA models. Both assume a reusable per-unit sensitivity ranking; both stop short of a closed control loop. Our result is a boundary condition on that assumption: under a full SLAM loop with a fixed back-end, the ranking does not transfer across sequences, so the allocation step it enables does not beat uniform out of distribution. We view this as complementary, not contradictory — it identifies where the recipe stops working.

**GFM PTQ baselines.** Quantized VGGT (2509.21302), VersaQ-3D (2601.20317), and VGGT-X (2509.25191) address reconstruction or memory-efficient inference without full-SLAM ATE under a fixed back-end. We use uniform W8/W8A8/W4 as ladder baselines.

**Proxy-based allocation.** MXSens (2607.17733) and HAWQ/GPTQ/AWQ-style methods allocate by Hessian or activation statistics. We include a magnitude-L1 proxy as a baseline; our finding is that at W4 no allocation signal — geometry or proxy — transfers for closed-loop SLAM.

**Orthogonal efficiency.** Co-Me (2511.14751) and LeanGate (2604.08718) reduce tokens/frames; complementary to quantization.

**Negative results and reproducibility.** Our methodological contribution — that a deterministic SLAM pipeline defeats seed-based variance estimation and needs a sequence bootstrap plus a perturbation floor — is in the spirit of empirical-rigor work in ML systems; we are not aware of a prior statement of it for feed-forward SLAM quantization.

---

## 3. Preliminaries & Problem Setup

**Pipeline.** MASt3R-SLAM runs a shared GFM trunk (encoder, dual decoders, regression heads) then a classical pose-graph / Sim(3) optimizer with loop closure. We quantize **only** the trunk forward pass; the back-end stays FP32 so any ATE change is due to trunk precision. This isolation is deliberate: a fixed back-end makes the trunk the only moving part, which is exactly what makes the sensitivity question well posed.

**Units.** A *unit* is a named quantizable block (e.g. `enc_blocks.19`, `dec_blocks2.8`, `downstream_head1`) as enumerated by `scripts/ptq.py`. There are 49 trunk units; downstream heads are always FP16 in mixed-precision experiments.

**Two sensitivity measures.** For each unit $u$ and sequence $s$:
- **quantize-one (`loi`):** unit $u$ at W4, rest FP16; score $(\text{ATE}_u - \text{ATE}_{\text{FP16}})/\text{ATE}_{\text{FP16}}$ (higher = more worth protecting).
- **protect-one (`loo`):** whole trunk W4, unit $u$ restored to FP16; score $(\text{ATE}_{\text{unif W4}} - \text{ATE}_u)/\text{ATE}_{\text{unif W4}}$ (higher = more worth protecting).

The two measures probe additive vs. interaction-dominated regimes respectively; we test both.

**Allocation.** Given scores $\{\delta_u\}$ and budget $K$: **greedy** protects the top-$K$ units (plus heads); **magnitude proxy** protects top-$K$ by weight L1 mass; **uniform** protects none. Fake W4 weight-only elsewhere (`fake_except`) so ATE isolates sensitivity without requiring sm_120 packed kernels.

---

## 4. Method: A Pre-Registered Transfer Test

Because our initial single-sequence result (geometry beats magnitude by 3.6% on TUM mean) proved fragile once error bars were added (§6.2), we pre-registered the transfer question before spending further compute.

**Stage A gate (pre-declared).** Profile all 49 units on 5 stable sequences (TUM desk/360/floor, EuRoC MH_01/V1_01) under both measures, and compute pairwise Spearman rank correlation of the per-unit scores. **PASS** iff, for either measure, the mean cross-dataset (TUM×EuRoC) Spearman $\geq 0.3$. Only on PASS would we proceed to Stage B (pooled leave-one-sequence-out allocation over 20 sequences, success = beat both uniform and magnitude on held-out sequences at Bonferroni-corrected 99.2%). Gate and criterion are in `scripts/profile_gate.py` and `memory-bank/activeContext.md`, committed before the pilot finished.

**Outcome.** The gate **failed decisively** (§6.3). Per the pre-registration, Stage B was not run; this manuscript reports the negative result. Pre-registration matters here specifically because a post-hoc reader could otherwise suspect the null was reached by insufficient tuning — the criterion and stopping rule were fixed in advance.

---

## 5. Experimental Setup

| Item | Setting |
|------|---------|
| Rig | RTX PRO 6000 Blackwell, sm_120, CUDA 12.8, torch 2.11.0+cu128 |
| GPU pin | GPU-2 (Track-A ladder, through 2026-09-26); GPU-1 (rescue profiling, 2026-09-27, `OMP_NUM_THREADS=12`/stream; ATE verified identical to uncapped, `results/thread_check.log`) |
| System | MASt3R-SLAM, `single_thread` (deterministic) |
| Data | TUM RGB-D fr1 (9 seqs); EuRoC (5-seq shortlist for transfer; 11 seqs extracted) |
| Metric | ATE RMSE (Sim3 align, evo) |
| Uncertainty | per-sequence paired bootstrap (10k) + perturbation noise floor; **not** seeds (deterministic) |
| Logging | Per-EXP JSON with driver/CUDA/torch/quant-lib versions (`scripts/explog.py`) |

---

## 6. Results

### 6.1 PTQ ladder — TUM fr1 mean ATE

Multi-seed (seeds 0,1,2; `results/20260924-multiseed-*.json`). **Seed std $\approx 0$ for every config** ($\leq 0.03$ mm): MASt3R-SLAM with `single_thread` is deterministic given fixed weights, so seed repeats do not estimate run-to-run noise (we replace them with a bootstrap and a perturbation floor, §6.5).

| config | mean ATE (m) | seed std (m) | vs FP16 |
|--------|-------------:|-------------:|--------:|
| FP16 | 0.02955 | 0.00003 | +0.0% |
| FP8 e4m3 | 0.02941 | 0.00000 | −0.5% |
| W8A8 | 0.0315 | — | +6.8% |
| W4 geom-protect $K{=}7$ | 0.03163 | 0.00000 | +7.0% |
| W4 mag-protect $K{=}7$ | 0.03276 | 0.00000 | +10.9% |
| W4 uniform | 0.03332 | 0.00000 | +12.8% |

On the TUM *mean*, FP8 is effectively free and geometry protection recovers most of the W4 gap. Both impressions dissolve under per-sequence uncertainty (§6.2) and out of distribution (§6.4).

### 6.2 Allocation gains on TUM are not significant

Uncertainty is estimated across sequences: paired per-sequence bootstrap of relative ATE differences (normalized by FP16), 10k resamples over the 9 fr1 sequences (`scripts/paired_bootstrap_tum.py` → `results/paired_bootstrap_tum.json`).

| comparison | mean rel. diff | 95% CI | first better on |
|------------|---------------:|--------|----------------:|
| geom-K7 vs mag-K7 | −1.4% | [−11.1%, +8.1%] | 5/9 |
| geom-K7 vs uniform W4 | −4.4% | [−12.8%, +5.3%] | 7/9 |
| mag-K7 vs uniform W4 | −3.0% | [−10.5%, +3.5%] | 5/9 |

Every CI spans zero. The earlier "geometry beats magnitude by 3.6%" mean-headline is driven by a few sequences; per-sequence, no allocation gap is distinguishable from zero. A budget sweep over $K\in\{3,5,7,9,11\}$ tells the same story: no pairwise difference is significant (K9 vs K7 $-2.9\%$ [$-8.5$, $+1.4$]; K5 vs K7 $+0.3\%$ [$-4.8$, $+5.1$]), so the apparent $K{=}9$ optimum and $K{=}11$ regression are not supported (`results/multiseed_h3_ksweep_summary.txt`; Fig. `k_budget_pareto` is descriptive only).

### 6.3 Main result: per-unit sensitivity does not transfer

Pre-registered Stage A gate (`results/profile_gate.json`). Mean pairwise Spearman of per-unit scores:

| measure | TUM×EuRoC (gate) | within TUM | within EuRoC |
|---------|-----------------:|-----------:|-------------:|
| quantize-one (`loi`) | **+0.011** | −0.136 | −0.043 |
| protect-one (`loo`) | **+0.005** | −0.032 | −0.152 |

The gate required $\geq 0.3$ cross-dataset; observed correlations are $\approx 0$ for both measures, and are *slightly negative within a single dataset*. Top-7 protect-set overlap between sequences ranges 0–3 of 7 — chance level for choosing 7 of 49. **The gate fails decisively; no pooled or held-out profile can produce a transferable protect set.**

The mechanism is visible in a single unit. `dec_blocks2.8` — the top-sensitivity unit on `desk`, which anchored our original protect set — ranks **1st on desk, 3rd on MH_01, but 47th–49th of 49 on 360, floor, and V1_01**, where quantizing it is among the *least* harmful choices. Under the protect-one measure the interaction is even starker: restoring only `dec_blocks2.8` to FP16 with the rest at W4 makes `desk` **31% worse** than uniform W4, i.e. effects are non-additive. Sensitivity is a property of the *trajectory being run*, not of the unit.

This directly explains §6.4: any protect set is fit to whatever sequence produced it and is uncorrelated with the demands of a new sequence, so out of distribution it behaves like a random selection of protected units.

### 6.4 W4 does not transfer to EuRoC under any allocation

EuRoC 5-sequence shortlist, ATE RMSE (m), Sim3 (`results/step1_summary.json`, `20260924-euroc-shortlist-*.json`):

| seq | FP16 | FP8 | W4 uniform | W4 mag-K7 | W4 geom-K7 (TUM-fit) |
|-----|-----:|----:|-----------:|----------:|---------------------:|
| V1_01_easy | 0.0395 | 0.0366 | 0.0411 | 0.0420 | 0.0467 |
| MH_01_easy | 0.0227 | 0.0313 | 0.0605 | 0.0608 | 0.0518 |
| MH_02_easy | 0.0171 | 0.0225 | 0.0575 | 0.0474 | 0.0564 |
| V1_02_medium | 0.0190 | 0.0176 | 0.0244 | 0.0297 | 0.0312 |
| V2_01_easy | 0.0204 | 0.0292 | 0.0284 | 0.0183 | 0.0473 |
| **mean vs FP16** | **—** | **+15.6%** | **+78.5%** | **+66.9%** | **+96.6%** |

**All W4 variants collapse.** The TUM-fit geometry set is worst on the mean but mixed per sequence (it beats uniform on MH_01/MH_02); with 5 sequences "geometry is worse" is not supported, but "W4 does not transfer, with or without allocation" is — and §6.3 says why. Note that FP8 is **no longer free** here ($+15.6\%$; MH_01 $+38\%$, V2_01 $+43\%$): the single-sequence V1_01 spot-check that once looked fine ($-7.4\%$) was unrepresentative. Single-dataset compression numbers flatter.

### 6.5 It is signal, not noise

Because seeds carry no variance, we estimate run-to-run noise by perturbing FP16 trunk weights multiplicatively, $w(1+10^{-3}\epsilon)$, $\epsilon\sim\mathcal{N}(0,1)$ — about $20\times$ below FP8 rounding error — over 4 draws (`ptq.py` method=`noise`; `scripts/run_step1_noise_baselines.sh`). The floor is std over {FP16, 4 draws} relative to FP16.

- **Mean ATE:** std 0.76% (TUM), 0.76% (EuRoC).
- **Per sequence:** most TUM below 1% (desk 0.24%, floor 0.13%, 360 0.65%), EuRoC 0.3–1.9%; **desk2 (6.5%) and teddy (6.1%) are chaotic** and unreliable single-run.
- **Effect sizes:** 18–47 of 49 units (`loi`) and 28–44 (`loo`) have $|\delta_u| > 2\times$ the sequence's floor (`profile_gate.json`).

**Implications.** (i) The $+15.6\%$ FP8 and all W4 collapses (§6.4) are orders of magnitude above the floor — real. (ii) Most per-unit effects (§6.3) are individually real, yet uncorrelated across sequences: this is genuine sequence-specificity, not measurement scatter. (iii) The desk profile was fit on a *stable* sequence (0.24% floor), so its ranking is trustworthy *for desk* — which makes its failure to transfer all the more pointed. (iv) desk2/teddy should be excluded from any single-run fitting.

### 6.6 Efficiency: memory yes, fused FP8 latency no

Track-A SLAM ATE uses fake quant (no real speedup). Separate microbenches (`scripts/bench_fp8.py`, GPU-2):

| path | large $4096^3$ | trunk-like stack | notes |
|------|----------------:|-----------------:|-------|
| `torch._scaled_mm` | 1.61× | **0.30×** | weight mem **2.00×** (944→472 MB) |
| ModelOpt real FP8 GEMM | 0.75× | 0.12–0.17× | slower than FP16 |
| torchao Float8Dynamic | 0.91× | 0.13–0.14× | same |
| Transformer Engine 2.19 | — | — | no sm_120 wheel; source build failed |

Under D1 (no deploy) we report **memory** ($2\times$ for FP8 weights), not FLOP/latency speedup, which current Blackwell FP8 software does not deliver for trunk-like shapes.

---

## 7. Discussion

**What we set out to show, and what we found instead.** We hypothesized that closed-loop ATE sensitivity would be a better, transferable allocation signal than weight proxies. The signal turned out not to be transferable at all: per-unit sensitivity is sequence-specific and interaction-dominated at W4-scale perturbations. This reframes the compression-for-SLAM problem. It is not "which proxy best ranks the units?" — no ranking is stable — but "why does a small weight change to the same unit help one trajectory and barely touch another?" We conjecture the closed loop is the culprit: quantization error interacts with data-association and pose-graph decisions differently depending on the trajectory's geometry and motion, so a static per-unit sensitivity cannot exist. A fixed FP32 back-end isolates the trunk but does not linearize its effect on ATE.

**Practical guidance.** (i) Do not fit a protect set on one workload and expect transfer; validate allocation on held-out sequences before claiming a win. (ii) Report compression robustness on more than one dataset — TUM-only numbers overstate it (FP8: $-0.5\%$ vs $+15.6\%$). (iii) For deterministic SLAM, seeds are not a variance estimate; use a sequence bootstrap and a perturbation floor.

**What still holds.** FP8-e4m3 halves trunk weight memory with near-lossless ATE *on TUM*; uniform W4 is the honest W4 baseline (no allocation reliably beats it out of distribution).

---

## 8. Limitations

(i) One SLAM system (MASt3R-SLAM), one precision regime of primary interest (W4 weight-only, fake-quant); the instability may weaken at W8 or with QAT. (ii) Stage A profiled 5 sequences per measure (49 units each) plus full TUM/EuRoC allocation runs; more sequences would tighten the correlation CIs but the observed $\approx 0$ leaves little room. (iii) Fake quant isolates accuracy, not wall-clock W4. (iv) FP8's EuRoC degradation is not yet decomposed into dataset shift (grayscale, motion blur) vs. calibration. (v) Asymmetric-precision arm cut; distillation dropped (D4). (vi) No edge deploy (D1); memory is the reported efficiency axis.

---

## 9. Conclusion

Compressing the GFM trunk of feed-forward SLAM by *closed-loop sensitivity* rests on the assumption that some units are reliably more fragile than others. For W4 on MASt3R-SLAM across TUM and EuRoC, that assumption is false: per-unit ATE sensitivity is uncorrelated across sequences (mean cross-dataset Spearman $+0.01$), individually real but sequence-specific, so no fitted or pooled protect set transfers and no allocation reliably beats uniform quantization out of distribution. FP8 halves weight memory near-losslessly on TUM but costs $+15.6\%$ on EuRoC, a reminder that single-dataset compression results flatter. We also show that deterministic SLAM defeats seed-based variance estimation and give a bootstrap-plus-perturbation-floor methodology in its place. The useful takeaway for the field is negative and specific: measure sensitivity if you like, but do not assume it transfers — validate allocation out of distribution, because for closed-loop SLAM the sensitivity signal lives in the trajectory, not the weights.

---

## Appendix

- **A.** Per-unit profiles: `results/profiles/profile_{loi,loo}_<seq>.txt`; desk `loi` from `20260923-phase2-sens-w4-full-fr1desk.json`. Gate: `results/profile_gate.json`.
- **B.** TUM multi-seed + K-sweep: `results/20260924-multiseed-*.json`, `multiseed_h3_ksweep_summary.txt`; bootstrap `results/paired_bootstrap_tum.json`.
- **C.** EuRoC shortlist + matched W4 baselines: `results/step1_summary.json`, `20260924-euroc-shortlist-*.json`.
- **D.** Noise floor: `results/multiseed_tum_noise_*`, `step1_summary.json`. Thread-cap equivalence: `results/thread_check.log`.
- **E.** Efficiency microbench: `results/20260924-benchfp8-*.json`.
- **F.** Reproducibility: `memory-bank/techContext.md`; `--no-deps` install landmine for quant stacks; SDPA-only attention.
