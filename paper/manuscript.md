# SLAM-Aware Mixed-Precision Compression of Geometric Foundation Models for Feed-Forward SLAM

> **Status:** skeleton / outline. Each section carries scope notes, which results/figures
> feed it, and TODOs. Numbers are pulled from `results/` and `figures/`; verify before
> submission. Target venue: vision/geometry (D2). Rig-only, no edge deploy (D1).

**Authors:** TBD
**Target:** CVPR/ICCV/3DV-class or equivalent
**One-line claim:** Allocating quantization bits by *measured downstream SLAM sensitivity*
(ATE/pointmap), not by weight-magnitude/Hessian proxies, gives a better accuracy–cost
trade-off when compressing the GFM trunk of feed-forward SLAM, with the classical back-end
held fixed for a clean ablation.

---

## Abstract
<!-- Write last. ~200 words. -->
- **Context:** feed-forward SLAM (MASt3R-SLAM) leans on a heavy geometric foundation model (GFM) trunk; deployment cost dominated by that forward pass.
- **Gap:** existing FM quantization optimizes reconstruction/perplexity proxies, not the SLAM metric that matters (trajectory + map error).
- **Method:** PTQ ladder + SLAM-aware mixed-precision bit allocation; back-end frozen FP32.
- **Result:** FP8-e4m3 matches FP16 within −0.4% mean ATE on TUM fr1; geometry-aware W4 protection beats a magnitude proxy by 3.6% rel. mean ATE at equal bit budget (K=7).
- **Takeaway:** downstream-aware allocation is the right objective for compression-for-SLAM.

**TODO:** finalize once §6 Pareto + energy/latency numbers land.

---

## 1. Introduction
- Feed-forward / GFM-based SLAM: what it is, why the trunk is the cost center.
- Compression is standard for LLMs/ViTs but the *objective* is transplanted blindly to geometry tasks.
- **Thesis:** the compression objective should be the *closed-loop* SLAM metric, measured, not proxied.
- **Contributions** (reframed 2026-09-23 after scoop-watch — task-aware MP for GFMs already exists, so we lead on the SLAM loop + allocator, not on "downstream-aware is new"):
  1. **Closed-SLAM-loop metric as the sensitivity signal** — bits allocated by measured **ATE/pointmap** with the classical back-end held **fixed FP32**. Prior GFM quant (QVGGT) stops at the forward-pass pose head and never closes the loop. **Primary wedge.**
  2. **Algorithmic allocator at matched budget** — greedy/ILP (§4.3) with a **head-to-head vs magnitude proxy and uniform** at equal K (H3); prior work hand-picks fragile blocks heuristically.
  3. Full-SLAM compression study on commodity Blackwell (MASt3R-SLAM): ATE/RPE + pointmap vs latency/peak-mem/energy Pareto. *(accuracy done; latency/energy PENDING — see §6.5 gap)*

**RISK (novelty.md R5): PARTIALLY SCOOPED.** QVGGT (2605.31124) and Mix-QVLA (2606.19565) already do task-aware mixed-precision. Distinction now rests on the three points above; do NOT claim downstream-aware allocation itself is new. See §2.

---

## 2. Related Work
<!-- Source: memory-bank/novelty.md (scoop-watch 2026-09-23). Verify all dates/claims before citing. -->
- **Task-aware MP for geometry FMs (closest prior — MUST engage head-on):**
  - **QVGGT** (2605.31124): PTQ W4A16 for VGGT; per-block sensitivity via downstream pose-head accuracy (AUC@30); keeps fragile blocks FP16. *Our distinction:* closes the full SLAM loop and reports ATE with a fixed FP32 back-end; uses an algorithmic (greedy/ILP) allocator with a matched-budget proxy ablation, vs QVGGT's hand-picked blocks.
  - **Mix-QVLA** (2606.19565): task-evidence-aware MP for VLA models — same philosophy, robotics domain. Cite as prior that task-aware > proxy.
- **GFM PTQ methods (baselines):** Quantized VGGT (2509.21302, ICLR'26; activation heavy-tails/calibration), VersaQ-3D (2601.20317, W4A4 VGGT + transform coding, reconstruction-only), VGGT-X (2509.25191, memory-efficient bf16).
- **Token/frame reduction (orthogonal efficiency):** Co-Me (2511.14751, confidence-guided token merging), LeanGate (2604.08718, feed-forward frame gating, 5×). Complementary — combinable with our quant, not competing.
- **Proxy-based MP (the thing we argue against):** MXSens (2607.17733, Hessian-guided), HAWQ-style, GPTQ, AWQ — contrast surrogate/Hessian objective vs measured closed-loop objective.
- **Distillation:** GFM distillation (2607.01851) — dropped (D4).
- **Distinction paragraph:** none optimize or report *full-SLAM ATE* under compression with a fixed classical back-end via an algorithmic downstream allocator.

**TODO:** re-run scoop-watch pre-submission (field moves monthly); keep novelty.md current.

---

## 3. Preliminaries & Problem Setup
- MASt3R-SLAM pipeline: GFM trunk (ViT + fusion/attention + regression heads) → classical back-end (pose graph, Sim(3)/SL(4) opt, loop closure).
- **Scope (CLAUDE.md):** compress trunk forward pass only; back-end stays FP32/CPU → fixed back-end = any metric delta attributable to trunk compression.
- Quantization notation: per-unit precision assignment; "unit" = quantizable block (define exactly, ref `scripts/allocate_bits.py`).
- Problem: choose per-unit bit-widths to minimize SLAM error under a bit-budget K.

---

## 4. Method

### 4.1 PTQ Ladder
- Configs: FP16 baseline → FP8-e4m3 (trunk) → W8 → W8A8 → W4. (`configs/*.yaml`)
- PTQ machinery: `scripts/ptq.py`. Calibration protocol, SDPA-only attention path (no flash-attn).

### 4.2 Sensitivity Profiling
- Per-unit sensitivity = measured ATE/pointmap degradation when that unit is dropped to low precision, others held high.
- Output: sensitivity profile → `figures/sensitivity_heatmap.{png,pdf}`, `results/sens_w4_full_units.txt`.

### 4.3 SLAM-Aware Bit Allocation (primary novelty)
- Given per-unit sensitivity + budget K, protect the K most SLAM-sensitive units at higher precision.
- Allocators (`scripts/allocate_bits.py`): **greedy**, **ILP** (matched budget), **magnitude-L1** (proxy baseline).
- Contrast with magnitude/Hessian: allocation driven by *downstream geometry*, not weight statistics.

### 4.4 (Stretch) SLAM-Structural Asymmetric Precision
- Novel wrinkle #2 (plan §6.6): precision asymmetry aligned to SLAM structure. **Status: not run.** Mark as future work unless time permits.

---

## 5. Experimental Setup
- **Rig:** RTX PRO 6000 Blackwell, sm_120, CUDA 12.8, torch 2.11.0+cu128, GPU-2 pinned. (techContext.md)
- **Datasets:** TUM RGB-D fr1 (primary), EuRoC V1_01_easy (transfer spot-check). *(EuRoC full 11-seq deferred.)*
- **Metrics:** ATE RMSE (primary), RPE, pointmap error; latency / peak-mem / energy-per-frame via NVML *(PENDING)*.
- **Logging discipline:** per-EXP JSON with driver/CUDA/torch/quant-lib versions, seed, config hash, capability (12,0), device. (`results/*.json`, `scripts/explog.py`)
- **Reproducibility:** seeds, config hashes, `--no-deps` install landmine noted.

---

## 6. Results

### 6.1 Baseline & PTQ Ladder — TUM fr1 mean ATE
<!-- Source: figures/shortlist_table.md, results/20260922-*, 20260923-* -->

| config | mean ATE (m) | vs FP16 | role |
|--------|-------------:|--------:|------|
| FP16 | 0.0295 | +0.0% | baseline |
| FP8 e4m3 | 0.0294 | −0.4% | accuracy champ |
| W8A8 | 0.0315 | +6.8% | uniform |
| W4 geom-protect K=7 | 0.0316 | +7.1% | method |
| W4 mag-protect K=7 | 0.0328 | +10.9% | proxy baseline |
| W4 uniform | 0.0333 | +12.8% | uniform |

### 6.2 SLAM-Aware Allocation vs Proxy (H3, matched K=7)
<!-- Source: figures/shortlist_table.md, results/*protect*, 20260922-phase4-* -->

| allocator | mean ATE | vs FP16 | vs uniform W4 |
|-----------|---------:|--------:|--------------:|
| Geometry greedy | 0.0316 | +7.1% | −5.1% |
| Magnitude L1 | 0.0328 | +10.9% | −1.7% |
| Uniform W4 | 0.0333 | +12.8% | — |

**Headline:** geometry beats magnitude by 3.6% rel. mean ATE at equal K=7 budget.

### 6.3 Sensitivity Profile
- `figures/sensitivity_heatmap.{png,pdf}` — which trunk units are SLAM-critical. Discuss structure (heads vs trunk layers).

### 6.4 Cross-Dataset Transfer (EuRoC V1_01_easy)
<!-- Source: results/20260923-phase2-euroc-* -->

| config | RMSE (m) | vs FP16 (0.0395) |
|--------|---------:|-----------------:|
| FP8 | 0.0366 | −7.4% |
| W4 geom-greedy K=7 | 0.0467 | +18.0% |

- FP8 transfers cleanly; W4-protect softer on this seq than TUM mean. Discuss why (seq characteristics, calibration transfer).

### 6.5 Efficiency Pareto (accuracy vs latency / mem / energy)
- `figures/shortlist_pareto_bars.{png,pdf}`.
- **GAP:** real latency/energy not yet measured (torchao/modelopt speedup gate open — mxfp8/cutlass .so load failures). Currently accuracy + peak-mem only. **Must close before claiming Pareto (D5 contribution).**

---

## 7. Discussion & Limitations
- Why downstream-aware wins: proxy objectives misrank geometry-critical units.
- **Limitations:** single primary dataset for allocation; EuRoC = 1 seq; latency/energy pending; asymmetric-precision arm unrun; distillation dropped (D4).
- Threats to validity: allocation may overfit TUM fr1 sensitivity; K=7 budget point (need budget sweep / Pareto over K).

---

## 8. Conclusion
- SLAM-aware bit allocation is the correct objective for GFM-for-SLAM compression; measured beats proxied. Recap headline deltas. Future: full EuRoC, energy Pareto, asymmetric precision, pruning.

---

## Appendix
- A. Full per-unit sensitivity table (`results/sens_w4_full_units.txt`).
- B. Allocator details (greedy/ILP/magnitude), matched-budget proof (`results/protect_*`).
- C. Reproducibility: env, versions, config hashes, seeds; Blackwell verify + `--no-deps` landmine.
- D. Per-sequence ATE logs.

---

## Submission blockers (track to zero)
1. [~] Novelty verification — DONE 2026-09-23: partially scooped (QVGGT, Mix-QVLA). Wedge reframed to closed-loop-ATE + fixed back-end + algorithmic allocator (§1, §2). Re-run pre-submission.
2. [ ] Latency/energy measurements — close speedup gate, fill §6.5 Pareto.
3. [ ] Budget sweep over K (not just K=7) for a real Pareto curve.
4. [ ] More EuRoC sequences (transfer claim rests on 1 seq).
5. [ ] Decide fate of asymmetric-precision arm (§4.4): run or cut.
