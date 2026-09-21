# Project plan — Quantized / efficient Geometric Foundation Models for edge SLAM

**Working title:** *Edge-GFM-SLAM: compressing the geometric backbone of feed-forward SLAM under an accuracy–latency–energy budget.*

**Rig:** `vm-130-131` (see `hardware-office-vm-130-131.md`). Single Blackwell GPU (physical **GPU-1**, `sm_120`), CUDA 12.8 via torch wheels, SDPA-only, ~96 GiB VRAM, ~1.5 TiB free.

**Status:** DRAFT v0.1 — grounded against literature as of 2026-09-22. Read §0 and §1 before committing.

**Decisions LOCKED (2026-09-22):** D1 = **no deploy for now** (rig-only Blackwell; Track B / Jetson deferred to a future Phase 3) · D2 = vision/geometry venue paper (3DV/WACV class), ~3 mo, method wrinkle required · D3 = **MASt3R-SLAM primary**, VGGT-SLAM large arm deferred · D4 = PTQ-first → pruning; distillation dropped · D5 = real-time = target, **Pareto frontier = contribution**. D1 pivots the wedge onto the §6.3 downstream-aware bit allocation. See `memory-bank/activeContext.md`.

---

## 0. Decisions I need from you (these change the plan materially)

| # | Decision | Why it matters | My default if you don't answer |
|---|----------|----------------|--------------------------------|
| D1 | **Which Jetson is the deploy target — and do you physically have one yet?** Thor (Blackwell) / AGX Orin / Orin NX / Orin Nano / none-yet | Thor shares your rig's Blackwell arch + native FP4/FP8 → clean prototyping story. Orin is Ampere (no native FP4, INT8-centric) → your desktop FP4 work won't transfer, INT8 becomes the headline. "None yet" means the deploy numbers are a Phase-3 procurement gate. | **Jetson AGX Thor** as primary target; Orin NX as a secondary "tighter budget" point if available. |
| D2 | **Deliverable + venue/timeline.** Conference paper (ICRA/IROS/CoRL/RSS/3DV — which deadline?), thesis chapter, or internal benchmark? | Sets rigor, #datasets, whether a novel *method* is required vs. a strong empirical study. | Target a **robotics/vision venue paper**, ~3–4 month runway, needs one novel method wrinkle + strong Pareto study. |
| D3 | **Primary SLAM system.** MASt3R-SLAM vs VGGT-SLAM as the anchor. | MASt3R-SLAM is lighter/real-time-oriented (better edge fit); VGGT-SLAM is heavier (more to compress, closer to VersaQ-3D's turf). | **MASt3R-SLAM primary**, VGGT/VGGT-SLAM as the "large-GFM" arm. |
| D4 | **Method ambition.** PTQ-only, PTQ+pruning, or full PTQ+pruning+distillation. | Distillation needs training data + compute + weeks; PTQ is days. | **PTQ-first**, pruning second, distillation as a stretch arm. |
| D5 | **Is real-time (≥ ~10 FPS on edge) a hard requirement, or is the contribution the Pareto characterization?** | Changes whether "we hit real-time" is a claim or a finding. | Treat **real-time as a target, the Pareto frontier as the contribution.** |

Everything below is written so it's actionable under the defaults, and flags where a different answer forks the plan.

---

## 1. Honest novelty assessment (read this first)

Your source framing ("model-compression of the GFM itself for edge is a near-empty corner, ~1–2 years less saturated") **was true in ~2024–2025 but is filling in fast during 2026.** Direct or near-direct neighbours now exist:

- **VersaQ-3D** (arXiv 2601.20317, ~mid-2026) — *first calibration-free, input-agnostic quantization for VGGT*, W4A4, transform-coding to tame activation outliers. **But**: it's an algorithm–**architecture co-design** (custom accelerator vs. Jetson Orin/Xavier), evaluated on reconstruction (Co3Dv2 pose + pointmap), **not** a full SLAM system with ATE on trajectory benchmarks.
- **Co-Me** (arXiv 2511.14751) — confidence-guided **token merging** for VGGT/Pi3, up to 21.5× speedup, training-free. This largely **claims the "token pruning" sub-corner** you listed.
- **VGGT-X** (arXiv 2509.25191) — memory-efficient VGGT (bf16, drop redundant intermediate features), for dense NVS.
- **Fast-FoundationStereo** (NVIDIA, arXiv 2512.11130) — KD + blockwise NAS + structured pruning to make a *stereo* foundation model real-time. Strong methodological template, different task.
- **GFM distillation for domain reconstruction** (e.g. lunar, arXiv 2607.01851) and a wave of efficiency-oriented VGGT variants (Streaming-4D-VGGT, Mamba-VGGT, VGGT-World @ 0.43B).

**What you still own (the defensible wedge):**

> A clean, reproducible, **commodity-hardware** study that compresses the GFM backbone **inside a full feed-forward SLAM loop** (MASt3R-SLAM / VGGT-SLAM) and characterises the **pose-accuracy (ATE/RPE) and pointmap-quality vs. latency / peak-memory / energy** Pareto frontier — **on the actual edge deploy target (Jetson), via the real deployment path (TensorRT / NVFP4-FP8)**, not a simulator or a custom ASIC.

Nobody in the list above does *all* of: (a) full SLAM ATE, (b) stock edge silicon, (c) energy-per-frame, (d) the desktop→TensorRT deployment reconciliation. That is a genuine, useful contribution — but note it's **more of a systems/empirical contribution than a "new quantizer" contribution.** To lift it to a top venue, add **one method wrinkle** (see §6.3 / §6.6). Candidate wrinkles:

1. **Downstream-aware (SLAM-aware) mixed-precision bit allocation** — allocate bits per layer/head by measured *ATE/pointmap* sensitivity, not by weight-Hessian/perplexity proxy (which is what LLM PTQ optimises). Solve a budgeted bit-allocation given a per-block sensitivity profile.
2. **SLAM-structural asymmetric precision** — keyframes (persistent map) at higher precision, tracking frames (transient localisation) at lower precision; or confidence-guided per-region precision. Exploits SLAM structure that pure per-image quantizers ignore.

Validate that neither wrinkle is already published before you lean on it (see §9 risk R5).

---

## 2. Problem statement & research questions

**Setup.** Feed-forward GFM-SLAM systems replace hand-built front-ends with a transformer that regresses pointmaps / matches / camera params (DUSt3R → MASt3R → VGGT lineage), wrapped in a light classical back-end (pose graph, Sim(3)/SL(4) optimisation, loop closure). The **GFM forward pass dominates compute and memory**; the back-end is comparatively cheap and stays FP32/CPU.

**RQs.**

- **RQ1 (feasibility/limits):** How far can the GFM backbone of a SLAM system be compressed (weight-only → weight+activation → FP8 → FP4/INT4, + pruning/distillation) before ATE/pointmap degrade beyond a usable threshold?
- **RQ2 (frontier):** What is the accuracy–latency–memory–energy Pareto frontier on edge silicon, and which compression configs are Pareto-dominant?
- **RQ3 (transfer):** Do desktop PTQ accuracy results predict on-Jetson TensorRT accuracy, or does the deployment path (different kernels/calibration) shift the frontier?
- **RQ4 (method):** Does SLAM-aware / downstream-aware bit allocation beat off-the-shelf LLM-style PTQ at equal budget for geometric metrics?

**Hypotheses to pre-register (so results are interpretable either way):**

- H1: **Weight-only INT8/INT4** on the ViT trunk is near-lossless for ATE; **activation quantization (esp. 4-bit)** is where geometry heads break — mixed precision (heads in FP16/FP8) recovers most of it.
- H2: On **Blackwell (rig + Thor)**, FP8 gives near-baseline accuracy with real speedup; NVFP4 is the aggressive frontier point.
- H3: Downstream-aware allocation shifts the frontier out vs. uniform / Hessian-only allocation at matched memory.

---

## 3. System & scope decisions

**What you compress:** the **GFM forward pass** — the ViT encoder/backbone, the fusion/attention blocks, and the regression heads (pointmap / depth / camera-token / matching). This is the >90% of latency and VRAM.

**What you do NOT touch (keep as reference):** the classical back-end — MASt3R-SLAM's pointmap matching + pose graph, or VGGT-SLAM's SL(4)-manifold optimisation, loop closure, global alignment. Compressing these is a different (crowded) "map-compression" problem and out of scope. Keeping them fixed makes the ablation clean: *any* metric change is attributable to GFM compression.

**Two measurement tracks (this is the crux of the experimental design):**

- **Track A — desktop accuracy (fast iteration, GPU-1).** PyTorch-side PTQ (bitsandbytes / GPTQ(Model) / AWQ / torchao / TensorRT-ModelOpt in fake-quant mode) to measure *accuracy* (ATE, pointmap) vs bit-width, per-layer sensitivity, mixed-precision configs. Cheap; single GPU is plenty.
- **Track B — edge deployment (latency/energy, on Jetson).** Export GFM → ONNX → **TensorRT** engine (FP16/INT8/FP8/NVFP4 via TensorRT-ModelOpt calibration) → run inside the SLAM loop on the Jetson → measure latency, peak unified memory, and **energy/frame** (`tegrastats`/`jetson-stats`).

**Track A ≠ Track B.** bitsandbytes/GPTQ kernels and calibration are *not* the same as TensorRT's. Weight-only results transfer reasonably; activation-quant results must be **re-validated for accuracy on the Jetson deployment**. RQ3 is exactly about this gap — make it a feature, not a bug.

**Anchor systems.** MASt3R-SLAM (pairwise, real-time-oriented, Davison lab, CVPR 2025) as the primary edge target; VGGT (1.2B, CVPR 2025) + VGGT-SLAM (SL(4) manifold, Maggio et al.) as the "large-GFM" arm where compression gains are largest and the story connects to VersaQ-3D.

---

## 4. Hardware & environment

### 4.1 Your rig (baked-in constraints from `hardware-office-vm-130-131.md`)

- **Pin GPU-1 always:** `export CUDA_VISIBLE_DEVICES=1` (GPU-0/2 are shared — do not use). Merge/quantize/eval all on GPU-1.
- **`sm_120` (Blackwell), CUDA 12.8, torch 2.11.0+cu128.** Verify capability `(12, 0)` before every EXP.
- **SDPA only — no flash-attn.** Any attention-quant path must respect the SDPA code path; do not install flash-attn.
- **Install landmine (same as your VLA work):** blind `pip install -e` on the SLAM repos may pull `torch==2.2.0` and break `sm_120`. Use `--no-deps`, then curated deps, then **re-verify cu128 + matmul on GPU-1**.
- **Disk OK now** (~1.5 TiB free) but Docker holds ~200 GiB of other images — datasets (TartanAir/ScanNet are large) go under `/office/dev_workspace/morshed`, not root free space you assume is infinite.
- **96 GiB VRAM is generous** — VGGT (1.2B) inference and even light QAT/distillation of the trunk fit comfortably single-GPU. Compute is *not* your bottleneck; **Jetson access + TensorRT export engineering is.**

### 4.2 Edge target options

| Target | Arch | Native low-precision | Mem | Power | Notes |
|--------|------|----------------------|-----|-------|-------|
| **Jetson AGX Thor** (T5000) | **Blackwell** | **NVFP4 + FP8** + INT8 | 128 GB LPDDR5X | 40–130 W | *Same family as your rig.* Cleanest prototyping→deploy story. JetPack 7.x, 2nd-gen Transformer Engine. |
| Jetson T3000 / T2000 | Blackwell | NVFP4/FP8/INT8 | 32 / 16 GB | lower | Announced 2026-07, GA ~Q1 2027 — likely too late unless timeline is long. |
| Jetson AGX Orin 64GB | Ampere | FP16 / INT8 | 64 GB | 15–60 W | No native FP4. INT8 becomes the headline; desktop FP4 work won't transfer. |
| Jetson Orin NX 16GB | Ampere | FP16 / INT8 | 16 GB | 10–40 W | Tight budget point; good "can it even fit" stress test. This is VersaQ-3D's baseline device. |

**If D1 = Thor:** your whole desktop FP8/NVFP4 characterisation is *architecturally faithful* to deployment — strong. **If D1 = Orin:** pivot the headline to INT8 (+ weight-only INT4), treat FP4 as "future Blackwell-edge" discussion, and expect a larger Track-A→Track-B accuracy shift.

### 4.3 Tooling risk gates (verify in Phase 0, before committing to a method)

Check *on GPU-1* which of these actually build/run `sm_120` kernels **with real (not just fake) speedup**:

- `bitsandbytes` (INT8, NF4) — Blackwell kernel support has been catching up; log the exact version.
- `GPTQModel` (maintained AutoGPTQ successor) / `AutoAWQ` — Marlin/Machete 4-bit kernels; verify sm_120.
- `torchao` — native FP8 / low-bit; check Blackwell path.
- **NVIDIA TensorRT-ModelOpt** (`nvidia-modelopt`) — FP8/NVFP4/INT8 PTQ+QAT, the bridge to TensorRT on both rig and Thor. This is the most deployment-relevant; prioritise it.

**Expected reality:** some PyTorch-side quant libs may give you *accurate fake-quant* but no real desktop speedup on `sm_120` yet. That's fine — accuracy comes from Track A, latency/energy from Track B (TensorRT). Just don't quote desktop wall-clock as your speedup number.

---

## 5. Datasets & metrics

### 5.1 Datasets

| Dataset | Role | GT available | Use |
|---------|------|--------------|-----|
| **TUM-RGBD** (fr1/2/3) | Primary ATE | pose + depth | Headline trajectory accuracy; matches MASt3R-SLAM/DROID reporting. |
| **EuRoC MAV** (MH, V1/V2) | ATE, harder motion | pose (Vicon/Leica) | Aggressive motion, mono from stereo. |
| **7-Scenes** | Pointmap + pose | pose + depth | Small indoor pointmap accuracy. |
| **Replica** | Pointmap acc/completeness | dense mesh | Best geometry GT for pointmap metrics. |
| **ScanNet** (subset) | Real indoor generalisation | pose + depth | Sensitivity to real-world texture/noise. |
| **TartanAir** / **KITTI** | Stress / scale | pose (+ depth) | Large-scale, outdoor stress point. Large download — plan disk. |

**Calibration set for PTQ:** a few hundred frames drawn from **held-out** sequences (a different scene/dataset than any eval sequence). Never calibrate on eval sequences. Log exactly which frames.

### 5.2 Metrics

**Trajectory (monocular GFM-SLAM is up-to-scale → align first):**
- **ATE RMSE** after **Sim(3)** alignment (scale+rot+trans); report per-sequence and mean.
- **RPE** (relative pose error) for drift over fixed windows.
- **Tracking success / completeness** — fraction of frames tracked; a config that "wins" ATE by dropping hard frames is not a win.

**Geometry / pointmap:**
- **Accuracy & completeness** (mean/median distance to GT mesh, Replica/7-Scenes).
- **Depth metrics** where available: abs-rel, RMSE, δ<1.25.
- **Normal consistency** (optional).

**Efficiency (the "edge" headline):**
- **Latency:** ms/frame and ms/keyframe; FPS. Report GFM-forward latency *and* end-to-end (incl. back-end).
- **Peak memory:** VRAM (rig) / peak unified memory (Jetson).
- **Model size:** MB on disk / in-mem.
- **Energy:** **J/frame** and avg power — `nvidia-smi`/NVML on rig; `tegrastats`/`jetson-stats` on Jetson. **Energy/frame is the metric that makes this an "edge" paper**, not just a speed paper.

**Evaluation protocol:** fixed seeds; **≥3 runs** per config (SLAM has nondeterminism); report mean ± std. Same input resolution / #views across configs unless that's the variable. Use the systems' official eval scripts for ATE where they exist, to stay comparable to published baselines.

---

## 6. Methods

### 6.1 Baselines (non-negotiable first)
Reproduce the **FP16/BF16** baseline ATE + pointmap for MASt3R-SLAM (and VGGT-SLAM) on TUM + EuRoC on GPU-1, matching published numbers within a small tolerance. **If you can't reproduce baseline, no compression result is meaningful.** This is the Phase-1 gate.

### 6.2 PTQ ladder (weakest → strongest pressure)
1. **W8 weight-only** (INT8) on ViT trunk, heads in FP16. Expect ~lossless.
2. **W4 weight-only** (INT4 via GPTQ/AWQ) on trunk, heads FP16. Watch pointmap.
3. **W8A8** (weight+activation INT8). First real activation-quant test.
4. **FP8 (e4m3)** weight+activation — Blackwell-native; expect strong accuracy+speed on rig/Thor.
5. **W4A4 / NVFP4** — the aggressive frontier; expect geometry-head breakage → needs mixed precision + outlier handling (cf. VersaQ-3D's transform coding).
6. **Mixed precision (default winner, H1):** trunk low-bit, **regression/camera-token heads + first/last blocks kept FP16/FP8.** VGGT-X's "FP32 only for critical MLP heads" supports this.

### 6.3 Downstream-aware bit allocation (novel wrinkle #1)
- **Sensitivity sweep:** quantize one block/head at a time to target bit-width, measure **ATE/pointmap** delta on a dev set → per-block sensitivity profile.
- **Allocation:** greedy or ILP to assign bit-widths under a memory/latency budget, minimising predicted geometric error. Contrast against uniform and Hessian/perplexity-proxy allocation (the LLM default).
- **Claim to test (H3):** matched-budget, geometry-metric-driven allocation Pareto-dominates off-the-shelf PTQ for SLAM.

### 6.4 Pruning
- **Token merging/pruning:** **Co-Me already owns confidence-guided token merging** — either reuse it as a *baseline/component* (cite, don't reinvent) or differentiate (e.g., precision-not-merging, or SLAM-loop-aware token budgeting). Don't claim it as novelty.
- **Structured channel/head pruning** of the ViT trunk + light fine-tune to recover. Composes with quantization (prune → quantize order, cf. "Prune-Quantize-Distill" ordering results).

### 6.5 Distillation (stretch arm — D4)
- Distill the ViT trunk into a smaller student that preserves pointmap + pose + matching outputs (Fast-FoundationStereo-style KD, adapted to GFM outputs). Highest effort/risk; needs data + single-GPU training days. Gate behind a positive PTQ+pruning result.

### 6.6 SLAM-structural asymmetric precision (novel wrinkle #2)
- **Keyframe vs tracking-frame precision:** keyframes (build persistent map) at higher precision; tracking frames (transient) at lower precision. Measure whether map quality is preserved while average per-frame cost drops.
- **Confidence-guided precision:** use the GFM's per-pixel confidence to route high-confidence regions to lower precision. (Differentiate from Co-Me, which merges tokens rather than varying precision.)

---

## 7. Phased plan with gates

Each phase has **entry** → **work** → **exit gate** → **deliverable**. Do not pass a gate on hope.

### Phase 0 — Bring-up & tooling gates (est. 3–5 days)
- Clone MASt3R-SLAM (+ VGGT/VGGT-SLAM) into `~/vla`-style workspace; install with `--no-deps` + curated deps; **re-verify `sm_120` + matmul on GPU-1** (the torch==2.2 landmine).
- Stand up datasets (TUM, EuRoC first) under workspace disk.
- Run the §4.3 tooling gates: which of bnb / GPTQModel / AWQ / torchao / TensorRT-ModelOpt actually run on `sm_120`.
- **Exit gate:** GFM forward runs on GPU-1 at expected FP16 speed; ≥2 viable quant toolchains confirmed; datasets load.
- **Deliverable:** `phase0_bringup.sh` (gated, like your existing pattern) + a tooling-support matrix note.

### Phase 1 — Baseline reproduction (est. 3–5 days)
- Reproduce FP16 ATE (TUM+EuRoC) and pointmap (Replica/7-Scenes) for the primary system; log versions/seeds.
- **Exit gate:** baseline ATE within tolerance of published numbers on GPU-1.
- **Deliverable:** baseline results table + `EXP` log entries.

### Phase 2 — Desktop PTQ accuracy sweep, Track A (est. 1.5–2 weeks)
- Run the §6.2 ladder + §6.6 mixed precision; build per-config accuracy table + the §6.3 sensitivity profile.
- **Exit gate:** at least one compressed config within an agreed ATE tolerance (e.g., ≤10–15% ATE RMSE increase) at meaningfully reduced size.
- **Deliverable:** accuracy-vs-bitwidth curves; sensitivity heatmap; shortlist of deploy candidates.

### Phase 3 — Deployment bridge, Track B on Jetson (est. 2–3 weeks; **highest engineering risk**)
- GFM → ONNX (handle dynamic shapes / variable views / custom ops) → TensorRT engine (FP16/INT8/FP8/NVFP4 via ModelOpt) → run inside SLAM loop on Jetson.
- Measure latency, peak unified memory, **energy/frame**; **re-validate accuracy on-device** (RQ3).
- **Exit gate:** ≥1 compressed config runs end-to-end on the Jetson with measured latency+energy and on-device ATE.
- **Deliverable:** Track-A↔Track-B reconciliation table (the RQ3 result — publishable on its own).

### Phase 4 — Novel method (est. 2–3 weeks)
- Implement + evaluate §6.3 (downstream-aware allocation) and/or §6.6 (asymmetric precision). Ablate vs uniform/off-the-shelf at matched budget.
- **Exit gate:** method beats the best off-the-shelf PTQ config on the accuracy–efficiency frontier (or a clean negative result with explanation).
- **Deliverable:** method section + ablation tables.

### Phase 5 — Pruning / distillation (est. 1–3 weeks, scope by D4)
- Structured pruning (+ recover); optional distillation arm. Compose with best quant config; test ordering.
- **Exit gate:** composition improves the frontier over quant-only, or is shown not to.

### Phase 6 — Full Pareto, ablations, write-up (est. 2–3 weeks)
- Assemble the accuracy–latency–energy–memory frontier across all configs, both devices; generalisation across datasets; failure analysis (where/why geometry breaks).
- **Deliverable:** paper draft + reproducible artifact.

---

## 8. Experiment matrix (fill as you go)

Config axis × dataset axis × metric axis:

- **Configs:** FP16 (baseline) · W8 · W4 · W8A8 · FP8 · NVFP4/W4A4 · +mixed-precision-heads · +downstream-aware-alloc · +asym-precision · +pruned · +distilled.
- **Systems:** MASt3R-SLAM (primary) · VGGT/VGGT-SLAM (large arm).
- **Datasets:** TUM · EuRoC · 7-Scenes · Replica · (ScanNet · TartanAir/KITTI stretch).
- **Devices:** rig GPU-1 (accuracy + reference latency) · Jetson (latency + energy + on-device accuracy).
- **Metrics:** ATE RMSE · RPE · tracking success · pointmap acc/comp · depth abs-rel/δ · latency (fwd + e2e) · peak mem · size · **J/frame**.

Not every cell must be filled — prioritise: full config ladder on TUM+EuRoC (both devices), then breadth on remaining datasets for the shortlisted configs.

---

## 9. Risks & mitigations

| ID | Risk | Likelihood | Mitigation |
|----|------|-----------|------------|
| R1 | **`sm_120` quant kernels immature** → no real desktop speedup | High | Split tracks: accuracy on desktop (fake-quant OK), latency/energy on Jetson via TensorRT. Don't quote desktop speedup. |
| R2 | **ONNX/TensorRT export of GFM is painful** (dynamic shapes, custom ops, variable views) | High | Budget real time (Phase 3). Start export early with a fixed-shape single-pair case; expand. Consider torch-TensorRT as fallback. Quantize *GFM forward only*; keep back-end separate. |
| R3 | **Activation/4-bit quant collapses geometry heads** | Med-High | Mixed precision (heads FP16/FP8) as default; outlier handling (SmoothQuant/transform-coding cf. VersaQ-3D); treat W4A4 as stretch, not baseline. |
| R4 | **No Jetson / late Jetson access** | ? (D1) | If none: Phase 3 becomes procurement gate; interim, report desktop-Blackwell FP8/FP4 as a faithful proxy (valid *only* if target is Thor). Cloud Jetson rental exists as fallback. |
| R5 | **Novelty erosion** — more GFM-compression papers land mid-project | Med-High | Lead with the systems/edge/energy contribution (harder to scoop) + one method wrinkle; keep a living related-work doc; move fast on Phases 0–3. |
| R6 | **Baseline won't reproduce on `sm_120`/SDPA** | Med | Phase-1 hard gate; engage the repos' issues; SDPA-only patches (you already do this for OpenVLA). |
| R7 | **Real-time not achievable on tightest Orin** | Med | Frame as Pareto finding; report the budget at which each device becomes viable. |

---

## 10. Timeline (contingent on D2)

Rough, assuming ~full-time and Jetson available by Phase 3:

- Weeks 1–2: Phase 0–1 (bring-up + baseline).
- Weeks 3–5: Phase 2 (desktop PTQ sweep).
- Weeks 5–8: Phase 3 (Jetson deploy bridge) — overlaps.
- Weeks 8–11: Phase 4 (novel method).
- Weeks 11–13: Phase 5 (pruning/distill, scoped).
- Weeks 13–16: Phase 6 (Pareto + write-up).

≈ 3.5–4 months to a submittable draft. Compress by dropping the distillation arm (D4) and the large-GFM (VGGT) arm if the deadline is tight.

---

## 11. Repo & logging (mirror your existing memory-bank/EXP discipline)

```
edge-gfm-slam/
  memory-bank/
    techContext.md        # rig + Jetson specs, tooling-support matrix
    activeContext.md       # current phase, gate status, next action
    novelty.md             # living related-work / scoop watch (§1)
  scripts/
    phase0_bringup.sh      # gated, sm_120 re-verify
    export_onnx.py         # GFM -> ONNX
    build_trt.py           # ONNX -> TensorRT (ModelOpt calib)
    eval_ate.py, eval_pointmap.py, measure_energy.py
  configs/                 # one file per quant config
  results/                 # per-EXP JSON: versions, seeds, config, metrics
  CLAUDE.md                # hard rules (GPU-1 pin, SDPA-only, --no-deps)
```

**Log on every EXP (per your plan §7 habit):** driver / CUDA / torch / quant-lib versions, `CUDA_VISIBLE_DEVICES`, GPU capability `(12,0)`, seed, config hash, dataset+sequence, all metrics, and **which device** (rig vs Jetson). Energy runs must log power-sampling method + duration.

---

## 12. First-week checklist (concrete, tailored to `vm-130-131`)

```bash
# 0. Pin GPU-1, activate env (your standard preamble)
export CUDA_VISIBLE_DEVICES=1
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate openvla   # or a fresh env: conda create -n edgeslam python=3.10

# 1. Re-verify Blackwell before anything (the landmine)
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))"
#   expect: 2.11.0+cu128 (12, 0) NVIDIA RTX PRO 6000 Blackwell ...

# 2. Clone systems WITHOUT letting them pull torch==2.2
cd /office/dev_workspace/morshed && mkdir -p edge-gfm-slam/ext && cd edge-gfm-slam/ext
git clone https://github.com/rmurai0610/MASt3R-SLAM.git   # verify current URL
# install curated deps, then:  pip install -e MASt3R-SLAM --no-deps
# re-run step 1 to confirm sm_120 survived.

# 3. Tooling gates on GPU-1 (log pass/fail + versions)
python -c "import bitsandbytes as bnb; print('bnb', bnb.__version__)"
pip show gptqmodel autoawq torchao nvidia-modelopt 2>/dev/null | grep -E 'Name|Version'

# 4. Datasets (start small): TUM fr1/fr2, EuRoC MH01 — under workspace disk, not /
# 5. Baseline FP16 ATE on TUM fr1 -> first EXP log entry.
```

**Week-1 exit:** baseline FP16 ATE on at least one TUM sequence reproduced on GPU-1, tooling-support matrix filled, D1/D2 answered.

---

## 13. Read-first reading list

**Core GFMs:** DUSt3R (CVPR 2024) · MASt3R (ECCV 2024) · VGGT (CVPR 2025).
**GFM-SLAM:** MASt3R-SLAM (CVPR 2025) · VGGT-SLAM (Maggio et al., arXiv 2505.12549 / NeurIPS 2026) · (context: VGGT-Long, VGGT-Motion, Mamba-VGGT).
**Direct competitors (compression):** VersaQ-3D (arXiv 2601.20317) · Co-Me (arXiv 2511.14751) · VGGT-X (arXiv 2509.25191) · Fast-FoundationStereo (arXiv 2512.11130) · GFM distillation (arXiv 2607.01851).
**Method background:** GPTQ, AWQ, SmoothQuant (activation outliers), NVFP4/FP8 Transformer-Engine docs, TensorRT-ModelOpt PTQ/QAT guide, "Prune-Quantize-Distill" ordering (arXiv 2604.04988).
**Edge deployment:** Jetson Thor / JetPack 7.x TensorRT docs; `jetson-stats`/`tegrastats` for energy.

> Keep `memory-bank/novelty.md` updated as you read — the scoop risk (R5) is real and the field is moving monthly.
