#!/usr/bin/env python3
"""bench_fp8.py — REAL FP8-e4m3 speedup microbench on Blackwell (GPU-2).

Why this exists (speedup gate): the Track-A accuracy runs use *fake* quant
(quantize->dequantize into fp16, ptq.py) so their compute path is fp16 and shows
NO real latency/energy gain. This script measures a REAL FP8 matmul path via
`torch._scaled_mm` (native float8_e4m3fn tensor cores) against fp16 `F.linear`,
in isolation from MASt3R-SLAM's multiprocessing (sidesteps the Int8Tensor pickling
wall). It produces the first real point for the §6.5 Pareto axis.

Scope (CLAUDE.md): Linear path of the GFM trunk only. SDPA/attention untouched.
Pins GPU-2, verifies Blackwell (12,0) before timing, EXP-logs full provenance.

Modes:
  kernel  bench a set of (M,K,N) Linear shapes: fp16 vs real-fp8. speedup + TFLOP/s.
  trunk   bench the MASt3R ViT-L trunk Linear inventory as one stack (real model if
          importable via --mast3r-root, else a shape spec): aggregate latency +
          weight-storage bytes (fp16 vs fp8).

Usage (on rig, GPU-2):
  source scripts/env_gpu2.sh
  python scripts/bench_fp8.py kernel --out results/
  python scripts/bench_fp8.py trunk  --mast3r-root ext/MASt3R-SLAM --out results/ --energy

NOTE: not runnable on pre-Ada GPUs (needs sm_89+/sm_90/sm_120 FP8). UNVERIFIED until
run on the rig — author machine is sm_75.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

_FP8_E4M3_MAX = 448.0
_FP8 = torch.float8_e4m3fn


# ---------- Blackwell / GPU-2 preflight (CLAUDE.md hard rule) ----------

def preflight() -> dict:
    if not torch.cuda.is_available():
        sys.exit("FATAL: CUDA not available.")
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"[preflight] torch {torch.__version__} cu{torch.version.cuda} "
          f"cap={cap} {name}")
    if cap[0] < 8 or (cap[0] == 8 and cap[1] < 9):
        sys.exit(f"FATAL: FP8 needs sm_89+ (Ada/Hopper/Blackwell); got sm_{cap[0]}{cap[1]}.")
    if cap != (12, 0):
        print(f"[preflight] WARNING: expected Blackwell (12,0), got {cap}. "
              "Torch may have been downgraded — check CLAUDE.md landmine.",
              file=sys.stderr)
    # real matmul smoke on the device
    _ = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
    return {"gpu_capability": list(cap), "gpu_name": name}


# ---------- real FP8 linear via torch._scaled_mm ----------

def quant_fp8_pertensor(x: torch.Tensor):
    """Per-tensor symmetric cast to float8_e4m3fn. Returns (q, dequant_scale)."""
    amax = x.detach().abs().amax().clamp(min=1e-12)
    scale = (amax / _FP8_E4M3_MAX).to(torch.float32)  # dequant scale
    q = (x / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(_FP8)
    return q, scale.reshape(())


class RealFP8Linear:
    """y = x @ W^T + b using native FP8 tensor cores (torch._scaled_mm).

    Weight pre-quantized once. Activations quantized per call (dynamic per-tensor).
    _scaled_mm computes (a*scale_a) @ (b*scale_b); b must be column-major.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None,
                 out_dtype=torch.float16):
        self.out_dtype = out_dtype
        wq, sw = quant_fp8_pertensor(weight)          # W: (out, in), row-major
        # _scaled_mm needs mat_b (K,N) column-major; W.t() of a contiguous (N,K) gives exactly that
        self.b_mat = wq.t()                           # (in, out), column-major
        self.scale_b = sw
        self.bias = None if bias is None else bias.to(out_dtype)
        self.out_features = weight.shape[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x2 = x.reshape(-1, orig[-1])
        xq, sa = quant_fp8_pertensor(x2)
        y = torch._scaled_mm(
            xq, self.b_mat,
            scale_a=sa, scale_b=self.scale_b,
            bias=None, out_dtype=self.out_dtype, use_fast_accum=True,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*orig[:-1], self.out_features)


# ---------- energy sampler (NVML) ----------

class PowerSampler:
    def __init__(self, gpu_index: int, hz: float = 50.0):
        self.hz, self.gpu_index = hz, gpu_index
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._pynvml = pynvml
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[energy] pynvml unavailable ({e}); skipping energy.", file=sys.stderr)

    def _loop(self):
        dt = 1.0 / self.hz
        while not self._stop.is_set():
            try:
                self.samples.append(self._pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0)
            except Exception:  # noqa: BLE001
                break
            time.sleep(dt)

    def __enter__(self):
        if self.ok:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *a):
        if self.ok:
            self._stop.set()
            if self._t:
                self._t.join(timeout=1.0)

    def summary(self):
        if not self.ok or not self.samples:
            return None
        return {"method": "pynvml.nvmlDeviceGetPowerUsage", "hz": self.hz,
                "n_samples": len(self.samples),
                "mean_w": statistics.mean(self.samples),
                "max_w": max(self.samples)}


# ---------- timing ----------

def time_fn(fn, warmup=25, iters=100, repeats=5) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        meds.append(start.elapsed_time(end) / iters)  # ms/call
    return {"ms_median": statistics.median(meds),
            "ms_min": min(meds), "ms_max": max(meds),
            "iters": iters, "repeats": repeats}


def bench_shape(M, K, N, dtype=torch.float16) -> dict:
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype)
    fp16 = time_fn(lambda: torch.nn.functional.linear(x, w, b))
    fp8lin = RealFP8Linear(w, b, out_dtype=dtype)
    fp8 = time_fn(lambda: fp8lin(x))
    flops = 2.0 * M * K * N
    return {
        "M": M, "K": K, "N": N,
        "fp16_ms": fp16["ms_median"], "fp8_ms": fp8["ms_median"],
        "speedup": fp16["ms_median"] / fp8["ms_median"],
        "fp16_tflops": flops / (fp16["ms_median"] * 1e-3) / 1e12,
        "fp8_tflops": flops / (fp8["ms_median"] * 1e-3) / 1e12,
    }


# MASt3R ViT-L trunk representative Linear shapes (in, out) per block, dim=1024, mlp=4.
# enc: 24 blocks; dec/dec2: 12 blocks each, dim 768. Token count for 512-res ~ 1024 (+specials).
# Used only when the real model is not importable.
def trunk_shape_spec(tokens: int = 1024, batch: int = 1) -> list[tuple[str, int, int, int]]:
    M = batch * tokens
    shapes = []
    # encoder ViT-L: dim 1024, mlp 4096, 24 blocks. qkv(1024->3072), proj(1024->1024), fc1(1024->4096), fc2(4096->1024)
    for i in range(24):
        shapes += [(f"enc.{i}.qkv", M, 1024, 3072), (f"enc.{i}.proj", M, 1024, 1024),
                   (f"enc.{i}.fc1", M, 1024, 4096), (f"enc.{i}.fc2", M, 4096, 1024)]
    # decoder x2: dim 768, mlp 3072, 12 blocks each; + cross-attn kv
    for tag in ("dec", "dec2"):
        for i in range(12):
            shapes += [(f"{tag}.{i}.qkv", M, 768, 2304), (f"{tag}.{i}.proj", M, 768, 768),
                       (f"{tag}.{i}.fc1", M, 768, 3072), (f"{tag}.{i}.fc2", M, 3072, 768)]
    return shapes


def weight_bytes(shapes) -> dict:
    fp16 = sum(K * N * 2 for _, _, K, N in shapes)
    fp8 = sum(K * N * 1 for _, _, K, N in shapes)
    return {"fp16_weight_mb": fp16 / 1e6, "fp8_weight_mb": fp8 / 1e6,
            "weight_mb_saved": (fp16 - fp8) / 1e6, "ratio": fp16 / fp8}


def run_kernel(args) -> dict:
    # square-ish + real ViT-L linear shapes
    shapes = [(4096, 4096, 4096), (8192, 1024, 3072), (8192, 1024, 4096),
              (8192, 4096, 1024), (1024, 1024, 1024)]
    if args.shapes:
        shapes = [tuple(int(v) for v in s.split(",")) for s in args.shapes]
    rows = [bench_shape(M, K, N) for (M, K, N) in shapes]
    return {"mode": "kernel", "shapes": rows,
            "mean_speedup": statistics.mean(r["speedup"] for r in rows)}


def run_trunk(args) -> dict:
    spec = trunk_shape_spec(tokens=args.tokens, batch=args.batch)
    # Time each Linear independently then sum (heterogeneous shapes across the trunk).
    per = []
    for name, M, K, N in spec:
        xx = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(N, K, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)
        t16 = time_fn(lambda: torch.nn.functional.linear(xx, w, b), warmup=10, iters=50, repeats=3)
        fl = RealFP8Linear(w, b)
        t8 = time_fn(lambda: fl(xx), warmup=10, iters=50, repeats=3)
        per.append({"unit": name, "fp16_ms": t16["ms_median"], "fp8_ms": t8["ms_median"],
                    "speedup": t16["ms_median"] / t8["ms_median"]})
        del xx, w, b, fl
        torch.cuda.empty_cache()
    tot16 = sum(p["fp16_ms"] for p in per)
    tot8 = sum(p["fp8_ms"] for p in per)
    out = {"mode": "trunk", "tokens": args.tokens, "batch": args.batch,
           "n_linears": len(per),
           "fp16_total_ms": tot16, "fp8_total_ms": tot8,
           "trunk_linear_speedup": tot16 / tot8,
           "weights": weight_bytes(spec),
           "per_unit": per if args.verbose else "(use --verbose)"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kernel", "trunk"])
    ap.add_argument("--out", default=None, help="results dir for EXP json")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--shapes", nargs="*", help="kernel mode: 'M,K,N' triples")
    ap.add_argument("--mast3r-root", default=None, help="(reserved) real-model trunk")
    ap.add_argument("--energy", action="store_true", help="sample GPU power via NVML")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    hw = preflight()

    torch.cuda.reset_peak_memory_stats()
    sampler = PowerSampler(args.gpu, hz=50.0) if args.energy else None
    if sampler:
        with sampler:
            result = run_kernel(args) if args.mode == "kernel" else run_trunk(args)
        result["energy"] = sampler.summary()
    else:
        result = run_kernel(args) if args.mode == "kernel" else run_trunk(args)

    result["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6

    # pretty print
    print(f"\n=== bench_fp8 [{args.mode}] ===")
    if args.mode == "kernel":
        for r in result["shapes"]:
            print(f"  {r['M']}x{r['K']}x{r['N']}: fp16 {r['fp16_ms']:.4f}ms "
                  f"({r['fp16_tflops']:.0f} TF) -> fp8 {r['fp8_ms']:.4f}ms "
                  f"({r['fp8_tflops']:.0f} TF)  speedup {r['speedup']:.2f}x")
        print(f"  MEAN speedup: {result['mean_speedup']:.2f}x")
    else:
        w = result["weights"]
        print(f"  {result['n_linears']} linears  fp16 {result['fp16_total_ms']:.3f}ms "
              f"-> fp8 {result['fp8_total_ms']:.3f}ms  speedup {result['trunk_linear_speedup']:.2f}x")
        print(f"  weight mem: {w['fp16_weight_mb']:.0f}MB -> {w['fp8_weight_mb']:.0f}MB "
              f"({w['ratio']:.2f}x smaller)")
    if result.get("energy"):
        print(f"  power: mean {result['energy']['mean_w']:.1f}W max {result['energy']['max_w']:.1f}W")

    if args.out:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from explog import provenance, write_exp
        rec = provenance(seed=args.seed, dataset=f"synthetic-{args.mode}", device="rig",
                         extra={"bench": "fp8_real_scaled_mm"})
        rec["gpu_capability"] = hw["gpu_capability"]
        rec["metrics"] = result
        rec["exp_id"] = f"{time.strftime('%Y%m%d')}-benchfp8-{args.mode}"
        write_exp(args.out, rec)


if __name__ == "__main__":
    main()
