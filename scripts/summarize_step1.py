#!/usr/bin/env python3
"""Summarize step 1: EuRoC W4 baselines (uniform / mag-K7 / geom-K7) and FP16 weight-noise floor.

Noise floor per sequence = std of ATE over {FP16, noise draws n1..n4}; reported relative to FP16.
Writes results/step1_summary.json and prints tables.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SEQS = ["V1_01_easy", "MH_01_easy", "MH_02_easy", "V1_02_medium", "V2_01_easy"]
DRAWS = [1, 2, 3, 4]
REL = "1e-3"


def euroc_rmse(tag: str, seq: str) -> float | None:
    p = R / f"phase2_euroc_{tag}_{seq}_ate.log"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "rmse":
            return float(parts[1])
    return None


def tum_ates(tag: str) -> dict[str, float]:
    p = R / f"multiseed_tum_{tag}_ates.txt"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return out


def noise_floor(base: dict[str, float], draws: list[dict[str, float]]) -> dict:
    per = {}
    for seq, b in base.items():
        vals = [b] + [d[seq] for d in draws if seq in d]
        if len(vals) < 2:
            continue
        per[seq] = {
            "n": len(vals),
            "std_rel": st.stdev(vals) / b,
            "range_rel": (max(vals) - min(vals)) / b,
        }
    draw_means = [sum(d.values()) / len(d) for d in draws if len(d) == len(base)]
    mean_vals = ([sum(base.values()) / len(base)] if base else []) + draw_means
    return {
        "per_seq": per,
        "median_seq_std_rel": st.median(v["std_rel"] for v in per.values()) if per else None,
        "mean_ate_std_rel": st.stdev(mean_vals) / mean_vals[0] if len(mean_vals) > 1 else None,
        "n_complete_draws": len(draw_means),
    }


def main() -> None:
    euroc = {}
    for name, tag in [
        ("fp16", "fp16"),
        ("fp8", "fp8_trunk"),
        ("w4_geom_k7", "w4_sens_protect_d3"),
        ("w4_mag_k7", "w4_mag_k7"),
        ("w4_uniform", "w4_trunk"),
    ] + [(f"noise_n{n}", f"noise_r{REL}_n{n}") for n in DRAWS]:
        euroc[name] = {s: v for s in SEQS if (v := euroc_rmse(tag, s)) is not None}

    fp16 = euroc["fp16"]
    print("EuRoC ATE RMSE (m); rel vs FP16 mean")
    for name, d in euroc.items():
        if not d:
            continue
        m = sum(d.values()) / len(d)
        fm = sum(fp16[s] for s in d) / len(d)
        cells = " ".join(f"{d.get(s, float('nan')):.4f}" for s in SEQS)
        print(f"  {name:12s} {cells}  mean={m:.4f} ({(m - fm) / fm:+.1%}, n={len(d)})")

    tum_fp16 = tum_ates("fp16_s0")
    tum_draws = [tum_ates(f"noise_r{REL}_n{n}_s0") for n in DRAWS]
    euroc_draws = [euroc[f"noise_n{n}"] for n in DRAWS]
    out = {
        "noise_rel": float(REL),
        "euroc": euroc,
        "euroc_noise_floor": noise_floor(fp16, euroc_draws),
        "tum_noise_floor": noise_floor(tum_fp16, tum_draws),
    }
    for k in ("euroc_noise_floor", "tum_noise_floor"):
        nf = out[k]
        print(f"{k}: median per-seq std={nf['median_seq_std_rel']}, mean-ATE std={nf['mean_ate_std_rel']}, complete draws={nf['n_complete_draws']}")
    (R / "step1_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
