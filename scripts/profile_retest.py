#!/usr/bin/env python3
"""Test-retest reliability of the quantize-one (loi) profile within a sequence.

Retest profile loin<n>: unit W4 on top of FP16 weight-noise draw n (rel 1e-3), scored vs that
noise draw's FP16 ATE. Compares Spearman(original loi, retest) and between retest draws, against
the cross-sequence correlations from profile_gate.json. Writes results/profile_retest.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import spearmanr

from profile_gate import R, profile, read_pairs

JOBS = [("rgbd_dataset_freiburg1_floor", [1, 2]), ("MH_01_easy", [1])]


def noise_ref(seq: str, n: int) -> float:
    if seq.startswith("rgbd_"):
        return read_pairs(R / f"multiseed_tum_noise_r1e-3_n{n}_s0_ates.txt")[seq]
    for line in (R / f"phase2_euroc_noise_r1e-3_n{n}_{seq}_ate.log").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "rmse":
            return float(parts[1])
    raise ValueError(seq)


def retest_profile(seq: str, n: int) -> dict[str, float]:
    ates = read_pairs(R / "profiles" / f"profile_loin{n}_{seq}.txt")
    ref = noise_ref(seq, n)
    return {k: (v - ref) / ref for k, v in ates.items() if "downstream_head" not in k}


def rho(a: dict[str, float], b: dict[str, float]) -> tuple[float, int, int]:
    units = sorted(set(a) & set(b))
    top_a = set(sorted(units, key=lambda u: -a[u])[:7])
    top_b = set(sorted(units, key=lambda u: -b[u])[:7])
    return spearmanr([a[u] for u in units], [b[u] for u in units]).statistic, len(units), len(top_a & top_b)


def main() -> None:
    out = {}
    for seq, draws in JOBS:
        orig = profile(seq, "loi")
        rets = {n: retest_profile(seq, n) for n in draws}
        for n, p in rets.items():
            r, k, ov = rho(orig, p)
            out[f"{seq}|loi_vs_loin{n}"] = {"rho": r, "n_units": k, "top7_overlap": ov}
            print(f"{seq:30s} loi vs loin{n}: rho={r:+.2f} (n={k}) top7 overlap={ov}/7")
        if len(rets) > 1:
            (n1, p1), (n2, p2) = list(rets.items())[:2]
            r, k, ov = rho(p1, p2)
            out[f"{seq}|loin{n1}_vs_loin{n2}"] = {"rho": r, "n_units": k, "top7_overlap": ov}
            print(f"{seq:30s} loin{n1} vs loin{n2}: rho={r:+.2f} (n={k}) top7 overlap={ov}/7")
    gate = json.loads((R / "profile_gate.json").read_text())
    print("reference: cross-sequence mean rho (loi):", gate["measures"]["loi"]["mean_rho"])
    (R / "profile_retest.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
