#!/usr/bin/env python3
"""Stage A gate: do per-unit W4 sensitivity profiles agree across sequences?

Score per unit = relative ATE change vs reference, signed so higher = more worth protecting:
  loi (quantize-one): (ate - fp16) / fp16
  loo (protect-one):  (uniform_w4 - ate) / uniform_w4
Gate (pre-declared): PASS if, for either measure, mean cross-dataset (TUM x EuRoC)
Spearman >= 0.3. Writes results/profile_gate.json.
"""
from __future__ import annotations

import itertools
import json
import statistics as st
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
TUM = ["rgbd_dataset_freiburg1_desk", "rgbd_dataset_freiburg1_360", "rgbd_dataset_freiburg1_floor"]
EUROC = ["MH_01_easy", "V1_01_easy"]
THRESH = 0.3


def read_pairs(p: Path) -> dict[str, float]:
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


def euroc_rmse(tag: str, seq: str) -> float:
    for line in (R / f"phase2_euroc_{tag}_{seq}_ate.log").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "rmse":
            return float(parts[1])
    raise ValueError(f"no rmse for {tag} {seq}")


def reference(seq: str, measure: str) -> float:
    if seq in EUROC:
        return euroc_rmse("fp16" if measure == "loi" else "w4_trunk", seq)
    tag = "fp16_s0" if measure == "loi" else "w4_trunk_s0"
    return read_pairs(R / f"multiseed_tum_{tag}_ates.txt")[seq]


def profile(seq: str, measure: str) -> dict[str, float]:
    if seq == "rgbd_dataset_freiburg1_desk" and measure == "loi":
        rec = json.loads((R / "20260923-phase2-sens-w4-full-fr1desk.json").read_text())
        ates = {k: float(v) for k, v in rec["metrics"]["ate_rmse_per_unit_m"].items()}
    else:
        ates = read_pairs(R / "profiles" / f"profile_{measure}_{seq}.txt")
    ates = {k: v for k, v in ates.items() if "downstream_head" not in k}
    ref = reference(seq, measure)
    if measure == "loi":
        return {k: (v - ref) / ref for k, v in ates.items()}
    return {k: (ref - v) / ref for k, v in ates.items()}


def main() -> None:
    noise = json.loads((R / "step1_summary.json").read_text())
    floor = {**noise["tum_noise_floor"]["per_seq"], **noise["euroc_noise_floor"]["per_seq"]}
    out = {"threshold": THRESH, "measures": {}}
    passed = False
    for measure in ("loi", "loo"):
        profs = {s: profile(s, measure) for s in TUM + EUROC}
        profs = {s: p for s, p in profs.items() if len(p) == 49}
        pairs = {}
        for a, b in itertools.combinations(profs, 2):
            units = sorted(set(profs[a]) & set(profs[b]))
            rho = spearmanr([profs[a][u] for u in units], [profs[b][u] for u in units]).statistic
            top_a = set(sorted(units, key=lambda u: -profs[a][u])[:7])
            top_b = set(sorted(units, key=lambda u: -profs[b][u])[:7])
            kind = "cross" if (a in EUROC) != (b in EUROC) else ("tum" if a in TUM else "euroc")
            pairs[f"{a}|{b}"] = {"rho": rho, "top7_overlap": len(top_a & top_b), "kind": kind}
        by_kind = {
            k: st.mean(v["rho"] for v in pairs.values() if v["kind"] == k)
            for k in ("tum", "euroc", "cross")
            if any(v["kind"] == k for v in pairs.values())
        }
        above = {
            s: sum(abs(d) > 2 * floor[s]["std_rel"] for d in p.values()) for s, p in profs.items() if s in floor
        }
        out["measures"][measure] = {
            "complete_seqs": list(profs),
            "pairs": pairs,
            "mean_rho": by_kind,
            "n_units_above_2x_noise": above,
        }
        cross = by_kind.get("cross")
        print(f"[{measure}] complete={len(profs)}/5 mean rho: {by_kind}")
        for k, v in pairs.items():
            a, b = (s.replace("rgbd_dataset_freiburg1_", "fr1/") for s in k.split("|"))
            print(f"   {a:12s} vs {b:12s} rho={v['rho']:+.2f} top7 overlap={v['top7_overlap']}/7 ({v['kind']})")
        print(f"   units with |delta| > 2x noise floor: {above}")
        if cross is not None and cross >= THRESH:
            passed = True
    out["gate_pass"] = passed
    print("GATE:", "PASS" if passed else "FAIL")
    (R / "profile_gate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
