#!/usr/bin/env python3
"""Paired per-sequence bootstrap over TUM fr1 for the multi-seed configs.

Seed repeats give ~0 std (MASt3R-SLAM single_thread is deterministic), so the
uncertainty that matters is across sequences: resample the 9 per-seq relative
differences (normalized by FP16) and report a 95% CI on their mean.

Usage:
    python scripts/paired_bootstrap_tum.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_BOOT = 10000
SEED_RUN = 1

PAIRS = [
    ("w4_geom_k7", "w4_mag_k7"),
    ("w4_geom_k7", "w4_trunk"),
    ("w4_mag_k7", "w4_trunk"),
    ("w4_geom_k9", "w4_geom_k11"),
    ("w4_geom_k9", "w4_geom_k7"),
    ("w4_geom_k5", "w4_geom_k7"),
]


def load(tag: str) -> dict[str, float]:
    out = {}
    for line in (RESULTS / f"multiseed_tum_{tag}_s{SEED_RUN}_ates.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = float(parts[1])
    return out


def main() -> None:
    random.seed(0)
    fp16 = load("fp16")
    seqs = sorted(fp16)
    rows = []
    for a, b in PAIRS:
        da, db = load(a), load(b)
        diffs = [(da[s] - db[s]) / fp16[s] for s in seqs]
        mean = sum(diffs) / len(diffs)
        boots = sorted(
            sum(random.choice(diffs) for _ in diffs) / len(diffs) for _ in range(N_BOOT)
        )
        lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]
        wins = sum(d < 0 for d in diffs)
        rows.append({"a": a, "b": b, "mean_rel_diff": mean, "ci95": [lo, hi],
                     "a_better_seqs": wins, "n_seqs": len(seqs)})
        print(f"{a:12s} vs {b:12s}: {mean:+.1%}  95%CI [{lo:+.1%},{hi:+.1%}]  {wins}/{len(seqs)}")
    out = RESULTS / "paired_bootstrap_tum.json"
    out.write_text(json.dumps({"n_boot": N_BOOT, "seed_run": SEED_RUN, "pairs": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
