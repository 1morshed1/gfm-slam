#!/usr/bin/env python3
"""eval_ate.py — trajectory ATE RMSE (Sim3) via evo.

Wraps the same `evo_ape tum ... -as` call used by the run_*.sh scripts so the
committed repo has a real entrypoint for the headline metric (plan §5.2).

Usage:
    python scripts/eval_ate.py --est traj.txt --gt gt.txt --align sim3
    python scripts/eval_ate.py --est traj.txt --gt gt.txt --out results/ate.json

Notes:
- Prefer Sim(3) for monocular / up-to-scale GFM-SLAM.
- Callers that need mean±std across runs should invoke this per run and aggregate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def evo_ape_rmse(est: Path, gt: Path, align: str = "sim3") -> dict:
    if not est.is_file():
        raise FileNotFoundError(f"missing estimate: {est}")
    if not gt.is_file():
        raise FileNotFoundError(f"missing gt: {gt}")

    cmd = ["evo_ape", "tum", str(gt), str(est), "--no_warnings"]
    if align == "sim3":
        cmd.append("-as")
    elif align == "se3":
        cmd.append("-a")
    elif align != "none":
        raise ValueError(f"unknown align={align}")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"evo_ape failed ({proc.returncode}):\n{out}")

    metrics: dict[str, float] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in {
            "rmse",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "sse",
        }:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                continue
    if "rmse" not in metrics:
        raise RuntimeError(f"could not parse rmse from evo output:\n{out}")
    metrics["align"] = align  # type: ignore[assignment]
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, help="estimated trajectory (TUM format)")
    ap.add_argument("--gt", required=True, help="ground-truth trajectory")
    ap.add_argument("--align", default="sim3", choices=["sim3", "se3", "none"])
    ap.add_argument("--out", help="optional JSON path")
    args = ap.parse_args()

    metrics = evo_ape_rmse(Path(args.est), Path(args.gt), align=args.align)
    print(f"ate_rmse_m={metrics['rmse']:.6f}  std={metrics.get('std', float('nan')):.6f}")
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"metrics": metrics, "est": str(args.est), "gt": str(args.gt)}, indent=2))
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
