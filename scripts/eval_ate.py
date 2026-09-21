#!/usr/bin/env python3
"""eval_ate.py — trajectory accuracy for a GFM-SLAM run.

STUB. Fill in Phase 1. Computes ATE RMSE (Sim(3)-aligned) and RPE against GT poses,
plus tracking success (fraction of frames tracked). Prefer the SLAM system's OFFICIAL
eval script where one exists, to stay comparable to published numbers.

Usage (target):
    python eval_ate.py --est traj.txt --gt gt.txt --align sim3 --out results/<exp>.json

Notes:
- Monocular GFM-SLAM is up-to-scale -> Sim(3) alignment (scale+rot+trans) before ATE.
- Report per-sequence and mean, >=3 runs, mean +/- std (SLAM is nondeterministic).
- A config that "wins" ATE by dropping hard frames is not a win -> always log tracking success.
"""
import argparse, json, sys


def align_sim3(est, gt):
    """Umeyama Sim(3) alignment. TODO: implement (or reuse evo / official script)."""
    raise NotImplementedError


def ate_rmse(est_aligned, gt):
    """ATE RMSE after alignment. TODO."""
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, help="estimated trajectory (TUM format)")
    ap.add_argument("--gt", required=True, help="ground-truth trajectory")
    ap.add_argument("--align", default="sim3", choices=["sim3", "se3", "none"])
    ap.add_argument("--out", help="results JSON path")
    args = ap.parse_args()
    print("STUB — implement in Phase 1. See docstring.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
