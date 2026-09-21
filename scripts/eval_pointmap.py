#!/usr/bin/env python3
"""eval_pointmap.py — pointmap / depth geometry metrics.

STUB. Fill in Phase 1/2. Compares predicted pointmaps against GT mesh/depth
(Replica, 7-Scenes).

Metrics:
- accuracy & completeness (mean/median distance to GT mesh)
- depth: abs-rel, RMSE, delta < 1.25
- normal consistency (optional)

Usage (target):
    python eval_pointmap.py --pred pred_dir --gt gt_mesh --out results/<exp>.json
"""
import argparse, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out")
    ap.parse_args()
    print("STUB — implement in Phase 1/2. See docstring.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
