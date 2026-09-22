#!/usr/bin/env python3
"""measure_energy.py — power / energy-per-frame on the rig via NVML.

STUB. Fill in Phase 2. With D1 = no deploy, energy is measured on the Blackwell rig
(GPU-2) via NVML/pynvml, NOT on Jetson. When Jetson is revisited, add a tegrastats path.

Approach:
- Sample GPU power (pynvml nvmlDeviceGetPowerUsage) at fixed rate during a SLAM run.
- Integrate power over wall-time -> Joules; divide by #frames -> J/frame.
- Log power-sampling method + rate + duration (per CLAUDE.md EXP discipline).

Usage (target):
    python measure_energy.py --gpu 2 --hz 50 --run "<slam cmd>" --out results/<exp>.json
"""
import argparse, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2, help="physical GPU index (pin to 2)")
    ap.add_argument("--hz", type=float, default=50.0, help="power sampling rate")
    ap.add_argument("--run", required=True, help="SLAM command to profile")
    ap.add_argument("--out")
    ap.parse_args()
    print("STUB — implement in Phase 2 (NVML on rig GPU-2). See docstring.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
