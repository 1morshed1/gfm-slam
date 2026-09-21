#!/usr/bin/env python3
"""build_trt.py — ONNX -> TensorRT engine (FP16/INT8/FP8/NVFP4 via ModelOpt calib).

STUB + DEFERRED (D1 = no deploy for now). Track-B / Phase 3 only. See export_onnx.py.
Calibration set: held-out frames from a DIFFERENT scene than any eval sequence. Never
calibrate on eval sequences. Log exactly which frames.
"""
import sys

if __name__ == "__main__":
    print("STUB + DEFERRED (Track B, Phase 3). See docstring.", file=sys.stderr)
    sys.exit(1)
