#!/usr/bin/env python3
"""export_onnx.py — GFM forward -> ONNX.

STUB + DEFERRED (D1 = no deploy for now). Not on the critical path. Kept so the
Track-B deployment bridge (Phase 3) can start from a fixed-shape single-pair case
if/when Jetson access happens.

Guidance when revived:
- Export the GFM FORWARD ONLY (ViT trunk + fusion + heads). Keep the classical back-end out.
- Start with a FIXED-SHAPE single image-pair; expand to dynamic shapes / variable views after.
- Watch for custom ops that ONNX can't lower -> may need torch-TensorRT fallback (risk R2).
"""
import sys

if __name__ == "__main__":
    print("STUB + DEFERRED (Track B, Phase 3). See docstring.", file=sys.stderr)
    sys.exit(1)
