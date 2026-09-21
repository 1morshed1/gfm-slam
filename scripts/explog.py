#!/usr/bin/env python3
"""explog.py — build a results/<exp-id>.json entry with the mandated provenance fields.

Import this from every eval script so no EXP forgets versions/seed/device (CLAUDE.md).

    from explog import provenance, write_exp
    rec = provenance(config_path="configs/fp16_baseline.yaml", seed=0,
                     dataset="tum_fr1", device="rig")
    rec["metrics"] = {"ate_rmse": ..., "tracking_success": ...}
    write_exp("results", rec)
"""
import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path


def _safe(fn, default="?"):
    try:
        return fn()
    except Exception:
        return default


def provenance(config_path=None, seed=None, dataset=None, device="rig", extra=None):
    import torch
    cfg_hash = None
    if config_path and Path(config_path).exists():
        cfg_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:12]
    rec = {
        "exp_id": time.strftime("%Y%m%d-%H%M%S"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": _safe(lambda: torch.version.cuda),
        "gpu_capability": _safe(lambda: list(torch.cuda.get_device_capability(0))),
        "gpu_name": _safe(lambda: torch.cuda.get_device_name(0)),
        "driver": _safe(lambda: subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).decode().strip().splitlines()[0]),
        "quant_libs": _quant_lib_versions(),
        "python": platform.python_version(),
        "config_path": config_path,
        "config_hash": cfg_hash,
        "seed": seed,
        "dataset": dataset,
        "metrics": {},
    }
    if extra:
        rec.update(extra)
    return rec


def _quant_lib_versions():
    import importlib
    out = {}
    for mod in ["bitsandbytes", "gptqmodel", "awq", "torchao", "modelopt"]:
        try:
            out[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception:
            out[mod] = None
    return out


def write_exp(results_dir, rec):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    p = Path(results_dir) / f"{rec['exp_id']}.json"
    p.write_text(json.dumps(rec, indent=2))
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    # smoke test: print a provenance record (no metrics)
    print(json.dumps(provenance(device="rig"), indent=2))
