#!/usr/bin/env bash
# Phase 0 bring-up — gated. Run on the rig (vm-130-131), GPU-2.
# Every step gates the next: a failure stops the script rather than limping forward.
# Idempotent-ish: safe to re-run; clone step skips if the dir exists.
set -euo pipefail

# --- config ---
WORKSPACE="${EDGESLAM_WORKSPACE:-/office/dev_workspace/morshed/projects/gfm-slam}"
EXT_DIR="${WORKSPACE}/ext"
MAST3R_URL="https://github.com/rmurai0610/MASt3R-SLAM.git"   # VERIFY current URL before first run
CONDA_ENV="${EDGESLAM_ENV:-edgeslam}"
CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"

echo "==> [0] Pin GPU-2"
export CUDA_VISIBLE_DEVICES=2

echo "==> [1] Activate conda env: ${CONDA_ENV}"
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate "${CONDA_ENV}"

verify_blackwell () {
  python - <<'PY'
import sys, torch
v, cap, name = torch.__version__, torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0)
print(f"torch={v} cap={cap} name={name}")
assert cap == (12, 0), f"FATAL: capability {cap} != (12,0) — torch was downgraded (the sm_120 landmine)."
# real matmul on GPU-2 (visible as cuda:0)
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
_ = (x @ x).sum().item()
print("matmul on GPU-2: OK")
PY
}

echo "==> [2] Verify Blackwell BEFORE any install"
verify_blackwell

echo "==> [3] Clone MASt3R-SLAM --recursive (--no-deps install; do NOT let it pull torch==2.2)"
mkdir -p "${EXT_DIR}"
if [ ! -d "${EXT_DIR}/MASt3R-SLAM" ]; then
  git clone --recursive "${MAST3R_URL}" "${EXT_DIR}/MASt3R-SLAM"
else
  echo "    MASt3R-SLAM already present — skipping clone."
  # ensure submodules if a prior non-recursive clone exists
  git -C "${EXT_DIR}/MASt3R-SLAM" submodule update --init --recursive
fi
echo "    >>> MANUAL GATE: install curated deps yourself (keep torch 2.11+cu128), THEN:"
echo "        pip install -e ${EXT_DIR}/MASt3R-SLAM/thirdparty/mast3r --no-deps"
echo "        pip install -e ${EXT_DIR}/MASt3R-SLAM/thirdparty/in3d --no-deps"
echo "        pip install --no-build-isolation -e ${EXT_DIR}/MASt3R-SLAM --no-deps"
echo "    (script does not auto-install to avoid a blind dep pull.)"

echo "==> [4] Re-verify Blackwell survived the install"
verify_blackwell

echo "==> [5] Tooling gates — probe quant toolchain on GPU-2 (log pass/fail + versions)"
python - <<'PY'
import importlib
for mod in ["bitsandbytes", "gptqmodel", "awq", "torchao", "modelopt"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:14s} OK  version={getattr(m,'__version__','?')}")
    except Exception as e:
        print(f"  {mod:14s} MISSING/FAIL: {type(e).__name__}: {e}")
print("  -> record results in memory-bank/techContext.md tooling matrix")
PY

echo "==> Phase 0 script done. Manual exit gate:"
echo "    [ ] MASt3R-SLAM installed --no-deps, Blackwell still (12,0)"
echo "    [ ] >=2 quant toolchains confirmed on sm_120"
echo "    [ ] TUM fr1 + EuRoC MH01 downloaded under ${WORKSPACE}/data"
