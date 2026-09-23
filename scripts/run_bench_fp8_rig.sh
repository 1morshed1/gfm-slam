#!/usr/bin/env bash
# Real FP8 speedup microbench on the rig (GPU-2). First real §6.5 Pareto point.
# Usage: bash scripts/run_bench_fp8_rig.sh [kernel|trunk]
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"
MODE="${1:-kernel}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/office/dev_workspace/morshed/miniconda3/envs/edgeslam/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u

# Blackwell verify (CLAUDE.md landmine)
python -c "import torch; cap=torch.cuda.get_device_capability(0); print('cap', cap, torch.cuda.get_device_name(0)); assert cap==(12,0), 'NOT Blackwell (12,0) — torch downgraded?'"

python "$ROOT/scripts/bench_fp8.py" "$MODE" --out "$RESULTS" --energy --verbose
echo "BENCH_FP8_${MODE}_DONE"
