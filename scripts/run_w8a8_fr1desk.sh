#!/usr/bin/env bash
# Phase-2: fr1/desk with fake W8A8 (weight+activation) on trunk.
# Usage: bash scripts/run_w8a8_fr1desk.sh [trunk|encoder]
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/ext/MASt3R-SLAM"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

SCOPE="${1:-trunk}"
TAG="w8a8_${SCOPE}"
SAVE="tum/calib_ptq/${TAG}_rgbd_dataset_freiburg1_desk"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export LD_LIBRARY_PATH="/office/dev_workspace/morshed/miniconda3/envs/edgeslam/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# spawn workers must import FakeIntLinear from ptq
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u

cd "$EXT"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"

python - <<PY
import sys
sys.path.insert(0, "$ROOT/scripts")
from ptq import patch_load_mast3r
patch_load_mast3r(scope="$SCOPE", method="fake", bits=8, act_bits=8)

import runpy
sys.argv = [
  "main.py",
  "--dataset", "datasets/tum/rgbd_dataset_freiburg1_desk/",
  "--no-viz",
  "--save-as", "$SAVE",
  "--config", "config/eval_calib.yaml",
]
runpy.run_path("main.py", run_name="__main__")
PY

TRAJ="$EXT/logs/$SAVE/rgbd_dataset_freiburg1_desk.txt"
if [ ! -f "$TRAJ" ]; then
  TRAJ=$(find "$EXT/logs/$SAVE" -name '*.txt' | head -1 || true)
fi
echo "TRAJ=$TRAJ"
evo_ape tum \
  "$EXT/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt" \
  "$TRAJ" \
  -as --no_warnings | tee "$RESULTS/phase2_${TAG}_fr1desk_ate.log"

echo "PHASE2_W8A8_DONE"
