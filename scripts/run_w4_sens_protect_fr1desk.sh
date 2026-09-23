#!/usr/bin/env bash
# §6.3 mixed-precision: W4 WO except protected sensitive units (heads + dec_blocks2.8).
# Usage: bash scripts/run_w4_sens_protect_fr1desk.sh
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/ext/MASt3R-SLAM"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

BITS="${1:-4}"
# comma-separated protect list (heads always added in ptq)
PROTECT="${2:-dec_blocks2.8}"
TAG="w${BITS}_sens_protect"
SAVE="tum/calib_ptq/${TAG}_rgbd_dataset_freiburg1_desk"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export LD_LIBRARY_PATH="/office/dev_workspace/morshed/miniconda3/envs/edgeslam/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u

cd "$EXT"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
echo "TAG=$TAG PROTECT=$PROTECT"

python - <<PY
import sys
sys.path.insert(0, "$ROOT/scripts")
from ptq import patch_load_mast3r
protect = [p for p in "$PROTECT".split(",") if p]
patch_load_mast3r(method="fake_except", bits=int("$BITS"), protect=protect)
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

echo "PHASE2_${TAG}_DESK_DONE"
