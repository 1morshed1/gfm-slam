#!/usr/bin/env bash
# Seeded full-TUM fr1 PTQ/protect run. Always writes a fresh traj under TAG_s${SEED}.
# Usage:
#   SEED=1 bash scripts/run_tum_seeded.sh fp16
#   SEED=0 bash scripts/run_tum_seeded.sh w4_uniform
#   SEED=2 bash scripts/run_tum_seeded.sh w4_geom   results/protect_greedy_k7.txt w4_geom_k7
#   SEED=0 bash scripts/run_tum_seeded.sh w4_mag    results/protect_magnitude_k7.txt w4_mag_k7
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/ext/MASt3R-SLAM"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

MODE="${1:?mode required: fp16|fp8|w4_uniform|w4_geom|w4_mag}"
PROTECT_FILE="${2:-}"
TAG_BASE="${3:-}"
SEED="${SEED:-0}"
PROTECT=""

case "$MODE" in
  fp16) TAG_BASE="${TAG_BASE:-fp16}"; METHOD=""; BITS=16 ;;
  fp8) TAG_BASE="${TAG_BASE:-fp8_trunk}"; METHOD="fake_fp8"; BITS=8 ;;
  w4_uniform) TAG_BASE="${TAG_BASE:-w4_trunk}"; METHOD="fake"; BITS=4 ;;
  w4_geom|w4_mag)
    METHOD="fake_except"; BITS=4
    if [ -z "$PROTECT_FILE" ]; then
      echo "FATAL: $MODE needs protect file arg"; exit 1
    fi
    PROTECT="$(tr -d '\n' < "$ROOT/$PROTECT_FILE" 2>/dev/null || tr -d '\n' < "$PROTECT_FILE")"
    TAG_BASE="${TAG_BASE:-${MODE}}"
    ;;
  *) echo "unknown MODE=$MODE"; exit 1 ;;
esac

TAG="${TAG_BASE}_s${SEED}"
ATE_FILE="$RESULTS/multiseed_tum_${TAG}_ates.txt"
: > "$ATE_FILE"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export GFM_SEED="$SEED"
export PYTHONHASHSEED="$SEED"
export LD_LIBRARY_PATH="/office/dev_workspace/morshed/miniconda3/envs/edgeslam/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u

cd "$EXT"
echo "MODE=$MODE TAG=$TAG SEED=$SEED"

datasets=(
  rgbd_dataset_freiburg1_360
  rgbd_dataset_freiburg1_desk
  rgbd_dataset_freiburg1_desk2
  rgbd_dataset_freiburg1_floor
  rgbd_dataset_freiburg1_plant
  rgbd_dataset_freiburg1_room
  rgbd_dataset_freiburg1_rpy
  rgbd_dataset_freiburg1_teddy
  rgbd_dataset_freiburg1_xyz
)

for dataset in "${datasets[@]}"; do
  ds="datasets/tum/$dataset"
  if [ ! -d "$ds" ]; then
    echo "SKIP missing $dataset" | tee -a "$ATE_FILE"
    continue
  fi
  save="tum/calib_ptq_ms/${TAG}/$dataset"
  traj="logs/$save/$dataset.txt"
  if [ ! -f "$traj" ]; then
    echo "==> RUN $TAG $dataset"
    python - <<PY
import os, random, sys
import numpy as np
import torch
sys.path.insert(0, "$ROOT/scripts")
seed = int(os.environ["GFM_SEED"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

mode = "$MODE"
if mode == "fp8":
    from ptq import patch_load_mast3r
    patch_load_mast3r(method="fake_fp8", scope="trunk")
elif mode == "w4_uniform":
    from ptq import patch_load_mast3r
    patch_load_mast3r(method="fake", bits=4, scope="trunk")
elif mode in ("w4_geom", "w4_mag"):
    from ptq import patch_load_mast3r
    protect = [p for p in """$PROTECT""".split(",") if p]
    patch_load_mast3r(method="fake_except", bits=4, protect=protect)
# fp16: no patch

import runpy
sys.argv = [
  "main.py",
  "--dataset", "$ds/",
  "--no-viz",
  "--save-as", "$save",
  "--config", "config/eval_calib.yaml",
]
runpy.run_path("main.py", run_name="__main__")
PY
  else
    echo "==> REUSE traj $traj"
  fi
  echo "==> ATE $dataset"
  rmse=$(python "$ROOT/scripts/eval_ate.py" --est "$traj" --gt "$ds/groundtruth.txt" --align sim3 \
    | awk -F= '/ate_rmse_m/{print $2}' | awk '{print $1}')
  echo "$dataset $rmse" | tee -a "$ATE_FILE"
done

python - <<PY
from pathlib import Path
p = Path("$ATE_FILE")
vals = []
for line in p.read_text().splitlines():
    parts = line.split()
    if len(parts) == 2:
        try:
            vals.append((parts[0], float(parts[1])))
        except ValueError:
            pass
mean = sum(v for _, v in vals) / len(vals) if vals else float("nan")
print(f"SEED=$SEED TAG=$TAG MEAN_ATE={mean:.6f} n={len(vals)}")
Path("$RESULTS/multiseed_tum_${TAG}_mean.txt").write_text(f"{mean:.8f}\n")
PY
echo "SEEDED_${TAG}_DONE"
