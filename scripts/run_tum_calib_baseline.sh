#!/usr/bin/env bash
# Run FP16 calib baseline on whatever TUM fr1 seqs are present; log ATE + mean.
# Usage: bash scripts/run_tum_calib_baseline.sh
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/ext/MASt3R-SLAM"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export LD_LIBRARY_PATH="/office/dev_workspace/morshed/miniconda3/envs/edgeslam/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
# conda activate scripts reference unset backups under set -u
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u

cd "$EXT"

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

ATE_FILE="$RESULTS/phase1_tum_calib_ates.txt"
: > "$ATE_FILE"

python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"

for dataset in "${datasets[@]}"; do
  ds="datasets/tum/$dataset"
  if [ ! -d "$ds" ]; then
    echo "SKIP missing $dataset" | tee -a "$ATE_FILE"
    continue
  fi
  logdir="logs/tum/calib/$dataset"
  traj="$logdir/$dataset.txt"
  if [ ! -f "$traj" ]; then
    echo "==> RUN $dataset"
    python main.py --dataset "$ds/" --no-viz --save-as "tum/calib/$dataset" --config config/eval_calib.yaml
  else
    echo "==> REUSE traj $traj"
  fi
  echo "==> ATE $dataset"
  # capture rmse line
  out=$(evo_ape tum "$ds/groundtruth.txt" "$traj" -as --no_warnings 2>/dev/null | tee /dev/stderr)
  rmse=$(echo "$out" | awk '/rmse/{print $2; exit}')
  echo "$dataset $rmse" | tee -a "$ATE_FILE"
done

python - <<'PY'
from pathlib import Path
p = Path("/office/dev_workspace/morshed/projects/gfm-slam/results/phase1_tum_calib_ates.txt")
vals = []
for line in p.read_text().splitlines():
    parts = line.split()
    if len(parts) == 2:
        try:
            vals.append((parts[0], float(parts[1])))
        except ValueError:
            pass
print("--- TUM calib ATE RMSE (m) ---")
for n,v in vals:
    print(f"  {n:40s} {v:.6f}")
if vals:
    mean = sum(v for _,v in vals)/len(vals)
    print(f"  {'MEAN':40s} {mean:.6f}   (paper Ours avg=0.030)")
    print(f"  n={len(vals)}/9")
PY
