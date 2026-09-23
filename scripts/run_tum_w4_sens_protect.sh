#!/usr/bin/env bash
# Full TUM fr1: W4 WO except sensitivity-protected units (heads always protected).
# Usage:
#   bash scripts/run_tum_w4_sens_protect.sh dec_blocks2.8
#   bash scripts/run_tum_w4_sens_protect.sh 'dec_blocks2.8,enc_blocks.19,...' w4_sens_protect_d3
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/ext/MASt3R-SLAM"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

BITS=4
PROTECT="${1:-dec_blocks2.8}"
TAG="${2:-w${BITS}_sens_protect}"
ATE_FILE="$RESULTS/phase2_tum_${TAG}_ates.txt"
: > "$ATE_FILE"

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
  save="tum/calib_ptq/${TAG}/$dataset"
  traj="logs/$save/$dataset.txt"
  if [ ! -f "$traj" ]; then
    echo "==> RUN $TAG $dataset"
    python - <<PY
import sys
sys.path.insert(0, "$ROOT/scripts")
from ptq import patch_load_mast3r
protect = [p for p in "$PROTECT".split(",") if p]
patch_load_mast3r(method="fake_except", bits=int("$BITS"), protect=protect)
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
  out=$(evo_ape tum "$ds/groundtruth.txt" "$traj" -as --no_warnings 2>/dev/null | tee /dev/stderr)
  rmse=$(echo "$out" | awk '/rmse/{print $2; exit}')
  echo "$dataset $rmse" | tee -a "$ATE_FILE"
done

python - <<PY
from pathlib import Path
p = Path("$ATE_FILE")
fp16 = {
  "rgbd_dataset_freiburg1_360": 0.048152,
  "rgbd_dataset_freiburg1_desk": 0.016127,
  "rgbd_dataset_freiburg1_desk2": 0.023523,
  "rgbd_dataset_freiburg1_floor": 0.024977,
  "rgbd_dataset_freiburg1_plant": 0.019573,
  "rgbd_dataset_freiburg1_room": 0.061240,
  "rgbd_dataset_freiburg1_rpy": 0.023064,
  "rgbd_dataset_freiburg1_teddy": 0.040289,
  "rgbd_dataset_freiburg1_xyz": 0.008898,
}
w4u = {  # uniform W4 trunk (prior EXP)
  "rgbd_dataset_freiburg1_360": 0.048306,
  "rgbd_dataset_freiburg1_desk": 0.013487,
  "rgbd_dataset_freiburg1_desk2": 0.030384,
  "rgbd_dataset_freiburg1_floor": 0.023929,
  "rgbd_dataset_freiburg1_plant": 0.022688,
  "rgbd_dataset_freiburg1_room": 0.070095,
  "rgbd_dataset_freiburg1_rpy": 0.0265,
  "rgbd_dataset_freiburg1_teddy": 0.05514,
  "rgbd_dataset_freiburg1_xyz": 0.009369,
}
vals = []
for line in p.read_text().splitlines():
    parts = line.split()
    if len(parts) == 2:
        try:
            vals.append((parts[0], float(parts[1])))
        except ValueError:
            pass
print(f"--- TUM calib ATE RMSE (m)  tag=$TAG vs FP16 / uniform W4 ---")
for n, v in vals:
    b = fp16[n]
    u = w4u[n]
    print(f"  {n:40s} {v:.6f}  fp16_d={(v-b)/b:+.1%}  vsW4u={(v-u)/u:+.1%}")
if vals:
    mean = sum(v for _, v in vals) / len(vals)
    fp16_mean = sum(fp16[n] for n, _ in vals) / len(vals)
    w4u_mean = sum(w4u[n] for n, _ in vals) / len(vals)
    print(f"  {'MEAN':40s} {mean:.6f}  fp16_d={(mean-fp16_mean)/fp16_mean:+.1%}  vsW4u={(mean-w4u_mean)/w4u_mean:+.1%}")
    print(f"  gate <=+15% vs FP16 | uniform W4 mean={w4u_mean:.4f}")
    print(f"  n={len(vals)}/9")
PY

echo "PHASE2_TUM_${TAG}_DONE"
