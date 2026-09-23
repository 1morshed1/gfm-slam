#!/usr/bin/env bash
# §6.3 leave-one-unit sensitivity on fr1/desk (fake int WO on one unit; rest FP16).
# Usage:
#   bash scripts/run_sensitivity_fr1desk.sh           # full per-block W4 (~51 runs)
#   bash scripts/run_sensitivity_fr1desk.sh 4 coarse  # 12 coarse groups (faster)
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/ext/MASt3R-SLAM"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

BITS="${1:-4}"
MODE="${2:-full}"   # full | coarse
TAG="sens_w${BITS}_${MODE}"
ATE_FILE="$RESULTS/phase2_${TAG}_fr1desk_ates.txt"
: > "$ATE_FILE"
FP16_RMSE=0.016127

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
echo "TAG=$TAG BITS=$BITS MODE=$MODE FP16_RMSE=$FP16_RMSE"

# Build unit list (JSON lines: name + comma-separated members)
UNIT_LIST="$RESULTS/${TAG}_units.txt"
python - <<PY
import sys
sys.path.insert(0, "$ROOT/scripts")
from mast3r_slam.mast3r_utils import load_mast3r
from ptq import list_sensitivity_units, coarse_unit_groups
model = load_mast3r(device="cuda")
mode = "$MODE"
path = "$UNIT_LIST"
if mode == "coarse":
    groups = coarse_unit_groups()
    # keep only units that exist
    present = set(list_sensitivity_units(model))
    lines = []
    for name, members in groups.items():
        hit = [u for u in members if u in present]
        if hit:
            lines.append(name + "\t" + ",".join(hit))
else:
    lines = [u + "\t" + u for u in list_sensitivity_units(model)]
open(path, "w").write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} units -> {path}")
for ln in lines[:5]:
    print(" ", ln)
if len(lines) > 5:
    print(f"  ... ({len(lines)} total)")
# free VRAM before SLAM loop
del model
import torch
torch.cuda.empty_cache()
PY

n=0
total=$(wc -l < "$UNIT_LIST")
while IFS=$'\t' read -r name members; do
  [ -z "$name" ] && continue
  n=$((n+1))
  safe="${name//./_}"
  save="tum/calib_ptq/${TAG}/${safe}"
  traj="logs/$save/rgbd_dataset_freiburg1_desk.txt"
  echo "==> [$n/$total] UNIT=$name members=$members"
  if [ ! -f "$traj" ]; then
    python - <<PY
import sys
sys.path.insert(0, "$ROOT/scripts")
from ptq import patch_load_mast3r
units = [u for u in "$members".split(",") if u]
patch_load_mast3r(method="fake_units", bits=int("$BITS"), units=units)
import runpy
sys.argv = [
  "main.py",
  "--dataset", "datasets/tum/rgbd_dataset_freiburg1_desk/",
  "--no-viz",
  "--save-as", "$save",
  "--config", "config/eval_calib.yaml",
]
runpy.run_path("main.py", run_name="__main__")
PY
  else
    echo "    REUSE traj $traj"
  fi
  if [ ! -f "$traj" ]; then
    traj=$(find "logs/$save" -name '*.txt' 2>/dev/null | head -1 || true)
  fi
  out=$(evo_ape tum \
    datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    "$traj" -as --no_warnings 2>/dev/null)
  rmse=$(echo "$out" | awk '/rmse/{print $2; exit}')
  echo "$name $rmse" | tee -a "$ATE_FILE"
done < "$UNIT_LIST"

python - <<PY
from pathlib import Path
fp16 = float("$FP16_RMSE")
rows = []
for line in Path("$ATE_FILE").read_text().splitlines():
    parts = line.split()
    if len(parts) == 2:
        try:
            rows.append((parts[0], float(parts[1])))
        except ValueError:
            pass
rows.sort(key=lambda x: (x[1] - fp16) / fp16, reverse=True)
print(f"--- §6.3 sensitivity W$BITS $MODE on fr1/desk (fp16={fp16:.6f}) ---")
print(f"{'unit':30s} {'rmse':>10s} {'d_vs_fp16':>10s}")
for name, v in rows:
    d = (v - fp16) / fp16
    print(f"{name:30s} {v:10.6f} {d:+10.1%}")
print(f"n={len(rows)}  most sensitive first")
PY

echo "PHASE2_${TAG}_DONE"
