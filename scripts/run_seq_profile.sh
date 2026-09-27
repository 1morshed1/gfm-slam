#!/usr/bin/env bash
# Per-sequence per-unit W4 sensitivity profile (49 trunk units; heads always FP16).
#   loi (quantize-one): only UNIT in W4, rest FP16      -> compare to FP16
#   loo (protect-one):  whole trunk W4, UNIT kept FP16  -> compare to uniform W4
# Usage: bash scripts/run_seq_profile.sh tum rgbd_dataset_freiburg1_floor loo
#        bash scripts/run_seq_profile.sh euroc MH_01_easy loi
# Resumable: reuses existing trajectories; rewrites the ATE file each time.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/ext/MASt3R-SLAM"
RESULTS="$ROOT/results"
DSET="${1:?tum|euroc}"
SEQ="${2:?sequence}"
MEASURE="${3:?loi|loo}"

case "$DSET" in
  tum) DS="datasets/tum/$SEQ"; GT="$DS/groundtruth.txt" ;;
  euroc) DS="datasets/euroc/$SEQ"; GT="groundtruths/euroc/$SEQ.txt" ;;
  *) echo "unknown DSET=$DSET"; exit 1 ;;
esac
case "$MEASURE" in loi|loo) ;; *) echo "unknown MEASURE=$MEASURE"; exit 1 ;; esac

mkdir -p "$RESULTS/profiles"
ATE_FILE="$RESULTS/profiles/profile_${MEASURE}_${SEQ}.txt"
: > "$ATE_FILE"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export LD_LIBRARY_PATH="/office/dev_workspace/morshed/miniconda3/envs/edgeslam/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u

cd "$EXT"
[ -d "$DS" ] || { echo "FATAL: missing $DS"; exit 1; }
[ -f "$GT" ] || { echo "FATAL: missing $GT"; exit 1; }

mapfile -t UNITS < <(cut -f1 "$RESULTS/sens_w4_full_units.txt" | grep -v downstream_head)
echo "PROFILE $DSET $SEQ $MEASURE units=${#UNITS[@]} gpu=$CUDA_VISIBLE_DEVICES"

for unit in "${UNITS[@]}"; do
  save="profile/${MEASURE}/${SEQ}/${unit//./_}"
  traj="logs/$save/$SEQ.txt"
  if [ ! -f "$traj" ]; then
    python - <<PY
import sys
sys.path.insert(0, "$ROOT/scripts")
from ptq import patch_load_mast3r
if "$MEASURE" == "loi":
    patch_load_mast3r(method="fake_units", bits=4, units=["$unit"])
else:
    patch_load_mast3r(method="fake_except", bits=4, protect=["$unit"])
import runpy
sys.argv = ["main.py", "--dataset", "$DS/", "--no-viz", "--save-as", "$save",
            "--config", "config/eval_calib.yaml"]
runpy.run_path("main.py", run_name="__main__")
PY
  fi
  [ -f "$traj" ] || traj=$(find "logs/$save" -name '*.txt' | head -1 || true)
  rmse=$(evo_ape tum "$GT" "$traj" -as --no_warnings 2>/dev/null | awk '/rmse/{print $2; exit}')
  echo "$unit $rmse" | tee -a "$ATE_FILE"
done
echo "PROFILE_DONE $DSET $SEQ $MEASURE"
