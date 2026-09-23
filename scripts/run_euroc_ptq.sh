#!/usr/bin/env bash
# EuRoC shortlist spot-check on V1_01_easy (calib).
# Usage:
#   bash scripts/run_euroc_ptq.sh fp16
#   bash scripts/run_euroc_ptq.sh fp8
#   bash scripts/run_euroc_ptq.sh w4_greedy
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/ext/MASt3R-SLAM"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

MODE="${1:-fp8}"
SEQ="${2:-V1_01_easy}"
DS="datasets/euroc/${SEQ}"
GT="$EXT/groundtruths/euroc/${SEQ}.txt"
PROTECT=""

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
if [ ! -d "$DS" ]; then
  echo "FATAL: missing $DS"
  exit 1
fi
if [ ! -f "$GT" ]; then
  echo "FATAL: missing $GT"
  exit 1
fi

case "$MODE" in
  fp16)
    TAG="fp16"
    SAVE="euroc/calib/${SEQ}"
    ;;
  fp8)
    TAG="fp8_trunk"
    SAVE="euroc/calib_ptq/${TAG}/${SEQ}"
    ;;
  w4_greedy)
    TAG="w4_sens_protect_d3"
    SAVE="euroc/calib_ptq/${TAG}/${SEQ}"
    PROTECT='dec_blocks2.8,enc_blocks.19,enc_blocks.22,enc_blocks.9,enc_blocks.7,enc_blocks.0,enc_blocks.14'
    ;;
  *)
    echo "unknown MODE=$MODE (fp16|fp8|w4_greedy)"
    exit 1
    ;;
esac

python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
echo "MODE=$MODE TAG=$TAG SEQ=$SEQ"

TRAJ="logs/$SAVE/${SEQ}.txt"
if [ ! -f "$TRAJ" ]; then
  echo "==> RUN $MODE $SEQ"
  python - <<PY
import sys
sys.path.insert(0, "$ROOT/scripts")
mode = "$MODE"
if mode == "fp8":
    from ptq import patch_load_mast3r
    patch_load_mast3r(method="fake_fp8", scope="trunk")
elif mode == "w4_greedy":
    from ptq import patch_load_mast3r
    protect = [p for p in "$PROTECT".split(",") if p]
    patch_load_mast3r(method="fake_except", bits=4, protect=protect)
# fp16: no patch
import runpy
sys.argv = [
  "main.py",
  "--dataset", "$DS/",
  "--no-viz",
  "--save-as", "$SAVE",
  "--config", "config/eval_calib.yaml",
]
runpy.run_path("main.py", run_name="__main__")
PY
else
  echo "==> REUSE traj $TRAJ"
fi

if [ ! -f "$TRAJ" ]; then
  TRAJ=$(find "logs/$SAVE" -name '*.txt' | head -1 || true)
fi
echo "TRAJ=$TRAJ"
evo_ape tum "$GT" "$TRAJ" -as --no_warnings | tee "$RESULTS/phase2_euroc_${TAG}_${SEQ}_ate.log"
echo "EUROCPTQ_${MODE}_DONE"
