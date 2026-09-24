#!/usr/bin/env bash
# Multi-seq EuRoC shortlist: FP16 / FP8 / W4 geom-greedy K=7.
# Reuses scripts/run_euroc_ptq.sh; writes per-mode summary EXP + table.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SEQS=(V1_01_easy MH_01_easy MH_02_easy V1_02_medium V2_01_easy)
MODES=(fp16 fp8 w4_greedy)
SUMMARY="$RESULTS/euroc_shortlist_summary.txt"
: > "$SUMMARY"
echo "EUROC_SHORTLIST_START $(date -Is)" | tee -a "$SUMMARY"

CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

for mode in "${MODES[@]}"; do
  declare -A RMSE=()
  for seq in "${SEQS[@]}"; do
    echo "==> $mode $seq" | tee -a "$SUMMARY"
    bash "$ROOT/scripts/run_euroc_ptq.sh" "$mode" "$seq" \
      2>&1 | tee "$RESULTS/phase2_euroc_${mode}_${seq}_run.log"
    # parse rmse from ate log
    case "$mode" in
      fp16) tag=fp16 ;;
      fp8) tag=fp8_trunk ;;
      w4_greedy) tag=w4_sens_protect_d3 ;;
    esac
    ate_log="$RESULTS/phase2_euroc_${tag}_${seq}_ate.log"
    rmse=$(awk '/rmse/{print $2; exit}' "$ate_log" 2>/dev/null || echo "")
    if [ -z "$rmse" ]; then
      echo "FATAL: no rmse in $ate_log" | tee -a "$SUMMARY"
      exit 1
    fi
    RMSE["$seq"]="$rmse"
    echo "  $seq rmse=$rmse" | tee -a "$SUMMARY"
  done

  python - <<PY
import json, os
from pathlib import Path
import sys
sys.path.insert(0, "$ROOT/scripts")
from explog import provenance, write_exp

mode = "$mode"
seqs = """${SEQS[*]}""".split()
rmse = {
$(for s in "${SEQS[@]}"; do echo "  \"$s\": ${RMSE[$s]},"; done)
}
vals = list(rmse.values())
mean = sum(vals) / len(vals)
fp16_path = Path("$RESULTS/20260924-euroc-shortlist-fp16.json")
fp16_mean = None
fp16_per = {}
if mode != "fp16" and fp16_path.exists():
    prev = json.loads(fp16_path.read_text())
    fp16_per = prev["metrics"]["ate_rmse_per_seq_m"]
    fp16_mean = prev["metrics"]["ate_rmse_mean_m"]

quant = {"method": "none"}
if mode == "fp8":
    quant = {"method": "fake_fp8_e4m3", "scope": "trunk"}
elif mode == "w4_greedy":
    quant = {
        "method": "fake_int4_wo_except_units",
        "bits": 4,
        "k": 7,
        "allocator": "greedy_ate",
        "protect_file": "results/protect_greedy_k7.txt",
    }

rec = provenance(
    config_path=None,
    seed=0,
    dataset="euroc_shortlist_5",
    device="rig",
    extra={"system": "mast3r-slam", "quant": quant},
)
rec["exp_id"] = f"20260924-euroc-shortlist-{mode.replace('_','-')}"
metrics = {
    "ate_rmse_per_seq_m": rmse,
    "ate_rmse_mean_m": mean,
    "n_seqs": len(rmse),
    "align": "sim3",
    "seqs": seqs,
}
if fp16_mean is not None:
    metrics["fp16_ate_mean_m"] = fp16_mean
    metrics["ate_rel_change_mean_vs_fp16"] = (mean - fp16_mean) / fp16_mean
    metrics["ate_rel_change_per_seq_vs_fp16"] = {
        s: (rmse[s] - fp16_per[s]) / fp16_per[s] for s in rmse if s in fp16_per
    }
rec["metrics"] = metrics
write_exp("$RESULTS", rec)
print(f"MODE={mode} mean={mean:.6f}" + (f" vs_fp16={metrics['ate_rel_change_mean_vs_fp16']:+.1%}" if fp16_mean else ""))
Path("$SUMMARY").open("a").write(
    f"MODE={mode} mean={mean:.6f}" + (f" vs_fp16={metrics.get('ate_rel_change_mean_vs_fp16', float('nan')):+.1%}" if fp16_mean else "") + "\n"
)
PY
  unset RMSE
done

echo "EUROC_SHORTLIST_DONE $(date -Is)" | tee -a "$SUMMARY"
