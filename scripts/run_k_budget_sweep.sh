#!/usr/bin/env bash
# Paper blocker 3: greedy W4 protect budget sweep K ∈ {3,5,7,9,11} on full TUM fr1.
# Reuses K=7 from results/20260923-phase2-tum-w4-sens-protect-d3.json (identical protect set).
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u

KS=(3 5 7 9 11)
SUMMARY="$RESULTS/k_budget_sweep_summary.txt"
: > "$SUMMARY"
echo "K_BUDGET_SWEEP_START $(date -Is)" | tee -a "$SUMMARY"

for K in "${KS[@]}"; do
  PROT_FILE="$RESULTS/protect_greedy_k${K}.txt"
  if [ ! -f "$PROT_FILE" ]; then
    python "$ROOT/scripts/allocate_bits.py" --method greedy --k "$K" \
      --write-protect "results/protect_greedy_k${K}.txt"
  fi
  PROTECT="$(tr -d '\n' < "$PROT_FILE")"
  TAG="w4_geom_k${K}"
  ATE_FILE="$RESULTS/phase2_tum_${TAG}_ates.txt"

  if [ "$K" = "7" ] && [ -f "$RESULTS/20260923-phase2-tum-w4-sens-protect-d3.json" ]; then
    echo "==> REUSE K=7 from sens-protect-d3 EXP" | tee -a "$SUMMARY"
    PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "$ROOT/scripts")
from explog import provenance, write_exp
src = json.loads(Path("$RESULTS/20260923-phase2-tum-w4-sens-protect-d3.json").read_text())
rec = provenance(
    config_path="configs/w4_sens_protect_d3.yaml",
    seed=0,
    dataset="tum_fr1_all",
    device="rig",
    extra={"system": "mast3r-slam", "quant": {
        "method": "fake_int4_wo_except_units",
        "bits": 4,
        "k": 7,
        "allocator": "greedy_ate",
        "protect": src["quant"]["protect"],
        "reused_from": "20260923-phase2-tum-w4-sens-protect-d3",
    }},
)
rec["exp_id"] = "20260924-ksweep-w4-geom-k7"
rec["metrics"] = {
    "ate_rmse_per_seq_m": src["metrics"]["ate_rmse_per_seq_m"],
    "ate_rmse_mean_m": src["metrics"]["ate_rmse_mean_m"],
    "fp16_ate_mean_m": src["metrics"]["fp16_ate_mean_m"],
    "uniform_w4_ate_mean_m": src["metrics"]["uniform_w4_ate_mean_m"],
    "ate_rel_change_mean_vs_fp16": src["metrics"]["ate_rel_change_mean_vs_fp16"],
    "ate_rel_change_mean_vs_uniform_w4": src["metrics"]["ate_rel_change_mean_vs_uniform_w4"],
    "k": 7,
    "n_seqs": 9,
}
write_exp("$RESULTS", rec)
mean = src["metrics"]["ate_rmse_mean_m"]
rel = src["metrics"]["ate_rel_change_mean_vs_fp16"]
print(f"K=7 mean={mean:.6f} vs_fp16={rel:+.1%}")
Path("$SUMMARY").open("a").write(f"K=7 mean={mean:.6f} vs_fp16={rel:+.1%} REUSE\n")
PY
    continue
  fi

  echo "==> RUN TAG=$TAG K=$K PROTECT=$PROTECT" | tee -a "$SUMMARY"
  bash "$ROOT/scripts/run_tum_w4_sens_protect.sh" "$PROTECT" "$TAG" \
    2>&1 | tee "$RESULTS/phase2_tum_${TAG}_run.log"

  python - <<PY
import json, re
from pathlib import Path
import sys
sys.path.insert(0, "$ROOT/scripts")
from explog import provenance, write_exp

ate_file = Path("$ATE_FILE")
per = {}
for line in ate_file.read_text().splitlines():
    parts = line.split()
    if len(parts) == 2:
        try:
            per[parts[0]] = float(parts[1])
        except ValueError:
            pass
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
w4u_mean = 0.033322
fp16_mean = sum(fp16[n] for n in per) / len(per) if per else float("nan")
mean = sum(per.values()) / len(per) if per else float("nan")
rel = (mean - fp16_mean) / fp16_mean if per else float("nan")
rel_u = (mean - w4u_mean) / w4u_mean if per else float("nan")
protect = [p for p in "$PROTECT".split(",") if p]
k = int("$K")
rec = provenance(
    config_path=None,
    seed=0,
    dataset="tum_fr1_all",
    device="rig",
    extra={"system": "mast3r-slam", "quant": {
        "method": "fake_int4_wo_except_units",
        "bits": 4,
        "k": k,
        "allocator": "greedy_ate",
        "protect": protect,
        "protect_file": "results/protect_greedy_k${K}.txt",
    }},
)
rec["exp_id"] = f"20260924-ksweep-w4-geom-k{k}"
rec["metrics"] = {
    "ate_rmse_per_seq_m": per,
    "ate_rmse_mean_m": mean,
    "fp16_ate_mean_m": fp16_mean,
    "uniform_w4_ate_mean_m": w4u_mean,
    "ate_rel_change_mean_vs_fp16": rel,
    "ate_rel_change_mean_vs_uniform_w4": rel_u,
    "k": k,
    "n_seqs": len(per),
}
write_exp("$RESULTS", rec)
line = f"K={k} mean={mean:.6f} vs_fp16={rel:+.1%} vs_w4u={rel_u:+.1%} n={len(per)}"
print(line)
Path("$SUMMARY").open("a").write(line + "\n")
PY
done

echo "K_BUDGET_SWEEP_DONE $(date -Is)" | tee -a "$SUMMARY"
