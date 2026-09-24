#!/usr/bin/env bash
# Review Pri-1: ≥3 seeds for H3 + K-sweep (+ FP16/FP8/W4-uniform anchors) on TUM fr1.
# Seeds default 0 1 2. Writes results/20260924-multiseed-*.json with mean±std.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SEEDS=(${SEEDS:-0 1 2})
SUMMARY="$RESULTS/multiseed_h3_ksweep_summary.txt"
: > "$SUMMARY"
echo "MULTISEED_START $(date -Is) seeds=${SEEDS[*]}" | tee -a "$SUMMARY"

# Ensure protect lists exist
for k in 3 5 7 9 11; do
  if [ ! -f "$RESULTS/protect_greedy_k${k}.txt" ]; then
    python "$ROOT/scripts/allocate_bits.py" --method greedy --k "$k" \
      --write-protect "results/protect_greedy_k${k}.txt"
  fi
done
if [ ! -f "$RESULTS/protect_magnitude_k7.txt" ]; then
  python "$ROOT/scripts/allocate_bits.py" --method magnitude --k 7 \
    --write-protect "results/protect_magnitude_k7.txt"
fi

run_cfg() {
  local mode="$1" protect="$2" tag="$3"
  for seed in "${SEEDS[@]}"; do
    echo "==> cfg=$tag seed=$seed" | tee -a "$SUMMARY"
    SEED="$seed" bash "$ROOT/scripts/run_tum_seeded.sh" "$mode" "$protect" "$tag" \
      2>&1 | tee "$RESULTS/multiseed_tum_${tag}_s${seed}_run.log"
  done
}

# Anchors + H3 + K-sweep (geom)
run_cfg fp16 "" fp16
run_cfg fp8 "" fp8_trunk
run_cfg w4_uniform "" w4_trunk
run_cfg w4_mag results/protect_magnitude_k7.txt w4_mag_k7
run_cfg w4_geom results/protect_greedy_k3.txt w4_geom_k3
run_cfg w4_geom results/protect_greedy_k5.txt w4_geom_k5
run_cfg w4_geom results/protect_greedy_k7.txt w4_geom_k7
run_cfg w4_geom results/protect_greedy_k9.txt w4_geom_k9
run_cfg w4_geom results/protect_greedy_k11.txt w4_geom_k11

# Aggregate mean ± std per config
CONDA_ROOT="${CONDA_ROOT:-/office/dev_workspace/morshed/miniconda3}"
set +u
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
conda activate edgeslam
set -u
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import json, statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts") if False else "")
ROOT = Path("/office/dev_workspace/morshed/projects/gfm-slam")
sys.path.insert(0, str(ROOT / "scripts"))
from explog import provenance, write_exp

RESULTS = ROOT / "results"
SEEDS = [int(x) for x in """0 1 2""".split()]
cfgs = [
    ("fp16", "fp16"),
    ("fp8_trunk", "fp8"),
    ("w4_trunk", "w4_uniform"),
    ("w4_mag_k7", "w4_mag_k7"),
    ("w4_geom_k3", "w4_geom_k3"),
    ("w4_geom_k5", "w4_geom_k5"),
    ("w4_geom_k7", "w4_geom_k7"),
    ("w4_geom_k9", "w4_geom_k9"),
    ("w4_geom_k11", "w4_geom_k11"),
]
rows = []
for tag, name in cfgs:
    means = []
    per_seed = {}
    for s in SEEDS:
        mf = RESULTS / f"multiseed_tum_{tag}_s{s}_mean.txt"
        atef = RESULTS / f"multiseed_tum_{tag}_s{s}_ates.txt"
        if not mf.exists():
            print(f"MISSING {mf}")
            continue
        m = float(mf.read_text().strip())
        means.append(m)
        per_seq = {}
        for line in atef.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    per_seq[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        per_seed[str(s)] = {"mean_ate_m": m, "per_seq": per_seq}
    if not means:
        continue
    mean = statistics.mean(means)
    std = statistics.stdev(means) if len(means) > 1 else 0.0
    rows.append((name, tag, mean, std, means))
    rec = provenance(seed=None, dataset="tum_fr1_all", device="rig",
                     extra={"system": "mast3r-slam", "quant": {"config": name, "tag": tag},
                            "n_runs": len(means), "seeds": SEEDS})
    rec["exp_id"] = f"20260924-multiseed-{name}"
    rec["metrics"] = {
        "ate_rmse_mean_m": mean,
        "ate_rmse_std_m": std,
        "ate_rmse_per_seed_m": means,
        "per_seed": per_seed,
        "n_runs": len(means),
        "n_seqs": 9,
    }
    write_exp(str(RESULTS), rec)
    print(f"{name:16s} mean={mean:.6f} ± {std:.6f}  seeds={means}")

# summary table
fp16 = next((r for r in rows if r[0] == "fp16"), None)
lines = ["config mean_m std_m vs_fp16", "-" * 48]
for name, tag, mean, std, means in rows:
    if fp16:
        rel = (mean - fp16[2]) / fp16[2]
        lines.append(f"{name:16s} {mean:.6f} {std:.6f} {rel:+.1%}")
    else:
        lines.append(f"{name:16s} {mean:.6f} {std:.6f}")
summary = RESULTS / "multiseed_h3_ksweep_summary.txt"
summary.write_text(summary.read_text() + "\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "MULTISEED_DONE $(date -Is)" | tee -a "$SUMMARY"
