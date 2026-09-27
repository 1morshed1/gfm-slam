#!/usr/bin/env bash
# Step 1 (no-regret): EuRoC uniform-W4 + magnitude-K7 baselines, and FP16 weight-noise
# floor (rel=1e-3, draws n=1..4) on EuRoC shortlist + full TUM fr1.
# Two parallel streams on one GPU; traj reuse makes reruns resumable.
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_step1_noise_baselines.sh
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NOISE_REL="${NOISE_REL:-1e-3}"
SEQS=(V1_01_easy MH_01_easy MH_02_easy V1_02_medium V2_01_easy)
DRAWS=(1 2 3 4)
LOG="$RESULTS/step1_noise_baselines.log"
echo "STEP1_START $(date -Is) gpu=$CUDA_VISIBLE_DEVICES rel=$NOISE_REL" | tee -a "$LOG"

euroc_stream() {
  for mode in w4_uniform w4_mag; do
    for seq in "${SEQS[@]}"; do
      bash "$ROOT/scripts/run_euroc_ptq.sh" "$mode" "$seq" > "$RESULTS/step1_euroc_${mode}_${seq}_run.log" 2>&1
      echo "EUROC $mode $seq done $(date -Is)" >> "$LOG"
    done
  done
  for n in "${DRAWS[@]}"; do
    for seq in "${SEQS[@]}"; do
      NOISE_SEED=$n bash "$ROOT/scripts/run_euroc_ptq.sh" noise "$seq" > "$RESULTS/step1_euroc_noise_n${n}_${seq}_run.log" 2>&1
      echo "EUROC noise n$n $seq done $(date -Is)" >> "$LOG"
    done
  done
  echo "EUROC_STREAM_DONE $(date -Is)" >> "$LOG"
}

tum_stream() {
  for n in "${DRAWS[@]}"; do
    SEED=0 NOISE_SEED=$n bash "$ROOT/scripts/run_tum_seeded.sh" noise > "$RESULTS/step1_tum_noise_n${n}_run.log" 2>&1
    echo "TUM noise n$n done $(date -Is)" >> "$LOG"
  done
  echo "TUM_STREAM_DONE $(date -Is)" >> "$LOG"
}

euroc_stream & PA=$!
tum_stream & PB=$!
wait $PA && RA=0 || RA=$?
wait $PB && RB=0 || RB=$?
echo "STEP1_DONE $(date -Is) euroc_rc=$RA tum_rc=$RB" | tee -a "$LOG"
