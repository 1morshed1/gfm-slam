#!/usr/bin/env bash
# Stage A pilot: per-unit profiles on 5 stable sequences, both measures, 3 parallel streams.
# desk loi already exists (results/20260923-phase2-sens-w4-full-fr1desk.json).
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_profile_pilot.sh
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
LOG="$ROOT/results/profile_pilot.log"
mkdir -p "$ROOT/results/profiles"
echo "PILOT_START $(date -Is) gpu=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"

# longest jobs first so the 3 streams finish together
JOBS="euroc MH_01_easy loo
euroc MH_01_easy loi
euroc V1_01_easy loo
euroc V1_01_easy loi
tum rgbd_dataset_freiburg1_desk loo
tum rgbd_dataset_freiburg1_360 loo
tum rgbd_dataset_freiburg1_360 loi
tum rgbd_dataset_freiburg1_floor loo
tum rgbd_dataset_freiburg1_floor loi"

export ROOT LOG
echo "$JOBS" | xargs -P 3 -L 1 bash -c '
  name="$1_$2_$3"
  if bash "$ROOT/scripts/run_seq_profile.sh" "$1" "$2" "$3" > "$ROOT/results/profiles/run_${3}_${2}.log" 2>&1; then
    echo "JOB_DONE $name $(date -Is)" >> "$LOG"
  else
    echo "JOB_FAIL $name $(date -Is)" >> "$LOG"
  fi' _
echo "PILOT_DONE $(date -Is)" | tee -a "$LOG"
