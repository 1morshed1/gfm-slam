#!/usr/bin/env bash
# Test-retest of the quantize-one profile: same sequence, W4 on one unit on top of an
# FP16 weight-noise draw (rel 1e-3) so a few rounding decisions flip. 3 parallel jobs.
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_profile_retest.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
LOG=$ROOT/results/profile_retest.log
mkdir -p "$ROOT/results/profiles"
echo "RETEST_START $(date -Is)" >> $LOG
run() { NOISE_SEED=$3 bash $ROOT/scripts/run_seq_profile.sh $1 $2 loin > $ROOT/results/profiles/run_loin$3_$2.log 2>&1 && echo "JOB_DONE $2 n$3 $(date -Is)" >> $LOG || echo "JOB_FAIL $2 n$3 $(date -Is)" >> $LOG; }
run euroc MH_01_easy 1 &
run tum rgbd_dataset_freiburg1_floor 1 &
run tum rgbd_dataset_freiburg1_floor 2 &
wait
echo "RETEST_DONE $(date -Is)" >> $LOG
