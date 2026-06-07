#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-.envs/pta/bin/python}"
BASE_OUT="${BASE_OUT:-analysis_outputs/fulltiming_parallel_v1}"
MAX_JOBS="${MAX_JOBS:-20}"
STAGES="${STAGES:-8 16 32 68}"
MODELS="${MODELS:-hd_powerlaw hd_turnover hd_phase_bpl}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$BASE_OUT"

job_count() {
  jobs -rp | wc -l | tr -d ' '
}

wait_for_slot() {
  while [ "$(job_count)" -ge "$MAX_JOBS" ]; do
    sleep 10
  done
}

launch_one() {
  local stage="$1"
  local model="$2"
  local nlive="$3"
  local dlogz="$4"
  local walks="$5"
  local stage_dir="$BASE_OUT/${stage}p"
  mkdir -p "$stage_dir"
  wait_for_slot
  {
    echo "START $(date -u '+%Y-%m-%dT%H:%M:%SZ') stage=${stage} model=${model}"
    "$PYTHON_BIN" analysis/fulltiming_staged_evidence.py \
      --max-pulsars "$stage" \
      --models "$model" \
      --nlive "$nlive" \
      --dlogz "$dlogz" \
      --walks "$walks" \
      --outdir "$stage_dir" \
      --skip-existing
    echo "DONE $(date -u '+%Y-%m-%dT%H:%M:%SZ') stage=${stage} model=${model}"
  } > "$stage_dir/${model}.worker.log" 2>&1 &
  echo "$! stage=${stage} model=${model}" >> "$BASE_OUT/workers.pid"
}

rm -f "$BASE_OUT/workers.pid"

for stage in $STAGES; do
  case "$stage" in
    8) nlive=256; dlogz=2.0; walks=8 ;;
    16) nlive=384; dlogz=1.5; walks=10 ;;
    32) nlive=512; dlogz=1.2; walks=12 ;;
    68) nlive=800; dlogz=1.0; walks=16 ;;
    *) nlive=256; dlogz=2.0; walks=8 ;;
  esac
  for model in $MODELS; do
    launch_one "$stage" "$model" "$nlive" "$dlogz" "$walks"
  done
done

wait
"$PYTHON_BIN" analysis/summarize_fulltiming_evidence.py "$BASE_OUT" > "$BASE_OUT/summary.log"
