#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-.envs/pta/bin/python}"
BASE_OUT="${BASE_OUT:-analysis_outputs/fulltiming_staged_v1}"

"$PYTHON_BIN" analysis/fulltiming_staged_evidence.py \
  --max-pulsars 8 \
  --nlive 256 \
  --dlogz 2.0 \
  --walks 8 \
  --outdir "$BASE_OUT/8p" \
  --skip-existing

"$PYTHON_BIN" analysis/fulltiming_staged_evidence.py \
  --max-pulsars 16 \
  --nlive 384 \
  --dlogz 1.5 \
  --walks 10 \
  --outdir "$BASE_OUT/16p" \
  --skip-existing

"$PYTHON_BIN" analysis/fulltiming_staged_evidence.py \
  --max-pulsars 32 \
  --nlive 512 \
  --dlogz 1.2 \
  --walks 12 \
  --outdir "$BASE_OUT/32p" \
  --skip-existing

"$PYTHON_BIN" analysis/fulltiming_staged_evidence.py \
  --max-pulsars 68 \
  --nlive 800 \
  --dlogz 1.0 \
  --walks 16 \
  --outdir "$BASE_OUT/68p" \
  --skip-existing
