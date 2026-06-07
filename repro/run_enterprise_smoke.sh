#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

${PYTHON:-python} analysis/smbhb_env_fit.py \
  --model pl \
  --max-pulsars 2 \
  --nlive 64 \
  --dlogz 5.0 \
  --dynesty-walks 5 \
  --ptmcmc-steps 0 \
  --outdir analysis_outputs/test_run_small
