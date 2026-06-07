#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

${PYTHON:-python} analysis/smbhb_env_fit.py \
  --model both \
  --nlive 1500 \
  --dlogz 0.1 \
  --dynesty-walks 30 \
  --ptmcmc-steps 500000 \
  --burn 50000 \
  --thin 50 \
  --outdir analysis_outputs/smbhb_unified
