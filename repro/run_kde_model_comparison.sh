#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python analysis/spectrum_kde_model_comparison.py \
  --samples 160000 \
  --posterior-samples 30000 \
  --seed 20260521
