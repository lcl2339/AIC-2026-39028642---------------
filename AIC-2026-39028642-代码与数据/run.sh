#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
output="${1:-pred_results.csv}"
python build_cache.py --split test
python infer_pixels.py
python readout.py --output "$output"
