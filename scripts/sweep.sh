#!/usr/bin/env bash
# Dose-response sweep: performance against how much structure we destroy.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p data/activations/logs
for arm in real rewire0.1 rewire0.25 rewire0.5 rewire1.0 random; do
  echo "=== $arm  $(date +%H:%M)"
  uv run python scripts/extract.py --arm "$arm" --windows 30000 --seed 1 \
    2>&1 | grep -v -e Warning -e sparse_csr | tail -6
done
echo "=== sweep complete $(date +%H:%M)"
