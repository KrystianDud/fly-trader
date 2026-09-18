#!/usr/bin/env bash
# Bootstrap a GPU box for the sweep. Written for the AWS Deep Learning AMI
# (Ubuntu 22.04), where the NVIDIA driver and CUDA are already present.
#
#   bash scripts/gpu_setup.sh
#
# Downloads the connectome directly from Google Cloud Storage rather than
# uploading 1 GB from a laptop: the instance has far more bandwidth than you do.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== GPU check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "No GPU visible. Wrong AMI? Use a Deep Learning AMI with NVIDIA drivers."
  exit 1
}

echo "== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== python deps"
uv sync
# the CPU wheel ships by default; swap in the CUDA build
uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124

echo "== connectome (about 1.1 GB)"
mkdir -p data/connectome data/market
base="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
[ -f data/connectome/annotations.feather ] || curl -sL -o data/connectome/annotations.feather \
  "$base/body-annotations-male-cns-v1.0-minconf-0.5.feather"
[ -f data/connectome/neurotransmitters.feather ] || curl -sL -o data/connectome/neurotransmitters.feather \
  "$base/body-neurotransmitters-male-cns-v1.0.feather"
[ -f data/connectome/weights.feather ] || curl -sL -o data/connectome/weights.feather \
  "$base/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ls -lh data/connectome/

echo "== market data"
if [ ! -f data/market/usdjpy_1m.parquet ]; then
  echo "MISSING: data/market/usdjpy_1m.parquet"
  echo "Copy it up from your laptop (22 MB):"
  echo "  scp -i KEY.pem data/market/usdjpy_1m.parquet ubuntu@HOST:~/fruit_fly/data/market/"
  exit 1
fi

echo "== torch sees CUDA?"
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "== ready. Benchmark first:"
echo "  PYTHONPATH=. uv run python scripts/extract.py --arm real --windows 256 --batch 256 --device cuda"
