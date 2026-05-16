#!/usr/bin/env bash
# Downloads the trained model into the repo's models/ directory.
# Run once before starting the inference server.
# From the repo root:
#   chmod +x Demo_Project/Server/download_model.sh
#   Demo_Project/Server/download_model.sh
# Or without chmod:
#   bash Demo_Project/Server/download_model.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$SCRIPT_DIR/rf_vent4.pkl"

MODEL_URL="https://huggingface.co/schauppi/UAS_MKL_I40_SS2026/resolve/main/rf_vent4.pkl"

echo "Downloading rf_vent4.pkl ..."
curl -L --progress-bar "$MODEL_URL" -o "$DEST"
echo "Saved -> $DEST"
echo "Done."
