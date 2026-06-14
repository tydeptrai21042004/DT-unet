#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-kvasir_seg}"
DATA_ROOT="${DATA_ROOT:-data}"
SOURCE_DIR="${SOURCE_DIR:-}"
ZIP_PATH="${ZIP_PATH:-}"
DOWNLOAD_URL="${DOWNLOAD_URL:-}"
DOWNLOAD_DST="${DOWNLOAD_DST:-}"
IMAGE_SIZE="${IMAGE_SIZE:-352}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_hc_ablation}"
SEEDS="${SEEDS:-42,1,2}"
ALLOW_INSECURE_DOWNLOAD="${ALLOW_INSECURE_DOWNLOAD:-0}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-0}"

cmd=(
  "$PYTHON_BIN" scripts/run_hc_ablation.py
  --dataset "$DATASET"
  --data-root "$DATA_ROOT"
  --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --output-root "$OUTPUT_ROOT"
  --seeds "$SEEDS"
)

if [[ -n "$SOURCE_DIR" ]]; then
  cmd+=(--source-dir "$SOURCE_DIR")
fi
if [[ -n "$ZIP_PATH" ]]; then
  cmd+=(--zip-path "$ZIP_PATH")
fi
if [[ -n "$DOWNLOAD_URL" ]]; then
  cmd+=(--download-url "$DOWNLOAD_URL")
fi
if [[ -n "$DOWNLOAD_DST" ]]; then
  cmd+=(--download-dst "$DOWNLOAD_DST")
fi
if [[ "$ALLOW_INSECURE_DOWNLOAD" == "1" ]]; then
  cmd+=(--allow-insecure-download)
fi
if [[ "$SAVE_PREDICTIONS" == "1" ]]; then
  cmd+=(--save-predictions)
fi
if [[ "$SAVE_VISUALIZATIONS" == "1" ]]; then
  cmd+=(--save-visualizations)
fi

printf '[RUN]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
