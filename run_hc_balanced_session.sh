#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

SESSION="${1:-${SESSION:-}}"
if [[ ! "$SESSION" =~ ^[1-4]$ ]]; then
  echo "Usage: bash run_hc_balanced_session.sh {1|2|3|4}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGE_SIZE="${IMAGE_SIZE:-352}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42,1,2}"
DATA_ROOT="${DATA_ROOT:-data}"
PAPER_CONFIG_DIR="${PAPER_CONFIG_DIR:-configs/paper_fair}"
HC_ABLATION_CONFIG_DIR="${HC_ABLATION_CONFIG_DIR:-configs/hc_ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_hc_session_${SESSION}}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
ALLOW_INSECURE_DOWNLOAD="${ALLOW_INSECURE_DOWNLOAD:-1}"
BUSI_ZIP_PATH="${BUSI_ZIP_PATH:-}"
AUTO_FIND_KAGGLE_INPUT="${AUTO_FIND_KAGGLE_INPUT:-1}"

HC_MODEL="proposal_hc_unet_no_gate"
NEW_DATASET=""
NEW_MODELS=""
EXISTING_DATASET=""
HC_ABLATIONS=""

case "$SESSION" in
  1)
    NEW_DATASET="isic2018"
    NEW_MODELS="unet,unetpp,pranet,acsnet,hardnet_mseg,proposal_hc_unet_no_gate"
    EXISTING_DATASET="kvasir_seg"
    HC_ABLATIONS="hc_reference"
    ;;
  2)
    NEW_DATASET="isic2018"
    NEW_MODELS="polyp_pvt,caranet,hsnet,cfanet,resunetpp"
    EXISTING_DATASET="cvc_clinicdb"
    HC_ABLATIONS="hc_without_hc_branch,hc_shared_kernel"
    ;;
  3)
    NEW_DATASET="busi"
    NEW_MODELS="unet,unetpp,pranet,acsnet,hardnet_mseg,proposal_hc_unet_no_gate"
    EXISTING_DATASET="cvc_colondb"
    HC_ABLATIONS="hc_learnable_h"
    ;;
  4)
    NEW_DATASET="busi"
    NEW_MODELS="polyp_pvt,caranet,hsnet,cfanet,resunetpp"
    EXISTING_DATASET=""
    HC_ABLATIONS="hc_kernel5,hc_identity_projection,hc_no_channel_expansion"
    ;;
esac

if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -q --upgrade pip
  "$PYTHON_BIN" -m pip install -q -r requirements.txt pytest
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  "$PYTHON_BIN" -m pytest -q \
    tests/test_hc_ablation_variants.py \
    tests/test_hc_operator_contracts.py \
    tests/test_dataset_expansion_and_seeds.py \
    tests/test_runtime_and_data_cli.py
fi

if [[ "$DEVICE" == cuda* ]]; then
  nvidia-smi
fi

rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

find_busi_zip() {
  local candidate=""

  if [[ -n "$BUSI_ZIP_PATH" ]]; then
    printf '%s\n' "$BUSI_ZIP_PATH"
    return 0
  fi

  if [[ "$AUTO_FIND_KAGGLE_INPUT" == "1" && -d /kaggle/input ]]; then
    candidate="$(find /kaggle/input -type f \
      \( -iname 'Dataset_BUSI.zip' -o -iname '*BUSI*.zip' \) \
      -print -quit 2>/dev/null || true)"
  fi

  printf '%s\n' "$candidate"
}

validate_zip_archive() {
  local archive_path="$1"

  [[ -f "$archive_path" ]] || {
    echo "ERROR: Dataset archive does not exist: $archive_path" >&2
    return 1
  }

  "$PYTHON_BIN" - "$archive_path" <<'PY_VALIDATE_ZIP'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
if not zipfile.is_zipfile(archive):
    raise SystemExit(f"ERROR: Not a valid ZIP archive: {archive}")
print(f"Validated dataset archive: {archive} ({archive.stat().st_size} bytes)")
PY_VALIDATE_ZIP
}

prepare_dataset() {
  local dataset_name="$1"
  local archive_path=""
  local prepare_args=(
    --dataset "$dataset_name"
    --data-root "$DATA_ROOT"
    --image-size "$IMAGE_SIZE"
    --force
  )

  echo "============================================================"
  echo "Prepare ${dataset_name}"
  echo "============================================================"

  if [[ "$dataset_name" == "busi" ]]; then
    archive_path="$(find_busi_zip)"
    if [[ -n "$archive_path" ]]; then
      validate_zip_archive "$archive_path"
      echo "Using supplied official BUSI archive: $archive_path"
      prepare_args+=(--zip-path "$archive_path")
    else
      echo "No local BUSI archive found; trying the registry official URL."
      echo "If the official server returns HTTP 403, upload Dataset_BUSI.zip"
      echo "as a Kaggle input or set BUSI_ZIP_PATH=/path/to/Dataset_BUSI.zip."
      if [[ "$ALLOW_INSECURE_DOWNLOAD" == "1" ]]; then
        prepare_args+=(--allow-insecure-download)
      fi
    fi
  elif [[ "$ALLOW_INSECURE_DOWNLOAD" == "1" ]]; then
    prepare_args+=(--allow-insecure-download)
  fi

  "$PYTHON_BIN" scripts/prepare_dataset.py "${prepare_args[@]}"

  "$PYTHON_BIN" scripts/make_splits.py \
    --dataset "$dataset_name" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --seed 42

  for split_name in train val test; do
    local split_path="$DATA_ROOT/splits/$dataset_name/$split_name.txt"
    [[ -s "$split_path" ]] || {
      echo "ERROR: Missing or empty split: $split_path" >&2
      exit 1
    }
    printf '%-8s %s\n' "$split_name:" "$(wc -l < "$split_path")"
  done
}

run_multi_seed() {
  local models="$1"
  local dataset_name="$2"
  local config_dir="$3"
  local output_dir="$4"
  local extra=()

  if [[ "$ALLOW_INSECURE_DOWNLOAD" == "1" ]]; then
    extra+=(--allow-insecure-download)
  fi

  "$PYTHON_BIN" scripts/benchmark_multi_seed.py \
    --models "$models" \
    --dataset "$dataset_name" \
    --config-dir "$config_dir" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --output-root "$output_dir" \
    --seeds "$SEEDS" \
    "${extra[@]}"
}

# Each session prepares only its required cross-domain dataset. ISIC 2018 and
# BUSI are downloaded automatically from registry-configured official archives.
prepare_dataset "$NEW_DATASET"

run_multi_seed \
  "$NEW_MODELS" \
  "$NEW_DATASET" \
  "$PAPER_CONFIG_DIR" \
  "$OUTPUT_ROOT/${NEW_DATASET}_comparison"

if [[ -n "$EXISTING_DATASET" ]]; then
  run_multi_seed \
    "$HC_MODEL" \
    "$EXISTING_DATASET" \
    "$PAPER_CONFIG_DIR" \
    "$OUTPUT_ROOT/${EXISTING_DATASET}_hc_only"
fi

run_multi_seed \
  "$HC_ABLATIONS" \
  "kvasir_seg" \
  "$HC_ABLATION_CONFIG_DIR" \
  "$OUTPUT_ROOT/hc_ablation_kvasir"

{
  echo "session=$SESSION"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo archive)"
  echo "new_dataset=$NEW_DATASET"
  echo "new_models=$NEW_MODELS"
  echo "existing_dataset=$EXISTING_DATASET"
  echo "hc_ablation_models=$HC_ABLATIONS"
  echo "image_size=$IMAGE_SIZE"
  echo "batch_size=$BATCH_SIZE"
  echo "epochs=$EPOCHS"
  echo "seeds=$SEEDS"
  echo "device=$DEVICE"
  echo "dataset_download_source=official_archive_or_registry"
  echo "busi_zip_path=${BUSI_ZIP_PATH:-auto}"
} > "$OUTPUT_ROOT/session_manifest.txt"

find "$OUTPUT_ROOT" -type f \
  \( -name '*.csv' -o -name '*.tex' -o -name '*.json' \) \
  | sort

RESULT_ZIP="/kaggle/working/hc_session_${SESSION}_results.zip"
if [[ ! -d /kaggle/working ]]; then
  RESULT_ZIP="$PROJECT_ROOT/hc_session_${SESSION}_results.zip"
fi

rm -f "$RESULT_ZIP"
zip -qr "$RESULT_ZIP" \
  "$OUTPUT_ROOT" \
  "$DATA_ROOT/splits/$NEW_DATASET" \
  configs/hc_ablation \
  "$PAPER_CONFIG_DIR"

ls -lh "$RESULT_ZIP"
echo "Session ${SESSION} completed: $RESULT_ZIP"
