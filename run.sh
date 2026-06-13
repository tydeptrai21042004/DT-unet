#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
ALLOW_INSECURE_DOWNLOAD="${ALLOW_INSECURE_DOWNLOAD:-0}"
KAGGLE_HANDLE="${KAGGLE_HANDLE:-}"
DATASET="${DATASET:-kvasir_seg}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
IMAGE_SIZE="${IMAGE_SIZE:-352}"
DEVICE="${DEVICE:-auto}"
MODELS_DEFAULT="unet,attention_unet,unet_cbam,unetpp,resunetpp,pranet,acsnet,hardnet_mseg,polyp_pvt,caranet,cfanet,hsnet,csca_unet,proposal_hf_unet"
MODELS="${MODELS:-$MODELS_DEFAULT}"
SEEDS="${SEEDS:-42,1337,2024}"

usage() {
  cat <<EOF2
Usage:
  bash run.sh install
  bash run.sh prepare [--source-dir PATH | --zip-path PATH | --download-url URL]
  bash run.sh splits
  bash run.sh train-one MODEL
  bash run.sh train-all
  bash run.sh eval-one MODEL [SPLIT]
  bash run.sh eval-all [SPLIT]
  bash run.sh benchmark
  bash run.sh benchmark-strict-no-aux
  bash run.sh benchmark-pretrained
  bash run.sh benchmark-seeds
  bash run.sh ablation
  bash run.sh placement-ablation
  bash run.sh hc-ablation
  bash run.sh aggregate-seeds
  bash run.sh export
  bash run.sh download-backbones

Environment overrides:
  PYTHON_BIN   Python executable (default: python)
  CONFIG_DIR   Config directory (default: configs; use configs/paper_fair, configs/strict_no_aux, or configs/paper_fair_pretrained)
  DATASET      Dataset key (default: kvasir_seg)
  DATA_ROOT    Data root (default: data)
  OUTPUT_ROOT  Output root (default: outputs)
  IMAGE_SIZE   Image size (default: 352)
  DEVICE       Device string or auto (default: auto)
  MODELS       Comma-separated model list for train-all/eval-all/benchmark
  SEEDS        Comma-separated seeds for benchmark-seeds/aggregate-seeds (default: 42,1337,2024)
  ALLOW_INSECURE_DOWNLOAD  Set to 1 to bypass TLS verification for dataset download when needed
  KAGGLE_HANDLE Optional Kaggle dataset handle override (owner/dataset)
EOF2
}

cmd_install() {
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r requirements.txt
}

cmd_prepare() {
  local extra=()
  if [[ "$ALLOW_INSECURE_DOWNLOAD" == "1" ]]; then
    extra+=(--allow-insecure-download)
  fi
  if [[ -n "$KAGGLE_HANDLE" ]]; then
    extra+=(--kaggle-handle "$KAGGLE_HANDLE")
  fi
  "$PYTHON_BIN" scripts/prepare_dataset.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    "${extra[@]}" \
    "$@"
}

cmd_splits() {
  "$PYTHON_BIN" scripts/make_splits.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE"
}

cmd_train_one() {
  local model="${1:?Missing model name}"
  "$PYTHON_BIN" scripts/train_one.py \
    --model "$model" \
    --dataset "$DATASET" \
    --config "$CONFIG_DIR/$model.yaml" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT"
}

cmd_train_all() {
  "$PYTHON_BIN" scripts/train_all.py \
    --models "$MODELS" \
    --dataset "$DATASET" \
    --config-dir "$CONFIG_DIR" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT"
}

cmd_eval_one() {
  local model="${1:?Missing model name}"
  local split="${2:-test}"
  "$PYTHON_BIN" scripts/eval_one.py \
    --model "$model" \
    --dataset "$DATASET" \
    --config "$CONFIG_DIR/$model.yaml" \
    --split "$split" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT"
}

cmd_eval_all() {
  local split="${1:-test}"
  "$PYTHON_BIN" scripts/eval_all.py \
    --models "$MODELS" \
    --dataset "$DATASET" \
    --config-dir "$CONFIG_DIR" \
    --split "$split" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT"
}

cmd_benchmark() {
  local extra=()
  if [[ "$ALLOW_INSECURE_DOWNLOAD" == "1" ]]; then
    extra+=(--allow-insecure-download)
  fi
  if [[ -n "$KAGGLE_HANDLE" ]]; then
    extra+=(--kaggle-handle "$KAGGLE_HANDLE")
  fi
  "$PYTHON_BIN" scripts/benchmark_all.py \
    --models "$MODELS" \
    --dataset "$DATASET" \
    --config-dir "$CONFIG_DIR" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT" \
    "${extra[@]}"
}


cmd_benchmark_strict_no_aux() {
  CONFIG_DIR="configs/strict_no_aux" OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_strict_no_aux}" cmd_benchmark "$@"
}

cmd_benchmark_pretrained() {
  CONFIG_DIR="configs/paper_fair_pretrained" OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_paper_fair_pretrained}" cmd_benchmark "$@"
}

cmd_download_backbones() {
  "$PYTHON_BIN" scripts/download_official_backbones.py "$@"
}

cmd_benchmark_seeds() {
  local extra=()
  if [[ "$ALLOW_INSECURE_DOWNLOAD" == "1" ]]; then
    extra+=(--allow-insecure-download)
  fi
  if [[ -n "$KAGGLE_HANDLE" ]]; then
    extra+=(--kaggle-handle "$KAGGLE_HANDLE")
  fi
  "$PYTHON_BIN" scripts/benchmark_multi_seed.py \
    --models "$MODELS" \
    --dataset "$DATASET" \
    --config-dir "$CONFIG_DIR" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT" \
    --seeds "$SEEDS" \
    "${extra[@]}" \
    "$@"
}

cmd_ablation() {
  local first_seed="${SEEDS%%,*}"
  "$PYTHON_BIN" scripts/run_compact_hf_ablation.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --seed "$first_seed" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT" \
    "$@"
}


cmd_placement_ablation() {
  local first_seed="${SEEDS%%,*}"
  "$PYTHON_BIN" scripts/run_hf_placement_ablation.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --seed "$first_seed" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT" \
    "$@"
}

cmd_hc_ablation() {
  local extra=()
  if [[ "$ALLOW_INSECURE_DOWNLOAD" == "1" ]]; then
    extra+=(--allow-insecure-download)
  fi
  if [[ -n "$KAGGLE_HANDLE" ]]; then
    extra+=(--kaggle-handle "$KAGGLE_HANDLE")
  fi
  "$PYTHON_BIN" scripts/run_hc_ablation.py \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT" \
    --seeds "$SEEDS" \
    "${extra[@]}" \
    "$@"
}

cmd_aggregate_seeds() {
  "$PYTHON_BIN" scripts/aggregate_seed_results.py \
    --output-root "$OUTPUT_ROOT" \
    --seeds "$SEEDS" \
    "$@"
}

cmd_export() {
  "$PYTHON_BIN" scripts/export_results.py --output-root "$OUTPUT_ROOT"
}

main() {
  local action="${1:-}"
  shift || true

  case "$action" in
    install) cmd_install "$@" ;;
    prepare) cmd_prepare "$@" ;;
    splits) cmd_splits "$@" ;;
    train-one) cmd_train_one "$@" ;;
    train-all) cmd_train_all "$@" ;;
    eval-one) cmd_eval_one "$@" ;;
    eval-all) cmd_eval_all "$@" ;;
    benchmark) cmd_benchmark "$@" ;;
    benchmark-strict-no-aux) cmd_benchmark_strict_no_aux "$@" ;;
    benchmark-pretrained) cmd_benchmark_pretrained "$@" ;;
    benchmark-seeds) cmd_benchmark_seeds "$@" ;;
    ablation) cmd_ablation "$@" ;;
    placement-ablation) cmd_placement_ablation "$@" ;;
    hc-ablation) cmd_hc_ablation "$@" ;;
    aggregate-seeds) cmd_aggregate_seeds "$@" ;;
    export) cmd_export "$@" ;;
    download-backbones) cmd_download_backbones "$@" ;;
    -h|--help|help|"") usage ;;
    *)
      echo "Unknown command: $action" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
