#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash cloud_train.sh [--config PATH] [--output-dir PATH] [--summary-dir PATH]
                      [--tuning PATH] [--seed N] [--update KEY=VALUE]...
                      [--no-amp]

Examples:
  bash cloud_train.sh \
    --config config/volleyball_s_obj2coco_d_bg2.yml \
    --output-dir runs/gradmotion/no-a/output \
    --summary-dir runs/gradmotion/no-a/summary

  bash cloud_train.sh \
    --config config/volleyball_s_obj2coco_d_bg2.yml \
    --output-dir runs/gradmotion/smoke-no-a/output \
    --summary-dir runs/gradmotion/smoke-no-a/summary \
    --update epochs=1 \
    --update checkpoint_freq=1
EOF
}

to_abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$PROJECT_ROOT/$1" ;;
  esac
}

CONFIG_PATH="config/volleyball_s_obj2coco_d_bg2.yml"
OUTPUT_DIR=""
SUMMARY_DIR=""
TUNING_PATH="weights/dfine_s_obj2coco.pth"
SEED="0"
USE_AMP=1
TRAIN_UPDATES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --summary-dir)
      SUMMARY_DIR="$2"
      shift 2
      ;;
    --tuning)
      TUNING_PATH="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --update)
      TRAIN_UPDATES+=("$2")
      shift 2
      ;;
    --no-amp)
      USE_AMP=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
TRAIN_ROOT="$PROJECT_ROOT/D-FINE"
TRAIN_ENTRY="$TRAIN_ROOT/train.py"
CONFIG_ABS="$(to_abs_path "$CONFIG_PATH")"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$PROJECT_ROOT/runs/gradmotion/default/output"
fi
if [ -z "$SUMMARY_DIR" ]; then
  SUMMARY_DIR="$PROJECT_ROOT/runs/gradmotion/default/summary"
fi

OUTPUT_DIR="$(to_abs_path "$OUTPUT_DIR")"
SUMMARY_DIR="$(to_abs_path "$SUMMARY_DIR")"

PERSONAL_ROOT="/personal/personal"
if [ ! -d "$PERSONAL_ROOT" ]; then
  PERSONAL_ROOT="/personal"
fi

IMAGES_ZIP="$PERSONAL_ROOT/images.zip"
NEGATIVE_ZIP="$PERSONAL_ROOT/negative_samples.zip"
DATASET_ROOT="$PROJECT_ROOT/coco"
IMAGES_DIR="$DATASET_ROOT/images"
ANNOTATIONS_DIR="$DATASET_ROOT/Annotations"
CONVERTED_DIR="$DATASET_ROOT/converted"
NEGATIVE_IMG_DIR="$IMAGES_DIR/negative_samples"

print_artifact_summary() {
  echo "[postflight] output_dir=$OUTPUT_DIR"
  if [ -d "$OUTPUT_DIR" ]; then
    find "$OUTPUT_DIR" -maxdepth 2 -type f \( -name 'launch.log' -o -name 'log.txt' -o -name '*.pth' -o -name 'events.out.tfevents*' -o -name '*.json' \) | sort || true
  else
    echo "[postflight] output_dir missing"
  fi

  echo "[postflight] summary_dir=$SUMMARY_DIR"
  if [ -d "$SUMMARY_DIR" ]; then
    find "$SUMMARY_DIR" -maxdepth 2 -type f | sort || true
  else
    echo "[postflight] summary_dir missing"
  fi
}

on_exit() {
  status=$?
  echo "[postflight] exit_status=$status"
  print_artifact_summary || true
  exit "$status"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_DIR" "$SUMMARY_DIR" "$IMAGES_DIR" "$ANNOTATIONS_DIR" "$CONVERTED_DIR/annotations"
exec > >(tee -a "$OUTPUT_DIR/launch.log") 2>&1

echo "=== D-FINE GradMotion Cloud Launch ==="
echo "[preflight] pwd=$(pwd)"
echo "[preflight] project_root=$PROJECT_ROOT"
echo "[preflight] train_root=$TRAIN_ROOT"
echo "[preflight] config=$CONFIG_ABS"
echo "[preflight] images_zip=$IMAGES_ZIP"
echo "[preflight] negative_zip=$NEGATIVE_ZIP"
echo "[preflight] output_dir=$OUTPUT_DIR"
echo "[preflight] summary_dir=$SUMMARY_DIR"
echo "[preflight] seed=$SEED"
echo "[preflight] no-A training uses train.json only; negative_samples.zip is OA-eval-only"

test -f "$TRAIN_ENTRY"
test -f "$CONFIG_ABS"
test -f "$IMAGES_ZIP"

echo "[preflight] repo root listing"
ls -la "$PROJECT_ROOT"

echo "[preflight] config file"
ls -l "$CONFIG_ABS"

echo "[preflight] personal data root"
ls -la "$PERSONAL_ROOT" || true

echo "[env] python"
python --version

echo "[env] ensure requirements"
python -m pip install --disable-pip-version-check -r "$TRAIN_ROOT/requirements.txt" -q

echo "[env] torch + tensorboard sanity"
python - <<'PY'
import torch
from torch.utils.tensorboard import SummaryWriter
print(f'torch={torch.__version__}')
print(f'cuda_available={torch.cuda.is_available()}')
print('tensorboard=ok')
PY

echo "[data] extract images.zip -> $DATASET_ROOT"
unzip -qo "$IMAGES_ZIP" -d "$DATASET_ROOT/"

if [ -f "$NEGATIVE_ZIP" ]; then
  echo "[data] extract negative_samples.zip for OA evaluation only"
  mkdir -p "$NEGATIVE_IMG_DIR"
  unzip -qo "$NEGATIVE_ZIP" -d "$NEGATIVE_IMG_DIR/"
  if [ -d "$NEGATIVE_IMG_DIR/negative_samples" ]; then
    NEGATIVE_IMG_DIR="$NEGATIVE_IMG_DIR/negative_samples"
  fi
else
  echo "[data] negative_samples.zip not found under $PERSONAL_ROOT"
fi

echo "[data] generate COCO annotations"
(
  cd "$PROJECT_ROOT"
  python -u benchmark/scripts/prepare_dataset.py
)

echo "[data] dataset counts"
echo "[data] image_count=$(find "$IMAGES_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ')"
echo "[data] xml_count=$(find "$ANNOTATIONS_DIR" -maxdepth 1 -type f -name '*.xml' | wc -l | tr -d ' ')"
echo "[data] negative_count=$(find "$NEGATIVE_IMG_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ' || true)"
echo "[data] converted annotations"
find "$CONVERTED_DIR/annotations" -maxdepth 1 -type f | sort || true

TRAIN_CMD=(python -u train.py -c "$CONFIG_ABS" --seed "$SEED" --output-dir "$OUTPUT_DIR" --summary-dir "$SUMMARY_DIR" -t "$TUNING_PATH" -u "negative_img_dir=$NEGATIVE_IMG_DIR")
if [ "$USE_AMP" -eq 1 ]; then
  TRAIN_CMD+=(--use-amp)
fi
for update_expr in "${TRAIN_UPDATES[@]}"; do
  TRAIN_CMD+=(-u "$update_expr")
done

echo "[train] negative_img_dir=$NEGATIVE_IMG_DIR"
echo "[train] command=${TRAIN_CMD[*]}"
(
  cd "$TRAIN_ROOT"
  "${TRAIN_CMD[@]}"
)

echo "[postflight] training finished successfully"
