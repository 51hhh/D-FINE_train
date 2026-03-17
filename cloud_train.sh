#!/bin/bash
# Cloud Training Setup & Launch Script for D-FINE Volleyball Detection
# Target: Gradmotion cloud server with /personal storage
# Usage: bash cloud_train.sh <config_name>
# Example: bash cloud_train.sh volleyball_s_E2_lr_noscale_bg6.yml
set -e

CONFIG_NAME="${1:-volleyball_s_E2_lr_noscale_bg6.yml}"
PERSONAL="/personal"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== D-FINE Cloud Training Setup ==="
echo "Config: $CONFIG_NAME"
echo "Project: $PROJECT_ROOT"

# Step 1: Setup data directory structure
echo "[1/5] Setting up data directories..."
mkdir -p "$PROJECT_ROOT/coco/images"
mkdir -p "$PROJECT_ROOT/coco/Annotations"
mkdir -p "$PROJECT_ROOT/coco/converted/annotations"
mkdir -p "$PROJECT_ROOT/coco/images/negative_samples"

# Step 2: Unzip training data from /personal
echo "[2/5] Extracting training data..."
if [ -f "$PERSONAL/images.zip" ]; then
    unzip -qo "$PERSONAL/images.zip" -d "$PROJECT_ROOT/coco/"
    echo "  images.zip extracted"
else
    echo "  WARNING: $PERSONAL/images.zip not found!"
fi

if [ -f "$PERSONAL/negative_samples.zip" ]; then
    unzip -qo "$PERSONAL/negative_samples.zip" -d "$PROJECT_ROOT/coco/images/negative_samples/"
    echo "  negative_samples.zip extracted"
else
    echo "  WARNING: $PERSONAL/negative_samples.zip not found!"
fi

# Step 3: Generate COCO annotations from XML
echo "[3/5] Generating COCO annotations..."
cd "$PROJECT_ROOT"
python benchmark/scripts/prepare_dataset.py

# Step 4: Generate train_with_negatives.json (Method A)
echo "[4/5] Generating negative-augmented training set..."
python -c "
import json
from pathlib import Path
from PIL import Image
from datetime import datetime

neg_dir = Path('coco/images/negative_samples')
train_file = Path('coco/converted/annotations/train.json')
output_file = Path('coco/converted/annotations/train_with_negatives.json')

train_data = json.load(open(train_file))
max_id = max(img['id'] for img in train_data['images'])

neg_imgs = []
img_id = max_id + 1
for p in sorted(neg_dir.glob('*.jpg')):
    try:
        w, h = Image.open(p).size
        neg_imgs.append({'id': img_id, 'file_name': f'negative_samples/{p.name}', 'width': w, 'height': h})
        img_id += 1
    except Exception:
        pass

merged = {
    'images': train_data['images'] + neg_imgs,
    'annotations': train_data['annotations'],
    'categories': train_data['categories'],
}
output_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
print(f'Generated: {len(train_data[\"images\"])} train + {len(neg_imgs)} neg = {len(merged[\"images\"])} total')
"

# Step 5: Install deps and launch training
echo "[5/5] Starting training..."
cd "$PROJECT_ROOT/D-FINE"
pip install -r requirements.txt -q 2>/dev/null || true

python train.py \
    -c "../config/$CONFIG_NAME" \
    --use-amp \
    --seed 0 \
    -t weights/dfine_s_obj2coco.pth
