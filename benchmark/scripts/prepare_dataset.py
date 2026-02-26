import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

"""
直接运行脚本即可，不需要命令行参数。
如果要调整路径或划分比例，请修改下面的配置常量。
"""

# ====== 可修改配置（脚本头部）======
IMAGES_DIR = "coco/images"
XML_DIR = "coco/Annotations"
OUTPUT_DIR = "coco/converted"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
# ==================================


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image_path(images_dir: Path, stem: str):
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_voc(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    file_name = root.findtext("filename", default="").strip()
    size = root.find("size")
    width = int(size.findtext("width", default="0")) if size is not None else 0
    height = int(size.findtext("height", default="0")) if size is not None else 0

    objects = []
    for obj in root.findall("object"):
        cls = obj.findtext("name", default="").strip()
        if not cls:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", default="0"))
        ymin = float(box.findtext("ymin", default="0"))
        xmax = float(box.findtext("xmax", default="0"))
        ymax = float(box.findtext("ymax", default="0"))
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append((cls, xmin, ymin, xmax, ymax))

    return file_name, width, height, objects


def to_coco(records, cat2id):
    images = []
    annotations = []
    ann_id = 1

    for img_id, rec in enumerate(records, start=1):
        images.append(
            {
                "id": img_id,
                "file_name": rec["file_name"],
                "width": rec["width"],
                "height": rec["height"],
            }
        )
        for cls, xmin, ymin, xmax, ymax in rec["objects"]:
            w = xmax - xmin
            h = ymax - ymin
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat2id[cls],
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": i, "name": n, "supercategory": "object"} for n, i in sorted(cat2id.items(), key=lambda x: x[1])]
    return {"images": images, "annotations": annotations, "categories": categories}


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_lines(lines, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main():
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test 比例之和必须为 1.0")

    images_dir = Path(IMAGES_DIR).resolve()
    xml_dir = Path(XML_DIR).resolve()
    out_dir = Path(OUTPUT_DIR).resolve()

    xml_files = sorted([p for p in xml_dir.glob("*.xml") if not p.name.startswith(".")])
    if not xml_files:
        raise FileNotFoundError(f"未找到 XML: {xml_dir}")

    records = []
    class_names = set()
    skipped = 0

    for xml_path in xml_files:
        file_name, width, height, objects = parse_voc(xml_path)
        if not objects:
            skipped += 1
            continue

        stem = Path(file_name).stem if file_name else xml_path.stem
        image_path = find_image_path(images_dir, stem)
        if image_path is None:
            skipped += 1
            continue

        if width <= 0 or height <= 0:
            # 兜底：从图片读取尺寸（避免额外依赖，先跳过）
            skipped += 1
            continue

        class_names.update([o[0] for o in objects])
        records.append(
            {
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "objects": objects,
                "image_abs": str(image_path).replace("\\", "/"),
            }
        )

    if not records:
        raise RuntimeError("没有可用样本，请检查 XML 与图片匹配以及尺寸字段")

    # D-FINE/RT-DETR 等模型期望类别ID从0开始
    cat2id = {name: i for i, name in enumerate(sorted(class_names))}

    rnd = random.Random(SEED)
    rnd.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_recs = records[:n_train]
    val_recs = records[n_train : n_train + n_val]
    test_recs = records[n_train + n_val :]

    save_json(to_coco(train_recs, cat2id), out_dir / "annotations" / "train.json")
    save_json(to_coco(val_recs, cat2id), out_dir / "annotations" / "val.json")
    save_json(to_coco(test_recs, cat2id), out_dir / "annotations" / "test.json")

    save_lines([r["image_abs"] for r in train_recs], out_dir / "splits" / "train.txt")
    save_lines([r["image_abs"] for r in val_recs], out_dir / "splits" / "val.txt")
    save_lines([r["image_abs"] for r in test_recs], out_dir / "splits" / "test.txt")

    ultra_yaml = out_dir / "ultralytics_data.yaml"
    ultra_yaml.parent.mkdir(parents=True, exist_ok=True)
    with ultra_yaml.open("w", encoding="utf-8") as f:
        f.write(f"train: {(out_dir / 'splits' / 'train.txt').as_posix()}\n")
        f.write(f"val: {(out_dir / 'splits' / 'val.txt').as_posix()}\n")
        f.write(f"test: {(out_dir / 'splits' / 'test.txt').as_posix()}\n")
        f.write(f"nc: {len(cat2id)}\n")
        f.write(f"names: {[n for n, _ in sorted(cat2id.items(), key=lambda x: x[1])]}\n")

    summary = {
        "images_total": n,
        "train": len(train_recs),
        "val": len(val_recs),
        "test": len(test_recs),
        "classes": sorted(class_names),
        "num_classes": len(class_names),
        "skipped_files": skipped,
        "output_dir": out_dir.as_posix(),
    }
    save_json(summary, out_dir / "summary.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
