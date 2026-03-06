import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# ==========================
# Hardcoded runtime settings
# ==========================
SETTINGS = {
    "images_dir": "coco/images",
    "xml_dir": "coco/Annotations",
    "output_dir": "coco/converted",
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
}


def parse_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    filename = (root.findtext("filename") or "").strip()
    size = root.find("size")
    if size is None:
        return None

    w = int(float(size.findtext("width", default="0")))
    h = int(float(size.findtext("height", default="0")))
    if w <= 0 or h <= 0:
        return None

    objs = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue

        xmin = float(box.findtext("xmin", default="0"))
        ymin = float(box.findtext("ymin", default="0"))
        xmax = float(box.findtext("xmax", default="0"))
        ymax = float(box.findtext("ymax", default="0"))
        if xmax <= xmin or ymax <= ymin:
            continue
        objs.append((name, xmin, ymin, xmax, ymax))

    if not objs:
        return None
    return filename, w, h, objs


def find_img(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def to_coco(samples, cls2id):
    images, annotations = [], []
    ann_id = 1
    for img_id, s in enumerate(samples, start=1):
        images.append({
            "id": img_id,
            "file_name": s["file_name"],
            "width": s["width"],
            "height": s["height"],
        })
        for cls_name, xmin, ymin, xmax, ymax in s["objects"]:
            w = xmax - xmin
            h = ymax - ymin
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls2id[cls_name],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [
        {"id": cid, "name": name, "supercategory": "object"}
        for name, cid in sorted(cls2id.items(), key=lambda x: x[1])
    ]
    return {"images": images, "annotations": annotations, "categories": categories}


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if abs(SETTINGS["train_ratio"] + SETTINGS["val_ratio"] + SETTINGS["test_ratio"] - 1.0) > 1e-6:
        raise ValueError("train/val/test ratio sum must be 1.0")

    images_dir = Path(SETTINGS["images_dir"]).resolve()
    xml_dir = Path(SETTINGS["xml_dir"]).resolve()
    out_dir = Path(SETTINGS["output_dir"]).resolve()

    xml_files = sorted([p for p in xml_dir.glob("*.xml") if not p.name.startswith(".")])
    if not xml_files:
        raise FileNotFoundError(f"No xml files found in {xml_dir}")

    samples = []
    cls_names = set()
    skipped = 0

    for xp in xml_files:
        parsed = parse_xml(xp)
        if parsed is None:
            skipped += 1
            continue

        filename, w, h, objs = parsed
        stem = Path(filename).stem if filename else xp.stem
        img_path = find_img(images_dir, stem)
        if img_path is None:
            skipped += 1
            continue

        cls_names.update([o[0] for o in objs])
        samples.append({
            "file_name": img_path.name,
            "width": w,
            "height": h,
            "objects": objs,
            "abs_path": img_path.as_posix(),
        })

    if not samples:
        raise RuntimeError("No valid samples generated")

    cls2id = {name: i for i, name in enumerate(sorted(cls_names))}

    random.Random(SETTINGS["seed"]).shuffle(samples)
    n = len(samples)
    n_train = int(n * SETTINGS["train_ratio"])
    n_val = int(n * SETTINGS["val_ratio"])

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    ann_dir = out_dir / "annotations"
    split_dir = out_dir / "splits"

    dump_json(ann_dir / "train.json", to_coco(train, cls2id))
    dump_json(ann_dir / "val.json", to_coco(val, cls2id))
    dump_json(ann_dir / "test.json", to_coco(test, cls2id))

    dump_lines(split_dir / "train.txt", [x["abs_path"] for x in train])
    dump_lines(split_dir / "val.txt", [x["abs_path"] for x in val])
    dump_lines(split_dir / "test.txt", [x["abs_path"] for x in test])

    ultralytics_yaml = out_dir / "ultralytics_data.yaml"
    ultralytics_yaml.write_text(
        "\n".join([
            f"train: {(split_dir / 'train.txt').as_posix()}",
            f"val: {(split_dir / 'val.txt').as_posix()}",
            f"test: {(split_dir / 'test.txt').as_posix()}",
            f"nc: {len(cls2id)}",
            f"names: {[name for name, _ in sorted(cls2id.items(), key=lambda x: x[1])]}",
        ]) + "\n",
        encoding="utf-8",
    )

    summary = {
        "images_total": n,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "classes": sorted(cls_names),
        "num_classes": len(cls_names),
        "skipped_files": skipped,
        "output_dir": out_dir.as_posix(),
    }
    dump_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
