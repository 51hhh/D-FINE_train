import argparse
import csv
import json
from pathlib import Path


def from_csv(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = int(row["image_id"])
            level = row["blur_level"].strip()
            out[image_id] = level
    return out


def from_filename_rules(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # Rule:
    # filename contains "_blurheavy" -> heavy
    # filename contains "_blurmild"  -> mild
    # else -> clear
    out = {}
    for im in gt.get("images", []):
        name = str(im.get("file_name", "")).lower()
        if "_blurheavy" in name:
            level = "heavy"
        elif "_blurmild" in name:
            level = "mild"
        else:
            level = "clear"
        out[int(im["id"])] = level
    return out


def main():
    parser = argparse.ArgumentParser(description="Create blur manifest JSON: {image_id: blur_level}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--from-csv", default="", help="CSV with columns image_id,blur_level")
    parser.add_argument("--from-gt", default="", help="COCO gt json, infer blur level from filename rules")
    args = parser.parse_args()

    if args.from_csv:
        data = from_csv(args.from_csv)
    elif args.from_gt:
        data = from_filename_rules(args.from_gt)
    else:
        raise ValueError("Either --from-csv or --from-gt is required")

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
