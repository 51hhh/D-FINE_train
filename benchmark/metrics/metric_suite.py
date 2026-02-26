import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def xywh_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def build_gt_index(gt):
    img_size = {}
    for im in gt["images"]:
        img_size[im["id"]] = (float(im["width"]), float(im["height"]))

    gt_by_img_cls = defaultdict(list)
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        image_id = ann["image_id"]
        cls = ann["category_id"]
        box = ann["bbox"]
        gt_by_img_cls[(image_id, cls)].append({"bbox": box, "id": ann["id"]})
    return img_size, gt_by_img_cls


def build_pred_index(pred):
    pred_by_img_cls = defaultdict(list)
    for det in pred:
        image_id = det["image_id"]
        cls = det["category_id"]
        pred_by_img_cls[(image_id, cls)].append(det)

    for k in pred_by_img_cls:
        pred_by_img_cls[k].sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return pred_by_img_cls


def compute_recall(gt_path, pred_path, iou_thr=0.5, small_area_ratio=None, image_filter=None):
    gt = load_json(gt_path)
    pred = load_json(pred_path)
    img_size, gt_by_img_cls = build_gt_index(gt)
    pred_by_img_cls = build_pred_index(pred)

    total_gt = 0
    total_tp = 0

    keys = set(gt_by_img_cls.keys())
    for key in keys:
        image_id, cls = key
        if image_filter is not None and image_id not in image_filter:
            continue

        gts = gt_by_img_cls[key]
        if small_area_ratio is not None:
            w, h = img_size[image_id]
            area_thr = w * h * small_area_ratio
            gts = [g for g in gts if g["bbox"][2] * g["bbox"][3] <= area_thr]

        if not gts:
            continue

        preds = pred_by_img_cls.get(key, [])
        matched = set()
        tp = 0

        for det in preds:
            best_iou = 0.0
            best_idx = -1
            for i, g in enumerate(gts):
                if i in matched:
                    continue
                iou = xywh_iou(det["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thr and best_idx >= 0:
                matched.add(best_idx)
                tp += 1

        total_tp += tp
        total_gt += len(gts)

    recall = (total_tp / total_gt) if total_gt > 0 else 0.0
    return {
        "total_gt": total_gt,
        "true_positive": total_tp,
        "recall": recall,
        "iou_thr": iou_thr,
        "small_area_ratio": small_area_ratio,
    }


def load_blur_manifest(path):
    data = load_json(path)
    if isinstance(data, dict):
        # {"image_id": "clear"}
        return {int(k): str(v) for k, v in data.items()}

    # [{"image_id": 1, "blur_level": "clear"}, ...]
    out = {}
    for item in data:
        out[int(item["image_id"])] = str(item["blur_level"])
    return out


def compute_blur_robustness(gt_path, pred_path, blur_manifest_path, iou_thr=0.5):
    blur_map = load_blur_manifest(blur_manifest_path)
    levels = sorted(set(blur_map.values()))
    by_level = {}

    for level in levels:
        image_filter = {img_id for img_id, lv in blur_map.items() if lv == level}
        by_level[level] = compute_recall(
            gt_path=gt_path,
            pred_path=pred_path,
            iou_thr=iou_thr,
            small_area_ratio=None,
            image_filter=image_filter,
        )

    clear = by_level.get("clear", {}).get("recall", None)
    drop = {}
    if clear is not None and clear > 0:
        for level, r in by_level.items():
            drop[level] = 1.0 - (r["recall"] / clear)

    return {
        "by_level": by_level,
        "recall_drop_vs_clear": drop,
        "iou_thr": iou_thr,
    }


def percentile(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return float(d0 + d1)


def compute_jitter(track_csv_path, min_track_len=8):
    tracks = defaultdict(list)
    with open(track_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame", "track_id", "x", "y", "w", "h"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in track csv: {sorted(missing)}")
        for row in reader:
            tid = int(row["track_id"])
            frame = int(row["frame"])
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h"])
            cx = x + w / 2.0
            cy = y + h / 2.0
            tracks[tid].append((frame, cx, cy, w, h))

    jerk_vals = []
    area_log_std_vals = []
    used_tracks = 0

    for tid, items in tracks.items():
        items.sort(key=lambda t: t[0])
        if len(items) < min_track_len:
            continue

        c = [(it[1], it[2]) for it in items]
        s = [max(1.0, it[3] * it[4]) for it in items]
        avg_size = sum((it[3] + it[4]) * 0.5 for it in items) / len(items)

        per_track_jerk = []
        for i in range(2, len(c)):
            ax = c[i][0] - 2 * c[i - 1][0] + c[i - 2][0]
            ay = c[i][1] - 2 * c[i - 1][1] + c[i - 2][1]
            j = math.sqrt(ax * ax + ay * ay)
            per_track_jerk.append(j / max(avg_size, 1e-6))

        if not per_track_jerk:
            continue

        logs = [math.log(v) for v in s]
        mu = sum(logs) / len(logs)
        var = sum((x - mu) ** 2 for x in logs) / len(logs)
        area_log_std = math.sqrt(var)

        jerk_vals.extend(per_track_jerk)
        area_log_std_vals.append(area_log_std)
        used_tracks += 1

    if not jerk_vals:
        return {
            "tracks_used": 0,
            "center_jerk_norm_mean": 0.0,
            "center_jerk_norm_median": 0.0,
            "center_jerk_norm_p90": 0.0,
            "area_log_std_mean": 0.0,
            "area_log_std_median": 0.0,
        }

    return {
        "tracks_used": used_tracks,
        "center_jerk_norm_mean": sum(jerk_vals) / len(jerk_vals),
        "center_jerk_norm_median": median(jerk_vals),
        "center_jerk_norm_p90": percentile(jerk_vals, 90),
        "area_log_std_mean": sum(area_log_std_vals) / len(area_log_std_vals) if area_log_std_vals else 0.0,
        "area_log_std_median": median(area_log_std_vals) if area_log_std_vals else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Metric suite for volleyball detector benchmarking")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_small = sub.add_parser("small-recall")
    p_small.add_argument("--gt", required=True)
    p_small.add_argument("--pred", required=True)
    p_small.add_argument("--iou-thr", type=float, default=0.5)
    p_small.add_argument("--small-area-ratio", type=float, default=0.01)
    p_small.add_argument("--output", required=True)

    p_blur = sub.add_parser("blur-robustness")
    p_blur.add_argument("--gt", required=True)
    p_blur.add_argument("--pred", required=True)
    p_blur.add_argument("--blur-manifest", required=True)
    p_blur.add_argument("--iou-thr", type=float, default=0.5)
    p_blur.add_argument("--output", required=True)

    p_jitter = sub.add_parser("jitter")
    p_jitter.add_argument("--tracks", required=True, help="CSV with columns: frame,track_id,x,y,w,h")
    p_jitter.add_argument("--min-track-len", type=int, default=8)
    p_jitter.add_argument("--output", required=True)

    p_all = sub.add_parser("all")
    p_all.add_argument("--gt", required=True)
    p_all.add_argument("--pred", required=True)
    p_all.add_argument("--tracks", required=True)
    p_all.add_argument("--blur-manifest", required=True)
    p_all.add_argument("--iou-thr", type=float, default=0.5)
    p_all.add_argument("--small-area-ratio", type=float, default=0.01)
    p_all.add_argument("--min-track-len", type=int, default=8)
    p_all.add_argument("--output", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.cmd == "small-recall":
        out = compute_recall(
            gt_path=args.gt,
            pred_path=args.pred,
            iou_thr=args.iou_thr,
            small_area_ratio=args.small_area_ratio,
        )
        save_json(out, args.output)
        return

    if args.cmd == "blur-robustness":
        out = compute_blur_robustness(
            gt_path=args.gt,
            pred_path=args.pred,
            blur_manifest_path=args.blur_manifest,
            iou_thr=args.iou_thr,
        )
        save_json(out, args.output)
        return

    if args.cmd == "jitter":
        out = compute_jitter(track_csv_path=args.tracks, min_track_len=args.min_track_len)
        save_json(out, args.output)
        return

    if args.cmd == "all":
        out = {
            "small_recall": compute_recall(
                gt_path=args.gt,
                pred_path=args.pred,
                iou_thr=args.iou_thr,
                small_area_ratio=args.small_area_ratio,
            ),
            "box_stability": compute_jitter(
                track_csv_path=args.tracks,
                min_track_len=args.min_track_len,
            ),
            "motion_blur_robustness": compute_blur_robustness(
                gt_path=args.gt,
                pred_path=args.pred,
                blur_manifest_path=args.blur_manifest,
                iou_thr=args.iou_thr,
            ),
        }
        save_json(out, args.output)
        return


if __name__ == "__main__":
    main()
