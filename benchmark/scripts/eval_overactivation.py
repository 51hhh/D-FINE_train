"""D-FINE Query Over-Activation 评价脚本

评估 D-FINE 单类模型的 Query Over-Activation 问题严重程度。
两部分评估:
  1. 在负样本图片(无目标)上计算误检指标
  2. 在 val 集上计算标准 mAP

用法:
  cd D-FINE_train
  python benchmark/scripts/eval_overactivation.py

输出:
  - 控制台表格报告
  - output/eval_overactivation/report.json
  - output/eval_overactivation/confidence_histogram.png
  - output/eval_overactivation/per_image_details.csv
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ==========================
# 配置 — 修改这里即可
# ==========================
SETTINGS = {
    "repo_dir": "./D-FINE",
    "config": "./config/volleyball_s_transfer.yml",
    "checkpoint": "./best_stg2.pth",
    "device": "cuda:0",
    "input_size": 640,

    # 负样本图片目录 (无排球的场景图片)
    "negative_img_dir": "./coco/images/negative_samples",

    # 标准 val 集 (用于 mAP)
    "val_ann_file": "./coco/converted/annotations/val.json",
    "val_img_dir": "./coco/images",

    # 评估阈值
    "conf_thresholds": [0.3, 0.5, 0.7],

    # 输出目录
    "output_dir": "./output/eval_overactivation",

    # 是否生成直方图 (需要 matplotlib)
    "save_histograms": True,

    # 图片后缀 (扫描负样本目录时使用)
    "img_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
}


# ==========================
# 模型加载 (复用 run_fo_eval.py 的模式)
# ==========================
def prepare_windows_readable_config(config_path: Path) -> Path:
    """D-FINE 的 yaml loader 在 Windows 上可能用 GBK 解码 UTF-8 注释导致报错。
    创建一个 ASCII-only 的临时副本。"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            text = config_path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise RuntimeError(f"Cannot decode config file: {config_path}")

    safe_text = text.encode("ascii", errors="ignore").decode("ascii")
    tmp_path = config_path.parent / f".eval_tmp_{config_path.stem}.yml"
    tmp_path.write_text(safe_text, encoding="utf-8")
    return tmp_path


def cleanup_temp_config(tmp_path: Path, original_config_path: Path):
    if tmp_path.resolve() == original_config_path.resolve():
        return
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass


def load_dfine_model(repo_dir: Path, config_path: str, checkpoint_path: str, device: str):
    """加载 D-FINE 模型和后处理器。"""
    sys.path.insert(0, str(repo_dir.resolve()))
    from src.core import YAMLConfig

    safe_config_path = prepare_windows_readable_config(Path(config_path))
    try:
        cfg = YAMLConfig(str(safe_config_path), resume=checkpoint_path)
    finally:
        cleanup_temp_config(safe_config_path, Path(config_path))

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state, strict=False)

    model = cfg.model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device).eval()
    return model, postprocessor


def unpack_predictions(pred_output):
    """兼容 deploy mode tuple 和 list[dict] 两种输出格式。"""
    if isinstance(pred_output, (tuple, list)) and len(pred_output) == 3 and torch.is_tensor(pred_output[0]):
        labels_b, boxes_b, scores_b = pred_output
        return labels_b[0], boxes_b[0], scores_b[0]
    if isinstance(pred_output, list) and len(pred_output) > 0 and isinstance(pred_output[0], dict):
        first = pred_output[0]
        return first["labels"], first["boxes"], first["scores"]
    raise TypeError(f"Unsupported prediction output type: {type(pred_output)}")


# ==========================
# 负样本评估
# ==========================
@torch.no_grad()
def evaluate_negatives(model, postprocessor, neg_img_dir: Path, input_size: int,
                       device: str, conf_thresholds: list, img_extensions: list):
    """在负样本图片上运行推理，收集所有检测分数。"""
    tfm = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])

    all_scores = []
    per_image = []

    img_files = sorted([
        f for f in neg_img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in img_extensions
    ])

    if not img_files:
        print(f"[WARNING] 负样本目录 {neg_img_dir} 中未找到图片")
        return all_scores, per_image

    print(f"\n--- 评估负样本 ({len(img_files)} 张图片) ---")
    for i, img_path in enumerate(img_files):
        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            tensor = tfm(image).unsqueeze(0).to(device)
            orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)

            outputs = model(tensor)
            pred_output = postprocessor(outputs, orig_size)
            _, boxes_t, scores_t = unpack_predictions(pred_output)

            scores_list = scores_t.cpu().tolist()
            boxes_list = boxes_t.cpu().tolist()
            all_scores.extend(scores_list)

            per_image.append({
                "filename": img_path.name,
                "total_detections": len(scores_list),
                "scores": scores_list,
                "boxes": boxes_list,
                "dets_at_threshold": {
                    str(t): sum(1 for s in scores_list if s >= t)
                    for t in conf_thresholds
                },
                "max_score": max(scores_list) if scores_list else 0.0,
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(img_files)}]")

        except Exception as e:
            print(f"  [ERROR] 跳过 {img_path.name}: {e}")

    return all_scores, per_image


def compute_overactivation_metrics(all_scores, per_image, thresholds):
    """从原始分数计算 Over-Activation 指标。"""
    n_images = len(per_image)
    report = {"num_negative_images": n_images}

    for t in thresholds:
        above = [s for s in all_scores if s >= t]
        total_fp = len(above)
        clean_images = sum(
            1 for img in per_image if img["dets_at_threshold"][str(t)] == 0
        )

        report[f"threshold_{t}"] = {
            "total_false_detections": total_fp,
            "fppi": round(total_fp / max(n_images, 1), 4),
            "clean_image_rate": round(clean_images / max(n_images, 1), 4),
            "mean_confidence": round(float(np.mean(above)), 4) if above else 0.0,
            "median_confidence": round(float(np.median(above)), 4) if above else 0.0,
        }

    if all_scores:
        report["max_confidence_overall"] = round(max(all_scores), 4)
        report["mean_confidence_overall"] = round(float(np.mean(all_scores)), 4)
    else:
        report["max_confidence_overall"] = 0.0
        report["mean_confidence_overall"] = 0.0

    return report


# ==========================
# 标准 mAP 评估
# ==========================
@torch.no_grad()
def evaluate_map(model, postprocessor, ann_file: Path, img_dir: Path,
                 input_size: int, device: str):
    """在 val 集上计算标准 COCO mAP。"""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("[WARNING] pycocotools 未安装，跳过 mAP 评估")
        return None

    tfm = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])

    coco_gt = COCO(str(ann_file))
    img_ids = coco_gt.getImgIds()
    results = []

    print(f"\n--- 评估 mAP ({len(img_ids)} 张图片) ---")
    for i, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]

        if not img_path.exists():
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            tensor = tfm(image).unsqueeze(0).to(device)
            orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)

            outputs = model(tensor)
            pred_output = postprocessor(outputs, orig_size)
            labels_t, boxes_t, scores_t = unpack_predictions(pred_output)

            labels = labels_t.cpu().tolist()
            boxes = boxes_t.cpu().tolist()
            scores = scores_t.cpu().tolist()

            # 获取 category_id 映射
            cat_ids = coco_gt.getCatIds()

            for label_id, box, score in zip(labels, boxes, scores):
                x1, y1, x2, y2 = box
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)

                # label_id 是 0-based，category_id 可能也是 0-based
                if label_id in cat_ids:
                    cat_id = label_id
                elif len(cat_ids) == 1:
                    cat_id = cat_ids[0]
                else:
                    cat_id = label_id

                results.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "score": round(score, 6),
                })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(img_ids)}]")

        except Exception as e:
            print(f"  [ERROR] 跳过 {img_path.name}: {e}")

    if not results:
        print("[WARNING] 无预测结果")
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP": round(coco_eval.stats[0], 4),
        "mAP_50": round(coco_eval.stats[1], 4),
        "mAP_75": round(coco_eval.stats[2], 4),
        "mAP_small": round(coco_eval.stats[3], 4),
        "mAP_medium": round(coco_eval.stats[4], 4),
        "mAP_large": round(coco_eval.stats[5], 4),
    }


# ==========================
# 报告生成
# ==========================
def print_report(oa_metrics, map_metrics, cfg):
    """打印控制台报告。"""
    thresholds = cfg["conf_thresholds"]

    print("\n" + "=" * 60)
    print("  D-FINE Query Over-Activation 评估报告")
    print("=" * 60)
    print(f"  模型: {cfg['checkpoint']}")
    print(f"  配置: {cfg['config']}")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if oa_metrics and oa_metrics["num_negative_images"] > 0:
        n = oa_metrics["num_negative_images"]
        print(f"\n--- Over-Activation 指标 ({n} 张负样本图片) ---")

        # 表头
        header = f"{'指标':<28}"
        for t in thresholds:
            header += f"{'@' + str(t):>10}"
        print(header)
        print("-" * (28 + 10 * len(thresholds)))

        # Total FP
        row = f"{'Total False Detections':<28}"
        for t in thresholds:
            v = oa_metrics[f"threshold_{t}"]["total_false_detections"]
            row += f"{v:>10}"
        print(row)

        # FPPI
        row = f"{'FPPI':<28}"
        for t in thresholds:
            v = oa_metrics[f"threshold_{t}"]["fppi"]
            row += f"{v:>10.2f}"
        print(row)

        # Clean Image Rate
        row = f"{'Clean Image Rate':<28}"
        for t in thresholds:
            v = oa_metrics[f"threshold_{t}"]["clean_image_rate"]
            row += f"{v * 100:>9.1f}%"
        print(row)

        # Mean FP Confidence
        row = f"{'Mean FP Confidence':<28}"
        for t in thresholds:
            v = oa_metrics[f"threshold_{t}"]["mean_confidence"]
            row += f"{v:>10.3f}"
        print(row)

        # Median FP Confidence
        row = f"{'Median FP Confidence':<28}"
        for t in thresholds:
            v = oa_metrics[f"threshold_{t}"]["median_confidence"]
            row += f"{v:>10.3f}"
        print(row)

        print(f"\n  Max Confidence (any threshold): {oa_metrics['max_confidence_overall']:.4f}")
        print(f"  Overall Mean Score:             {oa_metrics['mean_confidence_overall']:.4f}")
    else:
        print("\n--- 无负样本评估结果 ---")

    if map_metrics:
        print(f"\n--- 标准 mAP (val 集) ---")
        print(f"  mAP:    {map_metrics['mAP']:.4f}")
        print(f"  mAP@50: {map_metrics['mAP_50']:.4f}")
        print(f"  mAP@75: {map_metrics['mAP_75']:.4f}")
        print(f"  mAP_s:  {map_metrics['mAP_small']:.4f}")
        print(f"  mAP_m:  {map_metrics['mAP_medium']:.4f}")
        print(f"  mAP_l:  {map_metrics['mAP_large']:.4f}")

    print("=" * 60)


def save_histogram(all_scores, output_dir: Path, thresholds: list):
    """保存置信度分布直方图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装，跳过直方图生成")
        return

    if not all_scores:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_scores, bins=50, range=(0, 1), color="steelblue", edgecolor="white", alpha=0.8)

    for t in thresholds:
        ax.axvline(x=t, color="red", linestyle="--", alpha=0.6, label=f"threshold={t}")

    ax.set_xlabel("Detection Confidence Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence Distribution on Negative Images\n(All detections, no ground truth objects)", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    hist_path = output_dir / "confidence_histogram.png"
    fig.savefig(str(hist_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  直方图已保存: {hist_path}")


def save_per_image_csv(per_image, output_dir: Path, thresholds: list):
    """保存逐图 CSV 详情。"""
    csv_path = output_dir / "per_image_details.csv"

    fieldnames = ["filename", "total_detections", "max_score"]
    for t in thresholds:
        fieldnames.append(f"dets_at_{t}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for img in per_image:
            row = {
                "filename": img["filename"],
                "total_detections": img["total_detections"],
                "max_score": round(img["max_score"], 4),
            }
            for t in thresholds:
                row[f"dets_at_{t}"] = img["dets_at_threshold"][str(t)]
            writer.writerow(row)

    print(f"  逐图详情已保存: {csv_path}")


def print_worst_images(per_image, threshold: float, top_n: int = 10):
    """打印误检最严重的图片。"""
    key = str(threshold)
    sorted_imgs = sorted(per_image, key=lambda x: x["dets_at_threshold"][key], reverse=True)
    worst = sorted_imgs[:top_n]

    if not worst or worst[0]["dets_at_threshold"][key] == 0:
        print(f"\n--- 所有图片在 @{threshold} 下无误检 ---")
        return

    print(f"\n--- Top-{top_n} 最严重误检图片 (@{threshold}) ---")
    for i, img in enumerate(worst):
        n = img["dets_at_threshold"][key]
        if n == 0:
            break
        print(f"  {i+1:>2}. {img['filename']:<40}  {n:>3} dets  max={img['max_score']:.3f}")


# ==========================
# 主流程
# ==========================
@torch.no_grad()
def main():
    cfg = SETTINGS
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    neg_img_dir = Path(cfg["negative_img_dir"])
    val_ann_file = Path(cfg["val_ann_file"])
    val_img_dir = Path(cfg["val_img_dir"])

    # 加载模型
    print("加载模型...")
    model, postprocessor = load_dfine_model(
        Path(cfg["repo_dir"]), cfg["config"], cfg["checkpoint"], cfg["device"]
    )
    print("模型加载完成")

    # 1. 负样本评估
    oa_metrics = {}
    per_image = []
    all_scores = []

    if neg_img_dir.exists():
        all_scores, per_image = evaluate_negatives(
            model, postprocessor, neg_img_dir,
            cfg["input_size"], cfg["device"],
            cfg["conf_thresholds"], cfg["img_extensions"],
        )
        if per_image:
            oa_metrics = compute_overactivation_metrics(
                all_scores, per_image, cfg["conf_thresholds"]
            )
    else:
        print(f"\n[INFO] 负样本目录不存在: {neg_img_dir}")
        print("       请先收集误检场景图片放到该目录中")

    # 2. 标准 mAP 评估
    map_metrics = None
    if val_ann_file.exists():
        map_metrics = evaluate_map(
            model, postprocessor, val_ann_file, val_img_dir,
            cfg["input_size"], cfg["device"],
        )
    else:
        print(f"\n[INFO] Val 标注文件不存在: {val_ann_file}")

    # 3. 输出报告
    print_report(oa_metrics, map_metrics, cfg)

    if per_image:
        print_worst_images(per_image, cfg["conf_thresholds"][0])

    # 4. 保存文件
    full_report = {
        "settings": {
            "config": cfg["config"],
            "checkpoint": cfg["checkpoint"],
            "input_size": cfg["input_size"],
            "date": datetime.now().isoformat(),
        },
        "overactivation": oa_metrics,
        "map": map_metrics,
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 报告已保存: {report_path}")

    if per_image:
        save_per_image_csv(per_image, output_dir, cfg["conf_thresholds"])

    if cfg["save_histograms"] and all_scores:
        save_histogram(all_scores, output_dir, cfg["conf_thresholds"])

    print("\n评估完成。")


if __name__ == "__main__":
    main()
