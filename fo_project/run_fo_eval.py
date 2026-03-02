import json
import sys
from collections import defaultdict
from pathlib import Path

import fiftyone as fo
import fiftyone.core.labels as fol
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


# ==========================
# Hardcoded runtime settings
# ==========================
# 修改这里即可，无需命令行传参
SETTINGS = {
    "repo_dir": "./D-FINE",
    "config": "./config/volleyball_s_transfer.yml",
    "checkpoint": "./D-FINE/output/exp_s_transfer_obj2coco_aug/best_stg1.pth",  # 最新的最佳权重
    "img_root": "./coco/images",
    "ann_file": "./coco/converted/annotations/val.json",
    "dataset_name": "volleyball-val-s-b1",
    "eval_key": "eval",
    "device": "cuda:0",
    "input_size": 640,
    "conf_thres": 0.25,
    "limit": 0,  # 0 = all
    "overwrite": True,
    "no_app": False,
}


def load_dfine_model(repo_dir: Path, config_path: str, checkpoint_path: str, device: str):
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


def prepare_windows_readable_config(config_path: Path) -> Path:
    """
    D-FINE's yaml loader opens files without explicit encoding.
    On Windows this can default to GBK and fail on UTF-8 comments.
    This helper creates a GBK-readable temporary copy in the same directory.
    """
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

    # Build an ASCII-only temporary config so both UTF-8 and GBK default decoders can read it.
    safe_text = text.encode("ascii", errors="ignore").decode("ascii")
    tmp_path = config_path.parent / f".fo_tmp_{config_path.stem}.yml"
    tmp_path.write_text(safe_text, encoding="utf-8")
    return tmp_path


def cleanup_temp_config(tmp_path: Path, original_config_path: Path):
    # Do not remove if it's exactly the original file path
    if tmp_path.resolve() == original_config_path.resolve():
        return
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass


def load_coco(ann_file: Path):
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    return images, anns_by_image, cat_id_to_name


def clamp01(x):
    return max(0.0, min(1.0, x))


def coco_bbox_to_fo_detection(bbox_xywh, w, h, label, confidence=None):
    x, y, bw, bh = bbox_xywh
    nx = clamp01(x / w)
    ny = clamp01(y / h)
    nw = clamp01(bw / w)
    nh = clamp01(bh / h)
    kwargs = dict(label=label, bounding_box=[nx, ny, nw, nh])
    if confidence is not None:
        kwargs["confidence"] = float(confidence)
    return fol.Detection(**kwargs)


def xyxy_abs_to_fo_detection(box_xyxy, w, h, label, confidence):
    x1, y1, x2, y2 = box_xyxy
    x1 = clamp01(float(x1) / w)
    y1 = clamp01(float(y1) / h)
    x2 = clamp01(float(x2) / w)
    y2 = clamp01(float(y2) / h)
    bw = clamp01(x2 - x1)
    bh = clamp01(y2 - y1)
    return fol.Detection(
        label=label,
        bounding_box=[x1, y1, bw, bh],
        confidence=float(confidence),
    )


def unpack_predictions(pred_output):
    """
    Support both formats:
    1) deploy mode tuple: (labels, boxes, scores), each shape [B, Q, ...]
    2) training/eval mode list[dict]: [{'labels','boxes','scores'}, ...]
    Returns tensors for a single sample: labels_1d, boxes_2d, scores_1d
    """
    # deploy tuple
    if isinstance(pred_output, (tuple, list)) and len(pred_output) == 3 and torch.is_tensor(pred_output[0]):
        labels_b, boxes_b, scores_b = pred_output
        return labels_b[0], boxes_b[0], scores_b[0]

    # list of dicts
    if isinstance(pred_output, list) and len(pred_output) > 0 and isinstance(pred_output[0], dict):
        first = pred_output[0]
        return first["labels"], first["boxes"], first["scores"]

    raise TypeError(f"Unsupported prediction output type: {type(pred_output)}")


@torch.no_grad()
def run():
    cfg = SETTINGS
    repo_dir = Path(cfg["repo_dir"])
    img_root = Path(cfg["img_root"])
    ann_file = Path(cfg["ann_file"])

    if fo.dataset_exists(cfg["dataset_name"]):
        if cfg["overwrite"]:
            fo.delete_dataset(cfg["dataset_name"])
        else:
            raise RuntimeError(
                f"Dataset '{cfg['dataset_name']}' already exists. Set SETTINGS['overwrite']=True to replace it."
            )

    images, anns_by_image, cat_id_to_name = load_coco(ann_file)
    if cfg["limit"] and cfg["limit"] > 0:
        images = images[: cfg["limit"]]

    model, postprocessor = load_dfine_model(
        repo_dir, cfg["config"], cfg["checkpoint"], cfg["device"]
    )

    tfm = T.Compose(
        [
            T.Resize((cfg["input_size"], cfg["input_size"])),
            T.ToTensor(),
        ]
    )

    dataset = fo.Dataset(cfg["dataset_name"])

    for img_info in tqdm(images, desc="Building FiftyOne dataset"):
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]
        image_path = img_root / file_name

        if not image_path.exists():
            # Skip missing files instead of hard failing
            continue

        sample = fo.Sample(filepath=str(image_path.resolve()))

        # Ground truth
        gt_dets = []
        for ann in anns_by_image.get(image_id, []):
            cat_id = ann["category_id"]
            label = cat_id_to_name.get(cat_id, str(cat_id))
            gt_dets.append(coco_bbox_to_fo_detection(ann["bbox"], width, height, label))
        sample["ground_truth"] = fol.Detections(detections=gt_dets)

        # Prediction
        image = Image.open(image_path).convert("RGB")
        tensor = tfm(image).unsqueeze(0).to(cfg["device"])
        orig_target_sizes = torch.tensor(
            [[width, height]], dtype=torch.float32, device=cfg["device"]
        )

        outputs = model(tensor)
        pred_output = postprocessor(outputs, orig_target_sizes)
        labels_t, boxes_t, scores_t = unpack_predictions(pred_output)

        labels = labels_t.detach().cpu().tolist()
        boxes = boxes_t.detach().cpu().tolist()
        scores = scores_t.detach().cpu().tolist()

        pred_dets = []
        for label_id, box, score in zip(labels, boxes, scores):
            if score < cfg["conf_thres"]:
                continue

            # Handle both 0-based and 1-based label id conventions
            if label_id in cat_id_to_name:
                label_name = cat_id_to_name[label_id]
            elif (label_id + 1) in cat_id_to_name:
                label_name = cat_id_to_name[label_id + 1]
            else:
                label_name = str(label_id)

            pred_dets.append(xyxy_abs_to_fo_detection(box, width, height, label_name, score))

        sample["predictions"] = fol.Detections(detections=pred_dets)
        dataset.add_sample(sample)

    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key=cfg["eval_key"],
        compute_mAP=True,
    )

    try:
        print(f"mAP: {results.mAP():.6f}")
    except Exception:
        print("mAP: unavailable")
    print(f"Dataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")
    print("Use FiftyOne sidebar to inspect FP/FN and per-sample errors.")

    if not cfg["no_app"]:
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == "__main__":
    run()
