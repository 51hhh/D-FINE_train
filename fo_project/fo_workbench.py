import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fiftyone as fo
import fiftyone.core.labels as fol
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


# =========================================
# Hardcoded settings (edit this block only)
# =========================================
SETTINGS = {
    "repo_dir": "./D-FINE",
    "img_root": "./coco/images",
    "ann_file": "./coco/converted/annotations/val.json",
    "dataset_name": "volleyball-workbench",
    "overwrite": True,
    "persistent": True,
    "launch_app": True,
    # 质量检查参数
    "duplicate_md5_max_mb": 20,   # 文件大于该值则跳过 md5（避免太慢）
    "low_iou_threshold": 0.50,
    "hard_fn_threshold": 1,       # FN 数 >= 1 标记为 hard
    "hard_fp_threshold": 2,       # FP 数 >= 2 标记为 hard
    "limit": 0,                   # 0 = 全量
    # 多模型对比：每个模型一个预测字段
    "models": [
        {
            "name": "s_transfer",
            "pred_field": "pred_s",
            "eval_key": "eval_s",
            "config": "./config/volleyball_s_transfer_balance.yml",
            "checkpoint": "./D-FINE/output/dfine_hgnetv2_s_obj2custom/best_stg1.pth",
            "skip_if_missing": False,
            "device": "cuda:0",
            "input_size": 640,
            "conf_thres": 0.05,
        },
        {
            "name": "n_scratch",
            "pred_field": "pred_n",
            "eval_key": "eval_n",
            "config": "./config/volleyball_n_scratch_compare.yml",
            "checkpoint": "./D-FINE/output/exp_n_scratch_compare/best_stg2.pth",
            "skip_if_missing": True,
            "device": "cuda:0",
            "input_size": 512,
            "conf_thres": 0.05,
        },
    ],
}


def get_det_attr(det, key):
    try:
        return det.get_field(key)
    except Exception:
        try:
            return det.to_dict().get(key)
        except Exception:
            return None


def prepare_windows_readable_config(config_path: Path) -> Path:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

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
    # This keeps YAML structure and values, while dropping non-ASCII comment characters.
    safe_text = text.encode("ascii", errors="ignore").decode("ascii")
    tmp_path = config_path.parent / f".fo_tmp_{config_path.stem}.yml"
    tmp_path.write_text(safe_text, encoding="utf-8")
    return tmp_path


def cleanup_temp_config(tmp_path: Path, original_path: Path):
    if tmp_path.resolve() == original_path.resolve():
        return
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass


def load_dfine_model(repo_dir: Path, config_path: str, checkpoint_path: str, device: str):
    sys.path.insert(0, str(repo_dir.resolve()))
    from src.core import YAMLConfig

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg_path = Path(config_path)
    safe_cfg = prepare_windows_readable_config(cfg_path)
    try:
        cfg = YAMLConfig(str(safe_cfg), resume=checkpoint_path)
    finally:
        cleanup_temp_config(safe_cfg, cfg_path)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state, strict=False)
    model = cfg.model.deploy().to(device).eval()
    post = cfg.postprocessor.deploy().to(device).eval()
    return model, post


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


def coco_bbox_to_detection(bbox_xywh, w, h, label):
    x, y, bw, bh = bbox_xywh
    nx = clamp01(x / w)
    ny = clamp01(y / h)
    nw = clamp01(bw / w)
    nh = clamp01(bh / h)
    return fol.Detection(label=label, bounding_box=[nx, ny, nw, nh])


def xyxy_abs_to_detection(box_xyxy, w, h, label, confidence):
    x1, y1, x2, y2 = box_xyxy
    x1 = clamp01(float(x1) / w)
    y1 = clamp01(float(y1) / h)
    x2 = clamp01(float(x2) / w)
    y2 = clamp01(float(y2) / h)
    bw = clamp01(x2 - x1)
    bh = clamp01(y2 - y1)
    return fol.Detection(label=label, bounding_box=[x1, y1, bw, bh], confidence=float(confidence))


def unpack_predictions(pred_output):
    # deploy output
    if isinstance(pred_output, (tuple, list)) and len(pred_output) == 3 and torch.is_tensor(pred_output[0]):
        labels_b, boxes_b, scores_b = pred_output
        return labels_b[0], boxes_b[0], scores_b[0]
    # non-deploy output
    if isinstance(pred_output, list) and pred_output and isinstance(pred_output[0], dict):
        d = pred_output[0]
        return d["labels"], d["boxes"], d["scores"]
    raise TypeError(f"Unsupported prediction format: {type(pred_output)}")


def build_base_dataset(images, anns_by_image, cat_id_to_name, img_root: Path, dataset_name: str, limit: int):
    if limit and limit > 0:
        images = images[:limit]

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = SETTINGS["persistent"]

    for img in tqdm(images, desc="Load GT"):
        image_path = img_root / img["file_name"]
        if not image_path.exists():
            continue

        w, h = img["width"], img["height"]
        sample = fo.Sample(filepath=str(image_path.resolve()))
        sample["image_id"] = img["id"]
        sample["file_name"] = img["file_name"]

        gts = []
        for ann in anns_by_image.get(img["id"], []):
            label = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            gts.append(coco_bbox_to_detection(ann["bbox"], w, h, label))
        sample["ground_truth"] = fol.Detections(detections=gts)
        dataset.add_sample(sample)

    return dataset


@torch.no_grad()
def run_model_predictions(dataset, model_cfg, cat_id_to_name):
    repo_dir = Path(SETTINGS["repo_dir"])
    try:
        model, post = load_dfine_model(
            repo_dir=repo_dir,
            config_path=model_cfg["config"],
            checkpoint_path=model_cfg["checkpoint"],
            device=model_cfg["device"],
        )
    except FileNotFoundError as e:
        if model_cfg.get("skip_if_missing", False):
            print(f"[SKIP] {model_cfg.get('name', model_cfg['pred_field'])}: {e}")
            return False
        raise
    tfm = T.Compose([T.Resize((model_cfg["input_size"], model_cfg["input_size"])), T.ToTensor()])

    pred_field = model_cfg["pred_field"]
    conf_thres = model_cfg["conf_thres"]
    dev = model_cfg["device"]

    for sample in tqdm(dataset.iter_samples(progress=True, autosave=True), total=len(dataset), desc=f"Infer {pred_field}"):
        image = Image.open(sample.filepath).convert("RGB")
        w, h = image.size
        tensor = tfm(image).unsqueeze(0).to(dev)
        orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=dev)

        outputs = model(tensor)
        pred_out = post(outputs, orig_size)
        labels_t, boxes_t, scores_t = unpack_predictions(pred_out)

        labels = labels_t.detach().cpu().tolist()
        boxes = boxes_t.detach().cpu().tolist()
        scores = scores_t.detach().cpu().tolist()

        dets = []
        for label_id, box, score in zip(labels, boxes, scores):
            if score < conf_thres:
                continue
            if label_id in cat_id_to_name:
                label_name = cat_id_to_name[label_id]
            elif (label_id + 1) in cat_id_to_name:
                label_name = cat_id_to_name[label_id + 1]
            else:
                label_name = str(label_id)
            dets.append(xyxy_abs_to_detection(box, w, h, label_name, score))

        sample[pred_field] = fol.Detections(detections=dets)
    return True


def evaluate_and_tag(dataset, model_cfg):
    pred_field = model_cfg["pred_field"]
    eval_key = model_cfg["eval_key"]
    low_iou_thr = SETTINGS["low_iou_threshold"]

    results = dataset.evaluate_detections(
        pred_field,
        gt_field="ground_truth",
        eval_key=eval_key,
        compute_mAP=True,
    )
    try:
        print(f"[{pred_field}] mAP = {results.mAP():.6f}")
    except Exception:
        print(f"[{pred_field}] mAP unavailable")

    # sample-level boolean flags: has_fp / has_fn / low_conf / low_iou
    for sample in tqdm(dataset.iter_samples(progress=True, autosave=True), total=len(dataset), desc=f"Tag {pred_field}"):
        preds = getattr(sample, pred_field).detections if getattr(sample, pred_field, None) else []
        gts = sample.ground_truth.detections if sample.ground_truth else []

        fp = 0
        low_conf = 0
        low_iou = 0
        for d in preds:
            status = get_det_attr(d, eval_key)
            if status == "fp":
                fp += 1
            if (d.confidence or 0.0) < model_cfg["conf_thres"]:
                low_conf += 1
            iou_val = get_det_attr(d, f"{eval_key}_iou")
            if iou_val is not None and iou_val >= 0 and iou_val < low_iou_thr:
                low_iou += 1

        fn = 0
        for d in gts:
            status = get_det_attr(d, eval_key)
            if status == "fn":
                fn += 1

        sample[f"{pred_field}_fp"] = fp
        sample[f"{pred_field}_fn"] = fn
        sample[f"{pred_field}_low_conf"] = low_conf
        sample[f"{pred_field}_low_iou"] = low_iou
        sample[f"has_fp_{pred_field}"] = fp > 0
        sample[f"has_fn_{pred_field}"] = fn > 0
        sample[f"has_low_conf_{pred_field}"] = low_conf > 0
        sample[f"has_low_iou_{pred_field}"] = low_iou > 0


def add_comparison_fields(dataset):
    # 默认比较前两个模型，若只有一个模型则跳过
    models = SETTINGS["models"]
    if len(models) < 2:
        return
    a = models[0]["pred_field"]
    b = models[1]["pred_field"]

    for sample in dataset.iter_samples(progress=True, autosave=True):
        fp_a = getattr(sample, f"{a}_fp", 0) or 0
        fp_b = getattr(sample, f"{b}_fp", 0) or 0
        fn_a = getattr(sample, f"{a}_fn", 0) or 0
        fn_b = getattr(sample, f"{b}_fn", 0) or 0

        sample["cmp_fp_delta"] = fp_b - fp_a
        sample["cmp_fn_delta"] = fn_b - fn_a
        sample["cmp_better_a"] = (fp_a + fn_a) < (fp_b + fn_b)
        sample["cmp_better_b"] = (fp_b + fn_b) < (fp_a + fn_a)


def file_md5(path: Path, chunk=1 << 20):
    m = hashlib.md5()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


def quality_checks(dataset):
    # 1) dirty GT: bbox非法（越界/宽高<=0）
    for sample in dataset.iter_samples(progress=True, autosave=True):
        dirty = False
        if sample.ground_truth:
            for d in sample.ground_truth.detections:
                x, y, w, h = d.bounding_box
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > 1.0001 or y + h > 1.0001:
                    dirty = True
                    break
        sample["dirty_gt"] = dirty
        if dirty:
            sample.tags = sorted(set((sample.tags or []) + ["dirty_gt"]))

    # 2) duplicates: md5
    max_mb = SETTINGS["duplicate_md5_max_mb"]
    keys = []
    for sample in dataset:
        p = Path(sample.filepath)
        if not p.exists():
            sample["duplicate_key"] = ""
            continue
        if p.stat().st_size > max_mb * 1024 * 1024:
            sample["duplicate_key"] = ""
            continue
        k = file_md5(p)
        sample["duplicate_key"] = k
        sample.save()
        keys.append(k)

    c = Counter([k for k in keys if k])
    dup_keys = {k for k, v in c.items() if v > 1}
    if dup_keys:
        for sample in dataset.iter_samples(progress=True, autosave=True):
            if getattr(sample, "duplicate_key", "") in dup_keys:
                sample["is_duplicate"] = True
                sample.tags = sorted(set((sample.tags or []) + ["duplicate"]))
            else:
                sample["is_duplicate"] = False
    else:
        for sample in dataset.iter_samples(progress=True, autosave=True):
            sample["is_duplicate"] = False

    # 3) hard samples: FN或FP偏多
    fn_thr = SETTINGS["hard_fn_threshold"]
    fp_thr = SETTINGS["hard_fp_threshold"]
    for sample in dataset.iter_samples(progress=True, autosave=True):
        hard = False
        for m in SETTINGS["models"]:
            pred = m["pred_field"]
            if (getattr(sample, f"{pred}_fn", 0) or 0) >= fn_thr:
                hard = True
            if (getattr(sample, f"{pred}_fp", 0) or 0) >= fp_thr:
                hard = True
        sample["hard_sample"] = hard
        if hard:
            sample.tags = sorted(set((sample.tags or []) + ["hard"]))


def print_quick_filters():
    print("\nQuick filters in FiftyOne:")
    print("1) 误检 FP: sample field -> has_fp_pred_s / has_fp_pred_n == True")
    print("2) 漏检 FN: sample field -> has_fn_pred_s / has_fn_pred_n == True")
    print("3) 低置信度: has_low_conf_pred_* == True")
    print("4) 低IoU: has_low_iou_pred_* == True")
    print("5) 脏标注: dirty_gt == True")
    print("6) 重复图: is_duplicate == True")
    print("7) 难例: hard_sample == True")
    print("8) 模型对比: cmp_better_a / cmp_better_b")


@torch.no_grad()
def run():
    if fo.dataset_exists(SETTINGS["dataset_name"]):
        if SETTINGS["overwrite"]:
            fo.delete_dataset(SETTINGS["dataset_name"])
        else:
            raise RuntimeError(
                f"Dataset '{SETTINGS['dataset_name']}' exists. Set SETTINGS['overwrite']=True"
            )

    img_root = Path(SETTINGS["img_root"])
    ann_file = Path(SETTINGS["ann_file"])
    images, anns_by_image, cat_id_to_name = load_coco(ann_file)

    dataset = build_base_dataset(
        images=images,
        anns_by_image=anns_by_image,
        cat_id_to_name=cat_id_to_name,
        img_root=img_root,
        dataset_name=SETTINGS["dataset_name"],
        limit=SETTINGS["limit"],
    )

    for model_cfg in SETTINGS["models"]:
        ok = run_model_predictions(dataset, model_cfg, cat_id_to_name)
        if not ok:
            continue
        evaluate_and_tag(dataset, model_cfg)

    add_comparison_fields(dataset)
    quality_checks(dataset)

    print(f"\nDataset ready: {dataset.name}, samples={len(dataset)}")
    print_quick_filters()

    if SETTINGS["launch_app"]:
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == "__main__":
    run()
