"""
使用摄像头实时测试 D-FINE 的 .pth 模型。

用法：
  cd d:\robotmaster\2026\D-FINE_train
  python benchmark/scripts/test_camera_pth.py
  python benchmark/scripts/test_camera_pth.py --camera-id 1 --conf 0.5
  python benchmark/scripts/test_camera_pth.py --checkpoint E:/数据集/model/exp_s_finetune_neg_aug.pth

按键：
  q / ESC 退出
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_DIR = PROJECT_ROOT / "D-FINE"
if not DEFAULT_REPO_DIR.exists():
    DEFAULT_REPO_DIR = PROJECT_ROOT


def prepare_windows_readable_config(config_path: Path) -> Path:
    """处理 Windows 下 yaml 注释编码问题。"""
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
    tmp_path = config_path.parent / f".camera_tmp_{config_path.stem}.yml"
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


def load_dfine_model(repo_dir: Path, config_path: Path, checkpoint_path: Path, device: str):
    """加载 D-FINE 模型和后处理器。"""
    sys.path.insert(0, str(repo_dir.resolve()))
    from src.core import YAMLConfig

    safe_config_path = prepare_windows_readable_config(config_path)
    try:
        cfg = YAMLConfig(str(safe_config_path), resume=str(checkpoint_path))
    finally:
        cleanup_temp_config(safe_config_path, config_path)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state, strict=False)

    model = cfg.model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device).eval()
    return model, postprocessor


def unpack_predictions(pred_output):
    """兼容 deploy mode tuple 和 list[dict] 输出。"""
    if isinstance(pred_output, (tuple, list)) and len(pred_output) == 3 and torch.is_tensor(pred_output[0]):
        labels_b, boxes_b, scores_b = pred_output
        return labels_b[0], boxes_b[0], scores_b[0]

    if isinstance(pred_output, list) and len(pred_output) > 0 and isinstance(pred_output[0], dict):
        first = pred_output[0]
        return first["labels"], first["boxes"], first["scores"]

    raise TypeError(f"Unsupported prediction output type: {type(pred_output)}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=str(DEFAULT_REPO_DIR),
        help="D-FINE 源码目录",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "volleyball_s_transfer.yml"),
        help="模型配置文件",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="E:/数据集/model/exp_s_obj2coco_neg_d_bg2_last.pth",
        help=".pth 权重文件",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头编号")
    parser.add_argument("--input-size", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="推理设备，如 cpu / cuda:0",
    )
    return parser.parse_args()


@torch.no_grad()
def infer_frame(model, postprocessor, frame_bgr, input_size: int, device: str, conf_thr: float):
    """单帧推理。"""
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    tfm = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])

    tensor = tfm(image).unsqueeze(0).to(device)
    orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)

    outputs = model(tensor)
    pred_output = postprocessor(outputs, orig_size)
    labels_t, boxes_t, scores_t = unpack_predictions(pred_output)

    labels = labels_t.detach().cpu().tolist()
    boxes = boxes_t.detach().cpu().tolist()
    scores = scores_t.detach().cpu().tolist()

    results = []
    for label, box, score in zip(labels, boxes, scores):
        if score < conf_thr:
            continue
        x1, y1, x2, y2 = box
        results.append({
            "label": int(label),
            "score": float(score),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })
    return results


def draw_detections(frame, detections):
    """绘制检测框。"""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        label = det["label"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        text = f"id:{label} {score:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


@torch.no_grad()
def main():
    args = get_args()

    repo_dir = Path(args.repo_dir)
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")

    print("加载模型中...")
    model, postprocessor = load_dfine_model(
        repo_dir=repo_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    print(f"模型加载完成，设备: {args.device}")

    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.camera_id}")

    print("摄像头已打开，按 q 或 ESC 退出。")

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取摄像头画面失败")
            break

        detections = infer_frame(
            model=model,
            postprocessor=postprocessor,
            frame_bgr=frame,
            input_size=args.input_size,
            device=args.device,
            conf_thr=args.conf,
        )

        frame = draw_detections(frame, detections)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | dets: {len(detections)} | conf: {args.conf}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("D-FINE Camera Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("已退出。")


if __name__ == "__main__":
    main()