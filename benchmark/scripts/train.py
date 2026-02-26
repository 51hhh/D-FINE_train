import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

"""
统一训练入口（仅需改脚本头部）。
通过 PROFILE 文件切换模型配置，不再改多处代码。
"""

# ====== 头部配置（只改这里）======
PROFILE_FILE = "benchmark/configs/profiles/01_dfine_n_coco_pretrain.yaml"
DATASET_FILE = "benchmark/configs/dataset.yaml"
EXPERIMENTS_DIR = "benchmark/experiments"
AUTO_OPEN_TENSORBOARD = False
TENSORBOARD_PORT = 6006
# ================================


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def read_num_classes(summary_json: Path):
    if not summary_json.exists():
        raise FileNotFoundError(f"未找到 summary: {summary_json}")
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    return int(data["num_classes"])


def next_run_dir(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"run_(\d+)$", p.name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    run_id = max_id + 1
    run_dir = base_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def stream_to_log(cmd, cwd: Path, log_path: Path, extra_env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"[START] {datetime.now().isoformat()}\n")
        logf.write(f"[CWD] {cwd.as_posix()}\n")
        logf.write(f"[CMD] {' '.join(cmd)}\n\n")
        logf.flush()

        env = None
        if extra_env:
            env = dict(**extra_env)
            full_env = dict(**__import__("os").environ)
            full_env.update(env)
            env = full_env

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        ret = proc.wait()
        logf.write(f"\n[END] {datetime.now().isoformat()} code={ret}\n")

    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def build_dataset_override(model_type: str, dataset_cfg: dict, num_classes: int, root: Path, out_path: Path):
    images_dir = (root / dataset_cfg["images_dir"]).resolve().as_posix()
    ann_dir = (root / dataset_cfg["annotations_dir"]).resolve()
    train_json = (ann_dir / dataset_cfg["train_json"]).resolve().as_posix()
    val_json = (ann_dir / dataset_cfg["val_json"]).resolve().as_posix()

    # 只覆盖“数据路径与类别”，不覆盖 transforms / collate_fn / batch 等，
    # 避免把官方成熟配置中的图像张量化流程覆盖掉。
    data = {
        "task": "detection",
        "num_classes": num_classes,
        "remap_mscoco_category": False,
        "train_dataloader": {
            "dataset": {
                "img_folder": images_dir,
                "ann_file": train_json,
                "return_masks": False,
            }
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": images_dir,
                "ann_file": val_json,
                "return_masks": False,
            }
        },
    }

    dump_yaml(data, out_path)


def build_train_override(profile_cfg: dict, dataset_yaml: Path, model_output_dir: Path, out_path: Path, root: Path):
    repo_dir = (root / profile_cfg["repo_dir"]).resolve()
    base_config = (repo_dir / profile_cfg["base_config"]).resolve().as_posix()

    data = {
        "__include__": [base_config, dataset_yaml.resolve().as_posix()],
        "output_dir": model_output_dir.resolve().as_posix(),
    }

    if profile_cfg["model_type"] == "rtdetr_lyu" and "epochs" in profile_cfg:
        data["epoches"] = int(profile_cfg["epochs"])

    # 支持 profile 中的键路径覆盖，例如：
    # overrides:
    #   PResNet.pretrained: false
    #   HGNetv2.pretrained: false
    overrides = profile_cfg.get("overrides", {}) or {}
    for k, v in overrides.items():
        set_nested(data, str(k), v)

    # D-FINE 在 epoch == stop_epoch 时会尝试加载 best_stg1.pth。
    # 如果 stop_epoch <= 0，会在训练初期触发并导致文件不存在报错。
    if profile_cfg["model_type"] == "dfine":
        stop_epoch = (
            data.get("train_dataloader", {})
            .get("collate_fn", {})
            .get("stop_epoch", None)
        )
        if isinstance(stop_epoch, int) and stop_epoch <= 0:
            data.setdefault("train_dataloader", {}).setdefault("collate_fn", {})["stop_epoch"] = 9999

    dump_yaml(data, out_path)


def set_nested(data: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = data
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def maybe_open_tensorboard(logdir: Path, port: int):
    cmd = ["tensorboard", "--logdir", str(logdir), "--port", str(port)]
    print("[TB]", " ".join(cmd))
    subprocess.Popen(cmd)


def main():
    root = Path(__file__).resolve().parents[2]
    profile_path = (root / PROFILE_FILE).resolve()
    dataset_path = (root / DATASET_FILE).resolve()

    profile_cfg = load_yaml(profile_path)
    dataset_cfg = load_yaml(dataset_path)

    model_type = profile_cfg["model_type"]
    repo_dir = (root / profile_cfg["repo_dir"]).resolve()
    seed = int(profile_cfg.get("seed", 0))
    device = str(profile_cfg.get("device", "0"))
    use_amp = bool(profile_cfg.get("use_amp", True))
    pretrained = str(profile_cfg.get("pretrained", "")).strip()
    output_group = str(profile_cfg.get("output_group", profile_cfg["profile_name"]))

    summary_json = (root / dataset_cfg["summary_json"]).resolve()
    num_classes = read_num_classes(summary_json)

    exp_base = (root / EXPERIMENTS_DIR / output_group).resolve()
    run_id, run_dir = next_run_dir(exp_base)
    cfg_dir = run_dir / "configs"
    log_dir = run_dir / "logs"
    tb_dir = run_dir / "tensorboard"
    model_output_dir = run_dir / "model_output"
    for d in [cfg_dir, log_dir, tb_dir, model_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    dataset_override = cfg_dir / "dataset_override.yaml"
    train_override = cfg_dir / "train_override.yaml"
    build_dataset_override(model_type, dataset_cfg, num_classes, root, dataset_override)
    build_train_override(profile_cfg, dataset_override, model_output_dir, train_override, root)

    cmd = [sys.executable]
    if model_type == "dfine":
        cmd += ["train.py", "-c", str(train_override.resolve()), "--seed", str(seed)]
    elif model_type == "rtdetr_lyu":
        cmd += ["tools/train.py", "-c", str(train_override.resolve()), "--seed", str(seed)]
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")

    if use_amp:
        cmd.append("--use-amp")
    if pretrained:
        p = Path(pretrained)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"pretrained 不存在: {p}")
        cmd += ["-t", str(p)]

    metadata = {
        "profile_file": profile_path.as_posix(),
        "dataset_file": dataset_path.as_posix(),
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
        "model_type": model_type,
        "repo_dir": repo_dir.as_posix(),
        "command": cmd,
        "device": device,
        "created_at": datetime.now().isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # 训练前环境检查：确认当前 Python 的 torch 是否可见 CUDA
    cuda_check_cmd = [
        sys.executable,
        "-c",
        "import torch;print('torch',torch.__version__);print('cuda_available',torch.cuda.is_available());print('cuda_device_count',torch.cuda.device_count())",
    ]
    print("[INFO] CUDA precheck:")
    subprocess.run(cuda_check_cmd, check=False)

    tb_note = (
        f"TensorBoard logdir:\n{model_output_dir.as_posix()}\n\n"
        f"Run command:\n"
        f"tensorboard --logdir {model_output_dir.as_posix()} --port {TENSORBOARD_PORT}\n"
    )
    (run_dir / "tensorboard.txt").write_text(tb_note, encoding="utf-8")

    print(f"[INFO] Profile: {profile_cfg['profile_name']}")
    print(f"[INFO] Run dir: {run_dir.as_posix()}")
    print(f"[INFO] Train config: {train_override.as_posix()}")
    print(f"[INFO] Log file: {(log_dir / 'train.log').as_posix()}")
    print(f"[INFO] TensorBoard: {model_output_dir.as_posix()}")
    print(f"[INFO] Overrides: {profile_cfg.get('overrides', {})}")

    stream_to_log(
        cmd=cmd,
        cwd=repo_dir,
        log_path=log_dir / "train.log",
        extra_env={
            "CUDA_VISIBLE_DEVICES": device,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )

    if AUTO_OPEN_TENSORBOARD:
        maybe_open_tensorboard(model_output_dir, TENSORBOARD_PORT)


if __name__ == "__main__":
    main()
