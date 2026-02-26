import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

"""
统一评估/推理入口（GPU）。
使用 profile + dataset 配置，并调用官方仓库的 --test-only 流程。
"""

# ====== 头部配置（只改这里）======
PROFILE_FILE = "benchmark/configs/profiles/dfine_n.yaml"
DATASET_FILE = "benchmark/configs/dataset.yaml"
CHECKPOINT = ""  # 必填：模型权重绝对路径或相对路径
EXPERIMENTS_DIR = "benchmark/evaluations"
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
        if p.is_dir():
            m = re.match(r"eval_(\d+)$", p.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
    rid = max_id + 1
    out = base_dir / f"eval_{rid:04d}"
    out.mkdir(parents=True, exist_ok=False)
    return rid, out


def stream_to_log(cmd, cwd: Path, log_path: Path, extra_env=None):
    env = dict(__import__("os").environ)
    if extra_env:
        env.update(extra_env)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[START] {datetime.now().isoformat()}\n")
        f.write(f"[CWD] {cwd.as_posix()}\n")
        f.write(f"[CMD] {' '.join(cmd)}\n\n")
        f.flush()

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
            f.write(line)
        code = proc.wait()
        f.write(f"\n[END] {datetime.now().isoformat()} code={code}\n")
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def build_dataset_override(dataset_cfg: dict, num_classes: int, root: Path, out_path: Path):
    images_dir = (root / dataset_cfg["images_dir"]).resolve().as_posix()
    ann_dir = (root / dataset_cfg["annotations_dir"]).resolve()
    train_json = (ann_dir / dataset_cfg["train_json"]).resolve().as_posix()
    val_json = (ann_dir / dataset_cfg["val_json"]).resolve().as_posix()

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


def main():
    if not CHECKPOINT.strip():
        raise ValueError("请在脚本头部填写 CHECKPOINT")

    root = Path(__file__).resolve().parents[2]
    profile_path = (root / PROFILE_FILE).resolve()
    dataset_path = (root / DATASET_FILE).resolve()
    ckpt_path = (root / CHECKPOINT).resolve() if not Path(CHECKPOINT).is_absolute() else Path(CHECKPOINT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"权重不存在: {ckpt_path}")

    profile = load_yaml(profile_path)
    dataset_cfg = load_yaml(dataset_path)
    model_type = profile["model_type"]
    repo_dir = (root / profile["repo_dir"]).resolve()
    base_config = (repo_dir / profile["base_config"]).resolve().as_posix()
    device = str(profile.get("device", "0"))
    output_group = str(profile.get("output_group", profile["profile_name"]))

    num_classes = read_num_classes((root / dataset_cfg["summary_json"]).resolve())
    run_id, run_dir = next_run_dir((root / EXPERIMENTS_DIR / output_group).resolve())
    cfg_dir = run_dir / "configs"
    log_dir = run_dir / "logs"
    model_output = run_dir / "model_output"
    for d in [cfg_dir, log_dir, model_output]:
        d.mkdir(parents=True, exist_ok=True)

    ds_yaml = cfg_dir / "dataset_override.yaml"
    build_dataset_override(dataset_cfg, num_classes, root, ds_yaml)

    eval_yaml = cfg_dir / "eval_override.yaml"
    eval_data = {
        "__include__": [base_config, ds_yaml.resolve().as_posix()],
        "output_dir": model_output.resolve().as_posix(),
    }
    dump_yaml(eval_data, eval_yaml)

    cmd = [sys.executable]
    if model_type == "dfine":
        cmd += ["train.py", "-c", str(eval_yaml.resolve()), "--test-only", "-r", str(ckpt_path.resolve())]
    elif model_type == "rtdetr_lyu":
        cmd += ["tools/train.py", "-c", str(eval_yaml.resolve()), "-r", str(ckpt_path.resolve()), "--test-only"]
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")

    metadata = {
        "profile_file": profile_path.as_posix(),
        "dataset_file": dataset_path.as_posix(),
        "checkpoint": ckpt_path.as_posix(),
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
        "command": cmd,
        "device": device,
        "created_at": datetime.now().isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Eval dir: {run_dir.as_posix()}")
    print(f"[INFO] Using GPU: CUDA_VISIBLE_DEVICES={device}")
    print(f"[INFO] Eval config: {eval_yaml.as_posix()}")

    stream_to_log(
        cmd=cmd,
        cwd=repo_dir,
        log_path=log_dir / "eval.log",
        extra_env={"CUDA_VISIBLE_DEVICES": device},
    )


if __name__ == "__main__":
    main()
