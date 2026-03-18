#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = "config/volleyball_s_obj2coco_d_bg2.yml"
DEFAULT_RUN_DIR = "runs/gradmotion/no-a/run"
DEFAULT_TUNING = "weights/dfine_s_obj2coco.pth"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="GradMotion gm-run launcher for D-FINE no-A training."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root that contains D-FINE and cloud assets",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--tuning", default=DEFAULT_TUNING)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--update", action="append", default=[])
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use-amp", dest="use_amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    return parser.parse_args(argv)


def to_abs_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def resolve_personal_root() -> Path:
    personal_root = Path("/personal/personal")
    if personal_root.is_dir():
        return personal_root
    return Path("/personal")


def print_tree(title: str, root: Path, patterns: tuple[str, ...]) -> None:
    print(title, flush=True)
    if not root.exists():
        print(f"[gm-launcher] missing: {root}", flush=True)
        return
    matched = False
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                print(path.as_posix(), flush=True)
                matched = True
    if not matched:
        print(f"[gm-launcher] no files matched under {root}", flush=True)


def build_train_command(
    python_executable: str,
    project_root: Path,
    config: str,
    run_dir: Path,
    tuning: str,
    seed: int,
    updates: list[str],
    use_amp: bool,
    negative_img_dir: Path,
) -> list[str]:
    command = [
        python_executable,
        "-u",
        str(project_root / "D-FINE" / "train.py"),
        "-c",
        str(to_abs_path(project_root, config)),
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
        "--summary-dir",
        str(run_dir),
        "-t",
        str(to_abs_path(project_root, tuning)),
        "-u",
        f"negative_img_dir={negative_img_dir}",
    ]
    for update_expr in updates:
        command.extend(["-u", update_expr])
    if use_amp:
        command.append("--use-amp")
    return command


def main(argv=None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    train_root = project_root / "D-FINE"
    train_entry = train_root / "train.py"
    config_path = to_abs_path(project_root, args.config)
    run_dir = to_abs_path(project_root, args.run_dir)
    personal_root = resolve_personal_root()
    images_zip = personal_root / "images.zip"
    negative_zip = personal_root / "negative_samples.zip"
    dataset_root = project_root / "coco"
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "Annotations"
    converted_dir = dataset_root / "converted"
    negative_img_dir = images_dir / "negative_samples"

    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (converted_dir / "annotations").mkdir(parents=True, exist_ok=True)

    print("=== gm_dfine_launcher ===", flush=True)
    print(f"[gm-launcher] cwd={os.getcwd()}", flush=True)
    print(f"[gm-launcher] launcher={Path(__file__).resolve()}", flush=True)
    print(f"[gm-launcher] project_root={project_root}", flush=True)
    print(f"[gm-launcher] train_root={train_root}", flush=True)
    print(f"[gm-launcher] config={config_path}", flush=True)
    print(f"[gm-launcher] run_dir={run_dir}", flush=True)
    print(f"[gm-launcher] personal_root={personal_root}", flush=True)
    print(f"[gm-launcher] images_zip={images_zip}", flush=True)
    print(f"[gm-launcher] negative_zip={negative_zip}", flush=True)
    print(f"[gm-launcher] python={sys.executable}", flush=True)
    print("[gm-launcher] no-A training uses train.json only; negative samples stay OA-only", flush=True)

    if not train_entry.exists():
        raise FileNotFoundError(f"train.py not found: {train_entry}")
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not images_zip.exists():
        raise FileNotFoundError(f"images.zip not found: {images_zip}")

    print("[gm-launcher] repo root listing", flush=True)
    for path in sorted(project_root.iterdir()):
        print(path.as_posix(), flush=True)

    print("[gm-launcher] ensure requirements", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(train_root / "requirements.txt"), "-q"],
        cwd=project_root,
        check=True,
    )

    print("[gm-launcher] torch + tensorboard sanity", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; from torch.utils.tensorboard import SummaryWriter; print(f'torch={torch.__version__}'); print(f'cuda_available={torch.cuda.is_available()}'); print('tensorboard=ok')",
        ],
        cwd=project_root,
        check=True,
    )

    print("[gm-launcher] extract images.zip", flush=True)
    subprocess.run(["unzip", "-qo", str(images_zip), "-d", str(dataset_root)], cwd=project_root, check=True)

    if negative_zip.exists():
        print("[gm-launcher] extract negative_samples.zip for OA evaluation only", flush=True)
        negative_img_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["unzip", "-qo", str(negative_zip), "-d", str(negative_img_dir)], cwd=project_root, check=True)
        nested_negative_dir = negative_img_dir / "negative_samples"
        if nested_negative_dir.is_dir():
            negative_img_dir = nested_negative_dir
    else:
        print(f"[gm-launcher] negative zip missing under {personal_root}", flush=True)

    print("[gm-launcher] prepare dataset", flush=True)
    subprocess.run([sys.executable, "-u", str(project_root / "benchmark" / "scripts" / "prepare_dataset.py")], cwd=project_root, check=True)

    print(f"[gm-launcher] image_count={sum(1 for p in images_dir.iterdir() if p.is_file())}", flush=True)
    print(f"[gm-launcher] xml_count={sum(1 for p in annotations_dir.glob('*.xml') if p.is_file())}", flush=True)
    if negative_img_dir.exists():
        print(f"[gm-launcher] negative_count={sum(1 for p in negative_img_dir.iterdir() if p.is_file())}", flush=True)
    else:
        print("[gm-launcher] negative_count=0", flush=True)
    print_tree("[gm-launcher] converted annotations", converted_dir / "annotations", ("*.json",))

    command = build_train_command(
        python_executable=sys.executable,
        project_root=project_root,
        config=args.config,
        run_dir=run_dir,
        tuning=args.tuning,
        seed=args.seed,
        updates=args.update,
        use_amp=args.use_amp,
        negative_img_dir=negative_img_dir,
    )
    print(f"[gm-launcher] train_command={shlex.join(command)}", flush=True)

    completed = subprocess.run(command, cwd=train_root, check=False)
    print(f"[gm-launcher] train_returncode={completed.returncode}", flush=True)
    print_tree(
        "[gm-launcher] run dir artifacts",
        run_dir,
        ("log.txt", "*.pth", "events.out.tfevents*", "*.json", "*.txt"),
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
