#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = "config/volleyball_s_obj2coco_d_bg2.yml"
DEFAULT_OUTPUT_DIR = "runs/gradmotion/no-a/output"
DEFAULT_SUMMARY_DIR = "runs/gradmotion/no-a/summary"
DEFAULT_TUNING = "weights/dfine_s_obj2coco.pth"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="GradMotion gm-run launcher for D-FINE no-A training."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root that contains cloud_train.sh",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-dir", default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--tuning", default=DEFAULT_TUNING)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--update", action="append", default=[])
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use-amp", dest="use_amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    return parser.parse_args(argv)


def build_cloud_train_command(
    project_root: Path,
    config: str,
    output_dir: str,
    summary_dir: str,
    tuning: str,
    seed: int,
    updates: list[str],
    use_amp: bool,
) -> list[str]:
    command = [
        "bash",
        str(project_root / "cloud_train.sh"),
        "--config",
        config,
        "--output-dir",
        output_dir,
        "--summary-dir",
        summary_dir,
        "--tuning",
        tuning,
        "--seed",
        str(seed),
    ]
    for update_expr in updates:
        command.extend(["--update", update_expr])
    if not use_amp:
        command.append("--no-amp")
    return command


def main(argv=None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    cloud_train_path = project_root / "cloud_train.sh"

    if not cloud_train_path.exists():
        raise FileNotFoundError(f"cloud_train.sh not found: {cloud_train_path}")

    command = build_cloud_train_command(
        project_root=project_root,
        config=args.config,
        output_dir=args.output_dir,
        summary_dir=args.summary_dir,
        tuning=args.tuning,
        seed=args.seed,
        updates=args.update,
        use_amp=args.use_amp,
    )

    print("=== gm_dfine_launcher ===", flush=True)
    print(f"[gm-launcher] cwd={os.getcwd()}", flush=True)
    print(f"[gm-launcher] launcher={Path(__file__).resolve()}", flush=True)
    print(f"[gm-launcher] project_root={project_root}", flush=True)
    print(f"[gm-launcher] python={sys.executable}", flush=True)
    print(f"[gm-launcher] command={shlex.join(command)}", flush=True)

    completed = subprocess.run(command, cwd=project_root, check=False)
    print(f"[gm-launcher] returncode={completed.returncode}", flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
