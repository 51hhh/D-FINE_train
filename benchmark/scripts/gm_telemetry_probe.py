#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    print("GM_PROBE_START", flush=True)
    print(f"cwd={os.getcwd()}", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"version={sys.version}", flush=True)

    try:
        import torch
        print(f"torch={torch.__version__}", flush=True)
    except Exception as e:
        print(f"torch_import_failed={e}", flush=True)
        return 11

    try:
        from torch.utils.tensorboard import SummaryWriter
        print("summary_writer=ok", flush=True)
    except Exception as e:
        print(f"summary_writer_failed={e}", flush=True)
        return 12

    summary_dir = Path("gm_probe_summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(summary_dir.as_posix())
    for step in range(args.steps):
        loss = 1.0 / (step + 1)
        score = step / max(args.steps - 1, 1)
        writer.add_scalar("Probe/loss", loss, step)
        writer.add_scalar("Probe/score", score, step)
        writer.add_scalar("Probe/step", step, step)
        writer.flush()
        print(f"probe_step={step} loss={loss:.6f} score={score:.6f}", flush=True)
        time.sleep(args.sleep)

    Path("gm_probe_done.txt").write_text("ok\n", encoding="utf-8")

    writer.close()

    print("generated_files:", flush=True)
    for path in sorted(Path(".").glob("gm_probe*")):
        print(path.as_posix(), flush=True)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                print(child.as_posix(), flush=True)

    print("GM_PROBE_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
