import importlib.util
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = PROJECT_ROOT / "benchmark" / "scripts" / "gm_dfine_launcher.py"
TASK_JSON_PATH = PROJECT_ROOT / "task_smoke_test_final.json"


def load_launcher_module():
    spec = importlib.util.spec_from_file_location("gm_dfine_launcher", LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load launcher module from {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GMDFineLauncherTests(unittest.TestCase):
    def test_build_cloud_train_command_keeps_explicit_paths_and_updates(self):
        module = load_launcher_module()
        repo_root = Path("/workspace/isaaclab/D-FINE_train")

        command = module.build_cloud_train_command(
            project_root=repo_root,
            config="config/volleyball_s_obj2coco_d_bg2.yml",
            output_dir="runs/gradmotion/smoke-no-a/output",
            summary_dir="runs/gradmotion/smoke-no-a/summary",
            tuning="weights/dfine_s_obj2coco.pth",
            seed=7,
            updates=["epochs=1", "checkpoint_freq=1"],
            use_amp=True,
        )

        self.assertEqual(command[0], "bash")
        self.assertEqual(command[1], str(repo_root / "cloud_train.sh"))
        self.assertEqual(
            command[2:],
            [
                "--config",
                "config/volleyball_s_obj2coco_d_bg2.yml",
                "--output-dir",
                "runs/gradmotion/smoke-no-a/output",
                "--summary-dir",
                "runs/gradmotion/smoke-no-a/summary",
                "--tuning",
                "weights/dfine_s_obj2coco.pth",
                "--seed",
                "7",
                "--update",
                "epochs=1",
                "--update",
                "checkpoint_freq=1",
            ],
        )

    def test_build_cloud_train_command_adds_no_amp_flag_when_requested(self):
        module = load_launcher_module()
        command = module.build_cloud_train_command(
            project_root=Path("/workspace/isaaclab/D-FINE_train"),
            config="config/volleyball_s_obj2coco_d_bg2.yml",
            output_dir="runs/out",
            summary_dir="runs/summary",
            tuning="weights/dfine_s_obj2coco.pth",
            seed=0,
            updates=[],
            use_amp=False,
        )

        self.assertEqual(command[-1], "--no-amp")


class GitSmokeTemplateTests(unittest.TestCase):
    def test_git_smoke_template_uses_gm_run_launcher_and_hparams(self):
        payload = json.loads(TASK_JSON_PATH.read_text(encoding="utf-8"))
        code_info = payload["taskCodeInfo"]

        self.assertEqual(code_info["mainCodeName"], "gm_dfine_launcher.py")
        self.assertEqual(
            code_info["mainCodeUri"],
            "isaaclab/D-FINE_train/benchmark/scripts/gm_dfine_launcher.py",
        )
        self.assertEqual(
            code_info["hparamsPath"],
            "isaaclab/D-FINE_train/config/volleyball_s_obj2coco_d_bg2.yml",
        )
        self.assertIn(
            "gm-run D-FINE_train/benchmark/scripts/gm_dfine_launcher.py",
            code_info["startScript"],
        )
        self.assertIn("--output-dir runs/gradmotion/smoke-no-a/output", code_info["startScript"])
        self.assertIn("--summary-dir runs/gradmotion/smoke-no-a/summary", code_info["startScript"])


if __name__ == "__main__":
    unittest.main()
