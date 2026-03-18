import importlib.util
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = PROJECT_ROOT / "benchmark" / "scripts" / "gm_dfine_launcher.py"
SMOKE_TASK_JSON_PATH = PROJECT_ROOT / "task_smoke_test_final.json"
MAIN_TASK_JSON_PATH = PROJECT_ROOT / "task_cloud_train_no_A.json"


def load_launcher_module():
    spec = importlib.util.spec_from_file_location("gm_dfine_launcher", LAUNCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load launcher module from {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GMDFineLauncherTests(unittest.TestCase):
    def test_build_train_command_uses_single_run_dir_for_output_and_summary(self):
        module = load_launcher_module()
        repo_root = Path("/workspace/isaaclab/D-FINE_train")
        run_dir = repo_root / "runs/gradmotion/smoke-no-a/run"

        command = module.build_train_command(
            python_executable="python",
            project_root=repo_root,
            config="config/volleyball_s_obj2coco_d_bg2.yml",
            run_dir=run_dir,
            tuning="weights/dfine_s_obj2coco.pth",
            seed=7,
            updates=["epochs=1", "checkpoint_freq=1"],
            use_amp=True,
            negative_img_dir=repo_root / "coco/images/negative_samples",
        )

        self.assertEqual(
            command,
            [
                "python",
                "-u",
                str(repo_root / "D-FINE" / "train.py"),
                "-c",
                str(repo_root / "config/volleyball_s_obj2coco_d_bg2.yml"),
                "--seed",
                "7",
                "--output-dir",
                str(run_dir),
                "--summary-dir",
                str(run_dir),
                "-t",
                str(repo_root / "weights/dfine_s_obj2coco.pth"),
                "-u",
                f"negative_img_dir={repo_root / 'coco/images/negative_samples'}",
                "-u",
                "epochs=1",
                "-u",
                "checkpoint_freq=1",
                "--use-amp",
            ],
        )

    def test_build_train_command_omits_use_amp_when_disabled(self):
        module = load_launcher_module()
        repo_root = Path("/workspace/isaaclab/D-FINE_train")
        run_dir = repo_root / "runs/gradmotion/no-a-main/run"

        command = module.build_train_command(
            python_executable=sys.executable,
            project_root=repo_root,
            config="config/volleyball_s_obj2coco_d_bg2.yml",
            run_dir=run_dir,
            tuning="weights/dfine_s_obj2coco.pth",
            seed=0,
            updates=[],
            use_amp=False,
            negative_img_dir=repo_root / "coco/images/negative_samples",
        )

        self.assertNotIn("--use-amp", command)


class GitTaskTemplateTests(unittest.TestCase):
    def test_git_smoke_template_uses_direct_gm_run_entry_and_run_dir(self):
        payload = json.loads(SMOKE_TASK_JSON_PATH.read_text(encoding="utf-8"))
        code_info = payload["taskCodeInfo"]

        self.assertEqual(code_info["mainCodeName"], "gm_dfine_launcher.py")
        self.assertEqual(code_info["mainWorkDir"], "isaaclab/D-FINE_train/benchmark/scripts")
        self.assertEqual(
            code_info["mainCodeUri"],
            "isaaclab/D-FINE_train/benchmark/scripts/gm_dfine_launcher.py",
        )
        self.assertEqual(
            code_info["hparamsPath"],
            "isaaclab/D-FINE_train/config/volleyball_s_obj2coco_d_bg2.yml",
        )
        self.assertIn("gm-run gm_dfine_launcher.py", code_info["startScript"])
        self.assertIn("--run-dir runs/gradmotion/smoke-no-a/run", code_info["startScript"])
        self.assertNotIn("--output-dir", code_info["startScript"])
        self.assertNotIn("--summary-dir", code_info["startScript"])

    def test_git_main_template_uses_direct_gm_run_entry_and_run_dir(self):
        payload = json.loads(MAIN_TASK_JSON_PATH.read_text(encoding="utf-8"))
        code_info = payload["taskCodeInfo"]

        self.assertEqual(code_info["mainCodeName"], "gm_dfine_launcher.py")
        self.assertEqual(code_info["mainWorkDir"], "isaaclab/D-FINE_train/benchmark/scripts")
        self.assertEqual(
            code_info["mainCodeUri"],
            "isaaclab/D-FINE_train/benchmark/scripts/gm_dfine_launcher.py",
        )
        self.assertIn("gm-run gm_dfine_launcher.py", code_info["startScript"])
        self.assertIn("--run-dir runs/gradmotion/no-a-main/run", code_info["startScript"])
        self.assertNotIn("--output-dir", code_info["startScript"])
        self.assertNotIn("--summary-dir", code_info["startScript"])


if __name__ == "__main__":
    unittest.main()
