import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "volleyball_s_obj2coco_d_bg2.yml"


class Q100NoAConfigTests(unittest.TestCase):
    def read_config(self) -> str:
        return CONFIG_PATH.read_text(encoding="utf-8")

    def test_config_keeps_no_a_training_set(self):
        text = self.read_config()
        self.assertIn("ann_file: ../coco/converted/annotations/train.json", text)
        self.assertIn("negative_samples 仅用于 OA 评估，不并入训练集", text)

    def test_config_uses_q100_without_freeze(self):
        text = self.read_config()
        self.assertIn("num_queries: 100", text)
        self.assertIn("num_top_queries: 100", text)
        self.assertNotIn("freeze_at:", text)
        self.assertNotIn("freeze_norm:", text)

    def test_config_disables_oa_eval_by_default(self):
        text = self.read_config()
        self.assertIn("eval_overactivation: False", text)
        self.assertIn("oa_ap_min: 0.84", text)


if __name__ == "__main__":
    unittest.main()
