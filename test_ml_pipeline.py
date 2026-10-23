import json
import subprocess
import sys
import unittest
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent

class ModelPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable, str(ROOT / "train_model.py")], check=True, cwd=ROOT)
        cls.model = joblib.load(ROOT / "student_result_model.pkl")

    def test_model_and_metrics_artifacts_exist(self):
        self.assertTrue((ROOT / "student_result_model.pkl").exists())
        self.assertTrue((ROOT / "metrics.json").exists())

    def test_quality_threshold(self):
        metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(metrics["accuracy"], 0.80)

    def test_high_scores_pass_and_low_scores_fail(self):
        high = pd.DataFrame([{"attendance": 95, "internal_marks": 95,
                              "assignment_marks": 95, "previous_score": 95}])
        low = pd.DataFrame([{"attendance": 5, "internal_marks": 5,
                             "assignment_marks": 5, "previous_score": 5}])
        self.assertEqual(int(self.model.predict(high)[0]), 1)
        self.assertEqual(int(self.model.predict(low)[0]), 0)

if __name__ == "__main__":
    unittest.main()
