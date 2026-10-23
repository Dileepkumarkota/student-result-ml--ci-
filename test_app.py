import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "train_model.py")], check=True, cwd=ROOT)
from app import app

class FlaskAPITests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "healthy")

    def test_high_scores_pass(self):
        response = self.client.post("/predict", json={
            "attendance": 95, "internal_marks": 95,
            "assignment_marks": 95, "previous_score": 95
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "PASS")

    def test_low_scores_fail(self):
        response = self.client.post("/predict", json={
            "attendance": 5, "internal_marks": 5,
            "assignment_marks": 5, "previous_score": 5
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "FAIL")

    def test_missing_fields_returns_400(self):
        response = self.client.post("/predict", json={"attendance": 90})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
