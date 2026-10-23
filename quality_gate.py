"""Fail the CI workflow if model accuracy is below the required threshold."""
import json
from pathlib import Path

metrics_path = Path(__file__).resolve().parent / "metrics.json"
if not metrics_path.exists():
    raise SystemExit("FAIL: metrics.json is missing; train the model first.")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
accuracy = float(metrics.get("accuracy", 0))
threshold = 0.80
print(f"Model accuracy: {accuracy:.3f}; required minimum: {threshold:.2f}")
if accuracy < threshold:
    raise SystemExit("FAIL: model accuracy is below the quality-gate threshold.")
print("PASS: ML quality gate passed.")
