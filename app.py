"""Flask API for the student result prediction model."""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_result_model.pkl"
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]
app = Flask(__name__)
_model = None

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model artifact missing. Run train_model.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/")
def index():
    return jsonify({"service": "student-result-ml", "status": "running"})

@app.get("/health")
def health():
    if not MODEL_PATH.exists():
        return jsonify({"status": "unhealthy", "reason": "model artifact missing"}), 503
    try:
        load_model()
        return jsonify({"status": "healthy"})
    except Exception as exc:
        return jsonify({"status": "unhealthy", "reason": str(exc)}), 503

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    missing = [key for key in FEATURES if key not in payload]
    if missing:
        return jsonify({"error": "Missing required fields.", "missing_fields": missing}), 400
    try:
        values = {key: float(payload[key]) for key in FEATURES}
    except (TypeError, ValueError):
        return jsonify({"error": "All input values must be numeric."}), 400
    if any(value < 0 or value > 100 for value in values.values()):
        return jsonify({"error": "All input values must be between 0 and 100."}), 400
    try:
        prediction = int(load_model().predict(pd.DataFrame([values], columns=FEATURES))[0])
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
    return jsonify({"prediction": "PASS" if prediction == 1 else "FAIL",
                    "prediction_code": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
