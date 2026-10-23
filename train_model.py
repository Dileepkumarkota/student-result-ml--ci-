"""Train a reproducible student pass/fail classifier and save its artifacts."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]

def main():
    rng = np.random.default_rng(42)
    n = 1200
    data = pd.DataFrame({
        "attendance": rng.integers(0, 101, n),
        "internal_marks": rng.integers(0, 101, n),
        "assignment_marks": rng.integers(0, 101, n),
        "previous_score": rng.integers(0, 101, n),
    })
    weighted_score = (0.20 * data.attendance + 0.35 * data.internal_marks
                      + 0.20 * data.assignment_marks + 0.25 * data.previous_score)
    # Small deterministic threshold makes the classification learnable and reproducible.
    data["result"] = (weighted_score >= 50).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data["result"], test_size=0.2, random_state=42, stratify=data["result"]
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "test_samples": int(len(y_test)),
    }
    joblib.dump(model, ROOT / "student_result_model.pkl")
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    data.to_csv(ROOT / "student_results.csv", index=False)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
