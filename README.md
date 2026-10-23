# Student Result ML CI/CD — Practical 7

Automates ML training, quality-gate checks, unit tests, Docker image building, container startup, health verification, prediction API testing, log capture, and container cleanup using GitHub Actions.

## API
- `GET /` — service status
- `GET /health` — checks model availability
- `POST /predict` — JSON fields: `attendance`, `internal_marks`, `assignment_marks`, `previous_score` (each 0–100)

## Run locally
```bash
python -m pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest discover -v
python app.py
```

Example request:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"attendance":90,"internal_marks":85,"assignment_marks":88,"previous_score":80}'
```

## Docker
The workflow trains the model before building the image so `student_result_model.pkl` and `metrics.json` are present in the build context.

```bash
docker build -t student-result-ml:local .
docker run -d --name student-result-ml-container -p 5000:5000 student-result-ml:local
curl --fail http://localhost:5000/health
docker logs student-result-ml-container
docker rm -f student-result-ml-container
```

## Practical 7 CI stages
1. Checkout and set up Python.
2. Install dependencies, train model, and enforce accuracy threshold.
3. Run automated ML and API tests.
4. Build Docker image and start a container on port 5000.
5. Retry `/health` until ready and POST to `/predict`.
6. Always print container logs and clean up the container.

This practical validates the image inside the temporary GitHub Actions runner. It does **not** push an image to a registry or deploy to production.
