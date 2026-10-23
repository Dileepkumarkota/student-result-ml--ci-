# Student Result Prediction — MLOps Practicals 7 & 8

This repository contains the Student Result Prediction model-serving application, its automated tests, Docker packaging, and GitHub Actions workflow.

- **Practical 7:** Build and test the Docker image in GitHub Actions.
- **Practical 8:** Publish the already-tested image to GitHub Container Registry (GHCR), pull it back, and verify that the API still works.

## API
- `GET /` — service status
- `GET /health` — checks model availability
- `POST /predict` — JSON fields: `attendance`, `internal_marks`, `assignment_marks`, `previous_score` (each 0–100)

Example request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"attendance":90,"internal_marks":85,"assignment_marks":88,"previous_score":80}'
```

## Run locally
```bash
python -m pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest discover -v
python app.py
```

## Docker
The workflow trains the model before building the image so `student_result_model.pkl` and `metrics.json` are present in the build context.

```bash
docker build -t student-result-ml:local .
docker run -d --name student-result-ml-container -p 5000:5000 student-result-ml:local
curl --fail http://localhost:5000/health
docker rm -f student-result-ml-container
```

## Practical 8 — GHCR image
The GitHub Actions workflow automatically performs these steps on pushes to `main` (not on pull requests):
1. Train the model, enforce the ML quality gate, and run unit tests.
2. Build the Docker image and verify health and prediction endpoints.
3. Authenticate to GHCR using the repository-scoped `GITHUB_TOKEN` (no personal token is stored in the repository).
4. Publish both tags:
   - `ghcr.io/dileepkumarkota/student-result-prediction:v1`
   - `ghcr.io/dileepkumarkota/student-result-prediction:latest`
5. Pull the published `v1` image from GHCR, start a container from it, and repeat the health and prediction checks.

The workflow needs the repository's Actions permission to write packages; the workflow declares `packages: write`. The first GHCR package is normally private by default. Change package visibility in GitHub Packages settings only if public access is intended.

## Workflow
Open the [GitHub Actions runs](https://github.com/Dileepkumarkota/student-result-ml--ci-/actions) to see the actual build, test, publish, pull, and API validation result.
