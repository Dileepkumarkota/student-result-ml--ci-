FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py train_model.py ./
COPY student_result_model.pkl metrics.json ./
EXPOSE 5000
CMD ["python", "app.py"]
