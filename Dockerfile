FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8080"]
RUN apt-get update && apt-get install -y python3-dev gcc
RUN pip install --no-cache-dir praat-parselmouth

