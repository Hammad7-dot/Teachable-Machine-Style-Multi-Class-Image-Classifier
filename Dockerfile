FROM python:3.11-slim

# System deps needed by Pillow/scikit-image/tensorflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime data (SQLite db, uploaded images, model artifacts) lives here —
# mount a volume at /app/storage to persist across container restarts.
RUN mkdir -p /app/storage
VOLUME ["/app/storage"]

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
