FROM python:3.10-slim

WORKDIR /app

# Install system libs required by LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip first, then install with binary preference to avoid hash/compile issues
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy source code
COPY ml      /app/ml
COPY backend /app/backend
COPY outputs /app/outputs

# Create logs directory
RUN mkdir -p /app/logs

# Train models at build time
RUN python /app/ml/train_model.py

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
