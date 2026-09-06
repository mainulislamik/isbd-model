# ─── ISBD v1.00 — AI Training System (Dockerized) ───
# Multi-service build: Panel (FastAPI :8077) + Continuous 24/7 Trainer
FROM python:3.11-slim

# Minimal system deps
# - libglib2.0: OpenCV headless runtime
# - git: auto-commit checkpoints from trainer
# - docker.io (CLI only): panel controls trainer container via Docker socket
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (pinned to the exact versions the native venv uses)
RUN pip install --no-cache-dir \
    torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.52.4 \
    pillow==12.3.0 \
    numpy==2.4.6 \
    opencv-python-headless==5.0.0.93 \
    scikit-image==0.26.0 \
    ultralytics==8.4.140 \
    pypdf \
    python-docx \
    python-multipart==0.0.32 \
    pyyaml==6.0.3

# Copy application code (isbd package + templates + model weights)
COPY isbd/ ./isbd/
COPY yolov8n.pt ./yolov8n.pt

# Persistent volumes mount here: /app/checkpoints, /app/data, /app/samples
# Pre-create so ownership is right even on first run
RUN mkdir -p checkpoints data samples

EXPOSE 8077

# Default: run the panel (training runs as a separate compose service)
CMD ["python", "-m", "uvicorn", "isbd.panel:app", "--host", "0.0.0.0", "--port", "8077"]
