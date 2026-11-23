# Live (browser-mic) deployment of Lyra for the team.
# Runs the standalone live_server.py on a Linux container. The local owner app
# and its macOS-only features are NOT affected by this image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached.
COPY requirements.txt ./
# CPU-only PyTorch keeps the image and the container's memory use reasonable.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
    && pip install -r requirements.txt edge-tts

# Copy the application (app/, spike.py, config.yaml, ui/, live_ui/, live_server.py).
COPY . .

# Download the speech models at build time so cold starts are fast.
RUN python - <<'PY'
import numpy as np
from faster_whisper import WhisperModel
WhisperModel("base", device="cpu", compute_type="int8")
from spike import EcapaBackend
EcapaBackend()
from app.enrol.voiceprint import trim_to_speech
trim_to_speech(np.zeros(16000, dtype=np.float32), 16000)
print("models ready")
PY

# Cloud hosts inject PORT; live_server.py reads it automatically.
EXPOSE 8766
CMD ["python", "live_server.py", "--host", "0.0.0.0"]
