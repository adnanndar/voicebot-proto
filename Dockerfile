# Live (browser-mic) deployment of Lyra for the team.
# Runs the standalone live_server.py on a Linux container. The local owner app
# and its macOS-only features are NOT affected by this image.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt edge-tts

# Copy the application (app/, config.yaml, ui/, live_ui/, live_server.py, prompts).
COPY . .

EXPOSE 8766
CMD ["python", "live_server.py", "--port", "8766"]
