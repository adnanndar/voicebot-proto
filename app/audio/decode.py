"""Decode browser-uploaded audio to 16 kHz mono float32."""
from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path

import numpy as np

MIME_EXT = {
    "audio/webm": ".webm", "video/webm": ".webm",
    "audio/mp4": ".m4a", "video/mp4": ".m4a", "audio/mp4a-latm": ".m4a",
    "audio/aac": ".aac", "audio/x-m4a": ".m4a", "audio/mp4a": ".m4a",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/ogg": ".ogg", "application/ogg": ".ogg",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/flac": ".flac", "audio/x-flac": ".flac",
    "audio/aiff": ".aiff", "audio/x-aiff": ".aiff", "audio/aifc": ".aiff",
}


def guess_ext(mime: str, name: str = "") -> str:
    if mime:
        m = mime.split(";")[0].strip().lower()
        if m in MIME_EXT:
            return MIME_EXT[m]
    if name and "." in name:
        return Path(name).suffix.lower()
    return ".webm"


def decode_to_wav(b64: str, mime: str, name: str, workdir: Path) -> Path:
    """Decode uploaded audio to 16 kHz mono WAV with ffmpeg."""
    raw = base64.b64decode(b64)
    src = workdir / f"in{guess_ext(mime, name)}"
    dst = workdir / "out.wav"
    src.write_bytes(raw)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not dst.exists():
        raise ValueError(f"could not decode audio: {(r.stderr or '').strip()[:300]}")
    return dst


def decode_audio(b64: str, mime: str = "", name: str = "", sr: int = 16000) -> np.ndarray:
    """Base64 audio of any browser format -> mono float32 at `sr`."""
    if not b64:
        raise ValueError("no audio received")
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="lyra_audio_") as tmp:
        wav = decode_to_wav(b64, mime, name, Path(tmp))
        audio, _ = sf.read(str(wav), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=16000, target_sr=sr)
    return np.ascontiguousarray(audio, dtype=np.float32)
