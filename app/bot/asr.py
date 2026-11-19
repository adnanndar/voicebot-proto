"""Speech-to-text with faster-whisper. English only."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import AttrDict

LANGUAGE = "en"  # the assistant only understands English


class ASR:
    def __init__(self, cfg: AttrDict):
        self.model_name = cfg.bot.asr_model
        self.sr = cfg.audio.sample_rate
        self._model = None

    def _get(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio: np.ndarray) -> tuple[str, dict]:
        """Return (english_transcript, info)."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Normalise the level so quiet microphones still reach the model at a
        # comfortable volume (quiet audio used to come back as an empty string).
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-6:
            audio = (audio / peak * 0.95).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        try:
            sf.write(str(tmp), audio, self.sr, subtype="PCM_16")
            segments, _info = self._get().transcribe(
                str(tmp), language=LANGUAGE, beam_size=1, vad_filter=False,
                condition_on_previous_text=False, no_speech_threshold=1.0,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            return text, {"language": LANGUAGE}
        finally:
            tmp.unlink(missing_ok=True)
