"""Ticket 3 — per-recording quality checks before it becomes enrolment data."""
from __future__ import annotations

import numpy as np
from silero_vad import get_speech_timestamps, load_silero_vad

from ..config import AttrDict

_MODEL = None


def _vad():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_silero_vad()
    return _MODEL


def speech_mask(audio: np.ndarray, sr: int) -> np.ndarray:
    """Boolean mask of samples classified as speech."""
    stamps = get_speech_timestamps(audio, _vad(), sampling_rate=sr)
    mask = np.zeros(len(audio), dtype=bool)
    for s in stamps:
        mask[s["start"]:s["end"]] = True
    return mask


def validate(audio: np.ndarray, sr: int, cfg: AttrDict) -> tuple[bool, str]:
    """Return (ok, reason). `reason` is a short user-facing hint when not ok."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return False, "no audio recorded"

    dur = len(audio) / sr
    if dur < cfg.enrol.min_duration_sec:
        return False, "too short — read the whole prompt"

    peak = float(np.max(np.abs(audio)))
    if peak >= 0.99:
        return False, "clipping — move a little further from the mic"

    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_db = 20 * np.log10(rms + 1e-12)
    if rms_db < cfg.enrol.min_level_db:
        return False, "too quiet — move a bit closer"

    mask = speech_mask(audio, sr)
    speech_s = float(mask.sum()) / sr
    if speech_s < cfg.enrol.min_speech_sec:
        return False, "not enough speech — speak the full prompt clearly"

    speech_rms = float(np.sqrt(np.mean(audio[mask] ** 2))) if mask.any() else 0.0
    noise = audio[~mask]
    noise_rms = float(np.sqrt(np.mean(noise ** 2))) if noise.size else 0.0
    snr_db = 20 * np.log10(speech_rms / (noise_rms + 1e-12)) if noise_rms > 1e-6 else 60.0
    if snr_db < cfg.enrol.min_snr_db:
        return False, "too much background noise — find a quieter spot"

    return True, ""
