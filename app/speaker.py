"""Shared speaker embedding backend (ECAPA-TDNN or Resemblyzer).

Reuses the exact models from the ticket-1 spike. Embeddings are L2-normalised
so that cosine similarity is a plain dot product.
"""
from __future__ import annotations

import numpy as np

from spike import EcapaBackend, ResemblyzerBackend

_BACKENDS: dict = {}


def get_backend(name: str):
    if name not in _BACKENDS:
        _BACKENDS[name] = EcapaBackend() if name == "ecapa" else ResemblyzerBackend()
    return _BACKENDS[name]


def embed(name: str, wav: np.ndarray) -> np.ndarray:
    """L2-normalised speaker embedding for 16 kHz mono float32 audio.

    The audio is peak-normalised first so quiet and loud captures embed the
    same way (matches enrolment and live).
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1e-4:
        wav = wav * (0.7 / peak)
    v = np.asarray(get_backend(name).embed(wav), dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12) if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
