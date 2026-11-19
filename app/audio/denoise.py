"""Ticket 2 — noise suppression.

Uses `noisereduce` (spectral gating) as a dependency-free-in-practice stand-in
for WebRTC APM / RNNoise. It runs on ~`ns_chunk_sec` blocks. A production build
would swap this for RNNoise or WebRTC audio processing; the interface
(`reduce(array) -> array`) is the only contract the rest of the code relies on.
"""
from __future__ import annotations

import numpy as np


class Denoiser:
    def __init__(self, samplerate: int = 16000, chunk_sec: float = 0.5, prop_decrease: float = 0.85):
        self.sr = samplerate
        self.chunk = max(512, int(samplerate * chunk_sec))
        self.prop_decrease = prop_decrease
        self.noise_profile: np.ndarray | None = None
        self._nr = None
        try:
            import noisereduce as nr

            self._nr = nr
            # warm up: first call imports scipy bits and is ~1.5 s, later ~7 ms
            nr.reduce_noise(y=np.zeros(512, dtype=np.float32), sr=samplerate,
                            stationary=True, n_fft=512, win_length=512, hop_length=128)
        except ImportError:
            self._nr = None

    def set_noise_profile(self, y: np.ndarray) -> None:
        """Provide a sample of ambient noise (e.g. a fan/AC) to subtract."""
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.size >= self.sr // 4:  # at least 250 ms
            self.noise_profile = y

    def reduce(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) == 0 or self._nr is None:
            return y
        kwargs = dict(y=y, sr=self.sr, stationary=True, prop_decrease=self.prop_decrease,
                      n_fft=512, win_length=512, hop_length=128)
        if self.noise_profile is not None:
            kwargs["y_noise"] = self.noise_profile
        return np.asarray(self._nr.reduce_noise(**kwargs), dtype=np.float32)
