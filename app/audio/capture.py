"""Ticket 2 — microphone capture.

Captures 16 kHz mono float32 in fixed-size blocks on PortAudio's callback
thread and hands each block to `on_frame`. The heavy work (denoise + VAD) must
not happen here; the pipeline forwards frames to a queue instead.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class MicrophoneSource:
    def __init__(self, samplerate: int = 16000, channels: int = 1,
                 blocksize: int = 512, device=None):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self._stream = None
        self._on_frame: Callable[[np.ndarray], None] | None = None

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        import sounddevice as sd

        self._on_frame = on_frame

        def callback(indata, frames, time_info, status):
            if status:
                # input overflow / underflow: drop rather than crash
                pass
            if indata.size:
                x = indata[:, 0] if indata.ndim > 1 else indata
                # MUST copy: PortAudio reuses `indata` on the next callback, so
                # holding a reference would let the buffer mutate under us —
                # that is what distorted the queued audio.
                on_frame(np.array(x, dtype=np.float32, copy=True))

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
