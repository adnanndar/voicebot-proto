"""Ticket 2 — glue: mic/file -> denoise -> VAD -> utterances."""
from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np
from silero_vad import load_silero_vad

from ..config import AttrDict
from ..types import Utterance
from .capture import MicrophoneSource
from .denoise import Denoiser
from .vad import StreamingSegmenter


def load_vad_model():
    return load_silero_vad()


VAD_FRAME = 512  # Silero VAD only accepts exactly 512 samples at 16 kHz


def make_segmenter(model, cfg: AttrDict, on_utterance: Callable[[Utterance], None],
                   frame_len: int = VAD_FRAME) -> StreamingSegmenter:
    return StreamingSegmenter(
        model,
        samplerate=cfg.audio.sample_rate,
        frame_len=frame_len,
        on_utterance=on_utterance,
        **dict(cfg.vad),
    )


def iter_utterances(audio: np.ndarray, cfg: AttrDict, model=None) -> list[Utterance]:
    """Run the full denoise+segment chain over a complete array (file mode / tests).

    VAD runs on the denoised copy; the returned utterances keep the RAW audio,
    so speaker verification sees the same signal as enrolment.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    sr = cfg.audio.sample_rate
    frame = VAD_FRAME

    denoised = audio
    if cfg.audio.noise_suppression:
        denoiser = Denoiser(sr, cfg.audio.ns_chunk_sec, cfg.audio.ns_prop_decrease)
        denoised = np.concatenate(
            [denoiser.reduce(audio[i:i + denoiser.chunk]) for i in range(0, len(audio), denoiser.chunk)]
        )[:len(audio)]
        if len(denoised) < len(audio):  # never let VAD and raw drift apart
            denoised = np.concatenate([denoised, np.zeros(len(audio) - len(denoised), np.float32)])

    model = model or load_vad_model()
    out: list[Utterance] = []
    seg = make_segmenter(model, cfg, on_utterance=out.append, frame_len=frame)
    for i in range(0, len(audio) - frame + 1, frame):
        seg.process(audio[i:i + frame], denoised[i:i + frame])
    seg.flush()
    return out


class AudioPipeline:
    """Live microphone pipeline running on a background thread.

    Frames arrive on PortAudio's callback thread and are queued; a worker
    denoises them in chunks and feeds the segmenter. Completed utterances are
    emitted on the worker thread via `on_utterance`.
    """

    # Silero VAD only accepts exactly 512 samples at 16 kHz, regardless of the
    # microphone block size.
    VAD_FRAME = 512

    def __init__(self, cfg: AttrDict, on_utterance: Callable[[Utterance], None],
                 on_level: Callable[[float], None] | None = None,
                 on_speech: Callable[[bool], None] | None = None):
        self.cfg = cfg
        self.on_utterance = on_utterance
        self.on_level = on_level
        self.on_speech = on_speech
        self.muted = False  # AEC: set while the bot is speaking
        self._level_count = 0
        self._model = None
        self._segmenter = None
        self._denoiser = None
        self._profile_chunks: list[np.ndarray] = []
        self._profile_done = False
        self._frames: queue.Queue = queue.Queue(maxsize=4096)
        self._thread: threading.Thread | None = None
        self._running = False
        self._source: MicrophoneSource | None = None
        self._remainder = np.zeros(0, np.float32)

    # -- mic ------------------------------------------------------------
    def start(self) -> None:
        self._model = load_vad_model()
        self._segmenter = make_segmenter(self._model, self.cfg, self.on_utterance)
        if self.on_speech is not None:
            self._segmenter.on_speech_state = self.on_speech
        self._denoiser = Denoiser(self.cfg.audio.sample_rate, self.cfg.audio.ns_chunk_sec,
                                  self.cfg.audio.ns_prop_decrease) \
            if self.cfg.audio.noise_suppression else None
        self._profile_chunks = []
        self._profile_done = float(getattr(self.cfg.audio, "ns_profile_sec", 0.5)) <= 0
        self._remainder = np.zeros(0, np.float32)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-pipeline")
        self._thread.start()
        self._source = MicrophoneSource(
            samplerate=self.cfg.audio.sample_rate,
            channels=self.cfg.audio.channels,
            blocksize=self.cfg.audio.blocksize,
            device=self.cfg.audio.device,
        )
        self._source.start(self._enqueue)

    def capture_status(self) -> dict:
        seg = self._segmenter
        if seg is None:
            return {"active": False, "seconds": 0.0, "silence_seconds": 0.0}
        return seg.progress()

    def stop(self) -> None:
        self._running = False
        if self._source is not None:
            self._source.stop()
            self._source = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._segmenter is not None:
            self._segmenter.flush()

    def _enqueue(self, frame: np.ndarray) -> None:
        if self.on_level is not None:
            self._level_count += 1
            if self._level_count % 3 == 0:  # ~10 Hz at 32 ms frames
                rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
                self.on_level(rms)
        if self.muted:
            return
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            pass

    def _run(self) -> None:
        # Denoise in ~0.5 s chunks when noise suppression is on; otherwise pass
        # audio through as soon as it arrives (32 ms frames).
        chunk = self._denoiser.chunk if self._denoiser is not None else self.VAD_FRAME
        buf: list[np.ndarray] = []
        n = 0
        while self._running:
            try:
                f = self._frames.get(timeout=0.2)
            except queue.Empty:
                if buf:
                    self._flush(buf)
                    buf, n = [], 0
                continue
            buf.append(f)
            n += len(f)
            if n >= chunk:
                self._flush(buf)
                buf, n = [], 0
        if buf:
            self._flush(buf)

    def _flush(self, frames: list[np.ndarray]) -> None:
        if not frames:
            return
        # Keep a sample-level remainder so every block handed to the segmenter
        # is exactly 512 samples (Silero requirement), whatever the mic block
        # size or denoiser chunk size happens to be.
        raw = np.concatenate([self._remainder] + frames).astype(np.float32)
        frame = self.VAD_FRAME
        usable = len(raw) - (len(raw) % frame)
        if usable <= 0:
            self._remainder = raw
            return
        raw, self._remainder = np.ascontiguousarray(raw[:usable]), np.ascontiguousarray(raw[usable:])

        if self._denoiser is not None and self._denoiser.noise_profile is None and not self._profile_done:
            # Learn the ambient noise (fan/AC) ONLY from quiet audio captured
            # while no utterance is open. If the speaker starts immediately we
            # skip it, because subtracting a profile that contains speech is
            # what used to cut sentences short and empty the transcript.
            if self._segmenter.state != "silent":
                self._profile_done = True
                self._profile_chunks = []
            else:
                self._profile_chunks.append(raw)
                if len(self._profile_chunks) >= 4:
                    quietest = min(self._profile_chunks, key=lambda c: float(np.mean(c * c)))
                    rms = float(np.sqrt(np.mean(quietest * quietest)))
                    if rms <= 0.02:  # genuinely quiet (~ -34 dB or lower)
                        self._denoiser.set_noise_profile(quietest)
                    self._profile_done = True
                    self._profile_chunks = []

        vad = self._denoiser.reduce(raw) if self._denoiser is not None else raw
        # The VAD copy must be exactly as long as the audio we keep.
        if len(vad) < len(raw):
            vad = np.concatenate([vad, np.zeros(len(raw) - len(vad), np.float32)])
        elif len(vad) > len(raw):
            vad = vad[:len(raw)]
        gain = float(self.cfg.audio.gain_db)
        if gain:
            vad = vad * (10.0 ** (gain / 20.0))
        for i in range(0, len(raw), frame):
            self._segmenter.process(raw[i:i + frame], vad[i:i + frame])
