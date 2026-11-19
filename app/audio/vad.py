"""Ticket 2 — streaming speech segmentation on top of Silero VAD.

A small state machine over per-frame speech probabilities implements the
required timings: ~`start_ms` of speech to open an utterance, ~`end_ms` of
silence to close it, a `max_sec` hard cap, and `pre_roll_ms` of audio kept
before the speech start so the first words aren't clipped.

`process(frame)` is called with 512-sample (32 ms) mono float32 frames in
order; each completed utterance is handed to `on_utterance`.
"""
from __future__ import annotations

import collections
from typing import Callable

import numpy as np
import torch

from ..types import Utterance


class StreamingSegmenter:
    def __init__(self, model, samplerate: int = 16000, threshold: float = 0.5,
                 start_ms: int = 300, end_ms: int = 700, pre_roll_ms: int = 250,
                 max_sec: float = 15.0, frame_len: int = 512,
                 on_utterance: Callable[[Utterance], None] | None = None,
                 on_speech_state: Callable[[bool], None] | None = None,
                 noise_gate_ratio: float = 2.5):
        self.model = model
        self.sr = samplerate
        self.threshold = threshold
        self.frame_len = frame_len
        frame_ms = frame_len / samplerate * 1000.0
        self.start_frames = max(1, round(start_ms / frame_ms))
        self.end_frames = max(1, round(end_ms / frame_ms))
        self.pre_roll_frames = max(1, round(pre_roll_ms / frame_ms))
        self.max_frames = int(round(max_sec * samplerate / frame_len))
        self.on_utterance = on_utterance
        self.on_speech_state = on_speech_state
        self.noise_gate_ratio = float(noise_gate_ratio)
        self.reset()

    def reset(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        # history holds pre_roll + the opening speech run
        self.history = collections.deque(maxlen=self.pre_roll_frames + self.start_frames)
        self.candidate: list[np.ndarray] = []
        self.speech_count = 0
        self.silence_count = 0
        self.state = "silent"
        self.pos = 0  # samples seen so far
        # Ambient-noise floor, learned from frames Silero calls non-speech.
        self.noise_rms: float | None = 0.003

    def process(self, frame: np.ndarray, vad_frame: np.ndarray | None = None) -> None:
        """Feed one 512-sample frame. `frame` is the audio kept for the utterance
        (raw); `vad_frame`, if given, is the denoised copy used only for the
        speech/silence decision."""
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim > 1:
            frame = frame.mean(axis=1)
        n = len(frame)
        if n == 0:
            return
        self.pos += n

        vf = np.asarray(vad_frame, dtype=np.float32) if vad_frame is not None else frame
        if vf.ndim > 1:
            vf = vf.mean(axis=1)

        # --- adaptive noise gate ---
        # Silero decides speech vs. silence; the ambient floor acts as a
        # second opinion so stationary noise (fan/AC) that Silero occasionally
        # mistakes for speech cannot keep an utterance open forever.
        rms = float(np.sqrt(np.mean(vf * vf))) if vf.size else 0.0

        x = torch.from_numpy(vf)
        with torch.inference_mode():
            prob = float(self.model(x, self.sr).item())
        is_speech_prob = prob >= self.threshold

        if self.state == "silent" and not is_speech_prob:
            # Learn ambient only from frames Silero agrees are NOT speech, so a
            # sentence can never raise the floor above the speaker's own voice.
            if self.noise_rms is None:
                self.noise_rms = max(rms, 1e-4)
            else:
                self.noise_rms = 0.95 * self.noise_rms + 0.05 * min(rms, 0.2)
            self.noise_rms = min(self.noise_rms, 0.1)

        floor = max(self.noise_rms or 0.0, 0.002)
        gated = self.noise_gate_ratio > 1.0 and rms < floor * self.noise_gate_ratio
        is_speech = is_speech_prob and not gated

        self.history.append(frame)

        if self.state == "silent":
            if is_speech:
                self.speech_count += 1
                if self.speech_count >= self.start_frames:
                    self._begin()
            else:
                # decay rather than hard-reset, so a single noisy frame does
                # not throw away a sentence that is just starting
                self.speech_count = max(0, self.speech_count - 1)
            return

        # state == "speech"
        self.candidate.append(frame)
        if is_speech:
            # leaky counter: occasional noise spikes only slow the close, they
            # cannot reset it (a noisy room would otherwise hold the segment
            # open forever)
            self.silence_count = max(0, self.silence_count - 1)
        else:
            self.silence_count += 1
            if self.silence_count >= self.end_frames:
                self._finish(trailing_silence=self.end_frames)
                return
        if len(self.candidate) >= self.max_frames:
            self._finish(trailing_silence=0)

    def progress(self) -> dict:
        """Live state of the utterance being captured (for the UI)."""
        if self.state != "speech":
            return {"active": False, "seconds": 0.0, "silence_seconds": 0.0}
        return {
            "active": True,
            "seconds": len(self.candidate) * self.frame_len / self.sr,
            "silence_seconds": self.silence_count * self.frame_len / self.sr,
        }

    def flush(self) -> None:
        """Close any in-progress utterance (e.g. on stream stop)."""
        if self.state == "speech" and self.candidate:
            self._finish(trailing_silence=0)

    # ------------------------------------------------------------------
    def _begin(self) -> None:
        self.state = "speech"
        self.silence_count = 0
        self.candidate = list(self.history)
        if self.on_speech_state is not None:
            self.on_speech_state(True)

    def _finish(self, trailing_silence: int) -> None:
        frames = self.candidate
        if trailing_silence:
            frames = frames[:-trailing_silence]
        self.candidate = []
        self.speech_count = 0
        self.silence_count = 0
        self.state = "silent"
        self.history.clear()
        if self.on_speech_state is not None:
            self.on_speech_state(False)

        if not frames or len(frames) < self.start_frames:
            return
        audio = np.concatenate(frames).astype(np.float32, copy=False)
        end_sample = self.pos - trailing_silence * self.frame_len
        start_sample = end_sample - len(audio)
        utt = Utterance(
            audio=audio,
            start_s=start_sample / self.sr,
            end_s=end_sample / self.sr,
        )
        if self.on_utterance is not None:
            self.on_utterance(utt)
