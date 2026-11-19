"""Ticket 4 — decide owner / not-owner for every utterance, before ASR.

Rules:
  - score = cosine similarity between the utterance embedding and the voiceprint
    (both L2-normalised, so it's a dot product).
  - the WHOLE utterance is scored once, after the speaker has stopped talking.
  - score >= accept_threshold (35%) -> owner
  - score <  accept_threshold        -> not owner
  - this rule is applied to EVERY utterance, including follow-ups in an open
    conversation: only a >= 35% match is ever answered.
"""
from __future__ import annotations

import time

import numpy as np

from .. import speaker
from ..config import AttrDict
from ..types import Decision, Utterance


class SpeakerGate:
    def __init__(self, cfg: AttrDict, voiceprint: np.ndarray | None):
        self.cfg = cfg
        self.voiceprint = voiceprint
        self.accept = float(cfg.gate.accept_threshold)
        self.reject = float(cfg.gate.reject_threshold)
        self.min_utt = float(cfg.gate.min_utterance_sec)
        self.model = cfg.speaker.model
        self.sr = cfg.audio.sample_rate
        # Rejecting one utterance is not enough if someone keeps trying: after
        # several unknown voices in a row, pause recognition for a while.
        self.lockout_after = int(cfg.gate.get("lockout_after", 5))
        self.lockout_window = float(cfg.gate.get("lockout_window_sec", 60.0))
        self.lockout_sec = float(cfg.gate.get("lockout_sec", 60.0))
        self._rejects: list[float] = []
        self._locked_until = 0.0
        # kept for backwards compatibility with callers that read it
        self.last_owner_s = -float("inf")

    def has_voiceprint(self) -> bool:
        return self.voiceprint is not None

    def score(self, audio: np.ndarray) -> float:
        emb = speaker.embed(self.model, audio)
        return float(np.dot(emb, self.voiceprint))

    def decide(self, utt: Utterance) -> Decision:
        now = utt.end_s

        if self.voiceprint is None:
            return Decision(False, None, "no_voiceprint", "no voiceprint enrolled", "none")

        if utt.duration_s < self.min_utt:
            return Decision(False, None, "ignored", "too short to score reliably", "short")

        # score the WHOLE utterance (the speaker has stopped talking), as one recording
        score = self.score(utt.audio)
        pct = int(round(self.accept * 100))

        # already paused after repeated impostor attempts?
        mono = time.monotonic()
        if mono < self._locked_until:
            left = int(round(self._locked_until - mono))
            return Decision(
                False, round(float(score), 4), "locked",
                f"paused for {left}s after repeated unknown voices", "lockout",
            )

        # simple owner rule: 35% similarity or more is the owner, otherwise not.
        # No follow-up exception: every utterance must pass this bar.
        if score >= self.accept:
            self._rejects.clear()
            decision, reason = "owner", f"score >= {pct}% owner rule"
            self.last_owner_s = now
        else:
            self._rejects = [t for t in self._rejects if mono - t <= self.lockout_window]
            self._rejects.append(mono)
            if self.lockout_after > 0 and len(self._rejects) >= self.lockout_after:
                self._rejects.clear()
                self._locked_until = mono + self.lockout_sec
                return Decision(
                    False, round(float(score), 4), "locked",
                    f"paused for {int(self.lockout_sec)}s after repeated unknown voices",
                    "lockout",
                )
            decision, reason = "not_owner", f"score < {pct}% owner rule"

        return Decision(score >= self.accept, round(float(score), 4), decision, reason, "full")
