"""Small shared data types."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Utterance:
    """A detected speech segment. `audio` is 16 kHz mono float32."""
    audio: np.ndarray
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Decision:
    """The speaker-gate verdict for one utterance."""
    owner: bool
    score: float | None
    decision: str          # owner | not_owner | ignored | no_voiceprint
    reason: str
    method: str            # single | windowed_median | short
    transcript: str = ""

    def as_dict(self) -> dict:
        return {
            "owner": self.owner,
            "score": self.score,
            "decision": self.decision,
            "reason": self.reason,
            "method": self.method,
            "transcript": self.transcript,
        }
