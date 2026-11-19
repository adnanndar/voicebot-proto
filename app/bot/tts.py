"""Ticket 5 — text-to-speech. macOS `say` by default; print/null for testing."""
from __future__ import annotations

import subprocess
import threading

from ..config import AttrDict


class SayTTS:
    def __init__(self, voice: str | None = None):
        self.voice = voice
        self._proc: subprocess.Popen | None = None
        self._done = threading.Event()
        self._done.set()
        self._thread: threading.Thread | None = None

    def speak(self, text: str) -> None:
        self._done.clear()
        self._thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        self._thread.start()

    def _run(self, text: str) -> None:
        cmd = ["say"]
        if self.voice:
            cmd += ["-v", self.voice]
        cmd += [text]
        try:
            self._proc = subprocess.Popen(cmd)
            self._proc.wait()
        finally:
            self._proc = None
            self._done.set()

    def stop(self) -> None:
        p = self._proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                p.kill()
        self._done.set()

    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)


class PrintTTS:
    """Speaks to stdout (for headless testing)."""

    def __init__(self, voice=None):
        self._done = threading.Event()
        self._done.set()

    def speak(self, text: str) -> None:
        self._done.clear()
        print(f"\n  [TTS] {text}")
        self._done.set()

    def stop(self) -> None:
        self._done.set()

    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return True


class NullTTS(PrintTTS):
    def speak(self, text: str) -> None:
        pass


def make_tts(cfg: AttrDict):
    kind = cfg.bot.tts
    if kind == "say":
        return SayTTS(cfg.bot.tts_voice)
    if kind == "print":
        return PrintTTS()
    if kind == "null":
        return NullTTS()
    raise ValueError(f"unknown tts backend: {kind}")
