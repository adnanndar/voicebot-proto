#!/usr/bin/env python3
"""Warm macOS speech-synthesis worker for the live Lyra server.

The live server used to launch `say` for every reply. `say` spends roughly a
second loading the speech engine on every call, so the spoken reply lagged
noticeably behind the text. This worker keeps NSSpeechSynthesizer (the same
engine and default voice `say` uses) loaded and answers small requests over
stdin/stdout instead.

Protocol (binary, big-endian):
    request:  4-byte length + UTF-8 text
    response: 4-byte length + 16-bit PCM WAV (0 length means synthesis failed)

Only the live server uses this; the approved owner app is untouched.
"""
from __future__ import annotations

import struct
import sys
import tempfile
import time
from pathlib import Path

TIMEOUT_S = 30.0


def _read_exact(stream, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _pick_voice(appkit, name: str) -> str | None:
    """Map a display name (e.g. "Samantha") to a speech-synthesis voice id."""
    wanted = name.strip().lower()
    if not wanted:
        return None
    for ident in appkit.NSSpeechSynthesizer.availableVoices():
        attrs = appkit.NSSpeechSynthesizer.attributesForVoice_(ident) or {}
        display = str(attrs.get("VoiceName", "")).strip().lower()
        if wanted in (str(ident).lower(), display):
            return str(ident)
    return None


def main() -> int:
    import AppKit
    import Foundation
    import objc
    import soundfile as sf

    class Delegate(Foundation.NSObject):
        def init(self):
            self = objc.super(Delegate, self).init()
            self.done = False
            return self

        def speechSynthesizer_didFinishSpeaking_(self, synth, ok):
            self.done = True

    voice = None
    if "--voice" in sys.argv:
        try:
            voice = sys.argv[sys.argv.index("--voice") + 1]
        except IndexError:
            voice = None

    synth = AppKit.NSSpeechSynthesizer.alloc().init()
    if voice:
        ident = _pick_voice(AppKit, voice)
        if ident:
            synth.setVoice_(ident)
    delegate = Delegate.alloc().init()
    synth.setDelegate_(delegate)

    def synthesise(text: str) -> bytes:
        delegate.done = False
        with tempfile.TemporaryDirectory(prefix="lyra_tts_") as tmp:
            aiff = Path(tmp) / "say.aiff"
            url = Foundation.NSURL.fileURLWithPath_(str(aiff))
            if not synth.startSpeakingString_toURL_(text, url):
                return b""
            started = time.perf_counter()
            while not delegate.done:
                if time.perf_counter() - started > TIMEOUT_S:
                    return b""
                Foundation.NSRunLoop.currentRunLoop().runMode_beforeDate_(
                    Foundation.NSDefaultRunLoopMode,
                    Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.02),
                )
            try:
                data, sr = sf.read(str(aiff), dtype="int16")
            except Exception:
                return b""
            out = Path(tmp) / "out.wav"
            sf.write(str(out), data, sr, subtype="PCM_16")
            return out.read_bytes()

    # Load the engine and voice now so the first real reply is not slow.
    synthesise("Ready.")

    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        head = _read_exact(inp, 4)
        if head is None:
            break
        (size,) = struct.unpack(">I", head)
        payload = _read_exact(inp, size)
        if payload is None:
            break
        text = payload.decode("utf-8", "replace").strip()
        audio = synthesise(text) if text else b""
        out.write(struct.pack(">I", len(audio)))
        out.write(audio)
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
