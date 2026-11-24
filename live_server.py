#!/usr/bin/env python3
"""Standalone team version of Lyra.

A self-contained web app so teammates can talk to Lyra from their own browser,
using their own microphone and voice. It reuses the assistant's speech models
and behaviour (the "Hey Lyra" wake word, the 35% owner rule, greeting by name)
but is otherwise independent of the private, owner-only assistant:

  * nothing here reads or writes the owner's voiceprint or data/,
  * each visitor gets an in-memory session that enrols their own voice,
  * sessions are never written to disk and expire on their own.

This file does NOT import or modify the owner app's server.py / UI. It only
imports the shared model helpers from `app.*` read-only.

Run:  python team_server.py --port 8766
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import logging
import os
import re
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.bot.asr import ASR
from app.bot.brain import (
    USER_AGENT,
    OpenAICompatBrain,
    _build_messages,
    _clean,
    _system_for,
    make_brain,
)
from app.config import ROOT, AttrDict, load_config
from app.enrol.prompts import pick_prompts
from app.enrol.validate import validate
from app.enrol.voiceprint import enrol, parse_spoken_name, trim_to_speech
from app.gate.gate import SpeakerGate
from app.types import Utterance

log = logging.getLogger("team")
logging.basicConfig(level=logging.INFO)

UI_DIR = ROOT / "ui"          # the approved UI (index.html + style.css), read-only
LIVE_UI_DIR = ROOT / "live_ui"  # browser-mic app.js for the live version

# Extra Settings control for the live app. The approved HTML is served
# byte-for-byte apart from this one injected block, so the owner rule can be
# tuned here without editing the approved `ui/index.html`.
THRESHOLD_CONTROL = (
    '          <label class="field">Owner threshold (%)\n'
    '            <input type="range" id="threshold" min="20" max="90" step="1" />\n'
    '            <output id="threshold-out"></output>\n'
    '          </label>\n'
)
ANCHOR_CONTROL = (
    '          <label class="toggle"><input type="checkbox" id="timeout-enabled" /> auto-sleep when idle</label>\n'
)

SAMPLE_COUNT = 10
SESSION_TTL = 2 * 60 * 60
MAX_SESSIONS = 200
MAX_AUDIO_B64 = 12_000_000
MAX_SECONDS = 30.0
MIN_SECONDS = 0.4
ASK_LIMIT = 15
NAME_QUESTION = "What is your name?"

# ---------------------------------------------------------------- wake / stop
STOP_PHRASES = {
    "stop", "stop listening", "please stop", "stop it", "that's all", "thats all",
    "that is all", "i'm done", "im done", "goodbye", "good bye", "bye", "bye bye",
    "end conversation", "shut down", "shut up", "never mind", "cancel", "quiet",
}
_NORM_RE = re.compile(r"[^a-z' ]+")
_WAKE_RE = re.compile(
    r"^\s*(?:(?:hey|hi|okay|ok)[\s,]+)?"
    r"(?:lyra|lyrah|lira|lirra|laira|leera|lara)\b[\s,.:;!?\-]*",
    re.I,
)


def is_stop(text: str) -> bool:
    t = re.sub(r"\s+", " ", _NORM_RE.sub(" ", text.lower())).strip()
    if t in STOP_PHRASES:
        return True
    words = t.split()
    return 0 < len(words) <= 4 and words[-1] in {"stop", "goodbye", "bye"}


def split_wake(text: str) -> tuple[bool, str]:
    m = _WAKE_RE.match(text)
    if not m:
        return False, text.strip()
    return True, text[m.end():].strip(" ,.;:!?-")


def greeting(name: str, with_offer: bool = True) -> str:
    name = str(name or "").strip()
    named = name.lower() not in {"", "owner", "the owner"}
    if named:
        return f"Hey {name}, how are you? How can I help?" if with_offer else f"Hey {name}!"
    return "Hey! How can I help?" if with_offer else "Hey!"


# -------------------------------------------------------------- prompt words
CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "i'm": "i am", "im": "i am",
    "it's": "it is", "that's": "that is", "thats": "that is",
    "don't": "do not", "dont": "do not", "can't": "cannot", "cant": "cannot",
    "won't": "will not", "wont": "will not", "let's": "let us",
    "lets": "let us", "i'll": "i will", "ill": "i will", "we're": "we are",
    "you're": "you are", "they're": "they are", "there's": "there is",
    "o'clock": "oclock", "oclock": "oclock",
}
_TOKEN_RE = re.compile(r"[^a-z0-9'\s]+")


def tokens(text: str) -> list[str]:
    s = _TOKEN_RE.sub(" ", str(text or "").lower().replace("\u2019", "'"))
    out: list[str] = []
    for word in s.split():
        word = word.replace("'", "")
        mapped = CONTRACTIONS.get(word)
        out.extend((mapped or word).split())
    return out


def mark_words(prompt: str, transcript: str) -> list[dict]:
    spoken = set(tokens(transcript))
    words: list[dict] = []
    for word in str(prompt or "").split():
        need = tokens(word)
        words.append({"word": word, "ok": bool(need) and all(t in spoken for t in need)})
    return words


# ----------------------------------------------------------------- audio decode
MIME_EXT = {
    "audio/webm": ".webm", "video/webm": ".webm",
    "audio/mp4": ".m4a", "video/mp4": ".m4a", "audio/mp4a-latm": ".m4a",
    "audio/aac": ".aac", "audio/x-m4a": ".m4a", "audio/mp4a": ".m4a",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/ogg": ".ogg", "application/ogg": ".ogg",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/flac": ".flac", "audio/x-flac": ".flac",
    "audio/aiff": ".aiff", "audio/x-aiff": ".aiff", "audio/aifc": ".aiff",
}


def guess_ext(mime: str, name: str = "") -> str:
    if mime:
        m = mime.split(";")[0].strip().lower()
        if m in MIME_EXT:
            return MIME_EXT[m]
    if name and "." in name:
        return Path(name).suffix.lower()
    return ".webm"


def decode_audio(b64: str, mime: str = "", name: str = "", sr: int = 16000) -> np.ndarray:
    if not b64:
        raise ValueError("no audio received")
    import soundfile as sf

    raw = base64.b64decode(b64)
    with tempfile.TemporaryDirectory(prefix="team_audio_") as tmp:
        src = Path(tmp) / f"in{guess_ext(mime, name)}"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(raw)
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not dst.exists():
            raise ValueError(f"could not decode audio: {(r.stderr or '').strip()[:300]}")
        audio, _ = sf.read(str(dst), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=16000, target_sr=sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


def prep_enrol_audio(audio: np.ndarray) -> np.ndarray:
    """Give near-full-scale browser recordings some headroom.

    Phone/desktop browsers apply automatic gain control, so ordinary speech
    routinely peaks at full scale (0.99+). `validate` treats any such peak as
    clipping, which then blocks enrolment forever with "move further from the
    mic". Scaling those takes down to 0.7 keeps the speech intact (and is the
    same level `speaker.embed` already normalises to), so the check no longer
    fires on AGC audio. Quiet takes are left untouched, so "move closer" still
    works for genuinely low recordings.
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.size == 0:
        return a
    peak = float(np.max(np.abs(a)))
    if peak >= 0.95:
        a = a * (0.7 / peak)
    return a


# ------------------------------------------------------------------------ TTS
class WarmVoiceEngine:
    """Keeps the macOS speech engine loaded in a small worker process.

    Launching `say` per reply costs about a second of engine start-up every
    time, which made the spoken answer trail the text. The worker uses
    NSSpeechSynthesizer — the same engine and default voice as `say` — and
    answers length-prefixed requests over stdin/stdout. If it cannot start we
    fall back to the old one-shot `say` path in `speech_wav`.
    """

    def __init__(self, voice: str | None = None):
        self._voice = (voice or "").strip()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def _start(self) -> None:
        cmd = [sys.executable, str(ROOT / "live_tts.py")]
        if self._voice:
            cmd += ["--voice", self._voice]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except Exception:
                pass
        try:
            proc.kill()
        except Exception:
            pass

    def _read_exact(self, n: int, deadline: float) -> bytes | None:
        assert self._proc is not None and self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        buf = b""
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("speech worker timed out")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError("speech worker timed out")
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _request(self, text: str) -> bytes | None:
        if self._proc is None or self._proc.poll() is not None:
            self._start()
        assert self._proc is not None and self._proc.stdin is not None
        deadline = time.monotonic() + 45.0
        raw = text.encode("utf-8")
        self._proc.stdin.write(struct.pack(">I", len(raw)) + raw)
        self._proc.stdin.flush()
        head = self._read_exact(4, deadline)
        if head is None:
            raise OSError("speech worker exited")
        (size,) = struct.unpack(">I", head)
        audio = self._read_exact(size, deadline) if size else b""
        if audio is None:
            raise OSError("speech worker exited")
        return audio or None

    def speak(self, text: str) -> bytes | None:
        with self._lock:
            try:
                return self._request(text)
            except Exception:
                # One respawn in case the worker died (e.g. after a system sleep).
                self._stop()
                return self._request(text)


_WARM_VOICE: WarmVoiceEngine | None = None
_WARM_VOICE_LOCK = threading.Lock()


def _warm_voice(voice: str | None) -> WarmVoiceEngine:
    global _WARM_VOICE
    with _WARM_VOICE_LOCK:
        if _WARM_VOICE is None:
            _WARM_VOICE = WarmVoiceEngine(voice)
        return _WARM_VOICE


def speech_wav(text: str, cfg: AttrDict, timeout: float = 30.0) -> bytes | None:
    text = (text or "").strip()[:800]
    if not text:
        return None
    voice = cfg.bot.get("tts_voice")
    with tempfile.TemporaryDirectory(prefix="lyra_tts_") as tmp:
        wav = Path(tmp) / "out.wav"
        # macOS: the system `say` voice, kept warm between replies.
        if sys.platform == "darwin":
            try:
                audio = _warm_voice(voice).speak(text)
                if audio:
                    return audio
            except Exception:
                log.warning("warm speech worker failed; using one-shot say", exc_info=True)
            aiff = Path(tmp) / "say.aiff"
            cmd = ["say"]
            if voice:
                cmd += ["-v", str(voice)]
            cmd += ["-o", str(aiff), text]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                     "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                    check=True, capture_output=True, timeout=timeout,
                )
                return wav.read_bytes()
            except (subprocess.SubprocessError, OSError):
                return None
        # cloud (Linux, etc.): free neural TTS, no API key required.
        try:
            import asyncio

            import edge_tts

            mp3 = Path(tmp) / "tts.mp3"

            async def _gen():
                await edge_tts.Communicate(text, "en-US-AriaNeural").save(str(mp3))

            asyncio.run(_gen())
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                check=True, capture_output=True, timeout=timeout,
            )
            return wav.read_bytes()
        except Exception:
            return None


def wav_to_mp3(wav: bytes) -> bytes | None:
    """Compress a WAV clip to mono MP3 so the reply travels faster."""
    if not wav:
        return None
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0",
             "-ac", "1", "-ar", "24000", "-b:a", "64k", "-f", "mp3", "pipe:1"],
            input=wav, capture_output=True, timeout=30,
        )
        return r.stdout if r.returncode == 0 and r.stdout else None
    except (subprocess.SubprocessError, OSError):
        return None


# --------------------------------------------------------------------- session
class LiveASR(ASR):
    """ASR that prefers the Apple Metal GPU, falling back to CPU.

    Kept in this file so the owner app's `app/bot/asr.py` stays exactly as it
    was approved — this only speeds up the live team server.
    """

    def _get(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        last_err = None
        for device in ("mps", "cpu"):
            try:
                self._model = WhisperModel(self.model_name, device=device, compute_type="int8")
                return self._model
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(f"could not load whisper model: {last_err}")


class DemoSession:
    def __init__(self, sid: str, prompts: list[dict], cfg: AttrDict):
        self.id = sid
        self.cfg = cfg  # shared with the store, so settings are global
        self.stage = "name"
        self.name = ""
        self.prompts = prompts
        self.prompt_index = 0
        self.takes: list[np.ndarray] = []
        self.voiceprint: np.ndarray | None = None
        self.gate: SpeakerGate | None = None
        self.conversation_open = False
        self.history: list[tuple[str, str]] = []
        self.lock = threading.Lock()
        self.created = time.time()
        self.seen = time.time()
        self.last_owner = 0.0
        self.asks: list[float] = []

    @property
    def current_prompt(self) -> str:
        if self.stage == "prompts" and self.prompt_index < len(self.prompts):
            return self.prompts[self.prompt_index]["text"]
        return ""

    def state(self) -> dict:
        return {
            "sid": self.id,
            "stage": self.stage,
            "name": self.name,
            "prompt": self.current_prompt if self.stage == "prompts" else (
                NAME_QUESTION if self.stage == "name" else ""
            ),
            "index": self.prompt_index,
            "total": len(self.prompts),
            "conversation_open": self.conversation_open,
            "conversation": [
                {"role": role, "text": text} for role, text in self.history[-20:]
            ],
            # Same settings the approved app exposes (sleep timer, owner rule).
            "settings": {
                "accept_threshold": float(self.cfg.gate.accept_threshold),
                "reject_threshold": float(self.cfg.gate.reject_threshold),
                "conversation_window_sec": float(self.cfg.gate.conversation_window_sec),
                "timeout_minutes": float(self.cfg.timeout.minutes),
                "timeout_enabled": bool(self.cfg.timeout.enabled),
                "save_utterances": False,  # live sessions are never written to disk
            },
        }


class LiveHostedBrain(OpenAICompatBrain):
    """OpenAI-compatible backend with headroom for reasoning models.

    The approved brain caps replies at 120 tokens, which is plenty for classic
    models (gpt-4o-mini, Claude, the local Qwen) but too small for Groq's free
    reasoning models (GPT-OSS / Qwen): their internal reasoning counts against
    the same budget, so the answer gets truncated or comes back empty. This
    live-only subclass raises the cap; `_clean()` still trims the reply to at
    most four short spoken sentences, so the voice never runs long.
    """

    MAX_TOKENS = 512

    def __init__(self, url: str, model: str, api_key: str = "",
                 owner_name: str = "the owner", timeout: float = 20.0):
        # Unlike the parent, `url` here is already the full /chat/completions
        # endpoint (copied from the provider make_brain built), so store it as-is.
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.owner_name = owner_name
        self.timeout = timeout

    def reply(self, transcript: str, history: list, context: str = "") -> str:
        system = _system_for(self.owner_name)
        if context:
            system += "\n\n" + context
        payload = {
            "model": self.model,
            "messages": _build_messages(transcript, history, system),
            "max_tokens": self.MAX_TOKENS,
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = r.read().decode("utf-8", "replace")
        try:
            data = json.loads(body)
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            # some keyless endpoints answer with plain text
            text = body
        return _clean(text)


def _upgrade_hosted_backends(brain) -> None:
    """Swap Groq-bound backends for LiveHostedBrain (more token headroom).

    Only Groq is upgraded: the keyless Pollinations fallback and a real OpenAI
    key keep the approved 120-token behaviour. Everything else — provider order,
    the local model, the built-in rules — is left exactly as make_brain built it.
    """
    providers = getattr(brain, "providers", None)
    if not providers:
        return
    for i, provider in enumerate(providers):
        if isinstance(provider, OpenAICompatBrain) and "groq" in getattr(provider, "url", ""):
            providers[i] = LiveHostedBrain(
                provider.url, provider.model, provider.api_key,
                provider.owner_name, provider.timeout,
            )


class DemoStore:
    def __init__(self, cfg: AttrDict):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.debug.save_utterances = False
        # Live-only tuning; this copy is never written back to config.yaml.
        self.cfg.bot.asr_model = "base"  # much faster than "small", still accurate
        # Browser push-to-record clips start/stop right at the speech, so short
        # prompts ("Tell me a joke.", "Good morning.") are under the 1.0 s / 0.8 s
        # floors the desktop capture was tuned for. Without this, enrolment gets
        # stuck on the first short line and rejects every retry.
        self.cfg.enrol.min_duration_sec = 0.5
        self.cfg.enrol.min_speech_sec = 0.4
        # The shared demo must stay listening. Background voices and noise are
        # ignored one utterance at a time (still below the 35% owner rule), but
        # they must never pause the assistant, so disable the lockout here only.
        self.cfg.gate.lockout_after = 0
        # Browser clips are much shorter than desktop captures, so a short
        # owner phrase must still be scored and answered rather than silently
        # "ignored". MIN_SECONDS (0.4 s) already guards against empty clips.
        self.cfg.gate.min_utterance_sec = 0.2
        self._sessions: dict[str, DemoSession] = {}
        self._lock = threading.Lock()
        self._asr = None
        self._asr_lock = threading.Lock()
        self._brain = None
        self._brain_lock = threading.Lock()
        self._work = threading.Semaphore(2)
        self._tts_cache: dict[str, bytes | None] = {}
        self._tts_mp3_cache: dict[str, bytes | None] = {}
        self._tts_lock = threading.Lock()

    def _asr_engine(self):
        with self._asr_lock:
            if self._asr is None:
                self._asr = LiveASR(self.cfg)
            return self._asr

    def _brain_engine(self):
        with self._brain_lock:
            if self._brain is None:
                self._brain = make_brain(self.cfg)
                _upgrade_hosted_backends(self._brain)
            return self._brain

    def _say(self, text: str) -> str | None:
        key = text.strip()
        with self._tts_lock:
            if key in self._tts_cache:
                audio = self._tts_cache[key]
            else:
                audio = speech_wav(key, self.cfg)
                if len(self._tts_cache) < 32:
                    self._tts_cache[key] = audio
        return base64.b64encode(audio).decode() if audio else None

    def speak_mp3(self, text: str) -> bytes | None:
        """Reply audio as MP3 (~7x smaller than WAV, so it arrives sooner)."""
        key = text.strip()
        with self._tts_lock:
            if key in self._tts_mp3_cache:
                return self._tts_mp3_cache[key]
        wav_b64 = self._say(key)
        if not wav_b64:
            return None
        mp3 = wav_to_mp3(base64.b64decode(wav_b64))
        with self._tts_lock:
            if len(self._tts_mp3_cache) < 64:
                self._tts_mp3_cache[key] = mp3
        return mp3

    def _new(self) -> DemoSession:
        prompts = pick_prompts(self.cfg, SAMPLE_COUNT)
        if len(prompts) < SAMPLE_COUNT:
            raise RuntimeError("no prompts available")
        sid = uuid.uuid4().hex
        session = DemoSession(sid, prompts, self.cfg)
        with self._lock:
            self._prune_locked()
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions.values(), key=lambda s: s.seen)
                self._sessions.pop(oldest.id, None)
            self._sessions[sid] = session
        return session

    def _prune_locked(self) -> None:
        cutoff = time.time() - SESSION_TTL
        for sid in [s.id for s in self._sessions.values() if s.seen < cutoff]:
            self._sessions.pop(sid, None)

    def get(self, sid: str | None) -> DemoSession:
        if sid:
            with self._lock:
                session = self._sessions.get(sid)
            if session is not None:
                session.seen = time.time()
                return session
        return self._new()

    def reset(self, sid: str | None) -> DemoSession:
        if sid:
            with self._lock:
                self._sessions.pop(sid, None)
        return self._new()

    def update_settings(self, payload: dict) -> None:
        """Apply the same settings the approved app offers, plus the live-only
        owner-threshold slider (kept between 20% and 90% so the voice check
        can be tuned but never dropped or made impossibly strict)."""
        if "accept_threshold" in payload:
            try:
                t = float(payload["accept_threshold"])
            except (TypeError, ValueError):
                t = float(self.cfg.gate.accept_threshold)
            t = max(0.20, min(0.90, t))
            self.cfg.gate.accept_threshold = t
            with self._lock:
                sessions = list(self._sessions.values())
            for session in sessions:
                if session.gate is not None:
                    session.gate.accept = t
        if "timeout_minutes" in payload:
            try:
                minutes = float(payload["timeout_minutes"])
            except (TypeError, ValueError):
                minutes = float(self.cfg.timeout.minutes)
            self.cfg.timeout.minutes = max(1.0, min(60.0, minutes))
        if "timeout_enabled" in payload:
            self.cfg.timeout.enabled = bool(payload["timeout_enabled"])
        # Live sessions are never written to disk.
        self.cfg.debug.save_utterances = False

    def speak(self, text: str) -> str | None:
        return self._say(text)

    def warm(self) -> None:
        try:
            self._asr_engine()
            self._brain_engine()  # pre-load the answer model so the first reply is fast
            # pre-load the Silero VAD used by trim_to_speech
            try:
                trim_to_speech(np.zeros(16000, dtype=np.float32), 16000)
            except Exception:
                pass
            self._say(NAME_QUESTION)  # cache the first question in the system voice
        except Exception:
            pass

    def _audio(self, b64: str, mime: str) -> np.ndarray:
        if not b64:
            raise ValueError("no audio received")
        if len(b64) > MAX_AUDIO_B64:
            raise ValueError("that recording is too long — keep it under 30 seconds")
        audio = decode_audio(b64, mime)
        if len(audio) / 16000 > MAX_SECONDS:
            raise ValueError("that recording is too long — keep it under 30 seconds")
        return audio

    def enrol(self, session: DemoSession, b64: str, mime: str) -> dict:
        audio = prep_enrol_audio(self._audio(b64, mime))
        with session.lock:
            if session.stage == "chat":
                return {"ok": False, "reason": "you are already enrolled", **session.state()}

            if session.stage == "name":
                text = ""
                if len(audio) >= 4800:
                    try:
                        text, _ = self._asr_engine().transcribe(audio)
                    except Exception:
                        pass
                name = parse_spoken_name(text)
                if not name:
                    return {
                        "ok": False, "stage": "name", "transcript": text,
                        "reason": "I didn't catch a name — say something like "
                                  "\u201cMy name is Adnan\u201d.",
                        **session.state(),
                    }
                session.name = name
                session.stage = "prompts"
                session.prompt_index = 0
                prompt = session.current_prompt
                return {
                    "ok": True, "stage": "prompts", "name": name,
                    "transcript": text, "prompt": prompt,
                    "prompt_audio": self._say(prompt),
                    **session.state(),
                }

            ok, reason = validate(audio, 16000, self.cfg)
            prompt = session.current_prompt
            if not ok:
                return {"ok": False, "reason": reason, "prompt": prompt, **session.state()}

            transcript, _ = self._asr_engine().transcribe(audio)
            words = mark_words(prompt, transcript)
            session.takes.append(audio)
            session.prompt_index += 1

            if session.prompt_index >= len(session.prompts):
                vec, meta = enrol(session.takes, self.cfg.speaker.model, self.cfg)
                session.voiceprint = vec
                session.gate = SpeakerGate(self.cfg, vec)
                session.stage = "chat"
                session.takes = []
                session.last_owner = time.time()
                return {
                    "ok": True, "done": True, "stage": "chat", "name": session.name,
                    "transcript": transcript, "words": words,
                    "prompt_audio": self._say(
                        f"Nice to meet you, {session.name}. Ask me anything."
                    ),
                    **session.state(),
                }

            prompt = session.current_prompt
            return {
                "ok": True, "stage": "prompts", "transcript": transcript,
                "words": words, "prompt": prompt, "prompt_audio": self._say(prompt),
                **session.state(),
            }

    def ask(self, session: DemoSession, b64: str, mime: str, miccheck: bool = False) -> dict:
        if session.stage != "chat" or session.gate is None:
            return {"ok": False, "reason": "finish enrolment first", **session.state()}

        now = time.time()
        session.asks = [t for t in session.asks if now - t < 60]
        if len(session.asks) >= ASK_LIMIT:
            return {
                "ok": False, "reason": "that's a lot of questions — take a short break",
                **session.state(),
            }
        session.asks.append(now)

        audio = trim_to_speech(self._audio(b64, mime), 16000)
        if len(audio) / 16000 < MIN_SECONDS:
            return {
                "ok": False, "owner": False,
                "reason": "too short — hold the button and speak a full sentence",
                **session.state(),
            }

        with self._work:
            decision = session.gate.decide(
                Utterance(audio=audio, start_s=0.0, end_s=len(audio) / 16000)
            )
            # Mic check: always transcribe (it is the enrolled user checking
            # themselves) so the UI can show what was heard, regardless of score.
            if miccheck:
                try:
                    text, _ = self._asr_engine().transcribe(audio)
                except Exception:
                    text = ""
                return {
                    "ok": True, "owner": decision.owner, "score": decision.score,
                    "decision": decision.decision, "transcript": text, "miccheck": True,
                    **session.state(),
                }

            if not decision.owner:
                return {
                    "ok": True, "owner": False, "score": decision.score,
                    "decision": decision.decision, "reason": decision.reason,
                    **session.state(),
                }

            text, _ = self._asr_engine().transcribe(audio)
            if not text.strip():
                return {
                    "ok": True, "owner": True, "transcript": "",
                    # No server-side TTS here: the browser speaks the reply
                    # instantly (Web Speech API), so the wait after you stop
                    # talking stays close to one second.
                    "reply": "I heard you, but I couldn't make out the words.",
                    **session.state(),
                }

            session.last_owner = time.time()
            was_closed = not session.conversation_open
            had_wake, command = split_wake(text)

            if is_stop(command) and (session.conversation_open or had_wake):
                session.conversation_open = False
                reply = "Okay, stopping. Just speak whenever you need me again."
                session.history += [("user", text), ("bot", reply)]
                return {
                    "ok": True, "owner": True, "transcript": text, "reply": reply,
                    "stop": True, "score": decision.score, **session.state(),
                }

            # The owner's voice is enough to get an answer: open the conversation
            # and reply straight away. "Hey Lyra" still works (and adds a greeting
            # by name) but is no longer required before Lyra will talk back.
            session.conversation_open = True
            session.history.append(("user", text))
            session.history = session.history[-20:]

            brain = self._brain_engine()
            if hasattr(brain, "owner_name"):
                brain.owner_name = session.name or "the owner"
            else:
                for provider in getattr(brain, "providers", []):
                    if hasattr(provider, "owner_name"):
                        provider.owner_name = session.name or "the owner"

            if was_closed and had_wake:
                hello = greeting(session.name, with_offer=not bool(command.strip()))
                reply = f"{hello} {brain.reply(command, session.history)}" if command.strip() else hello
            else:
                reply = brain.reply(text, session.history)

            session.history.append(("bot", reply))
            session.history = session.history[-20:]
            return {
                "ok": True, "owner": True, "transcript": text, "reply": reply,
                "score": decision.score, "wake": was_closed, **session.state(),
            }

    def timeout_sweep(self) -> None:
        if not bool(self.cfg.timeout.enabled):
            return  # auto-sleep turned off in Settings
        minutes = float(self.cfg.timeout.minutes)
        cutoff = time.time() - minutes * 60
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            if session.conversation_open and session.last_owner < cutoff:
                session.conversation_open = False

    def ttl_sweep(self) -> None:
        with self._lock:
            self._prune_locked()


# ----------------------------------------------------------------------- server
cfg = load_config()
# Cloud hosts inject configuration as environment variables, not a `.env` file.
# load_config() only reads the `.env` file, so merge real process env vars here
# (OPENAI_* doubles as the Groq/any OpenAI-compatible backend — see README).
for _key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
             "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
    _val = os.environ.get(_key)
    if _val:
        cfg.env[_key] = _val.strip()
STORE = DemoStore(cfg)


def _sweep() -> None:
    while True:
        time.sleep(30)
        try:
            STORE.timeout_sweep()
            STORE.ttl_sweep()
        except Exception:
            log.exception("team sweep failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=STORE.warm, daemon=True).start()
    threading.Thread(target=_sweep, daemon=True).start()
    yield


app = FastAPI(title="Lyra — live", lifespan=lifespan)


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


def _err(e: Exception, code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "reason": str(e)}, status_code=code)


@app.get("/api/demo/session")
def api_session(sid: str | None = None):
    return STORE.get(sid).state()


@app.post("/api/demo/start")
def api_start(payload: dict):
    session = STORE.reset(payload.get("sid"))
    question = NAME_QUESTION
    return {"ok": True, "question": question, "audio": STORE.speak(question), **session.state()}


@app.post("/api/demo/settings")
def api_settings(payload: dict):
    """Update the approved settings (sleep timer, auto-sleep) for the live app."""
    try:
        STORE.update_settings(payload)
        session = STORE.get(payload.get("sid"))
        return {"ok": True, **session.state()}
    except Exception as e:
        log.exception("settings update failed")
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


@app.post("/api/demo/tts")
def api_tts(payload: dict):
    """Synthesise a reply with the system voice (the original Lyra voice).

    The conversation reply itself is returned as text first (fast), then the
    browser asks for the audio here. Audio is sent as MP3 so it crosses the
    tunnel quickly; the voice is still the same macOS `say` voice.
    """
    text = str(payload.get("text") or "").strip()[:800]
    if not text:
        return JSONResponse({"ok": False, "reason": "no text"}, status_code=400)
    try:
        audio = STORE.speak_mp3(text)
        if audio:
            return Response(content=audio, media_type="audio/mpeg")
        wav_b64 = STORE.speak(text)
        if wav_b64:
            return Response(content=base64.b64decode(wav_b64), media_type="audio/wav")
        return JSONResponse({"ok": False, "reason": "speech unavailable"}, status_code=503)
    except Exception as e:
        log.exception("tts failed")
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


@app.post("/api/demo/enrol")
def api_enrol(payload: dict):
    session = STORE.get(payload.get("sid"))
    try:
        result = STORE.enrol(session, payload.get("audio", ""), payload.get("mime", ""))
        log.info(
            "enrol sid=%s stage=%s ok=%s index=%s reason=%s",
            session.id[:8], result.get("stage"), result.get("ok"),
            result.get("index"), result.get("reason"),
        )
        return result
    except ValueError as e:
        log.info("enrol sid=%s rejected: %s", session.id[:8], e)
        return {"ok": False, "reason": str(e), **session.state()}
    except Exception as e:
        log.exception("team enrolment failed")
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


@app.post("/api/demo/ask")
def api_ask(payload: dict):
    session = STORE.get(payload.get("sid"))
    try:
        return STORE.ask(
            session, payload.get("audio", ""), payload.get("mime", ""),
            bool(payload.get("miccheck", False)),
        )
    except ValueError as e:
        return {"ok": False, "reason": str(e), **session.state()}
    except Exception as e:
        log.exception("team question failed")
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


@app.get("/")
def index():
    html = (UI_DIR / "index.html").read_text()
    if THRESHOLD_CONTROL not in html:
        if ANCHOR_CONTROL in html:
            html = html.replace(ANCHOR_CONTROL, THRESHOLD_CONTROL + ANCHOR_CONTROL)
        else:
            html = html.replace(
                '<div class="settings-grid">',
                '<div class="settings-grid">\n' + THRESHOLD_CONTROL,
            )
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/style.css")
def style():
    return FileResponse(UI_DIR / "style.css", headers={"Cache-Control": "no-store"})


@app.get("/app.js")
def app_js():
    return FileResponse(LIVE_UI_DIR / "app.js", headers={"Cache-Control": "no-store"})


def main() -> None:
    import os

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "8766"))
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
