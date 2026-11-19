"""Public web demo for Lyra.

Visitors talk to Lyra with their **own** browser microphone. Each visitor gets
an in-memory session:

1. answer "What is your name?" (spoken, transcribed server-side),
2. read three short prompts to build a private voiceprint for that session,
3. then talk to Lyra: the 35% owner rule applies to their own voiceprint,
   the "Hey Lyra" wake word starts a conversation, and replies come back as
   audio the browser plays.

Nothing is written to disk, the real owner voiceprint is never loaded, and
sessions expire on their own. This is a demo, not the private assistant.
"""
from __future__ import annotations

import base64
import copy
import threading
import time
import uuid

import numpy as np

from .audio.decode import decode_audio
from .bot.brain import make_brain
from .bot.engine import greeting, is_stop, split_wake
from .bot.tts import speech_wav
from .config import AttrDict
from .enrol.prompts import pick_prompts
from .enrol.validate import validate
from .enrol.voiceprint import enrol, parse_spoken_name, trim_to_speech
from .enrol.words import mark_words
from .gate.gate import SpeakerGate
from .types import Utterance

SAMPLE_COUNT = 3          # short enrolment for a demo session
SESSION_TTL = 2 * 60 * 60  # forget a visitor after two idle hours
MAX_SESSIONS = 200
MAX_AUDIO_B64 = 12_000_000  # ~9 MB of upload, generous for 30 s of speech
MAX_SECONDS = 30.0
MIN_SECONDS = 0.4
ASK_LIMIT = 15             # asks per minute, per visitor
NAME_QUESTION = "What is your name?"


class DemoSession:
    def __init__(self, sid: str, prompts: list[dict]):
        self.id = sid
        self.stage = "name"          # name -> prompts -> chat
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
        }


class DemoStore:
    """Owns every visitor session plus the shared (read-only) models."""

    def __init__(self, cfg: AttrDict):
        self.cfg = copy.deepcopy(cfg)
        # The demo never speaks through this machine's speakers and never
        # touches the owner's files: keep the model work in memory only.
        self.cfg.debug.save_utterances = False
        self._sessions: dict[str, DemoSession] = {}
        self._lock = threading.Lock()
        self._asr = None
        self._asr_lock = threading.Lock()
        self._brain = None
        self._brain_lock = threading.Lock()
        self._work = threading.Semaphore(2)   # keep the Mac responsive
        self._tts_cache: dict[str, bytes | None] = {}
        self._tts_lock = threading.Lock()

    # ------------------------------------------------------------- machinery
    def _asr_engine(self):
        with self._asr_lock:
            if self._asr is None:
                from .bot.asr import ASR

                self._asr = ASR(self.cfg)
            return self._asr

    def _brain_engine(self):
        with self._brain_lock:
            if self._brain is None:
                self._brain = make_brain(self.cfg)
            return self._brain

    def _say(self, text: str) -> str | None:
        """TTS bytes for a short line, cached (three prompts + the question)."""
        key = text.strip()
        with self._tts_lock:
            if key in self._tts_cache:
                audio = self._tts_cache[key]
            else:
                audio = speech_wav(key, self.cfg)
                if len(self._tts_cache) < 32:
                    self._tts_cache[key] = audio
        return base64.b64encode(audio).decode() if audio else None

    def _new(self) -> DemoSession:
        prompts = pick_prompts(self.cfg, SAMPLE_COUNT)
        if len(prompts) < SAMPLE_COUNT:
            raise RuntimeError("no prompts available")
        sid = uuid.uuid4().hex
        session = DemoSession(sid, prompts)
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

    # ----------------------------------------------------------------- public
    def speak(self, text: str) -> str | None:
        """Base64 WAV for a line (used for the spoken name question)."""
        return self._say(text)

    def warm(self) -> None:
        """Load the speech model and cache the name question, off the request path."""
        try:
            self._asr_engine()
            self._say(NAME_QUESTION)
        except Exception:
            pass

    # ------------------------------------------------------------------ input
    def _audio(self, b64: str, mime: str) -> np.ndarray:
        if not b64:
            raise ValueError("no audio received")
        if len(b64) > MAX_AUDIO_B64:
            raise ValueError("that recording is too long — keep it under 30 seconds")
        audio = decode_audio(b64, mime)
        if len(audio) / 16000 > MAX_SECONDS:
            raise ValueError("that recording is too long — keep it under 30 seconds")
        return audio

    # -------------------------------------------------------------- enrolment
    def enrol(self, session: DemoSession, b64: str, mime: str) -> dict:
        audio = self._audio(b64, mime)
        with session.lock:
            if session.stage == "chat":
                return {"ok": False, "reason": "you are already enrolled", **session.state()}

            if session.stage == "name":
                text = ""
                if len(audio) >= 4800:  # 0.3 s
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

    # -------------------------------------------------------------------- chat
    def ask(self, session: DemoSession, b64: str, mime: str) -> dict:
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
            decision = session.gate.decide(Utterance(audio=audio, start_s=0.0,
                                                     end_s=len(audio) / 16000))
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
                    "reply": "I heard you, but I couldn't make out the words.",
                    "audio": self._say("I heard you, but I couldn't make out the words."),
                    **session.state(),
                }

            session.last_owner = time.time()
            needs_wake = not session.conversation_open
            had_wake, command = split_wake(text)

            if is_stop(command) and (not needs_wake or had_wake):
                session.conversation_open = False
                reply = "Okay, stopping. Say \u201cHey Lyra\u201d when you need me."
                session.history += [("user", text), ("bot", reply)]
                return {
                    "ok": True, "owner": True, "transcript": text, "reply": reply,
                    "stop": True, "score": decision.score,
                    "audio": self._say(reply), **session.state(),
                }

            if needs_wake and not had_wake:
                return {
                    "ok": True, "owner": True, "transcript": text,
                    "wake_required": True, "score": decision.score,
                    "reply": "Say \u201cHey Lyra\u201d to start.",
                    **session.state(),
                }

            session.conversation_open = True
            session.history.append(("user", text))
            session.history = session.history[-20:]

            brain = self._brain_engine()
            # The shared brain personalises its answers with this name.
            if hasattr(brain, "owner_name"):
                brain.owner_name = session.name or "the owner"
            else:
                for provider in getattr(brain, "providers", []):
                    if hasattr(provider, "owner_name"):
                        provider.owner_name = session.name or "the owner"

            if needs_wake:
                hello = greeting(session.name, with_offer=not bool(command.strip()))
                reply = f"{hello} {brain.reply(command, session.history)}" if command.strip() else hello
            else:
                reply = brain.reply(text, session.history)

            session.history.append(("bot", reply))
            session.history = session.history[-20:]
            return {
                "ok": True, "owner": True, "transcript": text, "reply": reply,
                "score": decision.score, "wake": needs_wake,
                "audio": self._say(reply), **session.state(),
            }

    # ------------------------------------------------------------------ misc
    def timeout_sweep(self) -> None:
        """Close conversations that have been idle for the configured timeout."""
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
