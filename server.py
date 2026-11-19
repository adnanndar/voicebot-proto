#!/usr/bin/env python3
"""Voice assistant web UI: FastAPI + WebSocket.

The server owns the live pipeline (microphone -> voice activity -> owner check ->
English speech-to-text -> reply -> speech output) and streams state to the
browser, which also drives enrolment, the mic check and settings.

Run:
    python server.py            # http://127.0.0.1:8765
    python server.py --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import re
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.audio.decode import MIME_EXT, decode_to_wav, guess_ext
from app.audio.pipeline import AudioPipeline, iter_utterances
from app.bot.engine import Engine
from app.config import DATA_DIR, ROOT, load_config
from app.demo import DemoStore
from app.enrol.prompts import pick_prompts
from app.enrol.validate import validate
from app.enrol.voiceprint import (
    Voiceprint, add_samples, apply_thresholds, enrol, owner_name, parse_spoken_name,
)
from app.enrol import words
from app.gate.gate import SpeakerGate

log = logging.getLogger("voicebot")
UI_DIR = ROOT / "ui"
TMP_DIR = ROOT / "tmp"


def list_input_devices() -> list[dict]:
    """The microphones available on this machine (built-in, AirPods, headsets...)."""
    import sounddevice as sd

    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
    except Exception:
        return []
    out = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            out.append({
                "index": i,
                "name": d["name"],
                "channels": d.get("max_input_channels", 1),
                "is_default": i == default_in,
            })
    return out


class Runtime:
    """Owns the live pipeline, enrolment state and the data the UI renders."""

    def __init__(self) -> None:
        self.cfg = load_config()
        DATA_DIR.mkdir(exist_ok=True)
        self.vp = Voiceprint(DATA_DIR, self.cfg.speaker.model, encrypt=self.cfg.privacy.encrypt)
        self._vp_cache: dict | None = None
        _, meta = self._vp_meta()
        apply_thresholds(self.cfg, meta)

        # one lock for everything that opens the microphone directly
        self._audio_lock = threading.Lock()
        # one lock for start/stop of the live pipeline
        self._live_lock = threading.Lock()

        self.engine: Engine | None = None
        self.pipeline: AudioPipeline | None = None
        self._engine_thread: threading.Thread | None = None
        self.listening = False
        self.error = ""

        self.status = "idle"
        self.level = 0.0
        self.speech = False
        self.conversation: list[dict] = []
        self.last_decision: dict | None = None
        self.ignored_until = 0.0

        self.enrol = {
            "active": False, "mode": "enrol", "stage": "prompts", "owner_name": "",
            "prompts": [], "index": 0, "total": 0, "recordings": [], "last_reason": "",
        }
        self._record_stream = None
        self._record_chunks: list = []
        self._asr = None
        self._asr_lock = threading.Lock()
        self._tts = None
        self._devices_cache: list = []
        self._devices_time = 0.0

    # ----------------------------------------------------------- voiceprint
    def _vp_meta(self) -> tuple[bool, dict | None]:
        """Cached voiceprint info; re-read only after enrolment / delete / restore."""
        if self._vp_cache is None:
            vec, meta = self.vp.load()
            self._vp_cache = {
                "enrolled": vec is not None,
                "has_original": self.vp.has_original(),
                "meta": meta,
            }
        return self._vp_cache["enrolled"], self._vp_cache["meta"]

    def _invalidate_vp(self) -> None:
        self._vp_cache = None

    def _speak(self, text: str) -> None:
        """Speak a short line (used for the spoken name question)."""
        try:
            if self._tts is None:
                from app.bot.tts import make_tts

                self._tts = make_tts(self.cfg)
            self._tts.speak(text)
        except Exception:
            log.exception("could not speak enrolment prompt")

    def _ask_name(self) -> None:
        self.enrol["stage"] = "name"
        self.enrol["owner_name"] = ""
        self._speak("What is your name?")

    def _asr_engine(self):
        with self._asr_lock:
            if self._asr is None:
                from app.bot.asr import ASR
                self._asr = ASR(self.cfg)
            return self._asr

    def _devices(self) -> list[dict]:
        if time.time() - self._devices_time > 5.0:
            self._devices_cache = list_input_devices()
            self._devices_time = time.time()
        return self._devices_cache

    # -------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        enrolled, meta = self._vp_meta()
        status = self.status
        if time.time() < self.ignored_until:
            status = "ignored"
        ep = self.enrol["prompts"][self.enrol["index"]] \
            if self.enrol["active"] and self.enrol["index"] < len(self.enrol["prompts"]) else None
        return {
            "status": status,
            "listening": self.listening,
            "error": self.error,
            "owner_name": owner_name(meta, self.cfg.bot.get("owner_name", "the owner")),
            "conversation_open": bool(self.engine is not None and self.engine.conversation_open),
            "voiceprint": {
                "enrolled": enrolled,
                "has_original": bool(self._vp_cache and self._vp_cache["has_original"]),
                "meta": meta,
            },
            "privacy": {"encrypt": bool(self.cfg.privacy.encrypt),
                        "require_consent": bool(self.cfg.privacy.require_consent)},
            "settings": {
                "accept_threshold": self.cfg.gate.accept_threshold,
                "reject_threshold": self.cfg.gate.reject_threshold,
                "conversation_window_sec": self.cfg.gate.conversation_window_sec,
                "timeout_minutes": self.cfg.timeout.minutes,
                "timeout_enabled": bool(self.cfg.timeout.enabled),
                "save_utterances": bool(self.cfg.debug.save_utterances),
            },
            "live": {"level": self.level, "speech": self.speech},
            "capture": self._capture(),
            "close_after_sec": float(self.cfg.vad.end_ms) / 1000.0,
            "device": {"selected": self.cfg.audio.device, "list": self._devices()},
            "last_decision": self.last_decision,
            "conversation": list(self.conversation[-80:]),
            "enrol": {
                "active": self.enrol["active"],
                "mode": self.enrol["mode"],
                "stage": self.enrol.get("stage", "prompts"),
                "owner_name": self.enrol.get("owner_name", ""),
                "index": self.enrol["index"],
                "total": self.enrol["total"],
                "recording": self._record_stream is not None,
                "prompt": ep["text"] if ep else "",
                "tone": ep.get("tone") if ep else None,
            },
        }

    def _capture(self) -> dict:
        if self.pipeline is None:
            return {"active": False, "seconds": 0.0, "silence_seconds": 0.0}
        return self.pipeline.capture_status()

    # ------------------------------------------------------------- callbacks
    def _on_state(self, s: str) -> None:
        self.status = s

    def _on_event(self, ev: dict) -> None:
        kind = ev.get("kind")
        if kind == "decision":
            owner = bool(ev.get("owner"))
            self.last_decision = {
                "score": ev.get("score"),
                "decision": ev.get("decision"),
                "reason": ev.get("reason"),
                "duration_s": ev.get("duration_s"),
                "transcript": "",
            }
            if not owner:
                self.ignored_until = time.time() + 2.5
                self.conversation.append({
                    "role": "ignored", "score": ev.get("score"),
                    "decision": ev.get("decision"), "reason": ev.get("reason"),
                    "duration_s": ev.get("duration_s"), "text": "",
                })
        elif kind == "transcript":
            self.conversation.append({
                "role": "user", "text": ev.get("transcript", ""),
                "score": (self.last_decision or {}).get("score"),
            })
            if self.last_decision:
                self.last_decision["transcript"] = ev.get("transcript", "")
        elif kind == "reply":
            self.conversation.append({"role": "bot", "text": ev.get("reply", "")})
        elif kind == "stop_listening":
            # the owner said "stop": close the mic from a helper thread so the
            # engine thread that emitted this event never joins itself
            threading.Thread(target=self.stop_listening, daemon=True).start()
        if len(self.conversation) > 200:  # keep memory bounded on long sessions
            self.conversation = self.conversation[-100:]

    def _on_level(self, rms: float) -> None:
        self.level = rms

    def _on_speech(self, flag: bool) -> None:
        self.speech = flag

    # ------------------------------------------------------------- listening
    def start_listening(self) -> None:
        with self._live_lock:
            if self.listening:
                return
            if self.enrol["active"]:
                raise RuntimeError("finish or cancel enrolment before listening")
            if self._record_stream is not None:
                raise RuntimeError("a recording is in progress")
            self.error = ""
            self.conversation = []
            self.last_decision = None
            engine = Engine(self.cfg)
            engine.on_state = self._on_state
            engine.on_event = self._on_event
            pipeline = AudioPipeline(
                self.cfg, engine.on_utterance,
                on_level=self._on_level, on_speech=self._on_speech,
            )
            engine.pipeline = pipeline
            try:
                pipeline.start()
            except Exception as e:  # e.g. the selected microphone is unplugged
                self.error = f"Could not open the microphone: {e}"
                log.exception("microphone start failed")
                raise RuntimeError(self.error) from e
            self.engine, self.pipeline = engine, pipeline
            self.listening = True
            self._engine_thread = threading.Thread(target=self._run_engine, args=(engine,),
                                                   daemon=True, name="engine")
            self._engine_thread.start()

    def _run_engine(self, engine: Engine) -> None:
        try:
            engine.run()
        except Exception as e:
            self.error = f"Assistant stopped unexpectedly: {e}"
            log.exception("engine crashed")
            self.listening = False
            self.status = "idle"

    def stop_listening(self) -> None:
        with self._live_lock:
            if not self.listening and self.pipeline is None:
                return
            self.listening = False
            if self.engine:
                self.engine.stop()
            if self.pipeline:
                self.pipeline.stop()
            self.engine = None
            self.pipeline = None
            self._engine_thread = None
            self.status = "idle"
            self.speech = False

    # -------------------------------------------------------------- settings
    def update_settings(self, payload: dict) -> None:
        g = self.cfg.gate
        # The owner rule is fixed at 35%: the UI no longer exposes it, and a
        # crafted request must not be able to weaken it either.
        apply_thresholds(self.cfg, None)
        if "conversation_window_sec" in payload:
            g.conversation_window_sec = float(payload["conversation_window_sec"])
        if "timeout_minutes" in payload:
            self.cfg.timeout.minutes = float(payload["timeout_minutes"])
        if "timeout_enabled" in payload:
            self.cfg.timeout.enabled = bool(payload["timeout_enabled"])
        if "save_utterances" in payload:
            self.cfg.debug.save_utterances = bool(payload["save_utterances"])
        if "device" in payload:
            d = payload["device"]
            new_device = None if d in (None, "", "default") else int(d)
            if new_device != self.cfg.audio.device:
                self.cfg.audio.device = new_device
                if self.listening:  # restart the live pipeline on the new microphone
                    self.stop_listening()
                    self.start_listening()
        if self.engine is not None:
            self.engine.gate.accept = g.accept_threshold
            self.engine.gate.reject = g.reject_threshold

    # ------------------------------------------------------------ enrolment
    def enrol_start(self, mode: str, consent: bool = False) -> dict:
        if self.cfg.privacy.require_consent and not consent:
            raise RuntimeError("consent is required before enrolment")
        self.stop_listening()

        prompts = pick_prompts(self.cfg, 10)
        if not prompts:
            raise RuntimeError("no enrolment prompts available")

        enrolled, existing = self._vp_meta()
        owner = owner_name(existing, "") if enrolled else ""
        if mode == "add" and enrolled and owner:
            # adding samples keeps the known name: the name question is only
            # the first compulsory step of a fresh enrolment
            stage = "prompts"
        else:
            mode = "enrol" if not enrolled else mode
            stage = "name"
            owner = ""

        self.enrol = {
            "active": True, "mode": mode, "stage": stage, "owner_name": owner,
            "prompts": prompts, "index": 0, "total": len(prompts),
            "recordings": [], "last_reason": "",
        }
        if stage == "name":
            self._ask_name()
        return self.snapshot()

    def enrol_submit(self, b64: str, mime: str, name: str) -> dict:
        """Browser-uploaded audio."""
        import soundfile as sf

        self._require_enrol()
        TMP_DIR.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP_DIR, prefix="enrol_") as tmp:
            wav = decode_to_wav(b64, mime, name, Path(tmp))
            audio, _sr = sf.read(str(wav), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
        return self._enrol_from_audio(audio)

    def _require_enrol(self) -> None:
        if not self.enrol["active"]:
            raise RuntimeError("enrolment is not active")

    def enrol_record_start(self) -> None:
        """Start recording the current line on the selected microphone."""
        import sounddevice as sd

        self._require_enrol()
        with self._audio_lock:
            if self._record_stream is not None:
                raise RuntimeError("already recording")
            if self.listening:
                raise RuntimeError("stop listening before recording")
            self._record_chunks = []
            stream = sd.InputStream(
                samplerate=self.cfg.audio.sample_rate, channels=1, dtype="float32",
                blocksize=self.cfg.audio.blocksize, device=self.cfg.audio.device,
                callback=self._record_cb,
            )
            stream.start()
            self._record_stream = stream

    def _record_cb(self, indata, frames, time_info, status) -> None:
        self._record_chunks.append(np.ascontiguousarray(indata[:, 0].copy(), dtype=np.float32))
        if indata.size:
            self.level = float(np.sqrt(np.mean(indata * indata)))

    def _close_record_stream(self) -> np.ndarray:
        """Stop and close the recording stream; return the captured audio."""
        stream, self._record_stream = self._record_stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        chunks, self._record_chunks = self._record_chunks, []
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def enrol_record_stop(self) -> dict:
        with self._audio_lock:
            if self._record_stream is None:
                raise RuntimeError("not recording")
            audio = self._close_record_stream()
        return self._enrol_from_audio(audio)

    def _enrol_from_audio(self, audio: np.ndarray) -> dict:
        if self.enrol.get("stage") == "name":
            return self._capture_name(audio)

        ok, reason = validate(audio, self.cfg.audio.sample_rate, self.cfg)
        rms = float(np.sqrt(np.mean(audio * audio))) if len(audio) else 0.0
        level_db = 20 * np.log10(rms + 1e-12)

        words, transcript = [], ""
        prompt_text = ""
        if ok:
            prompt_text = self.enrol["prompts"][self.enrol["index"]]["text"]
            words, transcript = self._word_match(audio)
            self.enrol["recordings"].append(audio)
            self.enrol["index"] += 1
            self.enrol["last_reason"] = ""
            if self.enrol["index"] >= self.enrol["total"]:
                return self._enrol_finish()
        else:
            self.enrol["last_reason"] = reason

        return {
            "ok": ok, "reason": reason, "stage": "prompts",
            "level_db": round(float(level_db), 1),
            "index": self.enrol["index"], "total": self.enrol["total"],
            "progress": self.enrol["index"] / max(1, self.enrol["total"]),
            "done": False, "transcript": transcript, "words": words,
            "prompt": prompt_text,
            "owner_name": self.enrol.get("owner_name", ""),
        }

    def _capture_name(self, audio: np.ndarray) -> dict:
        """First enrolment step: capture the spoken answer to 'What is your name?'."""
        sr = self.cfg.audio.sample_rate
        rms = float(np.sqrt(np.mean(audio * audio))) if len(audio) else 0.0
        level_db = round(float(20 * np.log10(rms + 1e-12)), 1)
        text = ""
        if len(audio) >= int(sr * 0.3):
            try:
                text, _ = self._asr_engine().transcribe(audio)
            except Exception:
                log.exception("name transcription failed")

        name = parse_spoken_name(text)
        base = {
            "stage": "name", "level_db": level_db, "index": 0,
            "total": self.enrol["total"], "progress": 0.0, "done": False,
            "transcript": text, "words": [], "owner_name": "",
        }
        if not name:
            reason = ("I didn't catch a name. Tap record and say, for example, "
                      "\u201cMy name is Adnan\u201d.")
            self.enrol["last_reason"] = reason
            return {**base, "ok": False, "reason": reason}

        self.enrol["owner_name"] = name
        self.enrol["stage"] = "prompts"
        self.enrol["index"] = 0
        self.enrol["last_reason"] = ""
        prompt = self.enrol["prompts"][0]["text"] if self.enrol["prompts"] else ""
        return {
            **base, "ok": True, "reason": "", "stage": "prompts", "name": name,
            "owner_name": name, "prompt": prompt,
        }

    def _tokens(self, text: str) -> list[str]:
        return words.tokens(text)

    def _word_match(self, audio: np.ndarray) -> tuple[list[dict], str]:
        """Mark each prompt word green (heard) or red (not heard)."""
        if not (self.enrol["active"] and self.enrol["prompts"]):
            return [], ""
        prompt = self.enrol["prompts"][self.enrol["index"]]["text"]
        try:
            transcript, _ = self._asr_engine().transcribe(audio)
        except Exception:
            log.exception("enrolment transcription failed")
            return [], ""

        return words.mark_words(prompt, transcript), transcript

    def _enrol_finish(self) -> dict:
        recs = self.enrol["recordings"]
        mode = self.enrol["mode"]
        self.enrol["active"] = False
        model = self.cfg.speaker.model
        if not recs:
            return {"ok": False, "reason": "no good recordings", "done": True}

        try:
            vec, meta = self.vp.load()
            if mode == "add" and vec is not None:
                vec, meta = add_samples(vec, meta or {"num_samples": 0}, recs, model, self.cfg)
            else:
                # a fresh enrolment replaces the old frozen original as well
                for p in (self.vp.orig_vec_path, Path(str(self.vp.orig_vec_path) + ".enc")):
                    if p.exists():
                        p.unlink()
                vec, meta = enrol(recs, model, self.cfg)
        except ValueError as e:
            self.enrol["recordings"] = []
            return {"ok": False, "done": True, "reason": str(e), "stage": "prompts"}

        enrolled_name = str(self.enrol.get("owner_name") or owner_name(meta, ""))
        if enrolled_name:
            meta["owner_name"] = enrolled_name
        self.vp.save(vec, meta)
        self.cfg.bot.owner_name = enrolled_name or "the owner"
        self._invalidate_vp()
        apply_thresholds(self.cfg, meta)
        return {
            "ok": True, "done": True, "stage": "prompts",
            "owner_name": enrolled_name,
            "num_samples": meta.get("num_samples"),
            "self_check": meta.get("self_check"),
            "thresholds": meta.get("thresholds"),
        }

    def enrol_cancel(self) -> None:
        with self._audio_lock:
            self._close_record_stream()
        self.enrol["active"] = False
        self.enrol["recordings"] = []
        self.enrol["stage"] = "prompts"
        self.enrol["owner_name"] = ""

    def delete_voiceprint(self) -> None:
        self.stop_listening()
        self.vp.delete()
        self.cfg.bot.owner_name = "the owner"
        self._invalidate_vp()

    def restore_original(self) -> bool:
        self.stop_listening()
        ok = self.vp.restore_original()
        self._invalidate_vp()
        return ok

    # ------------------------------------------------------------- mic check
    def mic_check(self, seconds: float) -> dict:
        """Record a short clip from the selected microphone and report what is heard."""
        import sounddevice as sd

        if self.listening or self.enrol["active"]:
            raise RuntimeError("stop listening or enrolment before running a mic check")
        sr = self.cfg.audio.sample_rate
        with self._audio_lock:
            if self._record_stream is not None:
                raise RuntimeError("a recording is already in progress")
            audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32",
                           device=self.cfg.audio.device)
            sd.wait()
        audio = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio * audio)))
        result = {
            "ok": True,
            "db": round(float(20 * np.log10(rms + 1e-12)), 1),
            "peak": round(float(np.max(np.abs(audio))), 3) if audio.size else 0.0,
            "speech_found": False,
            "utterances": [],
        }
        utts = iter_utterances(audio, self.cfg)
        result["speech_found"] = len(utts) > 0
        if not utts:
            return result

        enrolled, _ = self._vp_meta()
        vec, _ = self.vp.load() if enrolled else (None, None)
        gate = SpeakerGate(self.cfg, vec) if vec is not None else None
        asr = self._asr_engine()
        for u in utts:
            text, _ = asr.transcribe(u.audio)
            score = gate.score(u.audio) if gate is not None else None
            result["utterances"].append({
                "duration_s": round(u.duration_s, 2),
                "score": None if score is None else round(float(score), 3),
                "transcript": text,
            })
        return result


RUNTIME = Runtime()
# Public web demo (browser microphone, per-visitor sessions). Separate from the
# owner runtime above: it never reads or writes the owner's voiceprint.
DEMO = DemoStore(RUNTIME.cfg)


# ------------------------------------------------------------------------ hub
class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        try:
            while True:
                await ws.receive_text()  # the client sends nothing; this keeps the socket open
        except WebSocketDisconnect:
            pass
        finally:
            self.clients.discard(ws)

    async def broadcast_loop(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            try:
                msg = json.dumps(RUNTIME.snapshot())
            except Exception:
                # never let one bad snapshot stop the live updates
                log.exception("snapshot failed")
                continue
            for ws in list(self.clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    self.clients.discard(ws)


HUB = Hub()


@asynccontextmanager
async def lifespan(app: FastAPI):
    TMP_DIR.mkdir(exist_ok=True)
    task = asyncio.create_task(HUB.broadcast_loop())
    sweep = asyncio.create_task(_demo_sweeper())
    # load the demo's speech model in the background so the first visitor
    # does not wait for it
    threading.Thread(target=DEMO.warm, daemon=True).start()
    yield
    task.cancel()
    sweep.cancel()
    RUNTIME.stop_listening()


async def _demo_sweeper() -> None:
    """Expire demo sessions and close idle conversations."""
    while True:
        await asyncio.sleep(30)
        try:
            DEMO.timeout_sweep()
            DEMO.ttl_sweep()
        except Exception:
            log.exception("demo sweep failed")


app = FastAPI(title="Lyra", lifespan=lifespan)

# Paths a visitor may reach through the public tunnel. Everything else — the
# owner UI, its API, the WebSocket and the live microphone — stays local.
_PUBLIC_FILES = {
    "/demo", "/demo.html", "/demo.js", "/demo.css",
    "/style.css", "/favicon.ico", "/favicon.svg",
}


def _is_public_request(request) -> bool:
    h = request.headers
    return bool(h.get("cf-connecting-ip") or h.get("cf-ray") or h.get("x-forwarded-for"))


@app.middleware("http")
async def ui_guard(request, call_next):
    """Serve only the demo to tunneled visitors; never cache UI files."""
    path = request.url.path
    public = _is_public_request(request)
    if public:
        if path == "/":
            return RedirectResponse("/demo", status_code=307)
        if path not in _PUBLIC_FILES and not path.startswith("/api/demo"):
            return JSONResponse(
                {"ok": False, "reason": "not available on the public demo"},
                status_code=403,
            )
    response = await call_next(request)
    if not path.startswith(("/api", "/ws")):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    if public:
        response.headers["X-Robots-Tag"] = "noindex"
    return response


def _err(e: Exception, code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "reason": str(e)}, status_code=code)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await HUB.register(ws)


# Plain `def` handlers run in a worker thread, so audio and model work never
# blocks the event loop that keeps the live UI connected.
@app.get("/api/state")
def api_state():
    return RUNTIME.snapshot()


@app.get("/api/devices")
def api_devices():
    return {"devices": list_input_devices(), "selected": RUNTIME.cfg.audio.device}


@app.post("/api/miccheck")
def api_miccheck(payload: dict):
    try:
        return RUNTIME.mic_check(float(payload.get("seconds", 4.0)))
    except RuntimeError as e:
        return _err(e)
    except Exception as e:
        log.exception("mic check failed")
        return _err(e, 500)


@app.post("/api/listen")
def api_listen(payload: dict):
    try:
        if payload.get("on"):
            RUNTIME.start_listening()
        else:
            RUNTIME.stop_listening()
    except RuntimeError as e:
        return _err(e)
    return RUNTIME.snapshot()


@app.post("/api/settings")
def api_settings(payload: dict):
    try:
        RUNTIME.update_settings(payload)
    except (RuntimeError, ValueError) as e:
        return _err(e)
    return RUNTIME.snapshot()


@app.post("/api/enrol/start")
def api_enrol_start(payload: dict):
    try:
        return RUNTIME.enrol_start(
            payload.get("mode", "enrol"),
            bool(payload.get("consent", False)),
        )
    except RuntimeError as e:
        return _err(e)


@app.post("/api/enrol/submit")
def api_enrol_submit(payload: dict):
    try:
        return RUNTIME.enrol_submit(payload.get("audio", ""), payload.get("mime", ""), payload.get("name", ""))
    except Exception as e:
        log.exception("enrol submit failed")
        return _err(e)


@app.post("/api/enrol/record/start")
def api_enrol_record_start():
    try:
        RUNTIME.enrol_record_start()
        return {"ok": True}
    except RuntimeError as e:  # expected refusals, e.g. a double click
        return _err(e)
    except Exception as e:
        log.exception("could not start recording")
        return _err(e, 500)


@app.post("/api/enrol/record/stop")
def api_enrol_record_stop():
    try:
        return RUNTIME.enrol_record_stop()
    except RuntimeError as e:
        return _err(e)
    except Exception as e:
        log.exception("could not stop recording")
        return _err(e, 500)


@app.post("/api/enrol/cancel")
def api_enrol_cancel():
    RUNTIME.enrol_cancel()
    return RUNTIME.snapshot()


@app.post("/api/profile/restore")
def api_profile_restore():
    RUNTIME.restore_original()
    return RUNTIME.snapshot()


@app.post("/api/profile/delete")
def api_profile_delete():
    RUNTIME.delete_voiceprint()
    return RUNTIME.snapshot()


# ------------------------------------------------------------------ web demo
@app.get("/demo")
def demo_page():
    return FileResponse(UI_DIR / "demo.html")


@app.get("/api/demo/session")
def api_demo_session(sid: str | None = None):
    return DEMO.get(sid).state()


@app.post("/api/demo/start")
def api_demo_start(payload: dict):
    session = DEMO.reset(payload.get("sid"))
    question = "What is your name?"
    return {
        "ok": True, "question": question, "audio": DEMO.speak(question),
        **session.state(),
    }


@app.post("/api/demo/enrol")
def api_demo_enrol(payload: dict):
    session = DEMO.get(payload.get("sid"))
    try:
        return DEMO.enrol(session, payload.get("audio", ""), payload.get("mime", ""))
    except ValueError as e:
        return {"ok": False, "reason": str(e), **session.state()}
    except Exception as e:
        log.exception("demo enrolment failed")
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


@app.post("/api/demo/ask")
def api_demo_ask(payload: dict):
    session = DEMO.get(payload.get("sid"))
    try:
        return DEMO.ask(session, payload.get("audio", ""), payload.get("mime", ""))
    except ValueError as e:
        return {"ok": False, "reason": str(e), **session.state()}
    except Exception as e:
        log.exception("demo question failed")
        return JSONResponse({"ok": False, "reason": str(e)}, status_code=500)


# static UI last, so /api and /ws take precedence
app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    print(f"\nvoice assistant UI  ->  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)
