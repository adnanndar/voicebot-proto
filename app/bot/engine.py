"""Ticket 5 — orchestrates capture -> gate -> ASR -> brain -> TTS.

State machine: idle -> listening -> verifying -> thinking -> speaking.
AEC for the prototype: while the bot is speaking the mic is muted (unless
barge-in is enabled, in which case an owner interrupt stops TTS).
"""
from __future__ import annotations

import json
import queue
import re
import time
from pathlib import Path
from typing import Callable

import soundfile as sf

from ..config import DATA_DIR, DEBUG_DIR, AttrDict
from ..enrol.voiceprint import Voiceprint, adapt, apply_thresholds, owner_name, trim_to_speech
from ..gate.gate import SpeakerGate
from ..types import Decision, Utterance
from .asr import ASR
from .brain import make_brain
from .tts import make_tts
from .. import speaker


class Engine:
    STATES = ("idle", "listening", "verifying", "thinking", "speaking", "standby")

    # short phrases that end the conversation
    STOP_PHRASES = {
        "stop", "stop listening", "please stop", "stop it", "that's all", "thats all",
        "that is all", "i'm done", "im done", "goodbye", "good bye", "bye", "bye bye",
        "end conversation", "shut down", "shut up", "never mind", "cancel", "quiet",
    }
    _NORM_RE = re.compile(r"[^a-z' ]+")
    # "Lyra" must start the utterance when a conversation is not already open.
    # A few spellings are accepted because speech-to-text may hear the name
    # slightly differently (Lira, Lyrah, Leera, Lara).
    _WAKE_RE = re.compile(
        r"^\s*(?:(?:hey|hi|okay|ok)[\s,]+)?"
        r"(?:lyra|lyrah|lira|lirra|laira|leera|lara)\b[\s,.:;!?\-]*",
        re.I,
    )

    def __init__(self, cfg: AttrDict):
        self.cfg = cfg
        self.state = "idle"

        self.voiceprint = Voiceprint(DATA_DIR, cfg.speaker.model, encrypt=cfg.privacy.encrypt)
        vec, meta = self.voiceprint.load()
        self.vp_vec = vec
        self.vp_meta = meta
        apply_thresholds(cfg, meta)  # use enrolment-calibrated thresholds
        # The name entered during enrolment is how the assistant addresses the owner.
        cfg.bot.owner_name = owner_name(meta, cfg.bot.get("owner_name", "the owner"))
        self.last_owner_time = time.time()
        # A conversation stays open after the owner is verified, so follow-up
        # questions keep working until the owner says "stop".
        self.conversation_open = False

        self.gate = SpeakerGate(cfg, self.vp_vec)
        self.asr = ASR(cfg)
        self.brain = make_brain(cfg)
        self.tts = make_tts(cfg)

        self.pipeline = None            # set by main.py when running live
        self.conversation: list[tuple[str, str]] = []
        self.running = False
        self._queue: queue.Queue = queue.Queue()
        self._utter_index = 0

        # optional hooks (ticket 7 UI attaches here)
        self.on_state: Callable[[str], None] | None = None
        self.on_event: Callable[[dict], None] | None = None

        DEBUG_DIR.mkdir(exist_ok=True)
        self._decisions_path = DEBUG_DIR / "decisions.jsonl"

    # ------------------------------------------------------------------ state
    def _set_state(self, s: str) -> None:
        self.state = s
        if self.on_state:
            self.on_state(s)

    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event:
            self.on_event({"kind": kind, "time": time.time(), **payload})

    # ------------------------------------------------------------ utterance in
    def on_utterance(self, utt: Utterance) -> None:
        self._queue.put(utt)

    def _save_debug_utterance(self, utt: Utterance) -> None:
        d = Path(self.cfg.debug.dir)
        d.mkdir(exist_ok=True)
        self._utter_index += 1
        path = d / f"utt_{self._utter_index:04d}_{utt.start_s:.2f}s.wav"
        sf.write(str(path), utt.audio, self.cfg.audio.sample_rate, subtype="PCM_16")

    # -------------------------------------------------------------- main loop
    def run(self) -> None:
        """Live loop: block, consuming utterances from the pipeline."""
        self.running = True
        self._set_state("listening")
        while self.running:
            try:
                utt = self._queue.get(timeout=1.0)
            except queue.Empty:
                self._check_timeout()
                continue
            if utt is None:
                break
            self.handle(utt)
            self._check_timeout()

    def _check_timeout(self) -> None:
        """Ticket-requested: go to standby after `minutes` with no owner voice."""
        t = self.cfg.timeout
        if not t.enabled or self.state == "standby":
            return
        if time.time() - self.last_owner_time >= float(t.minutes) * 60:
            self.conversation_open = False
            self._set_state("standby")
            self._emit("standby", {"minutes": t.minutes})

    def stop(self) -> None:
        self.running = False
        self.tts.stop()
        self._queue.put(None)

    # ------------------------------------------------------------- processing
    def _trim(self, utt: Utterance) -> Utterance:
        """Trim the utterance to its actual speech before scoring / ASR.

        Enrolment already trims to speech, so doing the same live keeps the
        comparison fair and stops silence from diluting the voice or confusing
        the speech-to-text model.
        """
        sr = self.cfg.audio.sample_rate
        audio = trim_to_speech(utt.audio, sr)
        if audio is utt.audio or len(audio) == len(utt.audio) or len(audio) < sr // 4:
            return utt
        dur = len(audio) / sr
        return Utterance(audio=audio, start_s=utt.end_s - dur, end_s=utt.end_s)

    def _is_stop(self, text: str) -> bool:
        t = re.sub(r"\s+", " ", self._NORM_RE.sub(" ", text.lower())).strip()
        if t in self.STOP_PHRASES:
            return True
        words = t.split()
        return 0 < len(words) <= 4 and words[-1] in {"stop", "goodbye", "bye"}

    def _split_wake(self, text: str) -> tuple[bool, str]:
        """Return (wake word present at the start, remaining request)."""
        m = self._WAKE_RE.match(text)
        if not m:
            return False, text.strip()
        return True, text[m.end():].strip(" ,.;:!?-")

    def _greeting(self, with_offer: bool = True) -> str:
        name = str(self.cfg.bot.get("owner_name", "the owner"))
        named = name.strip().lower() not in {"", "owner", "the owner"}
        if named:
            if with_offer:
                return f"Hey {name}, how are you? How can I help?"
            return f"Hey {name}!"
        return "Hey! How can I help?" if with_offer else "Hey!"

    def handle(self, utt: Utterance) -> Decision:
        """Process one utterance end to end. Synchronous."""
        was_standby = self.state == "standby"
        self._set_state("verifying")
        utt = self._trim(utt)
        d = self.gate.decide(utt)

        self._emit("decision", {**d.as_dict(), "duration_s": round(utt.duration_s, 3)})

        # Strict owner rule: every utterance must match >= 35%, even during an
        # open conversation. There is no follow-up relaxation.
        if not d.owner:
            self._log_decision(d, utt)
            self._set_state("standby" if was_standby else "listening")
            return d

        self.last_owner_time = time.time()
        # only owner audio is ever kept on disk (ticket 8: non-owner audio is dropped)
        if self.cfg.debug.save_utterances:
            self._save_debug_utterance(utt)
        d = self._process_owner(utt, d, was_standby)
        self._log_decision(d, utt)  # single log line, now with transcript
        return d

    def _process_owner(self, utt: Utterance, d: Decision, was_standby: bool) -> Decision:
        self._maybe_adapt(utt, d)
        self._set_state("thinking")
        text, info = self.asr.transcribe(utt.audio)
        d.transcript = text
        self._emit("transcript", {"transcript": text, **info})

        if not text.strip():
            reply = "I heard you, but I couldn't make out the words. Could you say that again?"
            self._emit("reply", {"reply": reply})
            self._speak(reply)
            self._set_state("listening")
            return d

        needs_wake = not self.conversation_open
        had_wake, command = self._split_wake(text)

        # A stop phrase ends the conversation. While sleeping, it still has to
        # start with the wake word (the UI Stop button also works).
        if self._is_stop(command) and (not needs_wake or had_wake):
            self.conversation_open = False
            reply = "Okay, stopping. Say \u201cHey Lyra\u201d when you need me."
            self.conversation.append(("user", text))
            self.conversation.append(("bot", reply))
            self._emit("reply", {"reply": reply, "stop": True})
            self._speak(reply)
            self._set_state("idle")
            self._emit("stop_listening", {})
            return d

        # Sleeping / new conversation: only an owner utterance that starts with
        # "Lyra" (or "Hey Lyra") may wake the assistant.
        if needs_wake and not had_wake:
            self._emit("wake_required", {"transcript": text})
            self._set_state("standby" if was_standby else "listening")
            return d

        self.conversation_open = True
        self.conversation.append(("user", text))
        self.conversation = self.conversation[-20:]  # bounded history

        if needs_wake:
            self._emit("wake", {"greeted": True})
            greeting = self._greeting(with_offer=not bool(command.strip()))
            if command.strip():
                answer = self.brain.reply(command, self.conversation)
                reply = f"{greeting} {answer}"
            else:
                reply = greeting
        else:
            reply = self.brain.reply(text, self.conversation)

        self.conversation.append(("bot", reply))
        self.conversation = self.conversation[-20:]
        self._emit("reply", {"reply": reply, **({"wake": True} if needs_wake else {})})

        self._speak(reply)
        self._set_state("listening")
        return d

    def _maybe_adapt(self, utt: Utterance, d: Decision) -> None:
        """Ticket 8 — EMA the voiceprint toward high-confidence owner utterances."""
        ad = self.cfg.speaker.adaptive
        if not ad.enabled or self.gate.voiceprint is None:
            return
        if d.score is None or d.score < float(ad.min_score):
            return
        emb = speaker.embed(self.cfg.speaker.model, utt.audio)
        self.gate.voiceprint = adapt(self.gate.voiceprint, emb, float(ad.weight))
        meta = dict(self.vp_meta or {})
        meta["adapted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["num_adaptations"] = meta.get("num_adaptations", 0) + 1
        self.vp_meta = meta
        self.voiceprint.update(self.gate.voiceprint, meta)
        self._emit("adapted", {"score": d.score})

    def _speak(self, reply: str) -> None:
        barge = bool(self.cfg.bot.barge_in)
        if self.pipeline and not barge:
            self.pipeline.muted = True
        try:
            self._set_state("speaking")
            self.tts.speak(reply)
            if not barge:
                self.tts.wait()
                return

            # barge-in: keep the mic live and watch for an owner interrupt
            while not self.tts.is_done():
                try:
                    u = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if u is None:
                    break
                d = self.gate.decide(u)
                self._log_decision(d, u)
                self._emit("decision", {**d.as_dict(), "duration_s": round(u.duration_s, 3), "barge_in": True})
                if d.owner:
                    self.tts.stop()
                    self._set_state("thinking")
                    text, info = self.asr.transcribe(u.audio)
                    d.transcript = text
                    self.conversation.append(("user", text))
                    reply = self.brain.reply(text, self.conversation)
                    self.conversation.append(("bot", reply))
                    self._emit("transcript", {"transcript": text, "barge_in": True, **info})
                    self._emit("reply", {"reply": reply, "barge_in": True})
                    self.tts.speak(reply)
            self.tts.wait()
        finally:
            if self.pipeline:
                self.pipeline.muted = False

    # ---------------------------------------------------------------- logging
    def _log_decision(self, d: Decision, utt: Utterance) -> None:
        record = {
            "time": time.time(),
            "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_s": round(utt.start_s, 3),
            "duration_s": round(utt.duration_s, 3),
            "score": d.score,
            "decision": d.decision,
            "reason": d.reason,
            "method": d.method,
            "transcript": d.transcript,
        }
        with open(self._decisions_path, "a") as f:
            f.write(json.dumps(record) + "\n")
