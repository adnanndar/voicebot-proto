"""Configuration loading: config.yaml + .env, merged over defaults."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # voiceprint + meta
DEBUG_DIR = ROOT / "debug"        # debug utterances + decision log


# Full defaults — config.yaml only overrides what it contains.
DEFAULTS = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "blocksize": 512,           # 32 ms @ 16 kHz
        "device": None,             # None = default input device
        "noise_suppression": False,
        "ns_chunk_sec": 0.5,        # denoise chunk size (adds ~this much latency)
        "ns_prop_decrease": 0.85,   # how aggressively to remove noise (0..1)
        "ns_profile_sec": 0.0,      # 0 = no learned noise profile (safer for VAD)
        "gain_db": 0.0,             # extra gain before VAD; 0 is best for normal mics
    },
    "vad": {
        "threshold": 0.5,
        "start_ms": 300,            # speech needed to open an utterance
        "end_ms": 2000,             # decide once the speaker has been silent for 2 s
        "max_sec": 30.0,            # backstop: close even if the room never goes quiet
        "pre_roll_ms": 600,         # keep before speech start so first words survive
        "noise_gate_ratio": 2.0,     # ignore anything within 2x the background noise floor
    },
    "speaker": {
        "model": "ecapa",           # ecapa | resemblyzer
        "adaptive": {               # ticket 8: slowly update the voiceprint
            "enabled": True,
            "weight": 0.02,         # EMA weight, kept small
            "min_score": 0.45,      # adapt only on utterances clearly above the 35% rule
        },
    },
    "privacy": {
        "encrypt": True,            # Fernet + macOS Keychain at rest
        "require_consent": True,    # consent screen before enrolment
    },
    "gate": {
        "accept_threshold": 0.35,   # >= 35% similarity -> owner
        "reject_threshold": 0.35,   # < 35% -> not owner
        "min_utterance_sec": 0.5,   # below this, scores are unreliable
        "conversation_window_sec": 20.0,
        "followup_score": 0.18,      # unused: strict 35% rule applies to every utterance
        "conversation_hold_sec": 300.0,  # unused: kept for config compatibility
        "lockout_after": 5,         # unrecognised attempts before a pause
        "lockout_window_sec": 60.0, # attempts counted within this window
        "lockout_sec": 60.0,        # pause length after too many attempts
    },
    "timeout": {
        "enabled": True,
        "minutes": 2,               # sleep after this long with no owner interaction
    },
    "enrol": {
        "min_speech_sec": 0.8,
        "min_duration_sec": 1.0,
        "min_level_db": -40.0,      # quietest accepted RMS (dBFS)
        "min_snr_db": 6.0,          # speech vs background
        "outlier_drop_gap": 0.15,   # cosine gap below others' mean => drop
        "add_min_similarity": 0.45,  # new samples must clearly match the owner
        "prompts_file": "app/enrol/prompts.yaml",
    },
    "bot": {
        "asr_model": "small",
        "language": "en",           # English only
        "brain": "auto",            # auto | openai | claude | pollinations | local
        "local_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "tts": "say",               # say | print | null
        "tts_voice": None,
        "barge_in": False,
        "owner_name": "the owner",
    },
    "debug": {
        "save_utterances": False,   # keep owner audio in debug/ only when turned on
        "dir": "debug",
    },
}


class AttrDict(dict):
    """dict with attribute access, recursively."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _deep_merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_dotenv() -> dict:
    env: dict = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_config(path: str | os.PathLike | None = None) -> AttrDict:
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else ROOT / "config.yaml"
    if p.exists():
        with open(p, "r") as f:
            user = yaml.safe_load(f) or {}
        _deep_merge(cfg, user)
    cfg = _wrap(cfg)
    cfg.env = _wrap(load_dotenv())
    cfg.path = p
    return cfg
