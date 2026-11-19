"""Ticket 3 — build, save, load and delete the owner voiceprint."""
from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import numpy as np

from .. import privacy, speaker
from ..config import AttrDict


def adapt(vec: np.ndarray, emb: np.ndarray, weight: float) -> np.ndarray:
    """Ticket 8 — one adaptive step: EMA toward a high-confidence embedding."""
    v = (1.0 - weight) * vec + weight * emb
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


# The owner rule: 35% similarity or more means "this is the owner".
OWNER_SIMILARITY = 0.35
NOT_OWNER_SIMILARITY = 0.35  # anything below the owner rule is someone else
# Extra samples are only accepted when they clearly match the enrolled voice,
# so a different speaker cannot slowly poison the voiceprint.
ADD_SIMILARITY = 0.45


def clean_owner_name(name: str | None) -> str:
    """Normalise the name the owner enters during enrolment."""
    s = re.sub(r"\s+", " ", str(name or "").strip())
    s = re.sub(r"[^\w .'\-]", "", s, flags=re.UNICODE)
    s = s.strip(" .'-")[:40].strip()
    return s if any(ch.isalpha() for ch in s) else ""


def owner_name(meta: dict | None, default: str = "the owner") -> str:
    """The enrolled owner's name, or a neutral default for older voiceprints."""
    if meta:
        name = clean_owner_name(meta.get("owner_name"))
        if name:
            return name
    return default


# Spoken answers to "What is your name?" usually start with one of these.
_NAME_PREFIX_RE = re.compile(
    r"^\s*(?:(?:hi|hello|hey)\s+)?(?:"
    r"my\s+name(?:\s+is|'s)?|the\s+name(?:\s+is)?|name\s+is|"
    r"i\s+am|i'?m|this\s+is|it'?s|it\s+is|call\s+me)\s+",
    re.I,
)
_NAME_STOPWORDS = {
    "and", "nice", "thank", "thanks", "please", "how", "what", "okay", "ok",
    "yeah", "sir", "madam", "ma'am", "meet", "to", "you",
    "my", "name", "the", "is",
}
_NAME_FILLERS = {"um", "uh", "erm", "hmm", "so", "like", "well"}
_NAME_FILLER_PREFIX_RE = re.compile(
    r"^(?:(?:um|uh|erm|hmm|so|like|well)\b[\s,]+)+", re.I
)


def parse_spoken_name(text: str) -> str:
    """Turn a spoken answer into a clean name.

    Examples: "My name is Adnan Bashir" -> "Adnan Bashir";
    "It's Adnan." -> "Adnan"; "A D N A N" -> "Adnan".
    """
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^\w\s.'\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _NAME_FILLER_PREFIX_RE.sub("", s)
    s = _NAME_PREFIX_RE.sub("", s).strip()

    tokens: list[str] = []
    for raw in s.split():
        low = raw.lower().strip(" .'-")
        if not low or low in _NAME_FILLERS:
            continue
        if low in _NAME_STOPWORDS:
            break
        tokens.append(raw.strip(" .'-"))
        if len(tokens) >= 6:  # enough for a spelled-out first name
            break

    if len(tokens) >= 2 and all(len(t) == 1 and t.isalpha() for t in tokens):
        tokens = ["".join(tokens)]  # "a d n a n" -> "adnan"
    else:
        tokens = tokens[:3]         # first, middle and last name

    name = clean_owner_name(" ".join(tokens))
    if not name:
        return ""
    if name.islower() or name.isupper():
        name = " ".join(p[:1].upper() + p[1:].lower() for p in name.split())
    return name


def calibrate(meta: dict) -> dict:
    """Apply the owner rule to a fresh voiceprint.

    35% similarity or more is the owner. The threshold is fixed so the rule is
    predictable; the adaptive voiceprint handles drift over time.
    """
    meta["thresholds"] = {"accept": OWNER_SIMILARITY, "reject": NOT_OWNER_SIMILARITY}
    return meta


def apply_thresholds(cfg, meta: dict | None) -> None:
    """Always enforce the owner rule, ignoring thresholds stored by older runs."""
    cfg.gate.accept_threshold = OWNER_SIMILARITY
    cfg.gate.reject_threshold = NOT_OWNER_SIMILARITY
    if meta is not None:  # keep metadata consistent for the UI and future loads
        meta["thresholds"] = {"accept": OWNER_SIMILARITY, "reject": NOT_OWNER_SIMILARITY}


_VAD_MODEL = None


def _vad_model():
    """Load the Silero VAD once and reuse it (avoids reloading for every clip)."""
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad

        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def trim_to_speech(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Trim a recording to its speech span (drop leading/trailing silence/noise),
    so enrolment embeddings are computed on the voice itself — consistent with
    the VAD-trimmed live utterances."""
    try:
        from silero_vad import get_speech_timestamps
    except ImportError:
        return audio
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    try:
        stamps = get_speech_timestamps(audio, _vad_model(), sampling_rate=sr)
    except Exception:
        return audio
    if not stamps:
        return audio
    start = stamps[0]["start"]
    end = stamps[-1]["end"]
    if end - start < sr // 4:  # less than 250 ms of speech: keep as-is
        return audio
    return np.ascontiguousarray(audio[start:end], dtype=np.float32)


def enrol(recordings: list[np.ndarray], model: str, cfg: AttrDict) -> tuple[np.ndarray, dict]:
    """Embed each recording, drop outliers, average and L2-normalise.

    Returns (voiceprint_vector, meta_dict).
    """
    if not recordings:
        raise ValueError("no recordings to enrol")

    embs = np.stack([speaker.embed(model, trim_to_speech(r)) for r in recordings])  # already normalised
    mean = embs.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)

    sims = [float(np.dot(e, mean)) for e in embs]
    center = float(np.median(sims))
    gap = float(cfg.enrol.outlier_drop_gap)
    kept = [e for e, s in zip(embs, sims) if s >= center - gap]

    # if that dropped everything (degenerate), keep them all
    if not kept:
        kept = list(embs)

    vec = np.mean(kept, axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-12)

    scores = [float(np.dot(e, vec)) for e in kept]
    meta = {
        "model": model,
        "version": 1,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(kept),
        "num_dropped": len(recordings) - len(kept),
        "self_check": {
            "mean": round(float(np.mean(scores)), 4),
            "min": round(float(np.min(scores)), 4),
        },
    }
    return vec.astype(np.float32), calibrate(meta)


def add_samples(vec: np.ndarray, meta: dict, recordings: list[np.ndarray],
                model: str, cfg: AttrDict) -> tuple[np.ndarray, dict]:
    """Strengthen an existing voiceprint with extra samples (ticket 3 'Add more').

    New recordings must clearly match the enrolled voice before they are added;
    this stops an impostor (or a noisy take) from slowly poisoning the print.
    Adding more genuine samples averages out microphone/room variation, which
    improves recognition accuracy over time.
    """
    if not recordings:
        raise ValueError("no recordings to add")

    # Score every take against the current voiceprint first.
    embs = np.stack([speaker.embed(model, trim_to_speech(r)) for r in recordings])
    sims = [float(np.dot(vec, e)) for e in embs]
    threshold = float(cfg.enrol.get("add_min_similarity", ADD_SIMILARITY))
    kept_idx = [i for i, s in enumerate(sims) if s >= threshold]
    if not kept_idx:
        best = max(sims)
        raise ValueError(
            "those samples do not match the enrolled voice "
            f"(best {best * 100:.0f}% match, need {threshold * 100:.0f}%) — "
            "use the same microphone and speak naturally"
        )
    accepted = [recordings[i] for i in kept_idx]

    existing = vec * float(meta.get("num_samples", 1))
    new_vec, new_meta = enrol(accepted, model, cfg)
    combined = existing + new_vec * new_meta["num_samples"]
    combined = combined / (np.linalg.norm(combined) + 1e-12)
    total = meta.get("num_samples", 0) + new_meta["num_samples"]
    meta = dict(meta)
    meta["num_samples"] = total
    meta["num_dropped"] = meta.get("num_dropped", 0) + new_meta["num_dropped"]
    meta["num_rejected_add"] = len(recordings) - len(accepted)
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["self_check"] = new_meta["self_check"]
    return combined.astype(np.float32), calibrate(meta)


class Voiceprint:
    """Loads/saves voiceprint.npy + meta.json under `data_dir`.

    Files are encrypted with Fernet (Keychain-held key) when available; a
    frozen `voiceprint.original.npy` is kept at first enrolment so an adaptive
    update can be undone.
    """

    def __init__(self, data_dir: Path, model: str = "ecapa", encrypt: bool = True):
        self.data_dir = Path(data_dir)
        self.model = model
        self.encrypt = encrypt
        self.vec_path = self.data_dir / "voiceprint.npy"
        self.meta_path = self.data_dir / "meta.json"
        self.orig_vec_path = self.data_dir / "voiceprint.original.npy"
        self._warned = False

    # ------------------------------------------------------------- encrypted io
    def _store(self, rel: Path, data: bytes) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.encrypt:
            enc = privacy.encrypt(data)
            if enc is not None:
                (self.data_dir / (rel.name + ".enc")).write_bytes(enc)
                return
            if not self._warned:
                print("  [privacy] encryption unavailable — storing plaintext")
                self._warned = True
        (self.data_dir / rel.name).write_bytes(data)

    def _read(self, rel: Path) -> bytes | None:
        enc_path = self.data_dir / (rel.name + ".enc")
        if enc_path.exists():
            dec = privacy.decrypt(enc_path.read_bytes())
            if dec is not None:
                return dec
        plain = self.data_dir / rel.name
        return plain.read_bytes() if plain.exists() else None

    @staticmethod
    def _npy_bytes(vec: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, np.asarray(vec, dtype=np.float32))
        return buf.getvalue()

    @staticmethod
    def _from_npy(data: bytes) -> np.ndarray:
        return np.load(io.BytesIO(data)).astype(np.float32)

    # ------------------------------------------------------------- public api
    def exists(self) -> bool:
        return self.vec_path.exists() or (self.data_dir / "voiceprint.npy.enc").exists()

    def has_original(self) -> bool:
        return self.orig_vec_path.exists() or (self.data_dir / "voiceprint.original.npy.enc").exists()

    def load(self) -> tuple[np.ndarray | None, dict | None]:
        if not self.exists():
            return None, None
        raw = self._read(self.vec_path)
        if raw is None:
            return None, None
        vec = self._from_npy(raw)
        meta = {}
        meta_raw = self._read(self.meta_path)
        if meta_raw:
            try:
                meta = json.loads(meta_raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                meta = {}
        return vec, meta

    def save(self, vec: np.ndarray, meta: dict) -> None:
        """Save the voiceprint. Freezes the original on first save."""
        if not self.has_original():
            self._store(self.orig_vec_path, self._npy_bytes(vec))
        self._store(self.vec_path, self._npy_bytes(vec))
        self._store(self.meta_path, json.dumps(meta, indent=2).encode("utf-8"))

    def update(self, vec: np.ndarray, meta: dict | None) -> None:
        """Save a new current voiceprint WITHOUT touching the frozen original."""
        if meta is None:
            meta = {}
        self._store(self.vec_path, self._npy_bytes(vec))
        self._store(self.meta_path, json.dumps(meta, indent=2).encode("utf-8"))

    def restore_original(self) -> bool:
        """Undo adaptive drift: put the frozen original back."""
        raw = self._read(self.orig_vec_path)
        if raw is None:
            return False
        self._store(self.vec_path, raw)
        return True

    def delete(self) -> bool:
        removed = False
        for rel in (self.vec_path, self.meta_path, self.orig_vec_path):
            for p in (self.data_dir / rel.name, self.data_dir / (rel.name + ".enc")):
                if p.exists():
                    p.unlink()
                    removed = True
        return removed
