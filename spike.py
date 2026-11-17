#!/usr/bin/env python3
"""Spike: local speaker verification (ECAPA vs Resemblyzer) + Silero VAD + faster-whisper.

Everything runs on CPU. No API keys required.

Usage:
    python spike.py embed --backend both
    python spike.py vad
    python spike.py asr --model small
    python spike.py all

Clips live in samples/ and are named owner_*.wav / other_*.wav.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
MODELS = ROOT / "models"
RESULTS = ROOT / "results.json"


# --------------------------------------------------------------------------- io
def load_wav(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read any wav/flac and return mono float32 at 16 kHz."""
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def find_clips(folder: Path) -> list[Path]:
    clips = sorted(p for p in folder.glob("*.wav") if p.is_file())
    if not clips:
        sys.exit(
            f"No .wav files in {folder}.\n"
            "Run scripts/make_fixtures.sh for synthetic clips, or put your own "
            "owner_*.wav / other_*.wav files there."
        )
    return clips


def label_of(path: Path) -> str:
    """owner_1.wav -> owner. Anything else -> other."""
    stem = path.stem.lower()
    if stem.startswith("owner"):
        return "owner"
    if stem.startswith("other"):
        return "other"
    return "unknown"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bench(fn, runs: int = 5, warmup: int = 1) -> dict:
    """Return p50/p95/mean wall time in ms over `runs` calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p95_idx = max(0, int(round(0.95 * (len(times) - 1))))
    return {
        "p50_ms": round(statistics.median(times), 1),
        "p95_ms": round(times[p95_idx], 1),
        "mean_ms": round(statistics.fmean(times), 1),
        "runs": runs,
    }


# ------------------------------------------------------------------- backends
class EcapaBackend:
    name = "ecapa"
    dim = 192

    def __init__(self) -> None:
        import torch

        try:  # speechbrain >= 1.0
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain 0.5.x
            from speechbrain.pretrained import EncoderClassifier

        t0 = time.perf_counter()
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(MODELS / "ecapa"),
            run_opts={"device": "cpu"},
        )
        self.model.eval()
        self._torch = torch
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)

    def embed(self, wav: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            x = self._torch.from_numpy(wav).unsqueeze(0)
            emb = self.model.encode_batch(x)
        return emb.squeeze().cpu().numpy()


class ResemblyzerBackend:
    name = "resemblyzer"
    dim = 256

    def __init__(self) -> None:
        from resemblyzer import VoiceEncoder

        t0 = time.perf_counter()
        self.model = VoiceEncoder(verbose=False)
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)

    def embed(self, wav: np.ndarray) -> np.ndarray:
        from resemblyzer import preprocess_wav

        w = preprocess_wav(wav, source_sr=SAMPLE_RATE)
        return self.model.embed_utterance(w)


BACKENDS = {"ecapa": EcapaBackend, "resemblyzer": ResemblyzerBackend}


# ------------------------------------------------------------------- stage 1
def run_embed(backend_names: list[str], results: dict) -> None:
    clips = find_clips(SAMPLES)
    print(f"\nLoaded {len(clips)} clips from {SAMPLES}")
    for c in clips:
        print(f"  - {c.name:<18} {label_of(c):<6} {len(load_wav(c)) / SAMPLE_RATE:5.2f}s")

    for name in backend_names:
        print(f"\n=== {name.upper()} ===============================================")
        backend = BACKENDS[name]()
        print(f"model load: {backend.load_ms} ms")

        t0 = time.perf_counter()
        embs = {c.name: backend.embed(load_wav(c)) for c in clips}
        total_ms = (time.perf_counter() - t0) * 1000
        per_clip = total_ms / len(clips)
        print(f"embedding:  {per_clip:.1f} ms/clip (first pass, includes warmup)")

        # timing on the longest clip, warm
        longest = max(clips, key=lambda p: len(load_wav(p)))
        wav = load_wav(longest)
        timing = bench(lambda: backend.embed(wav), runs=5)
        print(
            f"embedding:  p50={timing['p50_ms']} ms  p95={timing['p95_ms']} ms "
            f"(on {longest.name}, {len(wav) / SAMPLE_RATE:.1f}s audio)"
        )

        keys = list(embs)
        m = np.array([[cosine(embs[a], embs[b]) for b in keys] for a in keys])

        print("\n  cosine similarity matrix")
        print("  " + " " * 18 + "".join(f"{k[:10]:>11}" for k in keys))
        for i, a in enumerate(keys):
            print(f"  {a:<16}" + "".join(f"{m[i, j]:>11.3f}" for j in range(len(keys))))

        groups = {"owner-owner": [], "owner-other": [], "other-other": []}
        print("\n  pairs")
        for i, j in combinations(range(len(keys)), 2):
            la, lb = label_of(clips[i]), label_of(clips[j])
            kind = "-".join(sorted((la, lb), key=lambda s: (s != "owner", s)))
            groups.setdefault(kind, []).append(m[i, j])
            print(f"    {keys[i]:<16} vs {keys[j]:<16} {m[i, j]:+.3f}   [{kind}]")

        print("\n  summary")
        for kind, vals in groups.items():
            if vals:
                print(f"    {kind:<12} n={len(vals)}  min={min(vals):+.3f}  max={max(vals):+.3f}  mean={statistics.fmean(vals):+.3f}")

        oo = groups.get("owner-owner", [])
        ox = groups.get("owner-other", [])
        if oo and ox:
            margin = min(oo) - max(ox)
            verdict = "PASS" if margin > 0 else "FAIL"
            print(f"\n  margin (min owner-owner - max owner-other) = {margin:+.3f}  -> {verdict}")
            print("  (a useful threshold sits inside that gap, e.g. "
                  f"{(min(oo) + max(ox)) / 2:.2f})")

        results.setdefault("embed", {})[name] = {
            "load_ms": backend.load_ms,
            "p50_ms": timing["p50_ms"],
            "p95_ms": timing["p95_ms"],
            "matrix": m.tolist(),
            "clips": keys,
            "groups": {k: v for k, v in groups.items() if v},
        }


# ------------------------------------------------------------------- stage 2
def run_vad(results: dict, file: str | None = None) -> None:
    from silero_vad import get_speech_timestamps, load_silero_vad

    clip = Path(file) if file else pick_long_clip()
    wav = load_wav(clip)
    print(f"\n=== SILERO VAD ==================================================")
    print(f"clip: {clip.name}  {len(wav) / SAMPLE_RATE:.2f}s")

    t0 = time.perf_counter()
    model = load_silero_vad()
    print(f"model load: {(time.perf_counter() - t0) * 1000:.0f} ms")

    t0 = time.perf_counter()
    stamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLE_RATE)
    wall = time.perf_counter() - t0
    speech_s = sum(s["end"] - s["start"] for s in stamps) / SAMPLE_RATE
    print(f"speech segments: {len(stamps)}  ({speech_s:.2f}s of speech detected)")
    print(f"batch pass: {wall * 1000:.0f} ms for {len(wav) / SAMPLE_RATE:.2f}s audio "
          f"(RTF {wall / (len(wav) / SAMPLE_RATE):.4f})")

    # streaming-style: 512-sample frames at 16 kHz = 32 ms
    import torch

    frames = [wav[i:i + 512] for i in range(0, len(wav) - 511, 512)]
    tensor_frames = [torch.from_numpy(f) for f in frames]

    def stream():
        model.reset_states()
        for f in tensor_frames:
            model(f, SAMPLE_RATE)

    timing = bench(stream, runs=3)
    frame_ms = 512 / SAMPLE_RATE * 1000
    print(f"streaming: p50={timing['p50_ms']:.0f} ms for {len(frames)} frames "
          f"({timing['p50_ms'] / len(frames):.2f} ms per {frame_ms:.0f} ms frame) "
          f"-> {timing['p50_ms'] / (len(frames) * frame_ms) * 100:.1f}% of real time")

    results["vad"] = {
        "clip": clip.name,
        "batch_ms": round(wall * 1000, 1),
        "rtf": round(wall / (len(wav) / SAMPLE_RATE), 4),
        "stream_p50_ms": timing["p50_ms"],
        "stream_pct_realtime": round(timing["p50_ms"] / (len(frames) * frame_ms) * 100, 1),
        "segments": len(stamps),
    }


# ------------------------------------------------------------------- stage 3
def run_asr(results: dict, model_name: str = "small", file: str | None = None) -> None:
    from faster_whisper import WhisperModel

    clip = Path(file) if file else pick_long_clip()
    print(f"\n=== FASTER-WHISPER ({model_name}, int8, cpu) ====================")
    print(f"clip: {clip.name}  {len(load_wav(clip)) / SAMPLE_RATE:.2f}s")

    t0 = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print(f"model load: {(time.perf_counter() - t0) * 1000:.0f} ms")

    def transcribe():
        segments, _ = model.transcribe(
            str(clip), language="en", beam_size=1, vad_filter=False
        )
        return " ".join(s.text.strip() for s in segments)

    t0 = time.perf_counter()
    text = transcribe()
    first = time.perf_counter() - t0
    audio_s = len(load_wav(clip)) / SAMPLE_RATE
    print(f"\ntranscript: {text[:400]}{'...' if len(text) > 400 else ''}")
    print(f"\nfirst pass: {first * 1000:.0f} ms  (RTF {first / audio_s:.3f})")

    timing = bench(transcribe, runs=3)
    rtf = timing["p50_ms"] / 1000 / audio_s
    print(f"warm:       p50={timing['p50_ms']:.0f} ms  RTF {rtf:.3f}  "
          f"({1 / rtf:.1f}x faster than real time)")

    results.setdefault("asr", {})[model_name] = {
        "clip": clip.name,
        "audio_s": round(audio_s, 2),
        "p50_ms": timing["p50_ms"],
        "rtf": round(rtf, 3),
        "transcript": text,
    }


def pick_long_clip() -> Path:
    """Prefer samples/latency/*.wav so long clips don't pollute the speaker matrix."""
    lat = SAMPLES / "latency"
    if lat.is_dir():
        long_clips = sorted(lat.glob("*.wav"))
        if long_clips:
            return max(long_clips, key=lambda p: len(load_wav(p)))
    clips = find_clips(SAMPLES)
    return max(clips, key=lambda p: len(load_wav(p)))


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("embed", help="speaker embeddings + cosine similarities")
    p.add_argument("--backend", choices=["ecapa", "resemblyzer", "both"], default="both")

    p = sub.add_parser("vad", help="Silero VAD latency")
    p.add_argument("--file", help="60s clip to use (default: longest in samples/)")

    p = sub.add_parser("asr", help="faster-whisper latency")
    p.add_argument("--model", default="small", help="tiny|base|small|medium")
    p.add_argument("--file", help="clip to transcribe (default: longest in samples/)")

    sub.add_parser("all", help="run every stage")

    args = ap.parse_args()
    cmd = args.cmd or "embed"  # bare `python spike.py` runs the similarity table
    MODELS.mkdir(exist_ok=True)
    results: dict = {}
    if RESULTS.exists():  # merge so running stages separately doesn't erase earlier numbers
        try:
            results = json.loads(RESULTS.read_text())
        except json.JSONDecodeError:
            results = {}
    results["meta"] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    if cmd == "embed":
        backend = getattr(args, "backend", "both")
        names = ["ecapa", "resemblyzer"] if backend == "both" else [backend]
        run_embed(names, results)
    elif cmd == "vad":
        run_vad(results, getattr(args, "file", None))
    elif cmd == "asr":
        run_asr(results, getattr(args, "model", "small"), getattr(args, "file", None))
    else:
        run_embed(["ecapa", "resemblyzer"], results)
        run_vad(results)
        run_asr(results, "small")

    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
