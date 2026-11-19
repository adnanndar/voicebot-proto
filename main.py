#!/usr/bin/env python3
"""voicebot-proto — local voice assistant (tickets 2–5).

Usage:
    python main.py run               # live mic -> VAD -> gate -> ASR -> brain -> TTS
    python main.py run --file x.wav  # run the same pipeline over a recording
    python main.py enrol             # record prompts and save a voiceprint
    python main.py add-samples       # strengthen an existing voiceprint
    python main.py status            # show voiceprint + thresholds
    python main.py delete            # remove the voiceprint
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app.audio.pipeline import AudioPipeline, iter_utterances
from app.bot.engine import Engine
from app.config import DATA_DIR, load_config
from app.enrol.prompts import pick_prompts
from app.enrol.validate import validate
from app.enrol.voiceprint import (
    Voiceprint, add_samples, apply_thresholds, enrol, owner_name, parse_spoken_name,
)


# ---------------------------------------------------------------------- utils
def load_wav_mono(path: Path, sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def record_clip(seconds: float, cfg) -> np.ndarray:
    import sounddevice as sd

    sr = cfg.audio.sample_rate
    print(f"  ● recording {seconds:.0f}s…", end="", flush=True)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print(" done")
    return np.ascontiguousarray(np.squeeze(audio), dtype=np.float32)


def ask_name_voice(cfg) -> str:
    """The first enrolment step: ask 'What is your name?' out loud."""
    from app.bot.asr import ASR
    from app.bot.tts import make_tts

    tts = make_tts(cfg)
    asr = ASR(cfg)
    while True:
        print("\nFirst question (spoken aloud): “What is your name?”")
        tts.speak("What is your name?")
        tts.wait()
        input("  press Enter, then say your name…")
        audio = record_clip(4.0, cfg)
        text, _ = asr.transcribe(audio)
        print(f"  heard: {text or '(nothing)'}")
        name = parse_spoken_name(text)
        if name:
            print(f"  ✓ name: {name}")
            return name
        print("  ✗ I didn't catch a name — let's try again.")


# ---------------------------------------------------------------------- run
def run(cfg, args) -> None:
    if args.tts:
        cfg.bot.tts = args.tts
    engine = Engine(cfg)
    engine.on_state = lambda s: None  # UI hooks here later
    _print_status(cfg)

    if args.file:
        wav = load_wav_mono(Path(args.file), cfg.audio.sample_rate)
        print(f"file: {args.file}  ({len(wav) / cfg.audio.sample_rate:.1f}s)\n")
        utts = iter_utterances(wav, cfg)
        for i, u in enumerate(utts, 1):
            print(f"[{i}] utterance {u.start_s:.2f}s–{u.end_s:.2f}s ({u.duration_s:.2f}s)")
            d = engine.handle(u)
            score = "n/a" if d.score is None else f"{d.score:.3f}"
            print(f"    decision: {d.decision} (score {score})")
            if d.transcript:
                print(f"    heard: {d.transcript}")
        print("\ndone.")
        return

    pipeline = AudioPipeline(cfg, engine.on_utterance)
    engine.pipeline = pipeline
    pipeline.start()
    print("Listening…  (speak, then Ctrl+C to stop)\n")
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        engine.stop()
        pipeline.stop()


# ------------------------------------------------------------------- enrol
def _collect_recordings(cfg, prompts: list[dict], seconds: float) -> list[np.ndarray]:
    recordings = []
    for i, p in enumerate(prompts, 1):
        text = p["text"]
        tone = p.get("tone")
        hint = f"   [{tone} tone]" if tone else ""
        for attempt in range(3):
            print(f"\n[{i}/{len(prompts)}] “{text}”{hint}")
            input("  press Enter, then read the prompt aloud…")
            audio = record_clip(seconds, cfg)
            ok, reason = validate(audio, cfg.audio.sample_rate, cfg)
            if ok:
                recordings.append(audio)
                print(f"  ✓ ({len(audio) / cfg.audio.sample_rate:.1f}s)")
                break
            print(f"  ✗ {reason} — try again ({attempt + 1}/3)")
    return recordings


def enrol_cli(cfg, args) -> None:
    name = ask_name_voice(cfg)
    prompts = pick_prompts(cfg)
    if not prompts:
        print("no prompts found")
        return
    seconds = args.seconds
    recordings = _collect_recordings(cfg, prompts, seconds)
    if len(recordings) < 3:
        print(f"\nonly {len(recordings)} good recording(s) — aborting (need >= 3).")
        return
    vec, meta = enrol(recordings, cfg.speaker.model, cfg)
    meta["owner_name"] = name
    Voiceprint(DATA_DIR, cfg.speaker.model, encrypt=cfg.privacy.encrypt).save(vec, meta)
    print(f"\nsaved voiceprint for {name}: {DATA_DIR / 'voiceprint.npy'}")
    print(f"  samples={meta['num_samples']}  dropped={meta['num_dropped']}  "
          f"self-check mean={meta['self_check']['mean']} min={meta['self_check']['min']}")


def add_samples_cli(cfg, args) -> None:
    vp = Voiceprint(DATA_DIR, cfg.speaker.model, encrypt=cfg.privacy.encrypt)
    vec, meta = vp.load()
    if vec is None:
        print("no voiceprint to add to — run `python main.py enrol` first")
        return
    prompts = pick_prompts(cfg)
    recordings = _collect_recordings(cfg, prompts, args.seconds)
    if not recordings:
        print("no good recordings")
        return
    try:
        vec, meta = add_samples(vec, meta or {"num_samples": 0}, recordings,
                                cfg.speaker.model, cfg)
    except ValueError as e:
        print(f"\nnot added: {e}")
        return
    name = owner_name(meta, "")
    if name:
        meta["owner_name"] = name
    vp.save(vec, meta)
    print(f"\nupdated voiceprint: samples={meta['num_samples']}  "
          f"self-check mean={meta['self_check']['mean']}")


# ------------------------------------------------------------------ status
def _print_status(cfg) -> None:
    vp = Voiceprint(DATA_DIR, cfg.speaker.model, encrypt=cfg.privacy.encrypt)
    vec, meta = vp.load()
    apply_thresholds(cfg, meta)
    print("=" * 60)
    print(f"speaker model : {cfg.speaker.model}")
    print(f"thresholds    : accept {cfg.gate.accept_threshold} / reject {cfg.gate.reject_threshold}"
          f"  (owner rule: >= {int(cfg.gate.accept_threshold * 100)}% similarity)")
    print(f"voiceprint    : {'ENROLLED' if vec is not None else 'MISSING'}")
    if meta:
        print(f"  owner={owner_name(meta)}  samples={meta.get('num_samples')}  date={meta.get('date')}  "
              f"self-check mean={meta['self_check'].get('mean')} min={meta['self_check'].get('min')}")
    print(f"asr/tts/brain : {cfg.bot.asr_model} / {cfg.bot.tts} / {cfg.bot.brain}")
    print("=" * 60)


def status_cli(cfg, args) -> None:
    _print_status(cfg)


def delete_cli(cfg, args) -> None:
    vp = Voiceprint(DATA_DIR, cfg.speaker.model, encrypt=cfg.privacy.encrypt)
    if not vp.exists():
        print("no voiceprint to delete")
        return
    if input("delete the voiceprint and saved audio? [y/N] ").strip().lower() != "y":
        print("cancelled")
        return
    vp.delete()
    print("deleted")


def restore_cli(cfg, args) -> None:
    vp = Voiceprint(DATA_DIR, cfg.speaker.model, encrypt=cfg.privacy.encrypt)
    if not vp.has_original():
        print("no frozen original to restore")
        return
    if vp.restore_original():
        print("restored the original enrolment voiceprint")
    else:
        print("restore failed")


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")

    p = sub.add_parser("run", help="live mic (or --file) pipeline")
    p.add_argument("--file", help="run over a WAV file instead of the mic")
    p.add_argument("--tts", choices=["say", "print", "null"], help="override tts backend")
    p.add_argument("--config", help="path to config.yaml")

    p = sub.add_parser("enrol", help="record prompts and save a voiceprint")
    p.add_argument("--seconds", type=float, default=6.0, help="record length per prompt")
    p.add_argument("--config", help="path to config.yaml")

    p = sub.add_parser("add-samples", help="strengthen an existing voiceprint")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--config", help="path to config.yaml")

    sub.add_parser("status", help="show voiceprint + thresholds")
    p = sub.add_parser("delete", help="remove the voiceprint")
    p.add_argument("--config", help="path to config.yaml")
    sub.add_parser("restore", help="undo adaptive drift (restore original voiceprint)")

    args = ap.parse_args()
    cfg = load_config(args.config if hasattr(args, "config") else None)
    DATA_DIR.mkdir(exist_ok=True)

    handlers = {
        "run": run,
        "enrol": enrol_cli,
        "add-samples": add_samples_cli,
        "status": status_cli,
        "delete": delete_cli,
        "restore": restore_cli,
    }
    cmd = args.command or "run"
    handlers[cmd](cfg, args)


if __name__ == "__main__":
    main()
