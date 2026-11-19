#!/usr/bin/env python3
"""Ticket 6 — pick thresholds from labelled data, not guesses.

Segments every clip into utterances, scores each against the voiceprint
(single and sliding-window), and reports ROC / EER / FAR / FRR.

Usage:
    python eval.py --enrol eval/enrol --owner eval/owner --other eval/other
    python eval.py --enrol eval/enrol --owner eval/owner --other eval/other --background eval/bg
    python eval.py --voiceprint data/voiceprint.npy --owner eval/owner --other eval/other

Each directory holds .wav clips. `--enrol` builds the voiceprint; `--owner` and
`--other` clips are segmented and scored (owner = should accept, other = should
reject). With no args it falls back to the synthetic fixtures under samples/
(plumbing only — do not trust those numbers).

Outputs: eval_scores.csv (per-utterance), a summary table, and eval_roc.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from app import speaker
from app.audio.pipeline import iter_utterances
from app.config import DATA_DIR, load_config
from app.enrol.voiceprint import Voiceprint, enrol
from main import load_wav_mono


# ------------------------------------------------------------------ data
def wavs(path: Path) -> list[Path]:
    return sorted(path.glob("*.wav")) if path.is_dir() else []


def owner_clips_of(path: Path) -> list[Path]:
    return sorted(path.glob("owner_*.wav")) if path.is_dir() else []


def other_clips_of(path: Path) -> list[Path]:
    return sorted(path.glob("other_*.wav")) if path.is_dir() else []


def collect_clips(clips: list[Path], cfg) -> list[tuple[str, object]]:
    """Return [(name, Utterance), ...] by segmenting each clip."""
    out = []
    for c in clips:
        wav = load_wav_mono(c, cfg.audio.sample_rate)
        for u in iter_utterances(wav, cfg):
            out.append((c.name, u))
    return out


# ------------------------------------------------------------------ scoring
def score_single(audio: np.ndarray, vec: np.ndarray, model: str) -> float:
    emb = speaker.embed(model, audio)
    return float(np.dot(emb, vec))


def score_windowed(audio: np.ndarray, vec: np.ndarray, model: str, cfg) -> float:
    win = int(cfg.gate.window_sec * cfg.audio.sample_rate)
    hop = int(cfg.gate.hop_sec * cfg.audio.sample_rate)
    if len(audio) <= win:
        return score_single(audio, vec, model)
    scores = [score_single(audio[s:s + win], vec, model)
              for s in range(0, len(audio) - win + 1, hop)]
    return float(np.median(scores))


# ------------------------------------------------------------------ metrics
def metrics(pos: list[float], neg: list[float], accept: float = 0.45) -> dict:
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return {}

    lo = min(pos.min(), neg.min()) - 0.1
    hi = max(pos.max(), neg.max()) + 0.1
    ts = np.linspace(lo, hi, 2000)
    far = np.array([(neg >= t).mean() for t in ts])   # other accepted
    frr = np.array([(pos < t).mean() for t in ts])    # owner rejected

    # equal error rate
    i = int(np.abs(far - frr).argmin())
    eer = float((far[i] + frr[i]) / 2)

    # best threshold with FAR <= 2% (then report its FRR)
    best = None
    idx = np.where(far <= 0.02)[0]
    if len(idx):
        j = idx[int(frr[idx].argmin())]
        best = {"threshold": float(ts[j]), "far": float(far[j]), "frr": float(frr[j])}

    def at(t):
        return {"threshold": t, "far": float((neg >= t).mean()), "frr": float((pos < t).mean())}

    return {
        "n_owner": int(len(pos)), "n_other": int(len(neg)),
        "eer": eer, "best_far2": best, "at_accept": at(accept),
        "roc": {"t": ts.tolist(), "far": far.tolist(), "frr": frr.tolist()},
    }


def as_norm(scores: dict, cohort: list[float]):
    """AS-norm: (score - cohort_mean) / cohort_std, using an impostor cohort."""
    c = np.asarray(cohort, float)
    if len(c) < 2:
        return None
    mu, sd = float(c.mean()), float(c.std() + 1e-9)
    return {k: (float(v) - mu) / sd for k, v in scores.items()}


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enrol", help="dir of owner clips to build the voiceprint from")
    ap.add_argument("--voiceprint", help="path to an existing voiceprint.npy")
    ap.add_argument("--owner", help="dir of owner test clips (or samples/)")
    ap.add_argument("--other", help="dir of other test clips (or samples/)")
    ap.add_argument("--background", help="dir of background clips (reported separately)")
    ap.add_argument("--cohort", help="dir of impostor clips for AS-norm (optional)")
    ap.add_argument("--model", default=None, help="speaker model (default: config)")
    ap.add_argument("--accept", type=float, default=0.45)
    args = ap.parse_args()

    cfg = load_config()
    model = args.model or cfg.speaker.model
    sr = cfg.audio.sample_rate

    # voiceprint
    vp = Voiceprint(DATA_DIR, model, encrypt=cfg.privacy.encrypt)
    owner_clips = owner_clips_of(Path(args.owner or "samples"))
    other_clips = other_clips_of(Path(args.other or "samples"))

    if args.enrol:
        clips = wavs(Path(args.enrol))
        vec, _ = enrol([load_wav_mono(c, sr) for c in clips], model, cfg)
        print(f"voiceprint: built from {len(clips)} enrolment clips ({args.enrol})")
    elif args.voiceprint:
        vec = np.load(args.voiceprint).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        print(f"voiceprint: loaded {args.voiceprint}")
    elif vp.exists():
        vec, meta = vp.load()
        print(f"voiceprint: loaded from data/ (samples={meta and meta.get('num_samples')})")
    else:
        if len(owner_clips) >= 2:
            held_out = owner_clips[-1]
            n_enrol = len(owner_clips) - 1
            vec, _ = enrol([load_wav_mono(c, sr) for c in owner_clips[:-1]], model, cfg)
            owner_clips = [held_out]  # test only on the held-out clip
            print(f"voiceprint: leave-one-out enrol on {n_enrol} owner clips")
        elif owner_clips:
            vec, _ = enrol([load_wav_mono(c, sr) for c in owner_clips], model, cfg)
            print("warning: enrol == test (circular) — scores are inflated")
        else:
            sys.exit("no voiceprint and no clips found; run `python main.py enrol` or pass --enrol")

    bg_clips = wavs(Path(args.background)) if args.background else []
    print(f"test set: {len(owner_clips)} owner, {len(other_clips)} other, {len(bg_clips)} background clips")

    owner = collect_clips(owner_clips, cfg)
    other = collect_clips(other_clips, cfg)
    bg = collect_clips(bg_clips, cfg)

    # score everything
    rows = []
    pos_s, neg_s = [], []
    pos_w, neg_w = [], []
    bg_s = []

    for name, u in owner:
        s = score_single(u.audio, vec, model)
        w = score_windowed(u.audio, vec, model, cfg)
        pos_s.append(s); pos_w.append(w)
        rows.append({"file": name, "label": "owner", "dur_s": round(u.duration_s, 2), "score_single": round(s, 4), "score_windowed": round(w, 4)})
    for name, u in other:
        s = score_single(u.audio, vec, model)
        w = score_windowed(u.audio, vec, model, cfg)
        neg_s.append(s); neg_w.append(w)
        rows.append({"file": name, "label": "other", "dur_s": round(u.duration_s, 2), "score_single": round(s, 4), "score_windowed": round(w, 4)})
    for name, u in bg:
        s = score_single(u.audio, vec, model)
        bg_s.append(s)
        rows.append({"file": name, "label": "background", "dur_s": round(u.duration_s, 2), "score_single": round(s, 4), "score_windowed": round(score_windowed(u.audio, vec, model, cfg), 4)})

    # CSV
    import csv
    with open("eval_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "dur_s", "score_single", "score_windowed"])
        w.writeheader(); w.writerows(rows)

    # cohort (AS-norm)
    cohort = None
    if args.cohort:
        cohort = [score_single(u.audio, vec, model) for _, u in collect_clips(wavs(Path(args.cohort)), cfg)]
        print(f"cohort for AS-norm: {len(cohort)} impostor scores")

    # metrics
    ms = metrics(pos_s, neg_s, args.accept)
    mw = metrics(pos_w, neg_w, args.accept)
    m_bg = None
    if bg_s:
        # background acceptance rate at accept threshold
        m_bg = {"n": len(bg_s), "accept_rate": float(np.mean(np.array(bg_s) >= args.accept))}

    print("\n" + "=" * 72)
    print(f"{'method':<16} {'n_owner':>7} {'n_other':>7} {'EER':>7} {'acc@FAR≤2%':>10} {'FAR':>7} {'FRR':>7}")
    print("-" * 72)
    def line(name, m):
        if not m:
            return
        b = m.get("best_far2")
        print(f"{name:<16} {m['n_owner']:>7} {m['n_other']:>7} {m['eer']:>7.3f} "
              f"{b['threshold'] if b else '—':>10} {b['far'] if b else 0:>7.3f} {b['frr'] if b else 0:>7.3f}")
    line("single", ms)
    line("windowed", mw)
    print("-" * 72)
    for name, m in (("single", ms), ("windowed", mw)):
        if m:
            a = m["at_accept"]
            print(f"{name}: @ accept={args.accept}  FAR={a['far']:.3f}  FRR={a['frr']:.3f}   (EER {m['eer']:.3f})")
    if m_bg:
        print(f"background: {m_bg['n']} clips, {m_bg['accept_rate']*100:.1f}% would be accepted @ accept={args.accept}")
    print("=" * 72)

    # AS-norm comparison
    if cohort:
        print("\nAS-norm (score -> (score - cohort_mean)/cohort_std):")
        z_pos = as_norm({i: v for i, v in enumerate(pos_s)}, cohort)
        z_neg = as_norm({i: v for i, v in enumerate(neg_s)}, cohort)
        mz = metrics(list(z_pos.values()), list(z_neg.values()), args.accept)
        line("asnorm-single", mz)
        if mz:
            b = mz["best_far2"]
            print(f"  -> best threshold {b['threshold']:.2f} (z-score) gives FAR {b['far']:.3f}, FRR {b['frr']:.3f}")

    # ROC plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, m in (("single", ms), ("windowed", mw)):
            if m and m["roc"]:
                r = m["roc"]
                plt.plot(r["far"], 1 - np.array(r["frr"]), label=f"{name} (EER {m['eer']:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
        plt.xlabel("FAR (false accept rate)"); plt.ylabel("TPR (1 - false reject rate)")
        plt.title("Speaker gate ROC"); plt.legend(); plt.grid(alpha=0.3)
        plt.savefig("eval_roc.png", dpi=110); plt.close()
        print("\nwrote eval_roc.png and eval_scores.csv")
    except Exception as e:
        print(f"\n(no plot: {e}) wrote eval_scores.csv")


if __name__ == "__main__":
    main()
