# voicebot-proto — stack spike

Half-day spike to pick the local voice stack **before** building the product:

| Stage | Candidates | What we need to know |
|---|---|---|
| Speaker verification | SpeechBrain ECAPA-TDNN vs Resemblyzer | does owner–owner score clearly above owner–other, and how expensive is an embed? |
| VAD | Silero VAD | runs faster than real time on this Mac? |
| Speech-to-text | faster-whisper (base / small, int8) | real-time factor on this Mac? |
| Product shape | always-listening + speaker gate vs wake word | which to build first |

Everything runs **locally on CPU**. No API keys are required for the spike.

---

## 0. What is already set up

- `python@3.11` and `ffmpeg` installed via Homebrew
- `.venv/` created with `torch`, `speechbrain`, `resemblyzer`, `silero-vad`, `faster-whisper`
  (already done — you do not need to reinstall)
- `.env` created from `.env.example` (placeholders only; git-ignored)
- `spike.py` — the measurement script
- `scripts/make_fixtures.sh` — synthetic clips so you can run things right now
- `scripts/prepare_clips.sh` — converts your real recordings to 16 kHz mono wav

## 1. Activate the environment

```bash
cd "/Users/adnanbashir/Documents/Default Project/voicebot-proto"
source .venv/bin/activate
python --version        # should say 3.11.x
```

Every command below assumes this is active. To leave it: `deactivate`.

## 2. Smoke-test with synthetic voices (2 min)

This proves the install works before you spend time recording.

```bash
bash scripts/make_fixtures.sh      # writes samples/owner_*.wav, other_*.wav, samples/latency/long.wav
python spike.py                    # bare command = similarity table, both speaker models
python spike.py embed --backend both   # same thing, explicit
```

You should see a 5×5 cosine matrix and a `PASS` margin line. **The synthetic clips are
text-to-speech voices, not real people** — they only prove the plumbing. Do not put these
numbers in the decision note.

### Reference output on this machine (TTS voices — plumbing only)

Measured while setting this up (Apple M1, 16 GB, CPU-only, Python 3.11, torch 2.15):

| Check | ECAPA-TDNN | Resemblyzer |
|---|---|---|
| model load (warm) | ~1–2 s (20.8 s first run, includes download) | ~12 ms |
| embed, 5.3 s clip (p50) | 49 ms | 53 ms |
| owner–owner min | 0.854 | 0.934 |
| owner–other max | 0.336 | 0.822 |
| **margin** | **+0.518** | **+0.112** |

| Stage | Model | p50 | Real time? |
|---|---|---|---|
| VAD | Silero | 0.14 ms per 32 ms frame (0.4% of real time) | yes, by 200x |
| VAD | Silero, whole 36.6 s clip | 201 ms (RTF 0.0055) | yes |
| ASR | faster-whisper base int8, 5.3 s clip | 547 ms (RTF 0.10) | yes |
| ASR | faster-whisper small int8, 5.3 s clip | 1.19 s (RTF 0.22) | yes |
| ASR | faster-whisper small int8, 36.6 s clip | 3.7 s (RTF 0.10) | yes |

Two things this already tells us:

1. **ECAPA separates speakers much more cleanly** (margin 0.518 vs 0.112) at the same-ish
   embedding cost. That is the expected result and it survives into the decision.
2. **RTF is worse on short clips than on long ones** (small: 0.22 vs 0.10) because of fixed
   per-call overhead. Measure ASR on an utterance-length clip, not just on the 60 s clip:

   ```bash
   python spike.py asr --model small --file samples/owner_2.wav
   ```

## 3. Record the real sample clips (20 min)

First, get the synthetic fixtures out of the way so they don't end up in your real results:

```bash
mkdir -p samples/_synthetic && mv samples/*.wav samples/_synthetic/
```

You need 5 short clips plus 1 long one:

| File | Who | Length | Content |
|---|---|---|---|
| `owner_1.wav` | you | 5–10 s | e.g. "Hey, add milk to my shopping list" |
| `owner_2.wav` | you | 5–10 s | different sentence, **same room/mic** |
| `owner_3.wav` | you | 5–10 s | different sentence again |
| `other_1.wav` | someone else | 5–10 s | same kind of sentence |
| `other_2.wav` | a third person | 5–10 s | same kind of sentence |
| `samples/latency/long.wav` | you | 30–60 s | continuous speech with natural pauses |

Rules that matter more than the model choice:

1. **Same microphone and room for all five clips.** Mixing an iPhone clip with a laptop-mic
   clip makes speakers look artificially different and ruins the comparison.
2. **Trim to speech** — no long silence before/after.
3. **Ordinary speaking voice** at normal distance. Don't whisper or shout; note if a clip is
   noisy.
4. Record with QuickTime (File → New Audio Recording), Voice Memos, or `ffmpeg`.

Then convert everything to the expected format:

```bash
# pick a folder that contains your raw files, named owner_1.m4a, other_1.m4a, ...
bash scripts/prepare_clips.sh ~/Desktop/recordings
ls -la samples/
```

Each clip should be 5–10 s. Check with:

```bash
for f in samples/*.wav; do echo -n "$f "; ffprobe -v error -show_entries format=duration -of csv=p=0 "$f"; done
```

## 4. Run the speaker comparison

```bash
python spike.py embed --backend ecapa
python spike.py embed --backend resemblyzer
# or both at once:
python spike.py embed --backend both
```

Read three things:

1. **The margin** — `min(owner–owner) − max(owner–other)`. Positive means the model separates
   your voice from the others on this tiny sample.
2. **The threshold suggestion** — the midpoint of that gap. That is a starting value, not a
   calibrated threshold.
3. **`p50_ms` embedding time** — the cost of gating every utterance.

Reference expectations (real voices, same mic): owner–owner ≈ 0.6–0.85, owner–other ≈ 0.1–0.4.
If your margin is barely positive, the usual cause is different mics or clips that are too short.

## 5. Run VAD + speech-to-text latency

```bash
python spike.py vad
python spike.py asr --model base
python spike.py asr --model small
```

What to record:

- **VAD**: `% of real time` — should be far below 100%.
- **ASR**: `RTF` (real-time factor). RTF < 1 means faster than real time. Also read the
  transcript and check it looks sane.

Model load time is printed separately on purpose — cold start is not latency.

## 6. Optional: everything in one go

```bash
python spike.py all
```

Results are also written to `results.json` (git-ignored) so you can paste exact numbers
into the decision note.

---

## Decision note (fill this in)

> **Date:** …  **Machine:** Apple M1, 16 GB, macOS 26.6, CPU-only

| Stage | Chosen | Why | Measured p50 |
|---|---|---|---|
| VAD | Silero VAD | | ~… ms/frame (…% of real time) |
| Speaker embedding | ECAPA / Resemblyzer | | … ms per 5–10 s clip |
| Speech-to-text | faster-whisper … int8 | | RTF … |
| End-to-end (gate + ASR) | — | | ~… s per utterance |

**Speaker model:** … wins. Owner-owner margin was … (min owner-owner …, max owner-other …).
ECAPA is heavier (SpeechBrain + torch, ~… ms model load, ~… MB) but …; Resemblyzer is lighter
but older, less accurate on short clips, and needs old numpy pins.

**Shape:** start with **always-listening + speaker gate**, add the wake word later as an
optional stage between VAD and the gate. Rationale: the gate already gives "only respond to
the owner"; adding a wake word on top adds a second thing that can wrongly reject an utterance
before we know the gate's real false-reject rate. The pipeline is ordered VAD → embed → ASR,
so the wake word can be inserted after VAD without restructuring.

**Latency budget per utterance:** VAD … ms + embed … ms + ASR … ms ≈ … s, which is … for a
conversational response.

**Open risks:** only 5 clips (no threshold calibration); single mic; no background-noise
condition; ECAPA's dependency tree; wake word deferred.

---

## Troubleshooting

- **`No .wav files in samples/`** — run `bash scripts/make_fixtures.sh`, or drop your real
  clips in `samples/`.
- **ECAPA downloads on first run** (~80 MB) into `models/`. That's normal, and it isn't
  counted in the latency numbers.
- **Resemblyzer fails to import with `No module named 'pkg_resources'`** — fixed already by
  pinning `setuptools<81` (its `webrtcvad` dependency imports a module that newer setuptools
  removed). This is exactly the kind of rot that makes Resemblyzer a maintenance risk.
- **Resemblyzer fails in other ways** — it is an old package (2020). `numpy<2` is already
  pinned. If it still breaks, note it as a maintenance risk and move on — that alone can
  justify choosing ECAPA.
- **`say: voice not found`** in the fixture script — run `say -v '?'` and edit the voice lists
  at the top of `scripts/make_fixtures.sh`.
- **Numbers look too good** — check that all clips really come from the same microphone and
  that `other_*` clips are actually different people.
