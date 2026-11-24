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

**Shape:** always-listening pipeline with a **speaker gate** and a **wake word**. The gate
runs first, so a non-owner saying “Hey Lyra” is still ignored. The wake word is required only
when no conversation is open; after the owner wakes Lyra, follow-ups need no wake word until
“stop” or the idle timeout.

**Latency budget per utterance:** VAD … ms + embed … ms + ASR … ms ≈ … s, which is … for a
conversational response.

**Open risks:** only 5 clips (no threshold calibration); single mic; no background-noise
condition; ECAPA's dependency tree.

---

## 8. Voice assistant (tickets 2–5)

The runnable prototype built on the spike's choices. Local by default; a hosted Claude or
OpenAI key is optional for stronger answers.

```
main.py            CLI entry point (run / enrol / add-samples / status / delete)
app/audio/         capture + noise suppression + Silero VAD segmenter (ticket 2)
app/enrol/         prompts, quality checks, voiceprint build/save/load (ticket 3)
app/gate/          owner / not-owner decision + logging (ticket 4)
app/bot/           ASR -> brain (local model) -> TTS, state machine (ticket 5)
config.yaml        thresholds and settings
data/voiceprint.npy  the enrolled voiceprint (git-ignored)
debug/             saved utterances + decisions.jsonl (git-ignored)
```

### Setup (one-time)

```bash
source .venv/bin/activate
pip install sounddevice noisereduce fastapi "uvicorn[standard]" websockets
pip install mlx-lm              # local answer model (Apple Silicon)
brew install portaudio          # already done if this repo was set up
```

The first start downloads the local model
(`mlx-community/Qwen2.5-1.5B-Instruct-4bit`, ~1 GB) into the Hugging Face cache.

### Answer brain (best quality first)

`bot.brain: auto` uses the best backend available:

1. **Claude** when `ANTHROPIC_API_KEY` is set (default model `claude-sonnet-5-5`)
2. **OpenAI-compatible** when `OPENAI_API_KEY` is set (OpenAI, Groq, or any endpoint)
3. **Local MLX model** on this Mac (~1 GB, no network)
4. Built-in rules: time, date, live weather, arithmetic

Paste your keys into `.env` (all optional — any combination works):

```bash
# OpenAI — https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...
# optional: OPENAI_MODEL=gpt-4o-mini
# optional: OPENAI_BASE_URL=https://api.openai.com/v1

# ...or a FREE Groq key — https://console.groq.com/keys (OpenAI-compatible)
OPENAI_API_KEY=gsk-...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-120b       # or openai/gpt-oss-20b / qwen/qwen3.8-27b

# Anthropic — https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-...
# optional: ANTHROPIC_MODEL=claude-sonnet-5-5
```

When both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are set, Claude is tried first,
then the OpenAI-compatible endpoint, then the local model. A failed or exhausted
provider backs off for 90 s and the next one answers, so a missing key never
leaves Lyra silent.

When a hosted key is configured, transcripts leave the Mac for that provider.
With the local model, they never do.

### Enrol your voice, then run

```bash
python main.py enrol            # read 10 prompts (~2 min), saves data/voiceprint.npy
python main.py status           # voiceprint + thresholds
python main.py run              # live: mic -> VAD -> gate -> ASR -> brain -> say
python main.py run --tts print  # same, but prints replies instead of speaking
python main.py add-samples      # strengthen the voiceprint
python main.py delete           # remove it
```

Pipe a recording through the same pipeline (handy for testing):

```bash
python main.py run --file samples/owner_2.wav --tts print
```

### How each ticket landed

- **Ticket 2** — `app/audio/`: 16 kHz mono capture via PortAudio in 32 ms blocks; `noisereduce`
  as a stand-in for WebRTC/RNNoise; a streaming Silero state machine (300 ms speech to start,
  2 s silence to end, 30 s backstop, 600 ms pre-roll). Echo handling = mute the mic while the bot
  speaks. `debug.save_utterances` writes each segment to `debug/`.
- **Ticket 3** — `app/enrol/`: 10 prompts (a couple with a louder/quieter delivery), per-take
  checks (≥1.5 s speech, no clipping, level, SNR), outlier drop, mean embedding L2-normalised,
  self-check scores, saved as `voiceprint.npy` + `meta.json`. `enrol` / `add-samples` / `delete`.
- **Ticket 4** — `app/gate/`: cosine vs voiceprint with one strict rule — score >= 35% is the
  owner, anything lower is rejected. The rule applies to every utterance, including follow-ups;
  there is no conversation-window exception. Very short clips (< 0.5 s) are ignored because their
  scores are unreliable. Every decision is appended to `debug/decisions.jsonl`.
- **Ticket 5** — `app/bot/`: faster-whisper (small, English only) only on owner utterances;
  the brain prefers Claude when `ANTHROPIC_API_KEY` is set, then OpenAI, then the local MLX
  model (Qwen2.5 1.5B 4-bit on this Mac), then built-in rules (time, date, live weather,
  arithmetic). Backends are tried in order, so a question is never met with an echo. The name
  entered at enrolment is stored in the encrypted voiceprint metadata, used by the prompts and
  greetings, and shown in the UI. TTS is macOS `say`; states `idle → listening → verifying →
  thinking → speaking`; conversation window; barge-in behind `bot.barge_in` (owner-only, but
  without real AEC the bot can hear itself — leave off for now).

Measured on this machine (Apple M1): gate ~150 ms/utterance, ASR RTF ~0.1–0.22, VAD
~0.4% of real time, local answers 0.3–2.2 s — within the tickets' targets.

---

## 9. Demo UI (ticket 7)

A FastAPI + WebSocket screen to run the prototype and see why it answered or ignored
each utterance.

```bash
python server.py            # http://127.0.0.1:8765
```

```
server.py       FastAPI app: owns the live pipeline, streams state over /ws
ui/             the page (index.html / style.css / app.js)
```

- **Enrol my voice** — asks **“What is your name?” out loud first**, then steps through the
  10 prompts with a record button, a live level meter, ✓/retry per take (with a reason), and
  a progress bar. Each prompt is a single line whose words turn green/red. "Add samples"
  merges only takes that clearly match the enrolled voice; "Delete my voice" removes it.
- **Start / Stop listening** — runs mic → VAD → gate → wake word → ASR → brain → TTS on the
  server. While sleeping, only an owner utterance starting with **“Hey Lyra”** or **“Lyra”**
  wakes it, and the reply greets the owner by name.
- **Live panel** — mic level, a "speech" dot, the owner score as a bar with the fixed 35%
  owner rule marked, and status `Idle / Listening / Verifying / Ignored / Thinking / Speaking`.
- **Conversation log** — owner turns and bot replies; non-owner utterances show greyed out
  as `ignored (score 0.21)` with no transcript, and repeated attempts show the pause.
- **Settings** — idle timeout (default **2 minutes**), auto-sleep, and debug "save utterances".

Enrolment and live listening both use the server's microphone, so the takes match exactly.

---

## 10. Evaluation (ticket 6)

`eval.py` scores labelled clips through the gate and reports ROC / EER / FAR / FRR.

```bash
python eval.py --enrol eval/enrol --owner eval/owner --other eval/other --background eval/bg
```

- Each directory holds `.wav` clips. `--enrol` builds the voiceprint; `--owner` (should
  accept) and `--other` (should reject) clips are segmented into utterances and scored.
- Outputs `eval_scores.csv` (per-utterance scores) and `eval_roc.png` (ROC curve), plus a
  summary: EER, and the threshold that hits FAR ≤ 2% with its FRR.
- `--cohort DIR` adds AS-norm (score normalisation against an impostor cohort) to the report.
- With no args it falls back to the synthetic `samples/` fixtures (leave-one-out) so you can
  smoke-test the harness — those numbers are **plumbing only**.

### Results (fill in after recording the real test set)

Smoke test on synthetic TTS fixtures (not real voices — do not trust):

| method | n_owner | n_other | EER | FAR @ accept=0.45 | FRR @ accept=0.45 |
|---|---|---|---|---|---|
| single | 1 | 2 | 0.000 | 0.000 | 0.000 |
| windowed | 1 | 2 | 0.000 | 0.000 | 0.000 |

Real test set to record (kept local, not committed):

- **Owner**: 40+ utterances, different days, quiet/normal/loud, near/far, fan on/off.
- **Others**: 5–8 colleagues, ~10 each.
- **Background**: TV, YouTube, music with vocals, speakerphone, office chatter.
- **Hard**: owner talking over someone else; owner with a cold.

---

## 11. Privacy & post-prototype decisions (ticket 8)

**Done (privacy):**

- Voiceprint and audio stay on this device. Only owner-verified transcripts go to the brain;
  non-owner audio is dropped after scoring (only the score is logged, no transcript).
- The default `brain: auto` answers with the local model when no hosted key is set, so
  transcripts never leave the Mac. Setting `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` sends
  transcripts to that provider instead.
- Consent screen before enrolment, plus **Delete my voice**.
- Voiceprint encrypted at rest (`voiceprint.npy.enc`) with Fernet, key held in the macOS
  Keychain (`app/privacy.py`).

**Adaptive voiceprint (implemented):** high-confidence owner utterances (score ≥
`speaker.adaptive.min_score`) nudge the voiceprint with an EMA (weight `0.02`). A frozen
`voiceprint.original.npy` is kept at enrolment; **Restore original** (UI) or
`python main.py restore` undoes the drift.

**Anti-spoofing (documented, not built):** speaker verification alone can be fooled by a
recording of the owner. A replay/deepfake check (AASIST-style) or a "say this random word"
challenge is the future path. **This prototype is not a security lock** — do not use it to
gate anything sensitive.

**Multiple owners (later):** the data model supports several profiles (each would get its own
`voiceprint.npy` under a per-profile folder); the gate already returns a score so picking the
best-matching profile is a small change.

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

---

## 12. Troubleshooting voice recognition (read this first)

1. Open the UI. It uses the system's default microphone (change `audio.device` in
   `config.yaml` if you need a different input).
2. Click **🎤 Mic check**, then speak a full sentence for 4 seconds. It reports the level in
   dB, what it heard, and the voice match score.
   - "No speech heard" → check the input device or speak closer.
   - Speech heard, match ≥ 35% → recognition path works (the owner rule).
   - Speech heard, match low → re-enrol with the same mic you just tested.
3. Enrolment: the prompt appears as one line; each word turns **green** when it was heard,
   **red** when it was missed or mispronounced.

The processing pipeline (noise profile, VAD, gate) is tested on recorded audio:
whole-utterance owner clips score about 0.94 and other speakers about 0.03, with and
without fan noise. Live scores vary with microphone and distance, which is why the
owner rule is 35%.

---

## 13. How to use it (step by step)

**Start the app**

```bash
source .venv/bin/activate
python server.py          # open http://127.0.0.1:8765
```

The page opens with a three-step guide at the top.

### Step 1 — microphone

The app uses the **default system microphone**. To choose a different input, open
**System Settings → Sound → Input** (or set `audio.device` in `config.yaml`). The choice is
used for enrolment, the mic check, and live listening.

### Step 2 — enrol your voice (name + 10 samples)

1. Tick the consent box: *"I understand my voice stays on this device."*
2. Press **🎙 Enrol my voice**.
3. The **first question is asked out loud**: *“What is your name?”* Press **● Tap to answer**,
   say your name naturally (for example, “My name is Adnan”), then press **■ Stop**. The name
   is remembered and used to greet you every time you wake Lyra.
4. Ten prompts then appear, one at a time, as a **single line**. They are **picked at random
   from a pool of 30**, and every prompt is at least **4 words** long.
5. For each prompt: press **● Tap to record**, read the line in a natural voice, then press
   **■ Stop**. Keep each take roughly 3–8 seconds.
6. Words turn **green** when they were heard correctly and **red** when they were missed or
   mispronounced. A red word does not always reject the take; the take is judged on the whole
   sample.
7. Finish all 10. The assistant builds your voiceprint, runs a self-check, and saves it
   encrypted on this Mac, together with your name.

**Tips for a good voiceprint:** use the same microphone you will talk into later, keep a
normal speaking distance, avoid background TV/music, and do not whisper.

### Step 3 — check the microphone

Press **🎤 Mic check (4 s)** and say a full sentence. You will see:

- the input level in **dB**,
- what the assistant **heard**,
- your **match %**.

**35% or more means "this is you".** If you see "No speech heard", check the input device
or the level is too low. If the match is below 35%, re-enrol using the same microphone.

### Step 4 — wake Lyra and talk

Press **▶ Start listening**. Lyra waits in standby until it hears your voice say one of:

- **“Hey Lyra”**, or
- **“Lyra”**.

The utterance is checked against your enrolled voice **before** anything is answered. If it
is not you, Lyra stays silent. If it is you, Lyra greets you by name — for example,
“Hey Adnan, how are you? How can I help?” — and the conversation stays open.

1. waits until it hears speech,
2. keeps capturing while you talk, including short pauses,
3. waits **2 seconds of silence** after you stop,
4. compares the whole utterance with your enrolled voice,
5. **≥ 35% similarity → owner**; below 35% → ignored,
6. only for the owner: English speech-to-text, then the reply.

After the wake greeting you can ask follow-up questions without saying “Hey Lyra” again.
Say **“stop”** or **“goodbye”** to end the conversation, or let it sleep after the idle
timeout (default **2 minutes**, adjustable under **Settings**).

The **voice orb** pulses with your level and turns green while it hears speech.
The **capture panel** shows the elapsed time and the 2-second silence countdown.

### Everyday tips

- Speak in sentences. Very short noises such as a cough are ignored.
- Wait for the orb to finish the 2-second countdown; that is when the decision happens.
- Other people can talk near the device; their voice is scored, shown as *not the owner*,
  and never sent to speech-to-text. After **5 unknown attempts within 60 seconds**, recognition
  pauses for 60 seconds instead of letting someone keep trying.
- After **2 minutes** with no owner interaction Lyra sleeps. Say **“Hey Lyra”** to wake it.
- **Add samples** improves recognition later. Each new take must clearly match your enrolled
  voice (at least 45%), so a different speaker cannot poison the voiceprint.
  **Restore** undoes automatic adaptation; **Delete** removes your voiceprint completely.

**Behaviour notes**

- The assistant replies only to the enrolled owner.
- Speech recognition is English only (`bot.language: en`).
- Owner audio is kept in `debug/` only when "save audio" is turned on. Other people's audio is
  never saved.
- Background noise is reduced spectrally, and an adaptive noise gate ignores anything close
  to the ambient noise floor, so a fan or TV hiss does not open an utterance.
