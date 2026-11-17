#!/usr/bin/env bash
# Creates synthetic sample clips so the pipeline can be tested immediately.
# These are macOS text-to-speech voices, NOT real people: use them to prove the
# plumbing works, then replace samples/*.wav with your real recordings.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p samples samples/latency

AVAILABLE="$(say -v '?' | awk '{print $1}')"
pick_voice() {
  for v in "$@"; do
    if grep -qx "$v" <<<"$AVAILABLE"; then echo "$v"; return 0; fi
  done
  return 1
}

OWNER="$(pick_voice Samantha Ava Zoe Karen Moira Tessa)" \
  || { echo "No standard TTS voice found. Run 'say -v ?' and set OWNER manually."; exit 1; }
OTHER1="$(pick_voice Daniel Alex Fred Evan Tom Aaron)" \
  || { echo "Need a second distinct voice; see 'say -v ?'."; exit 1; }
OTHER2="$(pick_voice Victoria Serena Kate Fiona Susan Allison Karen Moira Tessa Fred)" \
  || { echo "Need a third distinct voice; see 'say -v ?'."; exit 1; }

echo "voices: owner=$OWNER  other1=$OTHER1  other2=$OTHER2"

mk() { # mk <outfile> <voice> <text>
  local out="samples/$1" voice="$2" text="$3"
  say -v "$voice" -r 175 -o /tmp/spike_tts.aiff -- "$text"
  ffmpeg -y -loglevel error -i /tmp/spike_tts.aiff -ac 1 -ar 16000 -c:a pcm_s16le "$out"
  echo "  $out  ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")s)"
}

# Owner: same voice, three different utterances.
mk owner_1.wav "$OWNER" "Hey, can you pull up my calendar for tomorrow morning and tell me what my first meeting is."
mk owner_2.wav "$OWNER" "I need to reschedule the design review to Thursday afternoon, and let the team know about the change."
mk owner_3.wav "$OWNER" "Remind me to pick up the dry cleaning on the way home, and order more coffee beans."

# Other people: different voices, same kind of content.
mk other_1.wav "$OTHER1" "Could you check whether the build finished, and send me the logs if anything failed."
mk other_2.wav "$OTHER2" "The weather looks good this weekend, so I was thinking we could take the kids to the park."

# One continuous clip for VAD / ASR latency (kept out of samples/ root).
LONG="This is a longer recording used to measure how fast the voice pipeline runs. \
I am going to speak continuously for a little while so that the voice activity detector \
has something realistic to work with. There are natural pauses between these sentences, \
and the level of my voice changes a bit as I go. The speech recognition model should \
produce a transcript that roughly matches what I am saying right now. Let me add a few \
more words to make this clip longer. I will mention a calendar, a meeting, a reminder \
about coffee, and a question about the weather. That should be enough audio for a first \
measurement of real time factors, so I will stop talking here."
say -v "$OWNER" -r 165 -o /tmp/spike_long.aiff -- "$LONG"
ffmpeg -y -loglevel error -i /tmp/spike_long.aiff -ac 1 -ar 16000 -c:a pcm_s16le samples/latency/long.wav
echo "  samples/latency/long.wav  ($(ffprobe -v error -show_entries format=duration -of csv=p=0 samples/latency/long.wav)s)"

echo "Done. Synthetic fixtures in samples/."
