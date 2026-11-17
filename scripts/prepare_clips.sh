#!/usr/bin/env bash
# Convert real recordings (m4a/mp3/mov/wav) into the format the spike expects:
# mono, 16 kHz, 16-bit PCM wav.
#
# Usage:
#   scripts/prepare_clips.sh ~/Desktop/recordings
#
# Name the source files owner_1.m4a, owner_2.m4a, other_1.m4a ... and this script
# writes samples/owner_1.wav etc.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:?usage: scripts/prepare_clips.sh <folder-with-recordings>}"
mkdir -p samples

shopt -s nullglob
files=("$SRC"/*.{m4a,mp3,mp4,mov,wav,flac,aiff})
if [ ${#files[@]} -eq 0 ]; then
  echo "No audio files found in $SRC"
  exit 1
fi

for f in "${files[@]}"; do
  name="$(basename "${f%.*}").wav"
  # highpass removes rumble; loudnorm evens out level between recordings
  ffmpeg -y -loglevel error -i "$f" -ac 1 -ar 16000 -c:a pcm_s16le \
    -af "highpass=f=80,loudnorm=I=-20:TP=-2:LRA=7" "samples/$name"
  echo "  $name  ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "samples/$name")s)"
done

echo "Done. Check that all clips are 5-10s and that names start with owner_ or other_."
