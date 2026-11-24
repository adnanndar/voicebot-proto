#!/bin/zsh
# Durable launcher for the Lyra live (browser-mic) server.
# Started by ~/Library/LaunchAgents/com.lyra.live.plist, which restarts it if
# it ever exits and keeps the Mac awake while the server is running.
set -e
# launchd starts with a minimal PATH; add Homebrew so ffmpeg/say are found.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
cd "/Users/adnanbashir/Documents/Default Project/voicebot-proto"
exec caffeinate -i -m -s .venv/bin/python live_server.py --port 8766
