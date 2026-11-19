"""Word-level comparison between a prompt and what speech-to-text heard.

Used by the owner enrolment flow and the public web demo, so a contraction
("what's") matches the words it stands for ("what is") instead of showing up
as a miss.
"""
from __future__ import annotations

import re

CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "i'm": "i am", "im": "i am",
    "it's": "it is", "that's": "that is", "thats": "that is",
    "don't": "do not", "dont": "do not", "can't": "cannot", "cant": "cannot",
    "won't": "will not", "wont": "will not", "let's": "let us",
    "lets": "let us", "i'll": "i will", "ill": "i will", "we're": "we are",
    "you're": "you are", "they're": "they are", "there's": "there is",
    "o'clock": "oclock", "oclock": "oclock",
}
_TOKEN_RE = re.compile(r"[^a-z0-9'\s]+")


def tokens(text: str) -> list[str]:
    """Lower-case words, with contractions expanded and punctuation removed."""
    s = _TOKEN_RE.sub(" ", str(text or "").lower().replace("\u2019", "'"))
    out: list[str] = []
    for word in s.split():
        word = word.replace("'", "")
        mapped = CONTRACTIONS.get(word)
        out.extend((mapped or word).split())
    return out


def mark_words(prompt: str, transcript: str) -> list[dict]:
    """One entry per prompt word, `ok` when the transcript contains it."""
    spoken = set(tokens(transcript))
    words: list[dict] = []
    for word in str(prompt or "").split():
        need = tokens(word)
        words.append({"word": word, "ok": bool(need) and all(t in spoken for t in need)})
    return words
