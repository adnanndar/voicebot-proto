"""Enrolment prompt pool.

Every prompt must be long enough to record: at least four words. Short lines
("Call mom.") produced unreliable voice samples, so they are filtered out even
if they are added to prompts.yaml later.
"""
from __future__ import annotations

import random

import yaml

from ..config import ROOT

MIN_PROMPT_WORDS = 4


def load_prompts(cfg) -> list[dict]:
    path = ROOT / cfg.enrol.prompts_file
    data = yaml.safe_load(path.read_text()) or {}
    prompts: list[dict] = []
    for item in data.get("prompts", []):
        if isinstance(item, str):
            item = {"text": item}
        text = " ".join(str(item.get("text", "")).split())
        if len(text.split()) < MIN_PROMPT_WORDS:
            continue
        prompt = dict(item)
        prompt["text"] = text
        prompts.append(prompt)
    return prompts


def pick_prompts(cfg, n: int = 10, rng=None) -> list[dict]:
    pool = load_prompts(cfg)
    if not pool:
        return []
    rng = rng or random
    return rng.sample(pool, min(n, len(pool)))
