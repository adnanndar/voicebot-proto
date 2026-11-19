"""The "brain": turns the owner's transcript into a spoken answer.

`bot.brain: auto` picks the best available backend, in order:

1. OpenAI-compatible chat API, when OPENAI_API_KEY is set
2. Local MLX model (Qwen2.5 1.5B, 4-bit) running on this Mac
3. Keyless online model (rate limited, so only a fallback)
4. Built-in rules: time, date, weather, arithmetic, small talk

A failing backend is skipped for a while and the next one answers, so a
clearly heard question never gets an echo of what was said.
"""
from __future__ import annotations

import ast
import datetime
import json
import logging
import operator
import re
import threading
import time
import urllib.request
from pathlib import Path

from ..config import AttrDict

log = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
WEATHER_URL = "https://wttr.in/?format=%l:+%t+%C"
USER_AGENT = "voicebot-proto/0.1"

_WEATHER_RE = re.compile(
    r"\b(weather|temperature|forecast|rain|raining|rainy|snow|snowing|sunny|"
    r"cloudy|humid|windy|umbrella|how (hot|cold)|degrees outside)\b"
)

_SYSTEM = (
    "You are a private, hands-free voice assistant for {owner}. "
    "Answer the question directly in at most four short sentences of plain spoken English, "
    "putting the key facts first. For a how-to or an explanation, include the important steps "
    "or details instead of being vague. Never use lists, markdown, emoji, or stage directions. "
    "If you are unsure, give your best short answer instead of refusing."
)


def _system_for(owner: str) -> str:
    named = bool(owner) and owner.strip().lower() not in {"", "owner", "the owner"}
    text = _SYSTEM.format(owner=owner or "the owner")
    if named:
        text += f" The owner's name is {owner}; address them by name when it feels natural."
    return text


# --------------------------------------------------------------------- helpers
def _clean(text: str, limit: int = 600) -> str:
    """Make model output safe and pleasant to speak aloud."""
    text = (text or "").strip()
    text = re.sub(r"^\s*(assistant|answer)\s*:\s*", "", text, flags=re.I)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"^\s*[-*\u2022]\s+", "", text, flags=re.M)
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    # keep spoken replies focused (at most four short sentences)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 4:
        text = " ".join(sentences[:4]).strip()
    if len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        text = (text[: cut if cut > 0 else limit]).rstrip(" ,;:-") + "."
    return text.strip()


def _build_messages(transcript: str, history: list, system: str) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": system}]
    turns = [(r, t) for r, t in (history or []) if t]
    for role, text in turns[-10:]:
        msgs.append({
            "role": "assistant" if role in {"bot", "assistant"} else "user",
            "content": text,
        })
    # The engine appends the current utterance before asking for a reply, so
    # only add it when the history does not already end with it.
    if not msgs or msgs[-1]["role"] != "user" or msgs[-1]["content"].strip() != transcript.strip():
        msgs.append({"role": "user", "content": transcript})
    return msgs


def _parse_weather(line: str) -> str:
    """Turn 'City, Region, CC: +15°C Sunny' into a spoken sentence."""
    m = re.match(r"^\s*(?P<place>.*?):\s*(?P<temp>[+-]?\d+)\s*°?C\s*(?P<cond>.*)$", line)
    if not m:
        return line.strip()
    city = m.group("place").split(",")[0].strip() or "your area"
    temp = m.group("temp").lstrip("+")
    cond = m.group("cond").strip().lower()
    if cond and cond not in {"unknown", "n/a"}:
        return f"It's {temp} degrees Celsius and {cond} in {city} right now."
    return f"It's {temp} degrees Celsius in {city} right now."


class WeatherLookup:
    """Current conditions from wttr.in, cached so we don't spam it."""

    def __init__(self, ttl: float = 300.0, timeout: float = 6.0):
        self.ttl = ttl
        self.timeout = timeout
        self._line: str | None = None
        self._at = 0.0
        self._tried = 0.0

    def get(self, max_age: float | None = None) -> str | None:
        now = time.time()
        age = now - self._at
        if self._line and age < (max_age or self.ttl):
            return self._line
        # if the last attempt failed, wait a minute before trying again
        if not self._line and now - self._tried < 60:
            return None
        self._tried = now
        try:
            req = urllib.request.Request(WEATHER_URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8", "replace").strip()
        except Exception:
            return None
        if not raw or "<" in raw[:1]:
            return None
        self._line = raw.splitlines()[0][:200]
        self._at = now
        return self._line


# -------------------------------------------------------------------- backends
class OpenAICompatBrain:
    """Any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 owner_name: str = "the owner", timeout: float = 20.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.owner_name = owner_name
        self.timeout = timeout

    def reply(self, transcript: str, history: list, context: str = "") -> str:
        system = _system_for(self.owner_name)
        if context:
            system += "\n\n" + context
        payload = {
            "model": self.model,
            "messages": _build_messages(transcript, history, system),
            "max_tokens": 120,
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = r.read().decode("utf-8", "replace")
        try:
            data = json.loads(body)
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            # some keyless endpoints answer with plain text
            text = body
        return _clean(text)


class ClaudeBrain:
    """Claude backend (needs ANTHROPIC_API_KEY and the anthropic package)."""

    def __init__(self, api_key: str, owner_name: str, model: str = "claude-sonnet-5-5"):
        self.api_key = api_key
        self.owner_name = owner_name
        self.model = model

    def reply(self, transcript: str, history: list, context: str = "") -> str:
        from anthropic import Anthropic

        system = _system_for(self.owner_name)
        if context:
            system += "\n\n" + context
        client = Anthropic(api_key=self.api_key)
        messages = _build_messages(transcript, history, system)[1:]
        resp = client.messages.create(
            model=self.model, max_tokens=1024, system=system, messages=messages
        )
        # adaptive thinking adds non-text blocks; speak only the text ones
        text = " ".join(
            getattr(block, "text", "")
            for block in resp.content
            if getattr(block, "type", "") == "text"
        )
        return _clean(text)


def _resolve_model_path(name: str) -> str:
    """Use the cached snapshot when present, so starting up needs no network."""
    p = Path(name).expanduser()
    if p.exists():
        return str(p)
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(name, local_files_only=True)
    except Exception:
        return name  # first run: mlx_lm downloads it


class LocalModelBrain:
    """Small instruct model running locally through MLX (Apple Silicon)."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL,
                 owner_name: str = "the owner", max_tokens: int = 110):
        self.model_name = model_name
        self.owner_name = owner_name
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warm(self) -> bool:
        """Load the model now so the first question is not delayed."""
        try:
            self._load()
        except Exception:
            return False
        return True

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from mlx_lm import load

            self._model, self._tokenizer = load(_resolve_model_path(self.model_name))

    def reply(self, transcript: str, history: list, context: str = "") -> str:
        self._load()
        from mlx_lm import generate

        system = _system_for(self.owner_name)
        if context:
            system += "\n\n" + context
        messages = _build_messages(transcript, history, system)
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        out = generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=self.max_tokens, verbose=False,
        )
        return _clean(out)


class RuleBrain:
    """Offline answers for the questions people ask most; no model needed."""

    def __init__(self, owner_name: str = "the owner", weather: WeatherLookup | None = None):
        self.owner_name = owner_name
        self.weather = weather or WeatherLookup()

    @property
    def _named(self) -> bool:
        return bool(self.owner_name) and self.owner_name.strip().lower() not in {
            "", "owner", "the owner"
        }

    def reply(self, transcript: str, history: list, context: str = "") -> str:
        t = (transcript or "").strip()
        if not t:
            return "I didn't catch that."
        low = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' %.+\-*/]", " ", t.lower())).strip()

        if re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b", low):
            if self._named:
                return f"Hello, {self.owner_name}! I'm listening. What would you like to know?"
            return "Hello! I'm listening. What would you like to know?"
        if re.search(r"\b(what('s| is)? my name|who am i|do you know my name)\b", low):
            if self._named:
                return f"Your name is {self.owner_name}."
            return "I don't know your name yet. Add it in the enrolment section and I'll use it."
        if "thank" in low:
            return "You're welcome."
        if "how are you" in low:
            return "I'm ready and listening. How can I help?"
        if "your name" in low or "who are you" in low:
            return "I'm your private voice assistant, running on this Mac."
        if "what can you do" in low or low.strip() in {"help", "what do you do"}:
            return ("I answer questions, tell you the time, date, weather and calculations, "
                    "and keep listening until you say stop.")
        if re.search(r"\b(are you (there|listening)|can you hear me)\b", low):
            return "Yes, I'm here and listening."

        if re.search(r"\b(what('s| is)? the time|what time is it|tell me the time|time now)\b", low) \
                or low.strip() in {"time", "the time"}:
            return f"It's {datetime.datetime.now():%H:%M}."

        now = datetime.datetime.now()
        if re.search(r"\b(what('s| is)? the date|what day is it|today'?s date|what is today)\b", low):
            return f"Today is {now:%A, %d %B %Y}."
        if re.search(r"\b(what('s| is)? the year|what year is it)\b", low):
            return f"It's {now:%Y}."
        if re.search(r"\b(what month is it|what('s| is)? the month)\b", low):
            return f"It's {now:%B}."

        if _WEATHER_RE.search(low):
            line = self.weather.get(max_age=120)
            if line:
                return _parse_weather(line)
            return "I couldn't reach the weather service just now. Please try again in a minute."

        math_reply = _try_math(low)
        if math_reply:
            return math_reply

        if re.search(r"\b(repeat that|say that again|what did you say)\b", low):
            for role, text in reversed(history or []):
                if role in {"bot", "assistant"}:
                    return text

        return ("That question needs my language model, which isn't loaded right now. "
                "I can still help with the time, date, weather, and arithmetic.")


def _try_math(low: str) -> str | None:
    """Answer simple spoken arithmetic without a model."""
    pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s*of\s*(\d+(?:\.\d+)?)", low)
    if pct:
        return f"That's {float(pct.group(1)) / 100 * float(pct.group(2)):.6g}."

    asked = re.search(
        r"\b(what is|what's|whats|how much is|calculate|compute)\b", low
    )
    spoken_op = re.search(
        r"\b(plus|minus|times|multiplied by|divided by|x)\b|%", low
    )
    if not (asked or spoken_op):
        return None

    expr = re.sub(
        r"\b(what is|what's|whats|how much is|calculate|compute|tell me|equals?|please)\b",
        " ",
        low,
    )
    expr = re.sub(r"\bplus\b|\band\b", "+", expr)
    expr = re.sub(r"\bminus\b", "-", expr)
    expr = re.sub(r"\btimes\b|\bmultiplied by\b|\bx\b", "*", expr)
    expr = re.sub(r"\bdivided by\b|\bover\b", "/", expr)
    # anything still spelled out means this is ordinary speech, not a sum
    if re.search(r"[a-z]", expr):
        return None
    expr = expr.replace("?", " ").replace("^", "**")
    expr = re.sub(r"[^0-9+\-*/().\s]", "", expr)
    if not re.search(r"\d", expr) or not re.search(r"[+\-*/]", expr):
        return None
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        value = _safe_eval(tree.body)
    except Exception:
        return None
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"That's {value:g}."


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if left is None or right is None:
            return None
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        val = _safe_eval(node.operand)
        if val is None:
            return None
        return -val if isinstance(node.op, ast.USub) else val
    return None


class AutoBrain:
    """Try each backend in turn; always end with the offline rules."""

    def __init__(self, providers: list, rules: RuleBrain, weather: WeatherLookup):
        self.providers = providers
        self.rules = rules
        self.weather = weather
        self._skip_until: dict[int, float] = {}

    def _context(self, transcript: str) -> str:
        now = datetime.datetime.now()
        lines = [f"Current local date and time: {now:%A, %d %B %Y, %H:%M}."]
        if _WEATHER_RE.search(transcript.lower()):
            line = self.weather.get()
            if line:
                lines.append(f"Live weather right now: {line}. Use it to answer accurately.")
        return "\n".join(lines)

    def reply(self, transcript: str, history: list) -> str:
        low = transcript.lower()
        # Live facts and arithmetic are answered exactly, not guessed at by a
        # small model. This also keeps those replies instant.
        if _WEATHER_RE.search(low):
            line = self.weather.get()
            if line:
                return _parse_weather(line)
        math_reply = _try_math(low)
        if math_reply:
            return math_reply

        context = self._context(transcript)
        now = time.monotonic()
        for i, provider in enumerate(self.providers):
            if now < self._skip_until.get(i, 0.0):
                continue
            try:
                text = provider.reply(transcript, history, context)
            except Exception as exc:
                # back off: a broken endpoint or exhausted free quota must not
                # stall every following turn
                log.warning("brain backend %s unavailable: %s", type(provider).__name__, exc)
                self._skip_until[i] = time.monotonic() + 90.0
                continue
            if text and text.strip():
                return text
        return self.rules.reply(transcript, history)


# Backwards-compatible name used by the old prototype/tests.
StubBrain = RuleBrain


def _has_anthropic() -> bool:
    try:
        import anthropic  # noqa: F401

        return True
    except Exception:
        return False


def make_brain(cfg: AttrDict):
    choice = str(cfg.bot.get("brain", "auto") or "auto").lower()
    owner = cfg.bot.get("owner_name", "the owner")
    weather = WeatherLookup()
    rules = RuleBrain(owner, weather)

    key = (cfg.env.get("OPENAI_API_KEY") or "").strip()
    base = (cfg.env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    model = (cfg.env.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    anthropic_key = (cfg.env.get("ANTHROPIC_API_KEY") or "").strip()
    anthropic_model = (cfg.env.get("ANTHROPIC_MODEL") or "claude-sonnet-5-5").strip()

    if choice in {"stub", "local", "rules"}:
        return rules

    if choice in {"claude", "anthropic"}:
        if not anthropic_key:
            raise RuntimeError(
                "brain=anthropic but ANTHROPIC_API_KEY is not set in .env"
            )
        if not _has_anthropic():
            raise RuntimeError("ANTHROPIC_API_KEY is set but the anthropic package is missing")
        return ClaudeBrain(anthropic_key, owner, anthropic_model)

    if choice == "openai":
        if not key:
            raise RuntimeError("brain=openai but OPENAI_API_KEY is not set in .env")
        return OpenAICompatBrain(base, model, key, owner)

    if choice == "pollinations":
        return OpenAICompatBrain(
            "https://text.pollinations.ai/openai", "openai", "", owner
        )

    # auto: best available, private local model included, rules last
    providers: list = []
    if anthropic_key:
        if _has_anthropic():
            providers.append(ClaudeBrain(anthropic_key, owner, anthropic_model))
        else:
            log.warning("ANTHROPIC_API_KEY is set but the anthropic package is missing")
    if key:
        providers.append(OpenAICompatBrain(base, model, key, owner))

    local_model = cfg.bot.get("local_model") or DEFAULT_LOCAL_MODEL
    local = LocalModelBrain(str(local_model), owner)
    if local.warm():  # only add it when the model actually loads
        providers.append(local)
    else:
        log.warning("local model %s unavailable; using fallbacks", local_model)

    # keyless endpoint: 1 reply / 45 s, so it is strictly a fallback
    providers.append(OpenAICompatBrain(
        "https://text.pollinations.ai/openai", "openai", "", owner, timeout=15.0
    ))
    log.info("brain backends: %s (owner: %s)", ", ".join(
        type(p).__name__ for p in providers), owner)
    return AutoBrain(providers, rules, weather)
