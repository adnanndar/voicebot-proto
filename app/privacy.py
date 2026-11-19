"""Ticket 8 — privacy.

Encrypts the voiceprint at rest with Fernet using a key stored in the macOS
Keychain (`security` CLI). If cryptography or the Keychain is unavailable, it
degrades to plaintext so the app still runs (with a warning from the caller).
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess

SERVICE = "voicebot-proto"
ACCOUNT = "voiceprint-key"


def _fernet():
    try:
        from cryptography.fernet import Fernet

        return Fernet
    except ImportError:
        return None


def available() -> bool:
    return _fernet() is not None and shutil.which("security") is not None


_KEY: bytes | None = None  # cached for the process lifetime; avoids a Keychain call per read


def _get_or_create_key() -> bytes:
    """Return a Fernet key (urlsafe-base64 of 32 bytes) from the Keychain, or make one."""
    global _KEY
    if _KEY is not None:
        return _KEY
    _KEY = _load_key_from_keychain()
    return _KEY


def _load_key_from_keychain() -> bytes:
    r = subprocess.run(
        ["security", "find-generic-password", "-s", SERVICE, "-a", ACCOUNT, "-w"],
        capture_output=True, text=True,
    )
    stored = r.stdout.strip() if r.returncode == 0 else ""
    if stored:
        try:
            base64.urlsafe_b64decode(stored)
            return stored.encode("ascii")
        except Exception:
            pass
    raw = os.urandom(32)
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    subprocess.run(
        ["security", "add-generic-password", "-U", "-s", SERVICE, "-a", ACCOUNT, "-w", encoded],
        capture_output=True, check=False,
    )
    return encoded.encode("ascii")


def encrypt(data: bytes) -> bytes | None:
    if not available():
        return None
    try:
        return _fernet()(_get_or_create_key()).encrypt(data)
    except Exception:
        return None


def decrypt(data: bytes) -> bytes | None:
    if not available():
        return None
    try:
        return _fernet()(_get_or_create_key()).decrypt(data)
    except Exception:
        return None
