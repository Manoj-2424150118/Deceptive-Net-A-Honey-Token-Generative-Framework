"""Canary tokens — decoy cloud & API credentials (standalone module).

Deceptive-Net's original token families cover cards, bank accounts and
login credentials. Modern breaches, however, leak *machine* secrets:
cloud access keys, personal access tokens, API keys and database
DSNs — the first thing attackers spend is usually a key, not a card.

This module generates format-faithful decoys for those families:

- AWS-style access key pairs (AKIA… id + 40-char secret)
- GitHub-style personal access tokens (ghp_…)
- Generic API keys (configurable prefix + hex secret)
- PostgreSQL DSNs with embedded decoy credentials

Design notes
------------
* Generation uses :mod:`secrets` (CSPRNG) — decoys must not be
  distinguishable by a weak PRNG, and must never collide with the
  deterministic patterns a scanner might fingerprint as "obviously
  fake".
* The registry stores only the SHA-256 fingerprint of each token,
  never the raw secret. Lookup hashes the candidate value the same
  way, so a registry/database leak does not hand an attacker the list
  of live decoys to avoid.
* Pure standard library — no numpy/pandas — so the module can be
  imported and unit-tested anywhere, including minimal CI images.

These are DECOYS. They are generated locally, belong to no real
account, and grant no access to anything. Their only job is to be
planted (in a repo, a config file, a leaked dump) and to call home —
via :meth:`CanaryRegistry.check` — when somebody tries to use one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

_ALNUM = string.ascii_uppercase + string.digits
_AWS_SECRET_ALPHABET = string.ascii_letters + string.digits + "/+"
_HEX = "0123456789abcdef"

# Shape validators: a value found in the wild can first be identified
# as "one of ours by shape" before the registry confirms it.
_PATTERNS = {
    "aws_access_key_id": re.compile(r"^AKIA[0-9A-Z]{16}$"),
    "github_pat": re.compile(r"^ghp_[A-Za-z0-9]{36}$"),
    "api_key": re.compile(r"^[a-z][a-z0-9_]*_[0-9a-f]{32}$"),
    "postgres_dsn": re.compile(
        r"^postgres://[a-z0-9_]+:[^@\s]+@[a-z0-9.\-]+:\d+/[a-z0-9_]+$"
    ),
}


def _rand(alphabet: str, length: int) -> str:
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_aws_keypair() -> Dict[str, str]:
    """A decoy AWS-style access key pair."""
    return {
        "family": "aws",
        "access_key_id": "AKIA" + _rand(_ALNUM, 16),
        "secret_access_key": _rand(_AWS_SECRET_ALPHABET, 40),
    }


def generate_github_pat() -> Dict[str, str]:
    """A decoy GitHub-style personal access token."""
    return {
        "family": "github_pat",
        "token": "ghp_" + _rand(string.ascii_letters + string.digits, 36),
    }


def generate_api_key(prefix: str = "sk_live") -> Dict[str, str]:
    """A decoy generic API key: ``<prefix>_<32 hex chars>``."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise ValueError("prefix must be lowercase letters, digits or underscores, starting with a letter")
    return {"family": "api_key", "token": f"{prefix}_{_rand(_HEX, 32)}"}


def generate_postgres_dsn(host: str = "db.internal.example.com") -> Dict[str, str]:
    """A decoy PostgreSQL DSN with embedded credentials."""
    user = "svc_" + _rand(string.ascii_lowercase, 8)
    password = _rand(string.ascii_letters + string.digits, 24)
    database = "app_" + _rand(string.ascii_lowercase, 6)
    dsn = f"postgres://{user}:{password}@{host}:5432/{database}"
    return {"family": "postgres_dsn", "dsn": dsn, "username": user}


_GENERATORS = {
    "aws": generate_aws_keypair,
    "github_pat": generate_github_pat,
    "api_key": generate_api_key,
    "postgres_dsn": generate_postgres_dsn,
}


def token_values(token: Dict[str, str]) -> List[str]:
    """Every string in a generated token that identifies it in the wild
    (the parts an attacker would actually submit somewhere)."""
    values = []
    for key in ("access_key_id", "secret_access_key", "token", "dsn"):
        if key in token:
            values.append(token[key])
    return values


def identify_family(value: str) -> Optional[str]:
    """Return the family whose *shape* a value matches, else None."""
    for family, pattern in _PATTERNS.items():
        if pattern.match(value):
            return family
    if re.fullmatch(r"[A-Za-z0-9/+]{40}", value):
        return "aws_secret_access_key"
    return None


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CanaryRegistry:
    """Fingerprint-keyed registry of planted canary tokens.

    Register a token when it is planted; when a suspicious value shows
    up (in a login attempt, an API call, a paste site), :meth:`check`
    tells you which plant it came from — watermark, actor, timestamp —
    without the registry ever storing the raw secrets.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self._by_fingerprint: Dict[str, dict] = {}
        self._plants: List[dict] = []
        self._storage_path = storage_path
        if storage_path:
            self._load()

    def _load(self) -> None:
        """Reload plants from the JSON store (fingerprints only)."""
        try:
            with open(self._storage_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        for record in data.get("plants", []):
            self._plants.append(record)
            for fp in record.get("fingerprints", []):
                self._by_fingerprint[fp] = record

    def _save(self) -> None:
        """Persist plants atomically. The file holds fingerprints only —
        never raw token values — so it is safe to keep on disk."""
        if not self._storage_path:
            return
        directory = os.path.dirname(os.path.abspath(self._storage_path))
        os.makedirs(directory, exist_ok=True)
        tmp_path = self._storage_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump({"plants": self._plants}, fh, indent=2)
        os.replace(tmp_path, self._storage_path)

    def register(self, token: Dict[str, str], *, planted_in: str,
                 actor: str = "unknown") -> dict:
        """Record where a generated token was planted. Returns the
        plant record (which contains fingerprints, never raw values)."""
        timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime(
            "%Y-%m-%d %H:%M:%S IST"
        )
        record = {
            "plant_id": f"PLT-{len(self._plants) + 1:04d}",
            "family": token.get("family", "unknown"),
            "planted_in": planted_in,
            "actor": actor,
            "timestamp": timestamp,
            "fingerprints": [_fingerprint(v) for v in token_values(token)],
        }
        for fp in record["fingerprints"]:
            self._by_fingerprint[fp] = record
        self._plants.append(record)
        self._save()
        return record

    def check(self, value: str) -> Optional[dict]:
        """If *value* is (part of) a planted canary token, return its
        plant record; else None."""
        return self._by_fingerprint.get(_fingerprint(value.strip()))

    def generate_and_register(self, family: str, *, planted_in: str,
                              actor: str = "unknown", **kwargs) -> Dict[str, str]:
        """Generate a token of *family* and register it in one call.
        Returns the raw token (for planting) — the registry keeps only
        its fingerprints."""
        if family not in _GENERATORS:
            raise ValueError(f"unknown family: {family!r} (have: {sorted(_GENERATORS)})")
        token = _GENERATORS[family](**kwargs)
        self.register(token, planted_in=planted_in, actor=actor)
        return token

    def get_plants(self) -> List[dict]:
        return list(self._plants)


# Module-level default registry, mirroring the project's pattern of a
# single shared registry instance for the running server.
canary_registry = CanaryRegistry()
