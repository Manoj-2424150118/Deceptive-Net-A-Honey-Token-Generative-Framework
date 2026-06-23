"""
Deceptive-Net – Audit Logger
==============================
Writes timestamped audit events to:
  - Rotating file  logs/audit.log
  - In-memory ring buffer (last 1000 events) for real-time dashboard
"""

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))
from logging.handlers import RotatingFileHandler
from threading import Lock

LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "audit.log")
os.makedirs(LOG_DIR, exist_ok=True)

_buffer: deque = deque(maxlen=1000)
_lock   = Lock()

# ── rotating file logger ──────────────────────────────────────────────────────
_file_logger = logging.getLogger("deceptive_net.audit")
_file_logger.setLevel(logging.INFO)
_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
_handler.setFormatter(logging.Formatter("%(message)s"))
_file_logger.addHandler(_handler)


def log_event(
    actor: str,
    action: str,
    resource: str = "",
    detail: str = "",
    severity: str = "INFO",   # INFO | WARN | ALERT
    ip: str = "unknown",
):
    """Write one audit event to file and ring buffer."""
    event = {
        "ts":       datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "actor":    actor,
        "action":   action,
        "resource": resource,
        "detail":   detail,
        "severity": severity,
        "ip":       ip,
    }
    line = json.dumps(event)
    _file_logger.info(line)
    with _lock:
        _buffer.append(event)


def get_recent_events(n: int = 100) -> list:
    """Return last n events from the in-memory ring buffer."""
    with _lock:
        events = list(_buffer)
    return events[-n:]


def get_alert_events(n: int = 50) -> list:
    """Return only ALERT/WARN severity events."""
    with _lock:
        events = list(_buffer)
    return [e for e in events if e.get("severity") in ("ALERT", "WARN")][-n:]
