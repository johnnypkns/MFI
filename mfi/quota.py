"""Server-side daily caps on paid API calls, so public visitors can't run up the bill.

Caps come from secrets (MAX_CLAUDE_RUNS_PER_DAY, MAX_APIFY_RUNS_PER_DAY; defaults 20 / 20) and are counted
in a small JSON file per UTC day. Per-session cap: 3 of each.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "data" / "quota.json"
DEFAULT = {"claude": 20, "apify": 20}
PER_SESSION = 3


def _cap(kind):
    key = f"MAX_{kind.upper()}_RUNS_PER_DAY"
    v = os.environ.get(key)
    if not v:
        try:
            import streamlit as st
            v = st.secrets.get(key)
        except Exception:
            v = None
    return int(v) if v else DEFAULT[kind]


def _load():
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        d = json.loads(FILE.read_text())
    except Exception:
        d = {}
    return (d if d.get("day") == day else {"day": day}), day


def allow(kind, session):
    """True and counts the call if under both caps; False otherwise."""
    used_s = session.get(f"q_{kind}", 0)
    if used_s >= PER_SESSION:
        return False
    d, _ = _load()
    if d.get(kind, 0) >= _cap(kind):
        return False
    d[kind] = d.get(kind, 0) + 1
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(d))
    session[f"q_{kind}"] = used_s + 1
    return True
