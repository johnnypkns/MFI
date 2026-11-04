import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import psycopg
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "postgresql://localhost/mfi"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def connect():
    return psycopg.connect(database_url(), autocommit=False)


def engine():
    url = database_url().replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url)


def apply_sql():
    """Apply every sql/*.sql file in order. All statements are idempotent."""
    with connect() as conn:
        for f in sorted((ROOT / "sql").glob("*.sql")):
            conn.execute(f.read_text())
        conn.commit()


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "no-git"


@contextmanager
def run(run_type: str, params: dict | None = None):
    """Open a research_runs row; mark ok/failed on exit. Yields run_id."""
    with connect() as conn:
        run_id = conn.execute(
            "INSERT INTO research_runs (run_type, code_version, params) VALUES (%s,%s,%s) RETURNING run_id",
            (run_type, code_version(), json.dumps(params or {}))).fetchone()[0]
        conn.commit()
    try:
        yield run_id
    except Exception as e:
        with connect() as conn:
            conn.execute("UPDATE research_runs SET status='failed', finished_at=now(), notes=%s WHERE run_id=%s",
                         (repr(e)[:2000], run_id))
            conn.commit()
        raise
    with connect() as conn:
        conn.execute("UPDATE research_runs SET status='ok', finished_at=now() WHERE run_id=%s", (run_id,))
        conn.commit()
