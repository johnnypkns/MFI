"""Download a registered source, archive the raw bytes, and record provenance.

Raw files are immutable: data/raw/<source_key>/<YYYY-MM-DD>/<filename>. If the bytes
match a file already registered (same sha256), nothing is re-downloaded into the DB
and the existing file_id is returned — loaders are then skipped.
"""
import hashlib
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests

from .db import ROOT, connect
from .sources import MADRID_CKAN, SOURCES

RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "madrid-franchise-intel/0.1 (research; low volume)"}


@dataclass
class Fetched:
    file_id: int
    path: Path
    is_new: bool


def resolve_url(key: str) -> str:
    s = SOURCES[key]
    if "url" in s:
        return s["url"]
    pkg = requests.get(MADRID_CKAN, params={"id": s["ckan_package"]}, headers=UA, timeout=60).json()["result"]
    hits = [r for r in pkg["resources"] if s["resource_match"](r)]
    if len(hits) != 1:
        raise RuntimeError(f"{key}: expected 1 matching resource in {s['ckan_package']}, found {len(hits)}")
    return hits[0]["url"]


def ensure_source(conn, key: str) -> int:
    s = SOURCES[key]
    cols = ["publisher", "title", "dataset_url", "license", "geographic_resolution",
            "data_type", "update_frequency", "confidence", "notes"]
    vals = [s.get(c) for c in cols]
    row = conn.execute(
        f"""INSERT INTO sources (source_key, {', '.join(cols)}) VALUES (%s, {', '.join(['%s'] * len(cols))})
            ON CONFLICT (source_key) DO UPDATE SET {', '.join(f'{c}=EXCLUDED.{c}' for c in cols)}
            RETURNING source_id""", [key, *vals]).fetchone()
    return row[0]


def fetch(key: str, run_id: int) -> Fetched:
    url = resolve_url(key)
    day_dir = RAW / key / date.today().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    dest = day_dir / url.rsplit("/", 1)[-1].split("?")[0]
    tmp = dest.with_suffix(dest.suffix + ".part")

    h, n = hashlib.sha256(), 0
    with requests.get(url, headers=UA, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                h.update(chunk)
                n += len(chunk)
    sha = h.hexdigest()

    with connect() as conn:
        source_id = ensure_source(conn, key)
        existing = conn.execute("SELECT file_id, local_path FROM source_files WHERE source_id=%s AND sha256=%s",
                                (source_id, sha)).fetchone()
        if existing:
            tmp.unlink()
            conn.commit()
            return Fetched(existing[0], Path(existing[1]), is_new=False)
        shutil.move(tmp, dest)
        file_id = conn.execute(
            """INSERT INTO source_files (source_id, run_id, url, local_path, sha256, bytes)
               VALUES (%s,%s,%s,%s,%s,%s) RETURNING file_id""",
            (source_id, run_id, url, str(dest), sha, n)).fetchone()[0]
        conn.commit()
    return Fetched(file_id, dest, is_new=True)


def set_file_meta(file_id: int, row_count: int, reference_date=None):
    with connect() as conn:
        conn.execute("UPDATE source_files SET row_count=%s, data_reference_date=%s WHERE file_id=%s",
                     (row_count, reference_date, file_id))
        conn.commit()
