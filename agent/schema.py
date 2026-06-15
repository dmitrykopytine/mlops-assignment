"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _bt(ident: str) -> str:
    """Backtick-quote an identifier for the (non-executable) sample hints."""
    return "`" + ident.replace("`", "``") + "`"


# Set to False to disable data sampling and render only the bare CREATE TABLE
# statements (the original schema format, no "Data samples for ..." hints).
ENABLE_SAMPLING = False

# Sampling knobs. Kept small on purpose: the samples live in the prefix-cached
# part of the prompt, so they are prefilled once per db_id and reused.
SAMPLE_ROWS = 10  # rows scanned per table (no ORDER BY -> cheap + deterministic)
SAMPLES_PER_COL = 2  # unique non-NULL values shown per column
SAMPLE_MAX_LEN = 60  # truncate long text/blob-ish values to keep the prefix lean


def _format_value(v: object) -> str:
    """Render one cell as a SQL-ish literal: numbers raw, strings quoted+escaped."""
    if isinstance(v, bool):  # sqlite has no bool, but be safe
        return str(int(v))
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, bytes):
        return "<blob>"
    s = str(v).replace("\n", " ").replace("\r", " ").strip()
    if len(s) > SAMPLE_MAX_LEN:
        s = s[:SAMPLE_MAX_LEN] + "..."
    return "'" + s.replace("'", "''") + "'"


def _data_samples(conn: sqlite3.Connection, table: str) -> str:
    """Per-column value samples for one table, or "" if empty/unavailable.

    Scans up to SAMPLE_ROWS rows (unordered), then for each column shows up to
    SAMPLES_PER_COL unique non-NULL values; if NULL occurs in the scanned rows,
    a trailing NULL is appended.
    """
    try:
        cur = conn.execute(f"SELECT * FROM {_q(table)} LIMIT {SAMPLE_ROWS}")
        rows = cur.fetchall()
        names = [d[0] for d in cur.description]
    except Exception:  # noqa: BLE001 - a bad table shouldn't break the schema
        return ""
    if not rows:
        return ""

    lines = [f"\nData samples for {_bt(table)}:"]
    for idx, name in enumerate(names):
        seen_null = False
        uniques: dict[str, None] = {}
        for row in rows:
            val = row[idx]
            if val is None:
                seen_null = True
            elif len(uniques) < SAMPLES_PER_COL:
                uniques.setdefault(_format_value(val), None)
        samples = list(uniques.keys())
        if seen_null:
            samples.append("NULL")
        if samples:
            lines.append(f"- {_bt(name)}: {', '.join(samples)}")
    return "\n".join(lines)


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    parts: list[str] = [f"-- Database: {db_id}"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            col_lines: list[str] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                col_lines.append(line)
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                col_lines.append(
                    f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(fk[4])})"
                )
            parts.append(",\n".join(col_lines))
            parts.append(");")
            if ENABLE_SAMPLING:
                samples = _data_samples(conn, t)
                if samples:
                    parts.append(samples)
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
