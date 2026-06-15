"""Replay SQLs from a run_eval.py DEBUG failure dump.

Paste a failure block (the chunk printed by run_eval.py when DEBUG is on - it
contains `Database:`, `Expected (gold) SQL:` and `Actual SQL:`) into the DUMP
string below, then just run:

    uv run python evals/sql.py

It parses out the database + both SQLs, runs them read-only against the
selected BIRD sqlite DB, and shows the rows plus whether the result sets match.
"""
from __future__ import annotations

import re
import sqlite3

from evals.run_eval import DB_DIR, canonicalize, matches

MAX_PREVIEW_ROWS = 30

# ---- Paste the failure dump between the triple quotes -----------------
DUMP = """
Database: financial

Expected (gold) SQL:
SELECT AVG(T1.A15) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE STRFTIME('%Y', T2.date) >= '1997' AND T1.A15 > 4000

Actual SQL:
SELECT AVG("A11") 
FROM "district" 
WHERE "A11" > 4000 
AND "district_id" IN (
    SELECT DISTINCT "district_id" 
    FROM "account" 
    WHERE "date" >= '1997-01-01.0'
)

"""
# ----------------------------------------------------------------------


def parse_block(text: str) -> tuple[str | None, str | None, str | None]:
    """Extract (database, gold_sql, actual_sql) from a pasted failure dump.

    Order-independent: `Database:` is matched anywhere; the gold SQL is taken
    between `Expected (gold) SQL:` and `Actual SQL:`; the actual SQL is taken
    from `Actual SQL:` to the end (trailing separators/blank lines trimmed).
    """
    db_match = re.search(r"^\s*Database:\s*(.+)$", text, re.MULTILINE)
    db = db_match.group(1).strip() if db_match else None

    gold_match = re.search(
        r"Expected \(gold\) SQL:\s*\n(.*?)\n\s*Actual SQL:",
        text,
        re.DOTALL,
    )
    gold = gold_match.group(1).strip() if gold_match else None

    actual_match = re.search(r"Actual SQL:\s*\n(.*)\Z", text, re.DOTALL)
    actual = _trim_sql(actual_match.group(1)) if actual_match else None

    return db, gold, actual


def _trim_sql(raw: str) -> str:
    """Drop trailing blank lines and a trailing `---` separator."""
    lines = raw.splitlines()
    while lines and (not lines[-1].strip() or lines[-1].strip() == "---"):
        lines.pop()
    return "\n".join(lines).strip()


def run(db: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[str], list[tuple] | None, str | None]:
    """Run sql read-only against db. Returns (ok, columns, rows, error)."""
    path = DB_DIR / f"{db}.sqlite"
    if not path.exists():
        return False, [], None, f"DB not found: {path}"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return True, cols, rows, None
    except Exception as e:  # noqa: BLE001
        return False, [], None, f"{type(e).__name__}: {e}"


def _render(label: str, db: str, sql: str | None) -> list[tuple] | None:
    print(f"===== {label} =====")
    if not sql or sql == "(none)":
        print("(no SQL)\n")
        return None
    print(sql)
    ok, cols, rows, err = run(db, sql)
    if not ok:
        print(f"-> ERROR: {err}\n")
        return None
    print(f"-> OK: {len(rows or [])} row(s)")
    if cols:
        print("   columns: " + ", ".join(cols))
    for row in (rows or [])[:MAX_PREVIEW_ROWS]:
        print("   " + " | ".join("" if c is None else str(c) for c in row))
    if rows and len(rows) > MAX_PREVIEW_ROWS:
        print(f"   ... ({len(rows) - MAX_PREVIEW_ROWS} more rows)")
    print()
    return rows


def main() -> None:
    db, gold, actual = parse_block(DUMP)

    if not db:
        raise SystemExit("Could not find a 'Database:' line in DUMP.")
    if gold is None and actual is None:
        raise SystemExit("Could not find 'Expected (gold) SQL:' or 'Actual SQL:' in DUMP.")

    print(f"\nDatabase: {db}\n")
    gold_rows = _render("Expected (gold) SQL", db, gold)
    actual_rows = _render("Actual SQL", db, actual)

    if gold_rows is not None and actual_rows is not None:
        ok = matches(gold_rows, actual_rows)
        print(f"Match (canonicalized row sets): {ok}")
        if not ok:
            print(f"   gold:   {len(canonicalize(gold_rows) or [])} canonical row(s)")
            print(f"   actual: {len(canonicalize(actual_rows) or [])} canonical row(s)")
    else:
        print("Match: n/a (one or both SQLs did not execute)")


if __name__ == "__main__":
    main()
