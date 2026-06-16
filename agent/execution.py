"""SQL execution helper (provided complete).

execute_sql() runs the agent's SQL against the target DB in read-only mode
and returns a structured ExecutionResult. The verify node consumes this
to decide whether the answer looks plausible.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from agent.schema import db_path


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    row_count: int = 0

    def render(
        self,
        max_rows: int = 10,
        max_cell_len: int = 80,
        max_chars: int = 1000,
    ) -> str:
        """Compact, length-bounded text rendering for prompt context.

        The schema already fills most of the model's 4096-token context, so an
        unbounded result (wide columns / long text cells) can push the verify or
        revise prompt over the limit and trigger a 400 from vLLM. We bound the
        output on three axes - rows, per-cell width, and total characters - so
        the prompt stays within budget regardless of what the query returned.
        """
        if not self.ok:
            err = self.error or ""
            if len(err) > max_chars:
                err = err[:max_chars] + " ...(truncated)"
            return f"ERROR: {err}"
        if self.row_count == 0:
            return "OK: 0 rows returned."

        def _cell(c: object) -> str:
            s = str(c)
            return s if len(s) <= max_cell_len else s[:max_cell_len] + "..."

        cols = ", ".join(self.columns or [])
        preview = "\n".join(
            " | ".join(_cell(c) for c in row) for row in (self.rows or [])[:max_rows]
        )
        more = f"\n... ({self.row_count - max_rows} more rows)" if self.row_count > max_rows else ""
        rendered = f"OK: {self.row_count} rows.\nCOLUMNS: {cols}\nFIRST ROWS:\n{preview}{more}"
        if len(rendered) > max_chars:
            rendered = rendered[:max_chars] + "\n...(output truncated)"
        return rendered


def execute_sql(db_id: str, sql: str, timeout_seconds: float = 5.0) -> ExecutionResult:
    """Run SQL against db_id's sqlite, return result or error."""
    path = db_path(db_id)
    try:
        with sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=timeout_seconds,
        ) as conn:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return ExecutionResult(ok=True, rows=rows, columns=cols, row_count=len(rows))
    except Exception as e:  # noqa: BLE001
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")
