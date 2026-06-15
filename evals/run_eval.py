"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"

# When DEBUG is on we (a) optionally restrict the run to a handful of evals and
# (b) print a detailed dump for every failed eval. Turn off for a clean run.
DEBUG = True

# 1-indexed line numbers in the eval set to run (matches the file's line
# numbers). Empty -> run the whole eval set as usual. Only honored when DEBUG.
DEBUG_EVALS_TO_RUN_LINE_NUM: list[int] = []


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _format_rows(rows: list[tuple] | None) -> str:
    """Render canonicalized rows for the DEBUG dump."""
    if rows is None:
        return "(query failed / no rows)"
    if not rows:
        return "(empty result set)"
    return "\n".join(" | ".join(cell for cell in row) for row in rows)


def _print_failure(line: int, r: dict) -> None:
    """Print a failed eval inline, right after its eval line in main()."""
    print("---")
    print(f"[line {line}] {r['question']}")
    print()
    print("Expected rows:")
    print(_format_rows(r.get("expected_rows")))
    print()
    print("Actual rows:")
    print(_format_rows(r.get("actual_rows")))
    print()
    print(f"Database: {r['db_id']}")
    print()
    print("Expected (gold) SQL:")
    print(r["gold_sql"])
    print()
    print("Actual SQL:")
    print(r["final_sql"] or "(none)")
    print()


def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question off the agent's *final* result only.

    We deliberately ignore the per-iteration history: whatever SQL the agent
    settled on (after however many revise loops) is what gets scored. A
    question that needed 3 iterations counts as a success only at iteration 3 -
    iterations 1 and 2 are treated as unsuccessful (see summarize()).

    DEBUG behavior lives here so main() stays untouched: eval_one tracks its own
    1-indexed call number (== eval-set line, since main iterates in file order).
    When DEBUG_EVALS_TO_RUN_LINE_NUM is non-empty, lines outside it are skipped
    (no agent call); failed evals are printed inline right after the eval line.
    """
    line = getattr(eval_one, "_line", 0) + 1
    eval_one._line = line  # type: ignore[attr-defined]

    db_id = question["db_id"]
    q_text = question["question"]
    gold_sql = question["gold_sql"]

    if DEBUG and DEBUG_EVALS_TO_RUN_LINE_NUM and line not in DEBUG_EVALS_TO_RUN_LINE_NUM:
        return {"line": line, "db_id": db_id, "question": q_text, "skipped": True}

    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    try:
        resp = httpx.post(
            agent_url,
            json={"question": q_text, "db": db_id, "tags": {"suite": "eval"}},
            timeout=120.0,
        ).json()
    except Exception as e:  # noqa: BLE001
        resp = {"sql": "", "iterations": 0, "ok": False, "error": f"request failed: {type(e).__name__}: {e}"}

    final_sql = resp.get("sql", "")
    pred_ok, pred_rows, pred_err = run_sql(db_id, final_sql) if final_sql else (False, None, "no SQL returned")
    final_correct = bool(gold_ok and pred_ok and matches(gold_rows, pred_rows))

    result = {
        "line": line,
        "question": q_text,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "final_sql": final_sql,
        "iterations": resp.get("iterations", 0),
        "gold_ok": gold_ok,
        "gold_error": gold_err,
        "agent_ok": bool(resp.get("ok")),
        "agent_error": resp.get("error"),
        "pred_ok": pred_ok,
        "pred_error": pred_err,
        "final_correct": final_correct,
        "expected_rows": canonicalize(gold_rows),
        "actual_rows": canonicalize(pred_rows),
    }

    if DEBUG and not final_correct:
        _print_failure(line, result)

    return result


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results from the final-iteration verdict.

    Per-iteration pass rate is built from the iteration count alone: a question
    that succeeded after `f` iterations contributes a success to iteration `f`
    and every later iteration (carry-forward), and nothing to iterations
    `< f`. So `iter_k` = "pass rate if the loop were allowed up to k iterations".
    """
    # DEBUG line selection may produce skipped stubs; they aren't scored.
    results = [r for r in results if not r.get("skipped")]
    n = len(results)
    max_iters = max((r["iterations"] for r in results), default=0)

    correct_at = {k: 0 for k in range(1, max_iters + 1)}
    for r in results:
        if r["final_correct"]:
            f = max(r["iterations"], 1)
            for k in range(f, max_iters + 1):
                correct_at[k] += 1

    overall_correct = sum(1 for r in results if r["final_correct"])

    iters_hist: dict[int, int] = {}
    for r in results:
        k = r["iterations"]
        iters_hist[k] = iters_hist.get(k, 0) + 1

    def rate(c: int) -> float:
        return round(c / n, 4) if n else 0.0

    return {
        "n": n,
        "overall_correct": overall_correct,
        "overall_pass_rate": rate(overall_correct),
        "pass_rate_by_iteration": {
            f"iter_{k}": rate(correct_at[k]) for k in range(1, max_iters + 1)
        },
        "correct_by_iteration": {
            f"iter_{k}": correct_at[k] for k in range(1, max_iters + 1)
        },
        "iterations_histogram": {str(k): iters_hist[k] for k in sorted(iters_hist)},
        "gold_exec_failures": sum(1 for r in results if not r["gold_ok"]),
        "agent_exec_failures": sum(1 for r in results if not r["agent_ok"]),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
