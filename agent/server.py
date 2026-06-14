"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager, nullcontext
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402

# Langfuse callback handler. If keys are set we initialize it; failures
# are NOT swallowed - a misconfigured Langfuse should not silently
# produce zero traces.
#
# langfuse 4.x: the CallbackHandler auto-instruments the LangGraph run (one
# nested span per node: generate_sql / verify / (revise)), while trace-level
# attributes - name, tags, metadata - are set with the module-level
# propagate_attributes() context manager wrapped around graph.invoke(). The
# README's `from langfuse.callback import CallbackHandler` snippet is the old
# v2 API; the v4 import lives under langfuse.langchain.
_lf_handler: Any = None
_lf_client: Any = None
_propagate_attributes: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler

    _lf_handler = CallbackHandler()
    _lf_client = get_client()
    _propagate_attributes = propagate_attributes


def _trace_attributes(req: "AnswerRequest"):
    """Name the trace and attach filterable tags/metadata for Phase 6.

    Tags are emitted as `key:value` strings (plus `db:<db>`) so traces are
    filterable in the Langfuse trace list; the same fields are duplicated into
    metadata for richer inspection. No-op when Langfuse is not configured.
    """
    if _propagate_attributes is None:
        return nullcontext()
    tags = [f"{k}:{v}" for k, v in req.tags.items()]
    tags.append(f"db:{req.db}")
    return _propagate_attributes(
        trace_name="agent_run",
        tags=tags,
        metadata={"db": req.db, **{k: str(v) for k, v in req.tags.items()}},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Flush buffered events on shutdown so no trace is lost when the server
    # stops (e.g. after a load-test / eval run). Per-request flushing is
    # avoided on purpose - it would add network latency to the SLO path.
    if _lf_client is not None:
        _lf_client.flush()


app = FastAPI(lifespan=lifespan)


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    state = AgentState(question=req.question, db_id=req.db)
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": req.tags,
    }
    try:
        with _trace_attributes(req):
            final = graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    if execution is None:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
