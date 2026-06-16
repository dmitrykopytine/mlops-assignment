"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 2

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm(max_tokens: int | None = None) -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default).

    `max_tokens` caps the completion length so a single H100 isn't tied up
    generating runaway output: SQL and the verify JSON are both short, so
    capping them keeps decode time (and KV-cache residency) low at ~10 RPS.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        max_tokens=max_tokens,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


_SQL_START = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Prompts ask for pure SQL, but real models (gpt-4o-mini and Qwen alike)
    occasionally wrap it in a fence or prepend a stray word, so we harden the
    parse rather than trust the instruction blindly:

    1. Prefer the first ```sql ... ``` block if one is present.
    2. Drop any leading prose before the first SELECT/WITH keyword.
    3. Strip a trailing semicolon so we hand a single clean statement to sqlite.
    """
    if not text:
        return ""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = (fenced.group(1) if fenced else text).strip()
    start = _SQL_START.search(candidate)
    if start:
        candidate = candidate[start.start():].strip()
    return candidate.rstrip().rstrip(";").strip()


def _parse_verify(text: str) -> tuple[bool, str]:
    """Defensively parse the verifier's {"ok": bool, "issue": str} reply.

    The model may wrap the JSON in prose or fences, so we regex out the verdict
    instead of a strict json.loads. If no verdict is found we accept (ok=True)
    rather than spend another LLM call on an ambiguous reply.
    """
    verdict = re.search(r'"?ok"?\s*[:=]\s*(true|false)', text, re.IGNORECASE)
    if verdict is None:
        return True, ""
    ok = verdict.group(1).lower() == "true"
    issue_match = re.search(r'"?issue"?\s*[:=]\s*"([^"]*)"', text, re.IGNORECASE)
    issue = issue_match.group(1).strip() if issue_match else ""
    return ok, issue


def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    The system message carries the static rules + schema (a prefix-cache hit
    across every request for this db_id); the user message puts the question on
    top with the STEP section last, so all three step prompts share the longest
    possible prefix. See prompts.py for the full layout rationale.
    """
    response = llm(max_tokens=512).invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM.format(schema=state.schema)),
        ("user", prompts.GENERATE_USER.format(question=state.question)),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: build messages from the VERIFY_*
    prompts, call llm(), parse the reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.
    """
    result = state.execution.render() if state.execution is not None else "(no result)"
    response = llm(max_tokens=128).invoke([
        ("system", prompts.VERIFY_SYSTEM.format(schema=state.schema)),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            result=result,
        )),
    ])
    ok, issue = _parse_verify(response.content)
    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{"node": "verify", "ok": ok, "issue": issue}],
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    result = state.execution.render() if state.execution is not None else "(no result)"
    response = llm(max_tokens=512).invoke([
        ("system", prompts.REVISE_SYSTEM.format(schema=state.schema)),
        ("user", prompts.REVISE_USER.format(
            question=state.question,
            sql=state.sql,
            result=result,
            issue=state.verify_issue or "(no issue given)",
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_execute(state: AgentState) -> str:
    """Gate the LLM verify call behind a cheap deterministic check.

    27/30 eval questions pass on the first try, yet every happy-path request was
    paying a `verify` LLM call that the eval shows rescues only ~1/30 answers.
    Under the open-loop load test that extra call is what keeps capacity below
    the 10 RPS target. So when the SQL executed cleanly and returned at least one
    row, we trust it and end; we only spend a verify (and possibly revise) call
    when execution errored or came back empty - the cases actually worth fixing.
    """
    ex = state.execution
    if ex is not None and ex.ok and ex.row_count > 0:
        return "end"
    return "verify"


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_conditional_edges(
        "execute",
        route_after_execute,
        {"verify": "verify", "end": END},
    )
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
