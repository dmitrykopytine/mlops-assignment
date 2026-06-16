"""Prompt templates for the agent nodes.

Layout is tuned for vLLM automatic prefix caching (single H100, ~10 RPS), so
the tokens are ordered most-static-first to changeable-last:

    [SYSTEM]  global rules (identical for every request)  ── shared by ALL calls
              + schema (identical for a given db_id)       ── shared by same-db calls
    [USER]    Question: <question>                         ── shared by the 3 steps
              ### STEP: <GENERATE|VERIFY|REVISE> + payload  ── unique tail

Concretely:
- Across different requests the long static SYSTEM prefix (rules + schema) is a
  cache hit, so the expensive schema tokens are only prefilled once per db_id.
- Within a single query, generate/verify/revise share
  `SYSTEM + "Question: <q>\n\n### STEP: "`, so each follow-up call re-prefills
  only its own short step section.
- The only changeable parts (step name, prior SQL, result, issue) live at the
  very end, after the shared prefix, exactly where the cache wants them.

`SYSTEM` is `.format(schema=...)`-ed; the `*_USER` templates are
`.format(...)`-ed with the per-step fields. Keep `{question}` first in every
user template so the per-query prefix stays identical across steps.
"""

SYSTEM = """You are a precise text-to-SQL agent for SQLite databases.

Rules:
- Dialect is SQLite.
- Use only the tables and columns in the schema below.
- Read-only: emit a single SELECT (or WITH ... SELECT) statement.
- Quote identifiers that contain spaces or are reserved words: `Table1`.`Column1`.
- Respond ONLY with requested columns/data, do not add extra to the query output - extra data will fail the eval.
- Every request ends with a STEP section. Produce exactly the OUTPUT it asks for and nothing else: no markdown fences, no comments, no prose.

When not sure about the data format, prepare for any format:
- Prefer LOWER(...) LIKE '%...%' over strict '=' for string comparisons.
- Gender can be 'female', 'f', 'F', etc.
- Filtering by date "Date" = '2010-07-19 19:39:08' may not work because the database expects second fraction part '.0' in the end: "Date" = '2010-07-19 19:39:08.0'.

Schema:
{schema}"""


# All three steps deliberately share one system prompt so that the
# `SYSTEM + schema` prefix is byte-identical across generate/verify/revise (and
# across every request for the same db_id), which is what lets vLLM's prefix
# cache reuse the expensive schema prefill. They're kept as separate names so
# each node references its own, but they intentionally point at the same value.
GENERATE_SQL_SYSTEM = SYSTEM
VERIFY_SYSTEM = SYSTEM
REVISE_SYSTEM = SYSTEM


# Available placeholders: {question}
GENERATE_USER = """Question: {question}

### STEP: GENERATE
Write one SQLite query that answers the question.
OUTPUT: the SQL query as plain text only."""


# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """Question: {question}

### STEP: VERIFY
Judge whether the executed query's result plausibly answers the question.

Query:
{sql}

Result:
{result}

Set ok=false when the query errored, returned zero rows although the question implies some should exist, or the columns/values clearly fail to answer the question. Otherwise set ok=true.
OUTPUT: one line of JSON: {{"ok": true|false, "issue": "<short reason, empty if ok>"}}"""


# Available placeholders: {question}, {sql}, {result}, {issue}
REVISE_USER = """Question: {question}

### STEP: REVISE
The previous query was rejected. Return a corrected query.

Previous query:
{sql}

Result:
{result}

Issue:
{issue}

OUTPUT: the corrected SQLite query as plain text only."""
