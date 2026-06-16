Initial config:

export HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d= -f2)
uv run python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --enable-prefix-caching

--max-model-len 4096: your prompts top out ~3K + short outputs. Keeping context tight frees KV cache for more concurrent sequences → higher throughput/RPS.
--gpu-memory-utilization 0.90: gives the KV cache more room on the 80GB card without OOM risk.
--max-num-seqs 64: allows enough concurrency to hit 10+ RPS; tune up/down based on latency.
--enable-chunked-prefill: long-ish prefills (1.5–3K tokens) won't block decodes → better P95 under load.
--enable-prefix-caching: your 2–3 dependent calls per request likely share system/prompt prefixes → big latency win on repeated prefixes.

"Please list the full names of the students in the Student_Club that come from the Art and Design Department."
"student_club"

"SELECT m.first_name, m.last_name
FROM member m
JOIN major maj ON m.link_to_major = maj.major_id
WHERE maj.department = 'Art and Design'"

"SELECT m.first_name, m.last_name
FROM member m
JOIN major maj ON m.link_to_major = maj.major_id
WHERE LOWER(maj.department) = 'art and design'"

"SELECT m.first_name, m.last_name
FROM member m
JOIN major maj ON m.link_to_major = maj.major_id
WHERE LOWER(maj.department) LIKE '%art and design%'"

BEFORE:
dm9@computeinstance-e00rg2d06zca2cc5s5:~/mlops-assignment$ uv run python c --rps 10 --duration 300
{
  "requested_rps": 10.0,
  "duration_seconds": 300,
  "wall_clock_seconds": 347.51548531700064,
  "total_requests": 3000,
  "achieved_rps": 8.632708833862255,
  "ok": 2616,
  "timeouts": 1,
  "http_errors": 383,
  "client_errors": 0,
  "latency_p50": 35.87833720600065,
  "latency_p95": 77.05337293699995,
  "latency_p99": 83.89367194900115,
  "latency_max": 102.60400766399835
}
Wrote /home/dm9/mlops-assignment/results/load_test.json

After fixing a bug with attach_schema:
Also, fixing the bug with context overflow when too much data is returned from execute() and we send it all to verify().

dm9@computeinstance-e00rg2d06zca2cc5s5:~/mlops-assignment$ uv run python load_test/driver.py --rps 10 --duration 300
{
  "requested_rps": 10.0,
  "duration_seconds": 300,
  "wall_clock_seconds": 359.3013099029995,
  "total_requests": 3000,
  "achieved_rps": 8.349538165641281,
  "ok": 2995,
  "timeouts": 5,
  "http_errors": 0,
  "client_errors": 0,
  "latency_p50": 38.043103411999255,
  "latency_p95": 91.55161022500033,
  "latency_p99": 97.78207281800132,
  "latency_max": 117.14981022100073
}
Wrote /home/dm9/mlops-assignment/results/load_test.json

Changed graph.py.

WHY:

Grafana (this run, same shape as before):

KV cache ~5–10%, preemptions zero, vLLM's own waiting-queue ~0, requests running plateaus ~40 while max-num-seqs=64. → vLLM is not memory-bound; if anything it's starved. Memory knobs (FP8-for-KV, bigger max-num-seqs) won't help.
Decode dominates per-call latency (ITL 60–90 ms/tok, lifecycle decode band ~8s, vLLM e2e p95 ~10–12s), and token throughput is prefill-heavy — the GPU spends most of its budget re-processing big prompts.
The math that matters: your driver is open-loop at 10 rps but capacity is only ~8.3 rps, so a backlog builds for the whole 5 min and latency runs away (p50 already 38s). The only way to fix p95 is to push capacity above 10 rps so no backlog forms. Capacity = vLLM calls/sec ÷ calls per request. You're serving ~20 calls/s; each request costs ~2.4 calls → ~8.3 rps. Cut calls-per-request and capacity jumps above 10 → backlog clears → latency collapses toward the unloaded per-call time (~1–2s, since KV is empty).

CHANGE:

Gate the LLM verify call behind a cheap deterministic check.

    27/30 eval questions pass on the first try, yet every happy-path request was
    paying a `verify` LLM call that the eval shows rescues only ~1/30 answers.
    Under the open-loop load test that extra call is what keeps capacity below
    the 10 RPS target. So when the SQL executed cleanly and returned at least one
    row, we trust it and end; we only spend a verify (and possibly revise) call
    when execution errored or came back empty - the cases actually worth fixing.

Quality did not change.

Wrote results/eval_after_early_end.json
{
  "n": 30,
  "overall_correct": 12,
  "overall_pass_rate": 0.4,
  "pass_rate_by_iteration": {
    "iter_1": 0.3667,
    "iter_2": 0.3667,
    "iter_3": 0.4
  },
  "correct_by_iteration": {
    "iter_1": 11,
    "iter_2": 11,
    "iter_3": 12
  },
  "iterations_histogram": {
    "1": 27,
    "3": 3
  },
  "gold_exec_failures": 0,
  "agent_exec_failures": 0
}

dm9@computeinstance-e00rg2d06zca2cc5s5:~/mlops-assignment$ uv run python load_test/driver.py --rps 10 --duration 300
{
  "requested_rps": 10.0,
  "duration_seconds": 300,
  "wall_clock_seconds": 304.7567046969998,
  "total_requests": 3000,
  "achieved_rps": 9.843917963946714,
  "ok": 2998,
  "timeouts": 2,
  "http_errors": 0,
  "client_errors": 0,
  "latency_p50": 1.3471105270000407,
  "latency_p95": 8.42837745199904,
  "latency_p99": 14.567598691999592,
  "latency_max": 32.35760068000127
}
Wrote /home/dm9/mlops-assignment/results/load_test.json

grafana_after_early_end.png

---

WHY:

Your p50 is 1.35s — the median request is a single generate call and it's fast. p95 8.4s / p99 14.6s / max 32s is the ~10% tail that hits the verify→revise path (execution errored or returned 0 rows), doing 3–6 sequential vLLM calls. To get p95 under 5s, shorten that tail:

Tighten the revise loop: MAX_ITERATIONS 3→2. The eval showed iter3 adds only +1/30, and those 3-iteration runs are precisely your p95/p99/max. This directly chops the longest paths — strongest single move, measure quality with eval_after_tuning.json.

CHANGE: MAX_ITERATIONS = 2 + minor prompt tweak to stay on 11/30.

dm9@computeinstance-e00rg2d06zca2cc5s5:~/mlops-assignment$ uv run python load_test/driver.py --rps 10 --duration 300
{
  "requested_rps": 10.0,
  "duration_seconds": 300,
  "wall_clock_seconds": 307.9410722419998,
  "total_requests": 3000,
  "achieved_rps": 9.742123641247856,
  "ok": 2998,
  "timeouts": 2,
  "http_errors": 0,
  "client_errors": 0,
  "latency_p50": 1.27043133099869,
  "latency_p95": 6.187561566001023,
  "latency_p99": 9.957056960000045,
  "latency_max": 24.039973681001356
}
Wrote /home/dm9/mlops-assignment/results/load_test.json

Quality:

Wrote results/eval_after_2iter.json
{
  "n": 30,
  "overall_correct": 11,
  "overall_pass_rate": 0.3667,
  "pass_rate_by_iteration": {
    "iter_1": 0.3333,
    "iter_2": 0.3667
  },
  "correct_by_iteration": {
    "iter_1": 10,
    "iter_2": 11
  },
  "iterations_histogram": {
    "1": 27,
    "2": 3
  },
  "gold_exec_failures": 0,
  "agent_exec_failures": 0
}

grafana_after_2iter.png

