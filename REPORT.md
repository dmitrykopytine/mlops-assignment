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
dm9@computeinstance-e00rg2d06zca2cc5s5:~/mlops-assignment$ uv run python load_test/driver.py --rps 10 --duration 300
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
