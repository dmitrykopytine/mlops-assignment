# 1. Serving configuration (Phase 1): chosen flags, one line of justification each

Flags used to launch vLLM (`scripts/start_vllm.sh`):

- `--max-model-len 4096`: The prompts are ~3K tokens plus short outputs (up to 512). A bigger value would likely not affect performance, but keeping it tight helps when debugging accidental context overuse.
- `--gpu-memory-utilization 0.90`: More room for the KV cache without OOM risk.
- `--max-num-seqs 64`: Allows enough concurrency to hit 10+ RPS.
- `--enable-chunked-prefill`: Long prefills won't block decodes -> better p95 under load.
- `--enable-prefix-caching`: Our requests share prefixes (system prompt + schema), so the system clearly benefits from it.

# 2. Baseline eval results (Phase 5): overall pass rate, per-iteration pass rate, brief commentary

Baseline eval: `results/eval_baseline.json`

- Overall pass rate: 11 out of 30.
- Per-iteration pass rate: 10 (iter 1), 10 (iter 2), 11 (iter 3).
- Attempted iterations: 1 (27 requests), 3 (3 requests).

The improvement on the later iterations comes from uncertainty in the data format. The LLM tries an explicit fixed-string condition like `department = 'Art and Design'`. When it fails (because such a string does not exist in exactly this form), it retries with `LOWER()` and `LIKE '%...%'` and succeeds. More on this in answer #4 about agent value below.

Side note on quality improvement attempts:

- I tried using a different external model (gpt-4o-mini and gpt-5.4).
- I tried adding data samples for each table to the prompt.
- Neither of these two changes improved the pass rate.
- I believe that to improve the base eval, the agent's architecture must be changed. See my thoughts on it in the answer to question #5.

# 3. Hitting the SLO (Phase 6): baseline performance vs. SLO, the iteration log, the final numbers

## Baseline vs. final numbers

|          | RPS  | Latency p95 | Timeouts      | HTTP errors | Eval pass rate |
| -------- | ---- | ----------- | ------------- | ----------- | -------------- |
| Baseline | 8.3  | 77 s        | 5 out of 3000 | 0           | 11 out of 30   |
| Final    | 10.0 | 4.3 s       | 2 out of 3030 | 0           | 10 out of 30   |

The SLO (10 RPS and p95 latency < 5 s) was achieved after 3 optimisations (see below).

## Initial condition (baseline before any optimisations)

- The prompts were already optimised to share the prefix as much as possible.
- Fixed 2 bugs leading to HTTP errors: a bug with `attach_schema` (in the provided code, which led to occasional schema generation failures), and a context overflow issue (to fix it, the execution results were truncated before being sent to the verify step).
- The KV cache has < 20% usage, a 90%+ hit rate, no preemptions, and vLLM's own waiting queue is ~0 -> vLLM is not memory-bound.

## Optimisation steps

When testing at 10 RPS (the capacity turned out to be 8.3 RPS), vLLM end-to-end latency (p95) according to Grafana was 5 s. Total agent request latency (p95) was 77 s. Since the system is not memory-bound, it was unlikely that I could push vLLM end-to-end latency much further. And for any agent request, these 5 s of vLLM would add up to a significant number if the number of LLM calls is large. So the main decision was to reduce the number of LLM calls per agent request.

### Optimisation 1 ('early end')

I stopped resending data for verification if the answer had no SQL errors and at least 1 row was returned. I could afford this because 27/30 eval questions pass verify on the first try, and evals worth revising always get revised because the SQL returned 0 rows.

Results: achieved 9.84 RPS (good!), p95 latency 8.4 s (much closer to the target!), eval grew from 11 to 12 (random rise, no quality decrease!).

Artifacts:

- `screenshots/grafana_after_early_end.png`
- `screenshots/grafana_after.png` is the same file as `grafana_after_early_end.png` because it was the key optimisation ('the change that moved the needle').
- `results/eval_after_early_end.json`

### Optimisation 2 ('2iter')

After the last optimisation, p50 is 1.35 s (perfect!) and p95 is 8.4 s (too big!), so we need to remove the long tail. Further reduction was done by reducing the number of iterations from 3 to 2. To compensate, I slightly modified the prompts to force the LLM to use more LIKE conditions and not assume it knows the exact titles/strings that it tried to guess.

Result: p95 latency dropped from 8.4 to 6.2 s (very good, but not enough!).

Artifacts:

- `screenshots/grafana_after_2iter.png`
- `results/eval_after_2iter.json`

### Optimisation 3 ('fp8')

According to Grafana, end-to-end vLLM latency (p95) was still high enough - close to the target of 5 s - while the vLLM call is only a part of the request processing.

So I decided to try fp8 quantization to replace the original BF16 - mostly as a decode-speed lever, since memory isn't the constraint.

Result: less time spent in the 'decode' stage (according to Grafana) => achieved the target SLO: 10.0 RPS, 4.3 s p95 latency.

I had to test at 10.1 RPS (not 10.0) to make the final result above 10.0.

It came at the price of quality reduction, though: eval pass dropped to 10 from 11 out of 30. (Even though there's some randomness in every eval run, the reported quality decrease is not random: running the eval multiple times before and after the last optimisation, I see that the pass rate dropped from quite stable values of 11-12 to 9-10.)

Artifacts:

- `screenshots/grafana_after_fp8.png`
- `results/eval_after_fp8.json`

### Detailed optimisation summary

| Optimisation      | RPS requested | RPS achieved | Latency p50/95/99 | Timeouts | HTTP err | Eval pass |
| ----------------- | ------------- | ------------ | ----------------- | -------- | -------- | --------- |
| Baseline          | 10            | 8.3          | 38/77/98 sec      | 5/3000   | 0        | 11/30     |
| After early end   | 10            | 9.84         | 1.3/8.4/14.6 sec  | 2/3000   | 0        | 12/30     |
| After 2iter       | 10            | 9.74         | 1.3/6.2/10.0 sec  | 2/3000   | 0        | 11/30     |
| After fp8 (final) | 10.1          | 10.03        | 0.9/4.3/6.7 sec   | 2/3030   | 0        | 10/30     |


# 4. Agent value (one paragraph): Did the loop actually help? How do you know? Cite the per-iteration pass rate.

The loop helped mostly in situations where the LLM is not sure about the exact value/title. Seeing 0 rows in the SQL result, it iterated to achieve the right selection.

Example for the eval "Please list the full names of the students in the Student_Club that come from the Art and Design Department.":
- `WHERE maj.department = 'Art and Design'` (iter 1)
- `WHERE LOWER(maj.department) = 'art and design'` (iter 2)
- `WHERE LOWER(maj.department) LIKE '%art and design%'` (iter 3)

I tried to convince the LLM to use `LOWER()` and `LIKE '%...%'` in all prompts from iter 1, but it just doesn't want to do it on the first attempt - at least not in all queries.

# 5. What you'd do with more time (be specific here! "Add Kubernetes" doesn't count)

With more time, I'd work on quality assessment (evals) and improvement. This would reduce RPS, but who needs a fast low-quality service?

1) The evals themselves are low quality (as discussed in Discord, different parts of the BIRD evals have a noise level of 15%-50%). I myself found 2 problems (out of 30 evals) in the 'gold' queries. So I would make at least 30-50 of my own evals that I can rely on.

2) I would use a richer agentic loop.

From what I debugged, most quality problems come from the uncertainty in the data format:
- 'F' or 'female'?
- 'Art and Design', 'Department of Art and Design', or 'Art and Design Department'?
- Does such a row exist at all?

So my agent would have a richer graph:
- proper planning
- data sampling and exploration
- getting all unique column values when needed
- data check (does the row exist?)
- partial SQL running for intermediate verification
- etc.

Or probably it would be a thinking LLM + tool use (SQLite as MCP)
