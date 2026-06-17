# 1. Serving configuration (Phase 1), your chosen flags, one line of justification each.

Flags used to launch VLLM (scripts/start_vllm.sh):

--max-model-len 4096: The prompts are ~3K tokens + short outputs (up to 512). Selecting bigger value would likely not affect performance, but keeping it tight helps in debugging of an accidental context overuse.

--gpu-memory-utilization 0.90: More room for KV cache without OOM risk.

--max-num-seqs 64: Allows enough concurrency to hit 10+ RPS.

--enable-chunked-prefill: Long prefills won't block decodes -> better P95 under load.

--enable-prefix-caching: Our requests share prefixes (system prompt + schema), so the system will definitely benefit from it.

# 2. Baseline eval results (Phase 5), overall pass rate, per-iteration pass rate, brief commentary.

Baseline eval: results/eval_baseline.json

- Overall pass rate: 11 out of 30.
- Per iteration pass rate: 10 (iter 1), 10 (iter 2), 11 (iter 3).
- Attempted iterations: 1 (27 requests), 3 (3 requests)

The improvement on the later iterations comes from uncertainty in the data format. LLM tries explicit fixed string condition like `department = 'Art and Design'`. When it fails (because such string does not exist in exactly this form), it repeats with `LOWER()`, `LIKE '%...%'` and succeeds. More info on this - in the answer #4 about agent value below.

Side note on quality improvement attempts:

- I tried using a different external model (gpt-4o-mini and gpt-5.4).
- I tried adding data samples for each table into the prompt.
- Neither of these 2 changes improved the pass rate.
- I believe, to improve a base eval, the agent's architecture must be changed. See my thoughts on it in the answer to the question #5.

# 3. Hitting the SLO (Phase 6), baseline performance vs. SLO, the iteration log, the final numbers.

## Baseline vs final numbers:


|          | RPS  | Latency p95 | Timeouts      | HTTP errors | Eval pass rate |
| -------- | ---- | ----------- | ------------- | ----------- | -------------- |
| Baseline | 8.3  | 77 s        | 5 out of 3000 | 0           | 11 out of 30   |
| Final    | 10.0 | 4.3 s       | 2 out of 3030 | 0           | 10 out of 30   |


The SLO (10 rps & p95 latency < 5 sec) was achieved after 3 optimisations (see below).

## Initial condition (baseline before any optimisations)

- The prompts were already optimized to share the prefix as much as possible.
- Fixed 2 bugs leading to http errors: a bug with attach_schema (in the provided code, led to occasional schema generation failures), and context overflow issue (to fix that, the execution results were truncated before sending them to verify step).
- KV cache has < 20% usage, 90%+ hit rate, no preemptions, vLLM's own waiting queue ~0 -> vLLM is not memory-bound.

## Optimisation steps

When testing at 10 rps (the capacity turned out to be 8.3 rps), VLLM end-to-end latency (p95) according to Grafana was 5 sec. Total agent request latency (p95) was 77 sec. Since the system is not memory restricted, it was unlikely that I can push VLLM end-to-end latency much further. And for any agent request these 5 sec of VLLM would add up to a significant number, if the number of LLM calls is big. So the main decision was to reduce the number of LLM calls per agent request.

### Optimisation 1 ('early end')

I stopped resending data for verification if the answer has no sql errors, and at least 1 row was returned. Why I could afford this - because 27/30 eval questions pass verify on the first try, and evals worth revising always get revised because SQL returned 0 rows.

Results: acheived 9.84 rps (good!), p95 latency 8.4 s (much closer to the target!), eval grew from 11 to 12 (random rise, no quality decrease!)

Artifacts:

- screenshots/grafana_after_early_end.png
- screenshots/grafana_after.png is the same file as grafana_after_early_end.png because is was the key optimisation ('the change that moved the needle').
- results/eval_after_early_end.json

### Optimisation 2 ('2iter')

After the last optimisation p50 is 1.35s (perfect!), p95 is 8.4s (too big!), so we need to remove the long tail. Further reduction was done by reducing the number of iteratinos from 3 to 2. To compensate for this, I slightly modified prompts to force LLM use more LIKE conditions and do not assume it knows exact titles/strings which it tried to guess.

Result: p95 latency dropped from 8.4 to 6.2 sec (very good but not enough!)

Artifacts:

- screenshots/grafana_after_2iter.png
- results/eval_after_2iter.json

### Optimisation 3 ('fp8')

According to Grafana, end-to-end VLLM latency (p95) was still high enough - close to the target 5s, while VLLM call is only a part(s) of the request processing.

So I decided to try fp8 quantization to replace the original BF16 - mostly as a decode-speed lever, since memory isn't the constraint.

Result: less time spent in 'decode' stage (accoring to Grafana) => acheived the target SLO: 10.0 RPS, 4.3s p95 latency

I had to test at 10.1 RPS (not 10.0) to make the final result above 10.0.

It came at a price of quality reduction though: eval pass dropped to 10 from 11 out of 30 (even though there's some randomness in every eval run, the reported quality decrease is not random: running eval multiple times before and after the last optimisation, I see that the pass rate dropped from quite stable values 11-12 to 9-10).

Artifacts:

- screenshots/grafana_after_fp8.png
- results/eval_after_fp8.json

### Detailed optimisation summary


| Optimisation      | RPS requested | RPS achieved | latency p50/95/99 | Timeouts | HTTP err | Eval pass |
| ----------------- | ------------- | ------------ | ----------------- | -------- | -------- | --------- |
| Baseline          | 10            | 8.3          | 38/77/98 sec      | 5/3000   | 0        | 11/30     |
| After early end   | 10            | 9.84         | 1.3/8.4/14.6 sec  | 2/3000   | 0        | 12/30     |
| After 2iter       | 10            | 9.74         | 1.3/6.2/10.0 sec  | 2/3000   | 0        | 11/30     |
| After fp8 (final) | 10.1          | 10.03        | 0.9/4.3/6.7 sec   | 2/3030   | 0        | 10/30     |


# 4. Agent value, one paragraph. Did the loop actually help? How do you know? Cite the per-iteration pass rate.

The loop helped mostly in the situations when LLM is not sure about the exact value/title. Seeing 0 rows in SQL result, it iterated to achieve the right selection.

Example for the eval "Please list the full names of the students in the Student_Club that come from the Art and Design Department.":
- WHERE maj.department = 'Art and Design'" (iter 1)
- WHERE LOWER(maj.department) = 'art and design'" (iter 2)
- WHERE LOWER(maj.department) LIKE '%art and design%'" (iter 3)

I tried to convinse LLM to use LOWER() and LIKE '%...%' in all prompts from iter 1, but it just does not want to do it at the first attempt - at least not in all queries.

# 5. What you'd do with more time, and be specific here! "Add Kubernetes" doesn't count.

With more time, I'd work on quality assessment (evals) and improvement. This would reduce RPS, but who needs fast low-quality service?

1) The evals themselves are low quality (as discussed in Discord, different parts of the BIRD evals have noise level at 15%-50%). I myself found 2 problems (out of 30 evals) in the 'gold' queries. So I would make at lease 30-50 my own evals which I can rely on.

2) So I would use a richer agentic loop.

From what I debugged, most quality problems come from the uncertainty in data format:
- 'F' or 'female'?
- 'Art and Design' or 'Department of Art and Design' or 'Art and Design Department'?
- Does such row exist at all?

So my agent would have a richer graph:
- proper planning
- data sampling and exploration
- getting all unique column values when needed
- data check (does the row exist?)
- partial SQL running for intermediate verification
- etc.
