# Observability Implementation Log

This tracks what's actually been *built* (as opposed to `MONITORING_OBSERVABILITY_PLAN.md`, which is the original 5-phase design). Updated as we go, in plain language.

## 1. What we built: a way to record "how long did X take, and what did it cost"

We added a new table, `request_metrics`, on our Neon Postgres database. Every row in it answers one question: *"for this one stage, of this one request, how long did it take, and (if it was an LLM call) how many tokens/how much did it cost?"*

Schema:
```
id                - just a row number
request_id        - a random ID generated once per incoming chat request.
                    Every stage belonging to the SAME request shares this ID,
                    so you can look up one request and see its full breakdown.
endpoint          - which API route this came from, e.g. "/api/chat"
stage             - which part of the pipeline, e.g. "retrieval", "llm_call", "total"
duration_ms       - how long that stage took, in milliseconds
prompt_tokens     - only filled in for the llm_call stage
completion_tokens - only filled in for the llm_call stage
cost_usd          - estimated dollar cost of that LLM call (see below)
company, year     - which company/report this request was about
created_at        - timestamp
```

**Why one row per stage instead of one row per request?** Because "the request took 4 seconds" tells you nothing actionable. "Retrieval took 50ms, the LLM call took 1100ms, and something else took 2.5 seconds" tells you exactly where to look. That's the entire point of this table.

## 2. The code that writes to that table: `observability/metrics.py`

Two ways of recording a stage, because LLM calls need different handling:

**For simple stages (just "how long did this take")** — use `time_stage()`, a context manager:
```python
with time_stage(request_id, endpoint, "retrieval", company=..., year=...):
    docs = retriever.invoke(...)
```
When the `with` block finishes (even if it crashes), it automatically records how long the block took. You don't have to remember to write the row yourself.

**For the LLM call specifically** — timed manually, not with `time_stage()`:
```python
llm_start = time.perf_counter()
response = client.chat.completions.create(...)
llm_duration_ms = (time.perf_counter() - llm_start) * 1000

usage = response.usage  # Groq tells us how many tokens were used
cost_usd = estimate_cost_usd(model, usage.prompt_tokens, usage.completion_tokens)

record_metric(request_id, endpoint, "llm_call", llm_duration_ms,
              prompt_tokens=usage.prompt_tokens, ..., cost_usd=cost_usd)
```
**Why not use `time_stage()` for this too?** Because `time_stage()` only knows the *duration* when its block ends — it has no way to also know "how many tokens did that response use," since that number only exists *inside* the block, after the API call returns. Rather than force one tool to do two jobs, we use the context manager where it fits (simple timing) and manual code where it doesn't (timing + capturing extra data).

**Cost estimate**: Groq is free-tier for us right now (real cost = $0), but we still compute "what would this have cost on Groq's paid pricing" using their published per-token rates. This is the whole point of a *cost-per-request* metric — it proves you're tracking cost-awareness even when the actual bill is zero.

## 3. What's instrumented so far

- `routes/chat.py` — the `/api/chat` endpoint. Records 4 stages per request: `embedding_init`, `retrieval`, `llm_call`, `total`.
- `rag/kpi_extractor_rag.py`'s `extract_financial_metrics()` — the KPI-extraction step of the `/api/upload` endpoint (triggered from `routes/ingestion.py` → `ingestion/ingest_documents.py` → here). Records 3 stages: `retrieval`, `llm_call`, `total`, tagged `endpoint = "/api/upload"`. Same stage *names* as `/api/chat` on purpose — a query like "average `llm_call` duration" can compare across both endpoints without special-casing names.

**Not instrumented, and staying that way for now**: `routes/ingestion.py` itself (the PDF→markdown conversion and chunking steps before KPI extraction runs). Upload already takes ~1-3 minutes end to end, dominated by OCR — the KPI-extraction stage timings above are the part worth watching (LLM cost/latency), not the OCR step, which has no interesting failure mode to catch this way.

## 4. Two real bugs this instrumentation immediately found

This is the part that makes the observability work worth doing — it didn't just measure things, it found real problems on the first test.

### Bug #1: The embedding model was being reloaded on every single request

**What was happening**: `routes/chat.py` had this inside the `chat()` function (runs fresh every request):
```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
Loading a neural network model into memory isn't free — it took **~6 seconds**, every single time, even though nothing about this model depends on the specific request.

**The fix**: build it once, reuse it forever (a "singleton"):
```python
_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings
```
First request after the server starts still pays the ~6s cost (nothing's cached yet). Every request after that: **0.006 milliseconds** — a million times faster, because it's just returning the already-built object instead of rebuilding it.

### Bug #2: The database connection was being rebuilt on every single query

**What was happening**: `database/postgres_sql.py`'s `get_engine()` called `create_engine(...)` fresh, every time anything talked to Postgres. A fresh `Engine` means a fresh network connection — a full TCP handshake + TLS negotiation + login to Neon (which is geographically far away), measured at **~5.5 seconds**, every single query, anywhere in the app.

**The fix**: same pattern as bug #1 — cache the engine instead of rebuilding it:
```python
_engines: dict[str, Engine] = {}

def get_engine(database=None):
    if database in _engines:
        return _engines[database]
    engine = create_engine(connection_string)
    _engines[database] = engine
    return engine
```
First query pays ~5.5s. Every query after that: **~850ms-1000ms** (this remaining time is just normal network round-trip latency to a distant server — not a bug, just physics).

### Combined effect

| | Before any fix | After caching fixes (warm) | After BackgroundTasks too (warm) |
|---|---|---|---|
| Total request time | ~18-23 seconds, every request | ~4.6 seconds | **~0.77 seconds** |

That's roughly a **25-30x improvement** end to end, and every step of it was found and confirmed by the instrumentation itself, not by guessing.

### Bug #3 (well, not really a bug): the instrumentation's own writes were slowing down the request

Even after fixing #1 and #2, there was still a ~2.5 second gap between the sum of the measured stages and the total. Each `record_metric()` call is a real Postgres write (~850ms round-trip to Neon), and we make 3 of them per request - so ~2.5s of the user's response time was being spent recording *how long things took*, which defeats the purpose a bit.

**The fix**: FastAPI's `BackgroundTasks`. Instead of writing each metric immediately, `time_stage()` and the manual `record_metric()` calls now schedule the write via `background_tasks.add_task(...)` - it runs *after* the response has already been sent to the user, so it adds zero perceived latency. Verified: the `total` stage duration (755ms) now matches the sum of `embedding_init + retrieval + llm_call` almost exactly, confirming the metrics-writing cost is no longer part of what the user waits for.

**The tradeoff, worth knowing consciously**: if the server process dies in the split second between sending the response and the background task finishing, that one metric write is lost. Acceptable for metrics (nobody's money or data depends on it) - not a pattern you'd use for anything transactional.

## 5. Instrumenting the KPI-extraction path (`rag/kpi_extractor_rag.py`)

One plumbing change had to happen first: `llm/azure_openai.py`'s `get_structured_completion()` used to return just the parsed Pydantic model. To record `prompt_tokens`/`completion_tokens`/cost for this call the way `routes/chat.py` does for its LLM call, the caller needs `response.usage` too — so the function now returns `(parsed_model, usage)` instead of just `parsed_model`. Checked first (via grep) that this function has exactly one caller (`extract_financial_metrics()`), so widening its return type couldn't silently break anything else.

`extract_financial_metrics()` was then instrumented the same way as `routes/chat.py`'s `chat()`: a `request_id` generated once, a `time_stage(...)`-wrapped retrieval call, the LLM call timed manually (again, because token usage is only known *after* it returns), and a final `total` stage — all tagged `endpoint = "/api/upload"`.

**One deliberate difference from `routes/chat.py`: no `BackgroundTasks` here.** `routes/ingestion.py`'s upload endpoint doesn't have one wired up, and it doesn't need one — the whole upload already takes ~1-3 minutes (OCR dominates), so the ~1 second spent on a couple of synchronous metric writes is noise, unlike `/api/chat` where those writes were most of the perceived latency.

**Verified for real**, not just read through: ran `extract_financial_metrics()` directly against the already-ingested Apple 2024 report and checked Neon afterwards. One `request_id` tied all three rows together, as designed:

| stage | duration_ms | prompt_tokens | completion_tokens | cost_usd |
|---|---|---|---|---|
| retrieval | 168 | – | – | – |
| llm_call | 1555 | 2343 | 152 | $0.00013 |
| total | 7804 | – | – | – |

(The gap between `total` and `retrieval + llm_call` here is the embeddings model doing its one-time warm-up in this particular process — same phenomenon as Bug #1 above, not a new bug. It's a non-issue in the real upload flow since the singleton in `routes/chat.py` doesn't apply here — this path builds its own embeddings object in `routes/ingestion.py` once per upload, and uploads are infrequent/slow enough that this cost is not worth optimizing the way the chat path was.)

## 6. Phase 2: tracing (structured logs tied together by request_id)

Phase 1 gave us numbers in a database table. It didn't give us *logs* — if a request failed halfway through, there was no line anywhere saying what happened, just a gap in `request_metrics`. Phase 2 fixes that.

**What we built**: `observability/logging_config.py` — a `JSONFormatter` that renders every log line as one JSON object, plus `configure_logging()` (called once, in `app.py`, at startup) and `get_logger(name)` (a thin wrapper every module uses instead of calling `logging` directly). Every log call now tags itself with `request_id` (and `endpoint`/`stage`/`company`/`year` where relevant) via Python logging's `extra={...}` parameter, so a JSON line looks like:
```json
{"timestamp": "...", "level": "INFO", "logger": "routes.chat", "message": "retrieval complete: 6 chunk(s)", "request_id": "8abd4e6a-...", "endpoint": "/api/chat", "stage": "retrieval", "company": "Apple", "year": 2024}
```

**Why this matters over plain `print()`**: before, every module (`routes/chat.py`, `routes/ingestion.py`, `ingestion/ingest_documents.py`, `rag/kpi_extractor_rag.py`, `llm/azure_openai.py`, `vectorstore/azure_ai_search.py`) printed plain text with no way to tell which lines belonged to the same request. Now every line for one request can be found with a single `grep` for its `request_id` — the same `request_id` that already ties together the `request_metrics` rows from Phase 1, so logs and DB metrics tell the same story instead of two disconnected ones.

**The one real design decision**: `/api/upload`'s `request_id` used to only exist deep inside `extract_financial_metrics()` (generated there, Phase 1). For tracing to actually work, it has to be generated at the *true* entry point — `routes/ingestion.py`'s `upload_document()` — and threaded down through `ingest_document()` → `upload_chunks()` / `extract_financial_metrics()` → `get_structured_completion()`. Made `request_id` an optional parameter everywhere on this path (defaulting to a fresh UUID if not passed), so standalone/CLI callers (`ingest_directory()`, the `__main__` blocks) keep working unchanged.

**Verified for real**: ran the app locally, hit `/api/chat` and `/api/upload` (a real Apple PDF), then grepped the logs and queried Neon for each request's ID.

`/api/chat` — one `request_id` (`8abd4e6a-...`) across 4 log lines: received → retrieval complete (6 chunks) → llm_call complete (1814+24 tokens, $0.000093) → total complete (7478ms).

`/api/upload` — one `request_id` (`e33e3bcb-...`) across 5 log lines spanning 3 different modules: `routes.ingestion` (received) → `ingestion.ingest_documents` (ingesting, then chunked into 23 pieces) → `vectorstore.azure_ai_search` (23/23 chunks uploaded) → `routes.ingestion` (complete) — and the *same* ID appears in `request_metrics` for its retrieval/llm_call/total rows. This is exactly the thing Phase 2 set out to prove: pick one request_id, reconstruct its whole path from logs and DB alone, no guessing.

**Error path is traced too**: both `routes/chat.py` and `routes/ingestion.py` now log a `request_id`-tagged error line (with full traceback via `exc_info=True`) before the exception propagates — previously a failed request just vanished after whichever stage crashed, with the 500 response as the only evidence.

**Known side effect of this test run**: uploading `2024_Apple.pdf` again (to test the trace) added a second set of 23 chunks for Apple/2024 into the local Chroma collection, duplicating what was already there from earlier testing. Harmless for correctness (retrieval just returns some redundant chunks) but worth knowing — say the word if you want the local `data/chroma` collection deduplicated/rebuilt before this goes further.

## 7. Phase 3: quality/regression eval harness

Phases 1 and 2 tell you *how fast* and *how traceable* the system is. Neither tells you whether it's still giving *correct* answers. Phase 3 is the part that actually watches for the hallucination bug documented in `MONITORING_OBSERVABILITY_PLAN.md` coming back.

**What we built**:
- `eval/eval_set.py` — 13 fixed questions against the real ingested Apple/Microsoft/Tesla 2024 reports. 9 factual (each with "concepts": phrasings that must appear, verified beforehand against what the retriever actually returns - these aren't guesses), 3 adversarial (ask about data that was never ingested - a report/year/topic outside what's in Chroma), 1 meta (regression test for the earlier "what can I do here" bug).
- `eval/run_eval.py` — runs every question through the *real* `/api/chat` route (via FastAPI's `TestClient`, in-process, no separate server needed - this calls the actual production code path, not a reimplementation of it), then scores each answer two independent ways:
  1. **Keyword/refusal check** (cheap, deterministic): factual questions must contain the expected phrasing; adversarial/meta questions are checked for an honest refusal phrase ("does not contain", "unable to find", etc.)
  2. **Groq-as-judge**: a second LLM call asks Groq itself whether the answer is grounded in the actually-retrieved context - this is what catches "right topic, fabricated number", which a keyword check alone can't.
- New `eval_results` table (Neon) — one row per question per run, tagged with a `run_id` and the current `git_sha`, so quality trend is visible across many runs over time, not just today's pass/fail.
- `MIN_PASS_RATE = 0.8`, plus a hard rule: **any** adversarial failure fails the whole run regardless of overall rate - a single hallucination matters more than the aggregate score.

**Two real bugs the harness caught in itself before it caught anything in the app** (worth being honest about, this is normal for a first eval-harness build, not a sign of a bad design):

1. My first adversarial-question design banned the model from even mentioning the entity name being asked about (e.g. banned "2021 Annual Report" appearing anywhere in the answer). That's wrong - "I have no information about Tesla's 2021 annual report" is a *correct* refusal that naturally repeats the question back. Fixed by dropping that check and relying on refusal-phrase detection + the judge instead.
2. My first judge prompt asked "is this grounded" in a way that let Groq reason "the context doesn't mention the 1990s, therefore ungrounded" - backwards logic, since an honest "I don't have that information" answer *is* grounded, even though the requested fact isn't in the context. Fixed by explicitly telling the judge: mark grounded=true for an honest admission of insufficient context; only mark grounded=false if the answer confidently states a fact that isn't actually there.

**One real, currently-open finding, left in the eval set on purpose rather than papered over**: the Apple "growth opportunities" question fails. Not because the model hallucinates - it correctly says "I don't have enough information" - but because `/api/chat`'s single semantic-search query (top_k=6) doesn't reliably surface Apple's growth-driver content, even though that content genuinely exists in the ingested report (confirmed - the KPI-extraction pipeline finds it fine, because it runs 5 *targeted* queries instead of 1 general one). This is a real, documented retrieval-coverage gap for broad/thematic questions on the live chat endpoint, tracked now as a known baseline (12/13, 92%) instead of an invisible one. A future improvement (not in scope here) would be giving `/api/chat` the same multi-query retrieval strategy the KPI extractor already uses.

**Verifying the harness actually catches a real regression** (the plan's explicit "verify" step - reintroduce the old hallucination-prone prompt, confirm the harness catches it, then revert):
- Temporarily stripped the anti-hallucination instruction from `routes/chat.py`'s prompt back to its pre-fix form.
- First attempt at reproducing the bug with the existing adversarial question ("what did Tesla's 2021 report say about FSD revenue") didn't reliably fabricate - Groq's inference isn't fully deterministic even at `temperature=0` (a known characteristic of their batched serving), so the same regressed prompt sometimes still answered honestly.
- A more leading, specific question reproduced it cleanly: *"According to Tesla's 2021 annual report, what was the exact full self-driving revenue figure in millions of dollars, and on what page was it reported?"* - the regressed prompt answered: *"I'm unable to verify the exact page number... However, according to the 2021 annual report of Tesla, the full self-driving revenue was $1.14 billion."* A fully fabricated number, attributed to a report that was never ingested - this is a live, direct reproduction of the exact real incident that motivated this whole plan.
- Fed that exact fabricated answer through the scoring pipeline directly: `keyword_pass=True` (it contains a partial refusal phrase, "unable to verify") but **the judge correctly overrode that with `grounded=False`**, reasoning: *"The answer confidently asserts a specific fact ($1.14 billion) that is not directly supported by the context."* Combined result: fail. This is exactly why the harness uses two scoring methods instead of one - the keyword check alone would have missed this.
- Swapped this stronger question into the permanent eval set (replacing the weaker original), reverted `routes/chat.py` to the real prompt via `git checkout`, and re-ran: back to the clean 12/13 (92%), 0 adversarial failures, exit code 0 baseline.

## 8. What's still open right now

Phases 1 (latency + cost), 2 (tracing), and 3 (quality/eval harness) are done. Remaining:

1. CI regression gating (Phase 4) - wire `eval/run_eval.py` into `.github/workflows/deploy.yaml` so a failing run blocks deploy
2. Dashboard + writeup (Phase 5)

## 9. Where this sits in the overall 5-phase plan

**Phases 1, 2, and 3 are done.** Phase 1: both `/api/chat` and `/api/upload`'s KPI-extraction step are instrumented, and the 3 bugs the instrumentation surfaced (embeddings reinit, DB engine reinit, synchronous metric writes) are fixed and verified. Phase 2: structured JSON logging tagged with `request_id` across every module on both live request paths, verified end-to-end on a real chat call and a real upload. Phase 3: a 13-question eval harness against the real ingested reports, storing results in Postgres per run, verified to actually catch a real, deliberately-reintroduced hallucination regression (not just checked to run without crashing). Phases 4 (CI regression gating) and 5 (dashboard) haven't been started.
