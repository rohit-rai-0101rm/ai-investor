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

## 6. What's still open right now

Nothing left in the original Phase 1 checklist (latency + cost tracking on both endpoints). Everything below is Phase 2+ work, not started yet:

1. Distributed tracing (Phase 2)
2. Quality/regression eval harness (Phase 3)
3. CI regression gating (Phase 4)
4. Dashboard + writeup (Phase 5)

## 7. Where this sits in the overall 5-phase plan

**Phase 1 (latency + cost tracking) is done** — both `/api/chat` and `/api/upload`'s KPI-extraction step are instrumented, and the 3 bugs the instrumentation surfaced (embeddings reinit, DB engine reinit, synchronous metric writes) are fixed and verified. Phases 2 (tracing), 3 (quality/eval harness), 4 (CI regression gating), and 5 (dashboard) haven't been started.
