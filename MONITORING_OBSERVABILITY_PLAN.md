# Monitoring & Observability Plan

Self-directed extension of the existing AI-Powered Investor Intelligence Platform — no tutorial for this part. Goal: add the production-AI concerns most portfolios skip — tracing, latency percentiles, cost-per-request, quality/regression metrics, and CI gating — on top of the RAG pipeline that already exists and is already deployed.

## Why this fits on top of the existing project

This project already has the real ingredients this challenge is testing observability *on*:
- A multi-stage pipeline (embed → retrieve → LLM call → DB save) worth tracing
- A live deployed endpoint (`/api/chat`, `/api/upload`) worth measuring latency on
- A real, already-documented incident (the hallucination bug we found and fixed on 2026-07-07 — Tesla chat citing a fabricated "2021 Annual Report") that's a genuine case study for *why* quality regression gating matters, not a hypothetical

No new project needed — this becomes a new capability layered onto `routes/chat.py`, `rag/kpi_extractor_rag.py`, and the CI workflow that already exist.

## Architecture — what gets added where

| Concern | Where it plugs in | Tool (free) |
|---|---|---|
| Tracing | Wrap each pipeline stage with a shared request ID; log stage boundaries | Structured JSON logging + correlation ID (stretch: OpenTelemetry) |
| Latency (p50/p95) | Time each stage + full endpoint calls | New `request_metrics` table on existing Neon Postgres |
| Cost-per-request | Groq's response includes `usage.prompt_tokens` / `completion_tokens` | Just arithmetic, stored alongside latency |
| Quality metrics | Groundedness/hallucination check, retrieval relevance | Small fixed eval set (~12-15 Q&A pairs) + Groq-as-judge scoring |
| Regression gating | Run eval set on every push; fail build if quality/latency regresses | New job in `.github/workflows/deploy.yaml`, gates the deploy step |
| Dashboard | Visualize the above | New "Observability" section in the existing `templates/dashboard.html` |

## Timeline (self-paced, assumes a few focused hours per session — compress or stretch freely)

### Phase 1 — Latency + cost tracking (Days 1-2)
- [ ] Add `request_metrics` table (Neon): `id, endpoint, stage, duration_ms, prompt_tokens, completion_tokens, cost_usd, created_at`
- [ ] Add a small `observability/metrics.py` helper: a context manager/decorator that times a code block and writes a row
- [ ] Instrument `routes/chat.py` and `rag/kpi_extractor_rag.py`: wrap retrieval, Groq call, and total request time
- [ ] Capture `response.usage` from Groq's API response for token counts; compute cost using Groq's published per-token pricing (report as "$0 - free tier" alongside "would cost $X on paid tier" for portfolio narrative)
- [ ] Verify: make a few real chat/upload calls, query `request_metrics` directly to confirm rows land correctly

### Phase 2 — Tracing (Days 3-4)
- [ ] Generate a `request_id` (UUID) per incoming request in `routes/chat.py` / `routes/ingestion.py`
- [ ] Thread that ID through to every log line and every `request_metrics` row for that request (add a `request_id` column)
- [ ] Structured logging: replace scattered `print()` calls with `logging` + JSON formatter, tagged with `request_id`
- [ ] Verify: pick one request_id, confirm you can reconstruct its full path (embed → retrieve → Groq → save) from logs/DB alone

### Phase 3 — Quality / eval harness (Days 5-8)
- [ ] Build a fixed eval set: ~12-15 questions against the 3 ingested companies (Apple/Microsoft/Tesla), each with expected keywords or facts that MUST appear, and at least 2-3 adversarial questions designed to tempt hallucination (like the one that caught the real bug)
- [ ] Write `eval/run_eval.py`: runs each question through `/api/chat` (or the underlying function directly), scores the answer
- [ ] Scoring approach: (a) keyword/fact presence check (cheap, deterministic), (b) Groq-as-judge: ask Groq itself "does this answer hallucinate/cite anything not in context?" for a second opinion
- [ ] Store eval run results (score, timestamp, git sha) in Postgres so quality trend over time is visible, not just pass/fail
- [ ] Verify: intentionally reintroduce the old hallucination-prone prompt temporarily, confirm the eval harness actually catches the regression before reverting

### Phase 4 — Regression gating in CI (Days 9-10)
- [ ] Add a new job to `.github/workflows/deploy.yaml`: `run-eval`, executes `eval/run_eval.py` against a test Chroma/Postgres instance (or against the same Neon dev data)
- [ ] Make the `deploy` job depend on `run-eval` passing (`needs: run-eval`)
- [ ] Set a concrete threshold (e.g., "at least 10/12 questions pass, zero hallucination flags") — build fails and blocks deploy if not met
- [ ] Verify: open a throwaway branch with a deliberately broken prompt, confirm CI blocks the deploy job

### Phase 5 — Dashboard + writeup (Days 11-12)
- [ ] Add an "Observability" section to `templates/dashboard.html`: p50/p95 latency, running cost total, latest eval score, small sparkline/table — reuse existing dark theme styling
- [ ] Add a short `OBSERVABILITY.md` (or a README section) telling the story: what broke in production, how it's now caught automatically — this is the artifact you actually show recruiters, not just the code
- [ ] Update `FREE_ALTERNATIVES_PLAN.md` cross-reference if relevant

## Definition of done

- Every `/api/chat` and `/api/upload` call produces a traceable row with latency + cost
- p50/p95 latency visible somewhere (dashboard or query)
- A fixed eval set runs automatically on every push and would have caught the real hallucination bug we hit
- CI blocks deployment on a quality regression, not just a build failure
- A short writeup exists explaining the real incident this was built to catch — this is the part that makes the project interview-story-worthy, not just resume-bullet-worthy
