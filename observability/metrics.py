import time
from contextlib import contextmanager

from fastapi import BackgroundTasks
from sqlalchemy import text

from database.postgres_sql import get_engine

# Groq's published per-million-token pricing (approximate, as of when this was
# written - Groq can change rates). We're on the free tier so nothing is
# actually billed, but computing "what this would cost" is the point of this
# metric: it demonstrates cost-awareness even at $0 real spend, and the number
# becomes meaningful the moment this ever runs on a paid tier.
GROQ_PRICING_PER_MILLION_TOKENS = {
    "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    """
    Estimate the USD cost of an LLM call using Groq's published pricing.

    Returns 0.0 for unknown models rather than raising - a missing price
    entry shouldn't break the request, it should just under-report cost.
    """
    pricing = GROQ_PRICING_PER_MILLION_TOKENS.get(model)

    if not pricing:
        return 0.0

    return (
        (prompt_tokens / 1_000_000) * pricing["prompt"]
        + (completion_tokens / 1_000_000) * pricing["completion"]
    )


def record_metric(
    request_id: str,
    endpoint: str,
    stage: str,
    duration_ms: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
    company: str | None = None,
    year: str | int | None = None
) -> None:
    """
    Insert one row into request_metrics for a single pipeline stage.

    Writes immediately/synchronously - call this directly when there's no
    FastAPI request in progress (e.g. Phase 3's eval harness). Inside a
    request handler, prefer scheduling it via BackgroundTasks instead (see
    time_stage() below) so the write happens after the response is already
    sent, rather than adding its own latency to the user's request.
    """
    engine = get_engine()

    query = text("""
        INSERT INTO request_metrics (
            request_id, endpoint, stage, duration_ms,
            prompt_tokens, completion_tokens, cost_usd, company, year
        ) VALUES (
            :request_id, :endpoint, :stage, :duration_ms,
            :prompt_tokens, :completion_tokens, :cost_usd, :company, :year
        )
    """)

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "request_id": request_id,
                "endpoint": endpoint,
                "stage": stage,
                "duration_ms": duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
                "company": company,
                "year": str(year) if year is not None else None
            }
        )


@contextmanager
def time_stage(
    request_id: str,
    endpoint: str,
    stage: str,
    background_tasks: BackgroundTasks | None = None,
    company: str | None = None,
    year: str | int | None = None
):
    """
    Time a block of code and record it as a pipeline stage.

    Use this for stages where only duration matters (e.g. retrieval). For
    LLM calls, time manually and call record_metric() directly instead -
    token usage/cost is only known after the call returns, which doesn't
    fit cleanly into a context manager's __exit__.

    Records the duration even if the block raises, via `finally` - a failed
    stage should still show up in the metrics, not vanish silently.

    Pass `background_tasks` (FastAPI's, from the request handler) to defer
    the actual Postgres write until after the response is sent - each write
    is a real network round-trip (~850ms to Neon), and without this, three
    stages per request means ~2.5s spent writing metrics instead of
    answering the user. Omit it (e.g. outside a request, like the eval
    harness) to fall back to writing immediately.
    """
    start = time.perf_counter()

    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000

        if background_tasks is not None:
            background_tasks.add_task(
                record_metric,
                request_id=request_id,
                endpoint=endpoint,
                stage=stage,
                duration_ms=duration_ms,
                company=company,
                year=year
            )
        else:
            record_metric(
                request_id=request_id,
                endpoint=endpoint,
                stage=stage,
                duration_ms=duration_ms,
                company=company,
                year=year
            )
