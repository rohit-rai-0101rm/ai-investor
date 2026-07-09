from sqlalchemy import text

from database.postgres_sql import get_engine, create_database


def create_tables() -> None:
    engine = get_engine()

    query = """
    CREATE TABLE IF NOT EXISTS financial_metrics (
        id SERIAL PRIMARY KEY,
        company VARCHAR(100),
        year VARCHAR(10),
        revenue TEXT,
        net_income TEXT,
        operating_income TEXT,
        cash_flow TEXT,
        total_assets TEXT,
        total_liabilities TEXT,
        risk_factors TEXT,
        growth_drivers TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    with engine.begin() as connection:
        connection.execute(text(query))

    print("financial_metrics table created.")

    # ===== MONITORING & OBSERVABILITY (Phase 1: latency + cost tracking) =====
    # One row per pipeline *stage* per request, not one row per request - this
    # is what makes "which stage is slow" answerable later, instead of only
    # knowing the total request took N ms. request_id ties stages of the same
    # request together now, and becomes the join key for Phase 2 tracing (logs
    # tagged with the same id) without needing a schema change then.
    metrics_query = """
    CREATE TABLE IF NOT EXISTS request_metrics (
        id SERIAL PRIMARY KEY,
        request_id VARCHAR(36) NOT NULL,
        endpoint VARCHAR(50) NOT NULL,
        stage VARCHAR(50) NOT NULL,
        duration_ms DOUBLE PRECISION NOT NULL,
        prompt_tokens INT,
        completion_tokens INT,
        cost_usd DOUBLE PRECISION,
        company VARCHAR(100),
        year VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Index matches how we'll actually query this: "give me p50/p95 for
    # endpoint X, stage Y, over some time range" - without it, that query
    # does a full table scan once this table has real volume.
    metrics_index_query = """
    CREATE INDEX IF NOT EXISTS idx_request_metrics_endpoint_stage_created
    ON request_metrics (endpoint, stage, created_at);
    """

    with engine.begin() as connection:
        connection.execute(text(metrics_query))
        connection.execute(text(metrics_index_query))

    print("request_metrics table created.")


if __name__ == "__main__":
    create_database()
    create_tables()