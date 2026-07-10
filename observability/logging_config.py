import json
import logging
import sys

# ===== MONITORING & OBSERVABILITY (Phase 2: tracing) =====
# Scattered print() calls across the request path (routes/chat.py,
# routes/ingestion.py, ingestion/ingest_documents.py, rag/kpi_extractor_rag.py,
# llm/azure_openai.py, vectorstore/azure_ai_search.py) all went to stdout as
# plain text with no way to tell which HTTP request a given line belonged to.
# Structured JSON logging + a request_id in `extra` fixes that: every line for
# one request can be found with a single grep for its request_id, the same
# way request_metrics rows are found with `WHERE request_id = ...`.

# Fields we tag log lines with, matching the columns on request_metrics so a
# request's story can be reconstructed from either logs or DB, not just one.
_CONTEXT_FIELDS = ("request_id", "endpoint", "stage", "company", "year")


class JSONFormatter(logging.Formatter):
    """Renders each log record as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Attach a single JSON-formatted stdout handler to the root logger.

    Call once, at process startup (app.py). Guarded against re-adding a
    handler on repeated calls (e.g. reload in dev) - would otherwise
    duplicate every log line.
    """
    root = logging.getLogger()

    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Standard logger lookup - kept as a thin wrapper (rather than importing
    `logging` directly everywhere) so call sites read as "this is part of
    the observability setup", and so the root-handler/formatter setup above
    is the only place that would ever need to change.
    """
    return logging.getLogger(name)
