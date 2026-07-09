import os
import time
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

# ===== OLD CODE (Azure AI Search) =====
# from vectorstore.azure_ai_search import AzureAISearchVectorStore, Retriever

# ===== FREE ALTERNATIVE (Chroma + local HuggingFace embeddings) =====
from langchain_huggingface import HuggingFaceEmbeddings
from vectorstore.azure_ai_search import ChromaVectorStore, Retriever
from llm.azure_openai import get_openai_client

# ===== MONITORING & OBSERVABILITY (Phase 1: latency + cost tracking) =====
from observability.metrics import time_stage, record_metric, estimate_cost_usd

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None


# ===== MONITORING & OBSERVABILITY (Phase 1 fix) =====
# Neither of these carries per-request state, so recreating them inside
# chat() on every call was pure waste - the first real latency measurement
# showed ~17s of an ~18s total request unaccounted for by retrieval/llm_call,
# which was this model reload happening on every single request. Lazy
# module-level singletons: built once (first request pays the cost, logged
# as "embedding_init" below), reused for the life of the process after that.
_embeddings = None
_vector_store = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def _get_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "investor-intelligence")
        )
    return _vector_store

@router.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    # request_id ties every stage of this one call together in
    # request_metrics - and will do the same job for log tracing in Phase 2,
    # so it's generated once here rather than per-stage.
    request_id = str(uuid.uuid4())
    endpoint = "/api/chat"
    request_start = time.perf_counter()

    try:
        # ===== OLD CODE (Azure AI Search) =====
        # vector_store = AzureAISearchVectorStore(
        #     endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        #     api_key=os.getenv("AZURE_SEARCH_API_KEY"),
        #     index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
        # )
        # retriever = Retriever(vector_store.client)

        # ===== FREE ALTERNATIVE (Chroma retriever needs the embeddings model too) =====
        # embedding_init: near-zero after the first request (cached singleton
        # above) - the gap between this and the old ~17s per-request cost is
        # exactly the kind of thing this metric exists to make visible.
        with time_stage(request_id, endpoint, "embedding_init", background_tasks=background_tasks, company=request.company, year=request.year):
            embeddings = _get_embeddings()
            vector_store = _get_vector_store()

        retriever = Retriever(vector_store.collection, embeddings)

        # Retrieve relevant context
        # top_k capped + each chunk truncated: Groq's free tier caps tokens-per-minute
        # (6k-12k depending on model) - same fix applied in rag/kpi_extractor_rag.py.
        context = ""
        with time_stage(request_id, endpoint, "retrieval", background_tasks=background_tasks, company=request.company, year=request.year):
            if request.company and request.year:
                docs = retriever.invoke(
                    query=request.question,
                    company=request.company,
                    year=request.year,
                    top_k=6
                )
            else:
                docs = retriever.invoke(
                    query=request.question,
                    top_k=6
                )

        max_chars_per_chunk = 1200
        context = "\n\n".join(doc.page_content[:max_chars_per_chunk] for doc in docs)

        # Build chat prompt – include retrieved context and the user question
        # Explicit "no outside knowledge" instruction: smaller free models will
        # otherwise blend in memorized facts (e.g. citing an old annual report
        # that was never actually ingested) instead of admitting a gap.
        # Also branches for meta/capability questions ("what can I do here?")
        # separately from data questions - without this, the strict
        # context-only rule made the model refuse to explain itself at all.
        prompt = (
            "You are the AI assistant for an investor intelligence platform. "
            "Users can ask you about financial data (revenue, net income, "
            "risk factors, growth drivers, etc.) extracted from annual "
            "reports they've uploaded, for any ingested company/year.\n\n"
            "If the user's question is about what you or this app can do "
            "in general (a greeting, capability question, or similar) rather "
            "than a specific data question, briefly explain that in 1-2 "
            "sentences without needing the context below.\n\n"
            "Otherwise, answer using ONLY the context below, which was "
            "retrieved from the company's actual ingested annual report. Do "
            "NOT use any outside knowledge about this company, even if you "
            "recognize it, and do NOT cite any report, year, or page number "
            "that is not explicitly present in the context below - doing so "
            "is fabrication. If the context does not contain enough "
            "information to answer, say so plainly instead of guessing.\n\n"
            f"Context:\n{context}\n\n"
            f"User Question: {request.question}\n\nAnswer:"
        )

        # ===== OLD CODE (Azure OpenAI deployment name) =====
        # client = get_openai_client()
        # response = client.chat.completions.create(
        #     model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        #     messages=[{"role": "user", "content": prompt}]
        # )

        # ===== FREE ALTERNATIVE (Groq model name) =====
        # temperature=0: makes answers consistent run-to-run for identical
        # questions - the default sampling was giving noticeably different
        # completeness on repeated identical queries.
        model = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
        client = get_openai_client()

        # Timed manually rather than via time_stage(): token usage/cost is
        # only known from `response.usage` after the call returns, so it has
        # to be recorded here, not in a context manager's __exit__.
        llm_start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        llm_duration_ms = (time.perf_counter() - llm_start) * 1000

        usage = response.usage
        cost_usd = estimate_cost_usd(model, usage.prompt_tokens, usage.completion_tokens)

        # Scheduled via background_tasks (not called directly) for the same
        # reason as the time_stage() calls above: each record_metric() write
        # is a real Postgres round-trip (~850ms) - deferring it until after
        # the response is sent means the user isn't waiting on it.
        background_tasks.add_task(
            record_metric,
            request_id=request_id,
            endpoint=endpoint,
            stage="llm_call",
            duration_ms=llm_duration_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost_usd,
            company=request.company,
            year=request.year
        )

        answer = response.choices[0].message.content

        background_tasks.add_task(
            record_metric,
            request_id=request_id,
            endpoint=endpoint,
            stage="total",
            duration_ms=(time.perf_counter() - request_start) * 1000,
            company=request.company,
            year=request.year
        )

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
