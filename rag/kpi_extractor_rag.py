# ===== OLD CODE (Azure AI Search retriever, duplicated from vectorstore/azure_ai_search.py) =====
# class Retriever:
#     def __init__(self, client):
#         self.client = client
#
#     def invoke(
#         self,
#         query: str,
#         company: str | None = None,
#         year: int | None = None,
#         top_k: int = 20
#     ) -> list:
#         """
#         Retrieve relevant chunks from Azure AI Search.
#         """
#         filter_expr = None
#
#         if company and year:
#             filter_expr = (
#                 f"company eq '{company}' "
#                 f"and year eq '{year}'"
#             )
#
#         results = (
#             self.client.search(
#                 search_text=query,
#                 top=top_k,
#                 filter=filter_expr
#             )
#             if filter_expr
#             else self.client.search(
#                 search_text=query,
#                 top=top_k
#             )
#         )
#
#         documents = []
#
#         for result in results:
#             content = result.get("content", "")
#             documents.append(
#                 SimpleNamespace(
#                     page_content=content
#                 )
#             )
#
#         return documents

# ===== FREE ALTERNATIVE (reuse the single Chroma-backed Retriever) =====
import os
import time
import uuid
from types import SimpleNamespace

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, field_validator, Field

from llm.azure_openai import get_structured_completion
from vectorstore.azure_ai_search import ChromaVectorStore, Retriever

# ===== MONITORING & OBSERVABILITY (Phase 1: latency + cost tracking) =====
from observability.metrics import time_stage, record_metric, estimate_cost_usd

load_dotenv()


class FinancialMetrics(BaseModel):
    # populate_by_name=True: Groq doesn't always use the exact alias key (e.g. it
    # sometimes returns "risk_factors" instead of "Top Risk Factors") - accept both.
    model_config = ConfigDict(populate_by_name=True)

    # `float` added to each numeric field: the Groq free-tier model returns
    # figures like 365.057 (billions) instead of ints/strings like Azure OpenAI did.
    revenue: str | int | float | None = Field(None, alias="Revenue")
    net_income: str | int | float | None = Field(None, alias="Net Income")
    operating_income: str | int | float | None = Field(None, alias="Operating Income")
    cash_flow: str | int | float | None = Field(None, alias="Cash Flow from Operating Activities")
    total_assets: str | int | float | None = Field(None, alias="Total Assets")
    total_liabilities: str | int | float | None = Field(None, alias="Total Liabilities")
    risk_factors: str | list | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list | None = Field(None, alias="Top Growth Drivers")


def retrieve_context(
    retriever: Retriever,
    company: str,
    year: int
) -> str:
    """
    Retrieve financial context from the vector store.
    """
    # One broad query missed most balance-sheet fields on larger documents
    # (e.g. Microsoft's 46 chunks vs Apple's 23) because only the top 6 chunks
    # by overall similarity were kept. Running a separate targeted query per
    # topic spreads retrieval across the document instead of clustering on
    # whatever the single broad query matched best.
    #
    # Company/year are NOT appended to the query text - the `where` filter
    # below already restricts to the right document, and appending them
    # diluted the embedding away from matching the literal statement wording
    # (e.g. "Total revenue", "Total assets" as they appear in the actual
    # income statement / balance sheet tables).
    queries = [
        "Total revenue Net income Operating income consolidated income statement",
        "Total assets Total liabilities balance sheet",
        "cash flow from operations net cash provided by operating activities",
        "risk factors that could materially affect our business",
        "growth drivers strategic priorities cloud and AI growth"
    ]

    # Groq's free tier caps tokens-per-minute (6k-12k depending on model), so both
    # the number of chunks per query and each chunk's length are capped to keep
    # the combined prompt well under that limit. Azure OpenAI didn't need this.
    max_chars_per_chunk = 800
    seen_chunks = set()
    combined_chunks = []

    for query in queries:
        documents = retriever.invoke(
            query=query,
            company=company,
            year=year,
            top_k=3
        )

        for doc in documents:
            snippet = doc.page_content[:max_chars_per_chunk]
            if snippet not in seen_chunks:
                seen_chunks.add(snippet)
                combined_chunks.append(snippet)

    return "\n\n".join(combined_chunks)


def build_extraction_prompt(
    company: str,
    year: int,
    context: str
) -> str:
    """
    Build KPI extraction prompt.
    """
    # Smaller free models (Groq) follow a concrete example far more reliably
    # than an abstract field list - without this, the model sometimes nests
    # the numbers under a sub-object instead of returning this flat shape.
    # Placeholder values are deliberately fake/non-numeric-looking (not real
    # financial figures) so the model can't mistake them for data to copy -
    # a realistic-looking example gets echoed back verbatim instead of replaced.
    example = """{
  "Revenue": "<value from context, or null>",
  "Net Income": "<value from context, or null>",
  "Operating Income": "<value from context, or null>",
  "Cash Flow from Operating Activities": "<value from context, or null>",
  "Total Assets": "<value from context, or null>",
  "Total Liabilities": "<value from context, or null>",
  "Top Risk Factors": ["<risk from context>", "<risk from context>", "<risk from context>"],
  "Top Growth Drivers": ["<driver from context>", "<driver from context>", "<driver from context>"]
}"""

    return f"""
You are an expert financial analyst.

Company: {company}
Year: {year}

Context:
{context}

Extract the following information:

1. Revenue
2. Net Income
3. Operating Income
4. Cash Flow from Operating Activities
5. Total Assets
6. Total Liabilities
7. Top Risk Factors
8. Top Growth Drivers

Instructions:

- Use only the provided context.
- Return null if unavailable.
- Financial values must match the report exactly.
- Risk factors should be concise.
- Growth drivers should be concise.
- Return valid JSON only.
- Your response MUST be a single flat JSON object with EXACTLY these 8 top-level
  keys - no nested objects, no extra keys. This is a SCHEMA TEMPLATE only -
  the placeholder text below is not real data, replace it with actual values
  extracted from the context above (or null if not found):

{example}
"""


def extract_financial_metrics(
    retriever: Retriever,
    company: str,
    year: int
) -> dict:
    """
    Extract KPIs using RAG.
    """
    # ===== MONITORING & OBSERVABILITY (Phase 1) =====
    # Same request_id/stage pattern as routes/chat.py's chat(), so retrieval,
    # llm_call and total stages line up in request_metrics regardless of which
    # endpoint produced them. No BackgroundTasks here - this doesn't run
    # inside a request handler that has one (routes/ingestion.py calls this
    # synchronously), and it doesn't need to: upload already takes ~2 minutes
    # end to end (OCR + chunking dominate), so a couple of ~850ms synchronous
    # metric writes are noise by comparison, unlike the chat endpoint where
    # they were most of the perceived latency.
    request_id = str(uuid.uuid4())
    endpoint = "/api/upload"
    request_start = time.perf_counter()

    with time_stage(request_id, endpoint, "retrieval", company=company, year=year):
        context = retrieve_context(
            retriever=retriever,
            company=company,
            year=year
        )

    prompt = build_extraction_prompt(
        company=company,
        year=year,
        context=context
    )

    # Timed manually rather than via time_stage(): token usage/cost is only
    # known from `usage` after the call returns, same reasoning as the
    # llm_call stage in routes/chat.py.
    llm_start = time.perf_counter()
    metrics, usage = get_structured_completion(
        prompt=prompt,
        response_model=FinancialMetrics
    )
    llm_duration_ms = (time.perf_counter() - llm_start) * 1000

    model = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
    cost_usd = estimate_cost_usd(model, usage.prompt_tokens, usage.completion_tokens)

    record_metric(
        request_id=request_id,
        endpoint=endpoint,
        stage="llm_call",
        duration_ms=llm_duration_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost_usd=cost_usd,
        company=company,
        year=year
    )

    record_metric(
        request_id=request_id,
        endpoint=endpoint,
        stage="total",
        duration_ms=(time.perf_counter() - request_start) * 1000,
        company=company,
        year=year
    )

    return metrics.model_dump()


def main() -> None:
    company = "Apple"
    year = 2024

    # ===== OLD CODE (Azure AI Search) =====
    # vector_store = AzureAISearchVectorStore(
    #     endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    #     api_key=os.getenv("AZURE_SEARCH_API_KEY"),
    #     index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
    # )
    #
    # retriever = Retriever(
    #     vector_store.client
    # )

    # ===== FREE ALTERNATIVE (Chroma needs an embeddings model to embed the query) =====
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = ChromaVectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "investor-intelligence")
    )

    retriever = Retriever(
        vector_store.collection,
        embeddings
    )

    results = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=year
    )

    print(f"\nExtracted KPIs for {company} {year}\n")

    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print("-" * 80)


    from database.save_metrics import save_metrics

    save_metrics(
        company=company,
        year=year,
        metrics=results
    )

if __name__ == "__main__":
    main()