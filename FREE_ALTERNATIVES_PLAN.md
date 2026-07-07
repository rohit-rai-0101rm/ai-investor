# Free Alternatives Plan (Deployable — Recruiter Demo)

Goal: swap every paid Azure dependency for a $0 alternative that can still be **deployed to a public URL**, not just run on your laptop. That rules out Ollama (needs real CPU/RAM the free hosting tiers don't give you) — everything below is chosen to survive on free hosting.

## Current paid stack

| Purpose | Current service | Used in |
|---|---|---|
| Chat LLM (KPI extraction) | Azure OpenAI (`AzureOpenAI` client) | [llm/azure_openai.py](llm/azure_openai.py) |
| Embeddings | Azure OpenAI Embeddings (`AzureOpenAIEmbeddings`) | [ingestion/ingest_documents.py](ingestion/ingest_documents.py), [ingestion/semantic_chunker.py](ingestion/semantic_chunker.py) |
| Vector store | Azure AI Search | [vectorstore/azure_ai_search.py](vectorstore/azure_ai_search.py), [vectorstore/create_index.py](vectorstore/create_index.py) |
| KPI storage | Azure PostgreSQL | [database/postgres_sql.py](database/postgres_sql.py) |

## Deployable free stack

| Purpose | Free replacement | Why this one |
|---|---|---|
| Chat LLM | **Groq API** (`llama-3.1-8b-instant` or `llama-3.3-70b-versatile`) | Free tier, generous rate limits, no credit card, OpenAI-SDK compatible (same `client.chat.completions.create` shape you already use) — actually runs on a public server, unlike Ollama |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`), run in-process on CPU | Free, no external API, light enough (~90MB) to run on a free web service's CPU |
| Vector store | **ChromaDB** (embedded, persisted to disk) | Free, no external service, supports the company/year metadata filter the code already uses |
| KPI storage | **Neon** or **Supabase** free-tier Postgres | Real always-on public Postgres, zero code change — `psycopg2`/`sqlalchemy` code stays identical, just repoint `POSTGRES_*` env vars |
| App hosting | **Hugging Face Spaces (Docker SDK)** or Render free web service | Public URL, free, HF Spaces is a recognizable signal for an AI/RAG project on a resume |

## Trade-off to know going in

Chroma's disk persistence is **ephemeral on most free hosts** (HF Spaces/Render free tier wipe the filesystem on redeploy/restart). For a recruiter demo this is fine if you either (a) re-run ingestion at container startup, or (b) seed a small pre-ingested dataset into the Docker image at build time so the demo always has data without a live ingestion step.

## File-by-file changes needed

1. **[llm/azure_openai.py](llm/azure_openai.py)** — replace `AzureOpenAI` client with plain `OpenAI` client pointed at `base_url="https://api.groq.com/openai/v1"`, `api_key=os.getenv("GROQ_API_KEY")`. Groq doesn't support the `.beta.chat.completions.parse()` structured-output call — the JSON-fallback path already in this file becomes the primary path (may need a small prompt tweak to reliably force valid JSON).
2. **[ingestion/ingest_documents.py](ingestion/ingest_documents.py)** — swap `AzureOpenAIEmbeddings` for `HuggingFaceEmbeddings` (from `langchain-huggingface`, model `all-MiniLM-L6-v2`). Note: this changes embedding dimensions from 1536 → 384, which affects the vector store schema (see #4).
3. **[ingestion/semantic_chunker.py](ingestion/semantic_chunker.py)** — no change needed; it just calls whatever `embeddings` object it's given.
4. **[vectorstore/azure_ai_search.py](vectorstore/azure_ai_search.py)** — rewrite `AzureAISearchVectorStore` and `Retriever` as a thin wrapper around a Chroma `Collection`: `upload_chunks` → `collection.add(ids=..., documents=..., embeddings=..., metadatas=[{"company":.., "year":..}])`; `Retriever.invoke` → `collection.query(query_embeddings=[...], n_results=top_k, where={...})`.
5. **[vectorstore/create_index.py](vectorstore/create_index.py)** — no longer needed (Chroma creates collections implicitly on first use). Delete or leave unused.
6. **[database/postgres_sql.py](database/postgres_sql.py)** — no code change. Just point `.env` at Neon/Supabase connection string.
7. **requirements.txt** — remove `azure-search-documents`; add `groq` (optional, or just reuse `openai` SDK with Groq's base_url), `chromadb`, `langchain-huggingface`, `sentence-transformers`.
8. **.env** — replace Azure vars with:
   - `GROQ_API_KEY=...`
   - `GROQ_CHAT_MODEL=llama-3.3-70b-versatile`
   - `CHROMA_PERSIST_DIR=./data/chroma`
   - `POSTGRES_HOST=...` (Neon/Supabase host), keep `POSTGRES_USER/PASSWORD/PORT/DATABASE`
9. **Dockerfile** — already exists ([dockerfile](dockerfile)); confirm it copies `data/chroma` (pre-seeded) if going with the "seed at build time" approach from the trade-off note above.

## Suggested order of implementation

1. Sign up for free Groq API key, swap `llm/azure_openai.py`, test one KPI extraction call end-to-end.
2. Swap embeddings to `sentence-transformers`, swap vector store to Chroma, re-run ingestion on `2024_Apple.pdf`.
3. Sign up for free Neon/Supabase Postgres, repoint `.env`, verify `save_metrics` writes correctly.
4. Update `README.md` setup instructions with the new free-stack steps.
5. Push a Docker image to Hugging Face Spaces, verify the public URL works end-to-end — this is the link you put on your resume/LinkedIn.

## Why this is good for recruiters

This keeps every skill the original Azure version demonstrates — RAG pipeline, structured extraction, vector search, Postgres persistence, Docker/K8s deployment — but replaces "requires an Azure subscription to run" with "clone, add a free API key, works." That's what people actually try when they click a resume link.
