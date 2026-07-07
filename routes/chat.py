import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ===== OLD CODE (Azure AI Search) =====
# from vectorstore.azure_ai_search import AzureAISearchVectorStore, Retriever

# ===== FREE ALTERNATIVE (Chroma + local HuggingFace embeddings) =====
from langchain_huggingface import HuggingFaceEmbeddings
from vectorstore.azure_ai_search import ChromaVectorStore, Retriever
from llm.azure_openai import get_openai_client

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # ===== OLD CODE (Azure AI Search) =====
        # vector_store = AzureAISearchVectorStore(
        #     endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        #     api_key=os.getenv("AZURE_SEARCH_API_KEY"),
        #     index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
        # )
        # retriever = Retriever(vector_store.client)

        # ===== FREE ALTERNATIVE (Chroma retriever needs the embeddings model too) =====
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = ChromaVectorStore(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "investor-intelligence")
        )
        retriever = Retriever(vector_store.collection, embeddings)

        # Retrieve relevant context
        # top_k capped + each chunk truncated: Groq's free tier caps tokens-per-minute
        # (6k-12k depending on model) - same fix applied in rag/kpi_extractor_rag.py.
        context = ""
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
        prompt = (
            "You are an expert financial analyst. Answer the user's question "
            "using ONLY the context below, which was retrieved from the "
            "company's actual ingested annual report.\n\n"
            "Do NOT use any outside knowledge about this company, even if you "
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
        client = get_openai_client()
        response = client.chat.completions.create(
            model=os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
