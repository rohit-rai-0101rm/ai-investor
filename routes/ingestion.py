import shutil
import uuid
from fastapi import APIRouter, File, UploadFile
from pathlib import Path
import os

# ===== OLD CODE (Azure OpenAI Embeddings + Azure AI Search) =====
# from langchain_openai import AzureOpenAIEmbeddings
# from vectorstore.azure_ai_search import AzureAISearchVectorStore

# ===== FREE ALTERNATIVE (local HuggingFace embeddings + Chroma) =====
from langchain_huggingface import HuggingFaceEmbeddings
from vectorstore.azure_ai_search import ChromaVectorStore
from ingestion.ingest_documents import ingest_document

# ===== MONITORING & OBSERVABILITY (Phase 2: tracing) =====
from observability.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    # Generated here, at the true entry point of this request, then threaded
    # down through ingest_document() -> extract_financial_metrics() - so the
    # PDF conversion/chunking/upload_chunks steps and the KPI-extraction
    # stages recorded in Phase 1 all share one ID, the same way request_id
    # ties every stage of one /api/chat call together.
    request_id = str(uuid.uuid4())
    endpoint = "/api/upload"

    upload_dir = Path("data/raw_pdfs")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = upload_dir / file.filename

    logger.info(
        f"upload received: {file.filename}",
        extra={"request_id": request_id, "endpoint": endpoint}
    )

    # Pre-existing bug (from the original tutorial code, unrelated to the
    # Azure->free swap): ingestion used to run INSIDE this `with` block,
    # reading the file while its write handle was still open and unflushed -
    # pymupdf would see a zero-byte file. Writing must fully close first.
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    # ===== OLD CODE (Azure OpenAI Embeddings + Azure AI Search) =====
    # embeddings = AzureOpenAIEmbeddings(
    #     model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    # )
    #
    # vector_store = AzureAISearchVectorStore(
    #     endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    #     api_key=os.getenv("AZURE_SEARCH_API_KEY"),
    #     index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
    # )

    # ===== FREE ALTERNATIVE (local HuggingFace embeddings + Chroma) =====
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = ChromaVectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "investor-intelligence")
    )

    try:
        ingest_document(
            pdf_path=str(file_path),
            embeddings=embeddings,
            vector_store=vector_store,
            request_id=request_id
        )
    except Exception as e:
        # Log with the request_id before letting the exception propagate -
        # otherwise a failed upload's story just stops wherever it crashed,
        # with no record tying the failure back to this specific request.
        logger.error(
            f"upload failed: {file.filename}: {e}",
            extra={"request_id": request_id, "endpoint": endpoint},
            exc_info=True
        )
        raise

    logger.info(
        f"upload complete: {file.filename}",
        extra={"request_id": request_id, "endpoint": endpoint}
    )

    return {
        "message": "Document uploaded successfully",
        "file_name": file.filename
    }