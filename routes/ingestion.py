import shutil
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

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    upload_dir = Path("data/raw_pdfs")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = upload_dir / file.filename

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

        ingest_document(
            pdf_path=str(file_path),
            embeddings=embeddings,
            vector_store=vector_store
        )

    return {
        "message": "Document uploaded successfully",
        "file_name": file.filename
    }