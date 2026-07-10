# ===== OLD CODE (Azure OpenAI Embeddings + Azure AI Search) =====
# from langchain_openai import AzureOpenAIEmbeddings
# from vectorstore.azure_ai_search import AzureAISearchVectorStore
# from vectorstore.azure_ai_search import Retriever

# ===== FREE ALTERNATIVE (local HuggingFace embeddings + Chroma) =====
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from vectorstore.azure_ai_search import ChromaVectorStore
from rag.kpi_extractor_rag import extract_financial_metrics
from database.save_metrics import save_metrics
from vectorstore.azure_ai_search import Retriever

# ===== MONITORING & OBSERVABILITY (Phase 2: tracing) =====
from observability.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)


def parse_company_year(pdf_file: Path) -> tuple[str, str]:
    """Parse company and year from a PDF filename.

    Supports names like `2024_Apple.pdf` and `2024_AnnualReport_Apple.pdf`.
    """
    stem = pdf_file.stem
    parts = stem.split("_")

    if parts and parts[0].isdigit():
        year = parts[0]
        company = parts[-1]
    elif len(parts) >= 2:
        company = parts[0]
        year = parts[1]
    else:
        company = stem
        year = ""

    return company, year


def ingest_document(
    pdf_path: str,
    embeddings,
    vector_store,
    request_id: str | None = None
) -> None:
    """
    Ingest a single PDF document.
    """
    # request_id defaults to a fresh one so CLI/standalone calls (ingest_directory()
    # below, or the __main__ block) still work without a caller having to pass
    # one - only routes/ingestion.py's live /api/upload path threads a real,
    # request-scoped ID through here.
    request_id = request_id or str(uuid.uuid4())

    pdf_file = Path(pdf_path)

    company, year = parse_company_year(pdf_file)
    logger.info(
        f"ingesting {pdf_file.name} as company={company!r}, year={year!r}",
        extra={"request_id": request_id, "company": company, "year": year}
    )

    converter = PDFToMarkdownConverter()

    markdown_file = converter.convert_pdf(
        pdf_path=pdf_path,
        output_dir="data/markdown"
    )

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    logger.info(
        f"generated {len(chunks)} chunks for {pdf_file.name}",
        extra={"request_id": request_id, "stage": "chunking", "company": company, "year": year}
    )

    vector_store.upload_chunks(
        chunks=chunks,
        embeddings=embeddings,
        company=company,
        year=year,
        source_file=pdf_file.name,
        request_id=request_id
    )

    # ===== OLD CODE (Azure AI Search retriever) =====
    # metrics = extract_financial_metrics(
    #     retriever=Retriever(vector_store.client),
    #     company=company,
    #     year=int(year) if year.isdigit() else None
    # )

    # ===== FREE ALTERNATIVE (Chroma retriever needs the embeddings model too) =====
    metrics = extract_financial_metrics(
        retriever=Retriever(vector_store.collection, embeddings),
        company=company,
        year=int(year) if year.isdigit() else None,
        request_id=request_id
    )

    # Persist metrics to PostgreSQL
    if metrics:
        save_metrics(company=company, year=int(year) if str(year).isdigit() else None, metrics=metrics)


def ingest_directory(input_dir: str) -> None:
    """
    Ingest all PDFs from a directory.
    """
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

    pdf_files = list(Path(input_dir).glob("*.pdf"))

    logger.info(f"found {len(pdf_files)} PDF(s) in {input_dir}")

    for pdf_file in pdf_files:
        ingest_document(
            pdf_path=str(pdf_file),
            embeddings=embeddings,
            vector_store=vector_store
        )


if __name__ == "__main__":
    ingest_directory("data/raw_pdfs")
    # ingest_document("data/raw_pdfs/2024_Apple.pdf")