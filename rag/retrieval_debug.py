# ===== OLD CODE (Azure AI Search) =====
# import os
# from vectorstore.azure_ai_search import AzureAISearchVectorStore
#
# def search_vectorstore(query: str, top: int = 5):
#     endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
#     api_key = os.getenv("AZURE_SEARCH_API_KEY")
#     index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
#
#     if not endpoint or not api_key or not index_name:
#         raise RuntimeError(
#             "Missing Azure Search configuration. Set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, and AZURE_SEARCH_INDEX_NAME in your .env."
#         )
#
#     store = AzureAISearchVectorStore(endpoint=endpoint, api_key=api_key, index_name=index_name)
#     results = list(store.client.search(query, top=top))
#
#     print(f"Query: {query!r}")
#     print(f"Top: {top}")
#     print(f"Results: {len(results)}\n")
#
#     for idx, result in enumerate(results, start=1):
#         content = None
#         try:
#             content = result.get("content")
#         except Exception:
#             content = getattr(result, "content", None)
#
#         if content is None:
#             try:
#                 content = result["content"]
#             except Exception:
#                 content = str(result)
#
#         snippet = content.strip().replace("\n", " ") if isinstance(content, str) else "<no content>"
#         if len(snippet) > 350:
#             snippet = snippet[:350].rstrip() + "..."
#
#         print(f"Result {idx}")
#         print(f"  content snippet: {snippet}")
#         print("  " + "-" * 60)
#
#     if not results:
#         print("No results returned. Verify your index contents or try a different query.")
#
#     return results

# ===== FREE ALTERNATIVE (ChromaDB, needs an embeddings model to embed the query) =====
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from vectorstore.azure_ai_search import ChromaVectorStore


def search_vectorstore(query: str, top: int = 5):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    store = ChromaVectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "investor-intelligence")
    )

    query_vector = embeddings.embed_query(query)
    raw_results = store.collection.query(
        query_embeddings=[query_vector],
        n_results=top
    )

    documents = raw_results.get("documents", [[]])[0]

    print(f"Query: {query!r}")
    print(f"Top: {top}")
    print(f"Results: {len(documents)}\n")

    for idx, content in enumerate(documents, start=1):
        snippet = content.strip().replace("\n", " ") if isinstance(content, str) else "<no content>"
        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."

        print(f"Result {idx}")
        print(f"  content snippet: {snippet}")
        print("  " + "-" * 60)

    if not documents:
        print("No results returned. Verify your collection contents or try a different query.")

    return documents


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m rag.retrieval_debug \"your query here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_vectorstore(query)


if __name__ == "__main__":
    main()
