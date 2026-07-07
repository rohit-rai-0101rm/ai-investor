import os
import uuid

from types import SimpleNamespace

import chromadb


# ===== OLD CODE (Azure AI Search) =====
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents import SearchClient
#
# class AzureAISearchVectorStore:
#     """Azure AI Search vector store."""
#
#     def __init__(
#         self,
#         endpoint: str,
#         api_key: str,
#         index_name: str
#     ) -> None:
#         self.client = SearchClient(
#             endpoint=endpoint,
#             index_name=index_name,
#             credential=AzureKeyCredential(api_key)
#         )
#
#     def upload_chunks(
#         self,
#         chunks,
#         embeddings,
#         company: str,
#         year: str,
#         source_file: str
#     ) -> None:
#         """
#         Upload chunks to Azure AI Search.
#         """
#         documents = []
#
#         for chunk in chunks:
#             vector = embeddings.embed_query(chunk.page_content)
#
#             documents.append(
#                 {
#                     "id": str(uuid.uuid4()),
#                     "company": company,
#                     "year": year,
#                     "source_file": source_file,
#                     "content": chunk.page_content,
#                     "content_vector": vector
#                 }
#             )
#
#         result = self.client.upload_documents(documents)
#
#         uploaded = sum(item.succeeded for item in result)
#
#         print(f"Uploaded {uploaded}/{len(documents)} chunks.")
#
# class Retriever:
#     """Simple wrapper around Azure Search client for retrieving relevant chunks.
#     Mirrors the Retriever used in the RAG extractor.
#     """
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
#         """Retrieve relevant chunks from Azure AI Search.
#         Returns a list of SimpleNamespace objects with `page_content`.
#         """
#         filter_expr = None
#         if company and year:
#             filter_expr = (
#                 f"company eq '{company}' "
#                 f"and year eq '{year}'"
#             )
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
#         documents = []
#         for result in results:
#             content = result.get("content", "")
#             documents.append(SimpleNamespace(page_content=content))
#         return documents


# ===== FREE ALTERNATIVE (ChromaDB, local/embedded vector store) =====
class ChromaVectorStore:
    """Local ChromaDB vector store (free, no external service)."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None
    ) -> None:
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "investor-intelligence")

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upload_chunks(
        self,
        chunks,
        embeddings,
        company: str,
        year: str,
        source_file: str
    ) -> None:
        """
        Upload chunks to ChromaDB.
        """
        ids = []
        documents = []
        vectors = []
        metadatas = []

        for chunk in chunks:
            vector = embeddings.embed_query(chunk.page_content)

            ids.append(str(uuid.uuid4()))
            documents.append(chunk.page_content)
            vectors.append(vector)
            metadatas.append(
                {
                    "company": company,
                    "year": str(year),
                    "source_file": source_file
                }
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas
        )

        print(f"Uploaded {len(documents)}/{len(documents)} chunks.")


class Retriever:
    """Simple wrapper around a Chroma collection for retrieving relevant chunks."""

    def __init__(self, collection, embeddings):
        self.collection = collection
        self.embeddings = embeddings

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 20
    ) -> list:
        """Retrieve relevant chunks from ChromaDB.
        Returns a list of SimpleNamespace objects with `page_content`.
        """
        where = None
        if company and year:
            where = {
                "$and": [
                    {"company": company},
                    {"year": str(year)}
                ]
            }

        query_vector = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where
        )

        documents = []
        for content in results.get("documents", [[]])[0]:
            documents.append(SimpleNamespace(page_content=content))
        return documents
