from dotenv import load_dotenv
import os

import chromadb

load_dotenv()


# ===== OLD CODE (Azure AI Search) =====
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.indexes import SearchIndexClient
# from azure.search.documents.indexes.models import (
#     HnswAlgorithmConfiguration,
#     SearchField,
#     SearchFieldDataType,
#     SearchIndex,
#     SimpleField,
#     VectorSearch,
#     VectorSearchProfile
# )
#
# def create_index(
#     endpoint: str,
#     api_key: str,
#     index_name: str,
#     embedding_dimensions: int = 1536
# ) -> None:
#     """
#     Create Azure AI Search index.
#
#     Args:
#         endpoint: Azure AI Search endpoint.
#         api_key: Azure AI Search API key.
#         index_name: Index name.
#         embedding_dimensions: Embedding dimensions.
#     """
#     client = SearchIndexClient(
#         endpoint=endpoint,
#         credential=AzureKeyCredential(api_key)
#     )
#
#     fields = [
#         SimpleField(name="id", type=SearchFieldDataType.String, key=True),
#         SimpleField(name="company", type=SearchFieldDataType.String, filterable=True),
#         SimpleField(name="year", type=SearchFieldDataType.String, filterable=True),
#         SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True),
#         SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
#         SearchField(
#             name="content_vector",
#             type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
#             vector_search_dimensions=embedding_dimensions,
#             vector_search_profile_name="vector-profile"
#         )
#     ]
#
#     vector_search = VectorSearch(
#         algorithms=[
#             HnswAlgorithmConfiguration(name="hnsw-config")
#         ],
#         profiles=[
#             VectorSearchProfile(
#                 name="vector-profile",
#                 algorithm_configuration_name="hnsw-config"
#             )
#         ]
#     )
#
#     index = SearchIndex(
#         name=index_name,
#         fields=fields,
#         vector_search=vector_search
#     )
#
#     client.create_or_update_index(index)
#
#     print(f"Index '{index_name}' created successfully.")
#
#
# if __name__ == "__main__":
#     create_index(
#         endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
#         api_key=os.getenv("AZURE_SEARCH_API_KEY"),
#         index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
#     )


# ===== FREE ALTERNATIVE (ChromaDB collections are created implicitly) =====
def create_index(
    endpoint: str | None = None,
    api_key: str | None = None,
    index_name: str | None = None,
    embedding_dimensions: int = 384
) -> None:
    """
    Ensure the ChromaDB collection exists.

    ChromaDB doesn't need a predefined schema/index like Azure AI Search -
    `get_or_create_collection` creates it on first use. This function is kept
    so `app.py`'s startup call doesn't need to change.

    Args:
        endpoint: Unused, kept for call-site compatibility.
        api_key: Unused, kept for call-site compatibility.
        index_name: Chroma collection name. Defaults to CHROMA_COLLECTION_NAME.
        embedding_dimensions: Unused, ChromaDB infers dimensions from the first insert.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    collection_name = index_name or os.getenv("CHROMA_COLLECTION_NAME", "investor-intelligence")

    client = chromadb.PersistentClient(path=persist_dir)
    client.get_or_create_collection(name=collection_name)

    print(f"Chroma collection '{collection_name}' ready at '{persist_dir}'.")


if __name__ == "__main__":
    create_index()
