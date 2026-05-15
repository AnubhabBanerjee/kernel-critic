import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# based on the user prompt, this method returns the most relevant CUDA doc
def get_relevant_cuda_docs(query: str) -> str:
    """
    Standardized RAG tool to retrieve NVIDIA documentation context.
    Expects environment variables to be pre-loaded by the application entry point.
    """
    active_env = os.getenv("ACTIVE_ENV", "local").lower()
    
    # 1. Dynamic selection of embedding credentials
    if active_env == "internal":
        model = os.getenv("INTERNAL_EMBED_MODEL")
        base_url = os.getenv("INTERNAL_EMBED_BASE_URL")
        api_key = os.getenv("INTERNAL_EMBED_API_KEY", "not-needed")
    elif active_env == "openai":
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        base_url = None
        api_key = os.getenv("OPENAI_API_KEY")
    else: # Default to local
        model = os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text")
        base_url = os.getenv("LOCAL_EMBED_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("LOCAL_EMBED_API_KEY", "ollama")

    embeddings = OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key
    )

    # 2. Connect to the local Vector Database
    # We use ChromaDB for its lightweight footprint and persistent storage.
    db_path = os.getenv("CHROMA_DB_DIR", "./db")
    
    if not os.path.exists(db_path):
        return "Warning: Vector database directory not found. Proceeding without RAG context."

    # opens the "Digital Library" (./db). It uses the embeddings to understand the documents stored in that folder.
    vector_db = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )

    # 3. Similarity Search
    # We retrieve the top 3 snippets (k=3) to provide dense technical context
    # while staying within the LLM's optimal attention window.
    try:
        docs: List[Document] = vector_db.similarity_search(query, k=3)
        
        # 4. Format for the LLM
        # Each chunk is separated by a clear delimiter to help the model 
        # distinguish between different sections of the NVIDIA documentation.
        formatted_context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return formatted_context if formatted_context else "No relevant documentation found."
        
    except Exception as e:
        return f"Retrieval Error: {str(e)}"