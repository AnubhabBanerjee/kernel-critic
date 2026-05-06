import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Environment Setup
load_dotenv()

def run_ingestion():
    # Configuration from .env
    data_path = "./data"
    db_path = os.getenv("CHROMA_DB_DIR", "./db")
    
    print(f"--- Starting Ingestion from {data_path} ---")

    # 2. Load Documents
    # Uses PyPDFLoader for documentation PDFs or local text files
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} document pages.")

    # 3. Chunking
    # We use a 1000-character chunk with overlap to preserve technical context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} text chunks.")

    # 4. Embedding & Storage
    # Selecting the model based on your ACTIVE_ENV (Local/Internal/Cloud)
    active_env = os.getenv("ACTIVE_ENV", "local").lower()
    
    if active_env == "openai":
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif active_env == "internal":
        embeddings = OpenAIEmbeddings(
            model=os.getenv("INTERNAL_EMBED_MODEL"),
            base_url=os.getenv("INTERNAL_EMBED_BASE_URL"),
            api_key=os.getenv("INTERNAL_EMBED_API_KEY", "not-needed"),
        )
    else:
        # Default to local Ollama-compatible embeddings
        embeddings = OpenAIEmbeddings(
            model=os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("LOCAL_EMBED_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LOCAL_EMBED_API_KEY", "ollama"),
        )

    # Create the vector store and persist it to the db/ directory
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    print(f"✅ Ingestion complete. Vector index saved to {db_path}")

if __name__ == "__main__":
    run_ingestion()