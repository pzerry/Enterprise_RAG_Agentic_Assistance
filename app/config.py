"""Load credentials and ingestion defaults; optional .env values override them."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Shared service configuration populated from the process environment."""

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_rag_hybrid_v1")

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    
   
settings = Settings()
# Conservative application budgets, below the observed project quotas.
settings.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-2-preview')
settings.EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '3072'))
settings.EMBEDDING_RPM = int(os.getenv('EMBEDDING_RPM', '80'))
settings.EMBEDDING_TPM = int(os.getenv('EMBEDDING_TPM', '24000'))
settings.EMBEDDING_RPD = int(os.getenv('EMBEDDING_RPD', '1000'))
settings.EMBEDDING_MAX_ATTEMPTS = int(os.getenv('EMBEDDING_MAX_ATTEMPTS', '6'))
settings.EMBEDDING_STATE = os.getenv('EMBEDDING_STATE', '.ingestion/embeddings.sqlite3')
settings.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1500'))
