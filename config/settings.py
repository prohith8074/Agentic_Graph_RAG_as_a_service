"""
Centralized configuration management for the Lyzr Challenge RAG system.
Loads environment variables and provides default configurations.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration."""

    # API Keys
    LLAMA_CLOUD_API_KEY: Optional[str] = os.getenv("LLAMA_CLOUD_API_KEY")
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    NOMIC_API_KEY: Optional[str] = os.getenv("NOMIC_API_KEY")
    NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI")
    NEO4J_USER: Optional[str] = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")

    # LLM Models
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "gemma-7b-it")
    COHERE_MODEL: str = os.getenv("COHERE_MODEL", "command-r-plus")
    # Set the default embedding provider to "cohere"
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "cohere")

    # File Paths
    PDF_FILE_PATH: str = os.getenv("PDF_FILE_PATH", "/content/2501.00309v2.pdf")
    CONTEXTUAL_DATA_PATH: Optional[str] = os.getenv("CONTEXTUAL_DATA_PATH")

    # Vector Database (Qdrant)
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "RAG_Marketing")
    # Set the default vector size to 1024 for Cohere embeddings
    QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))  # Default to Cohere embedding dimension

    # Legacy ChromaDB (deprecated, use Qdrant)
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "transformer_rag")

    # Processing Configuration
    CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "2048"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "5"))
    RATE_LIMIT_DELAY: int = int(os.getenv("RATE_LIMIT_DELAY", "10"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = '%(asctime)s [%(levelname)s] %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

    # Opik Configuration
    OPIK_API_KEY: Optional[str] = os.getenv("OPIK_API_KEY")
    OPIK_PROJECT_NAME: str = os.getenv("OPIK_PROJECT_NAME", "lyzr-challenge-rag")

    # Redis Configuration (Redis Cloud - 30MB free tier)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis-17527.c212.ap-south-1-1.ec2.redns.redis-cloud.com")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "17527"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")

    # Memory Configuration
    SHORT_TERM_MEMORY_TTL: int = int(os.getenv("SHORT_TERM_MEMORY_TTL", "3600"))  # 1 hour
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))

    # Database Configuration
    DEFAULT_DATABASE: str = os.getenv("DEFAULT_DATABASE", "neo4j")  # 'neo4j' (Neptune support disabled)
    # NEPTUNE_ENDPOINT: Optional[str] = os.getenv("NEPTUNE_ENDPOINT")  # Commented out - Neptune support disabled
    # NEPTUNE_PORT: Optional[int] = int(os.getenv("NEPTUNE_PORT", "8182"))  # Commented out - Neptune support disabled

    @classmethod
    def validate(cls) -> None:
        """Validate that all required settings are present."""
        required_settings = [
            ("LLAMA_CLOUD_API_KEY", cls.LLAMA_CLOUD_API_KEY),
            ("COHERE_API_KEY", cls.COHERE_API_KEY),
            ("GROQ_API_KEY", cls.GROQ_API_KEY),
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("NEO4J_URI", cls.NEO4J_URI),
            ("NEO4J_USER", cls.NEO4J_USER),
            ("NEO4J_PASSWORD", cls.NEO4J_PASSWORD),
        ]

        missing_settings = [name for name, value in required_settings if not value]
        if missing_settings:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_settings)}")

# Global settings instance
settings = Settings()