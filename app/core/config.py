"""Application settings, loaded from environment variables and an optional ``.env`` file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Each value is read from the environment variable with the same name in upper case
    (for example ``QDRANT_URL``), then from ``.env``, then falls back to the default.

    Attributes:
        qdrant_url: URL of the Qdrant vector database.
        qdrant_collection: Name of the Qdrant collection holding the document chunks.
        embedding_model: Name of the sentence-transformers model used for embeddings.
        ollama_url: URL of the local Ollama server.
        ollama_model: Name of the Ollama model used to generate answers.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sepa_rulebook"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"


def get_settings() -> Settings:
    """Build the application settings.

    Returns:
        A new ``Settings`` instance reflecting the current environment.

    Raises:
        pydantic.ValidationError: If an environment variable has an invalid value.
    """
    return Settings()
