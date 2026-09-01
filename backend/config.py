import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- General ---
    APP_NAME: str = "AI Customer Support Agent"
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # --- LLM provider: "openai" | "anthropic" | "groq" ---
    LLM_PROVIDER: str = "openai"

    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_CHAT_MODEL: str = "claude-sonnet-4-6"

    GROQ_API_KEY: str = ""
    GROQ_CHAT_MODEL: str = "openai/gpt-oss-20b"

    # ChromaDB uses its built-in local embedding model.
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # --- Storage ---
    DATABASE_URL: str = "sqlite:///./data/app.db"
    CHROMA_DIR: str = "./data/chroma"
    UPLOAD_DIR: str = "./data/uploads"

    # --- RAG ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVER_K: int = 4

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        extra = "ignore"


settings = Settings()

os.makedirs(settings.CHROMA_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "")) or ".", exist_ok=True)
