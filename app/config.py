"""
Configuration management for KnowledgeOps AI
"""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "KnowledgeOps AI"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost/knowledgeops",
        env="DATABASE_URL"
    )
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # LLM — xAI Grok (https://console.x.ai)
    xai_api_key: Optional[str] = Field(default=None, env="XAI_API_KEY")
    xai_model: str = Field(default="grok-beta", env="XAI_MODEL")

    # Embeddings — sentence-transformers runs locally, no API key needed
    # Produces 768-dim vectors; change model here to experiment
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5", env="EMBEDDING_MODEL"
    )
    
    # Vector Database
    vector_db_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost/vectordb",
        env="VECTOR_DB_URL"
    )
    
    # File Upload
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    # CrewAI — optional self-call URL so the Retrieval Specialist agent can
    # hit /retrieve for targeted re-searches during a crew run.
    # In Docker Compose set to http://app:8000; leave empty to disable the tool.
    api_base_url: Optional[str] = Field(default=None, env="API_BASE_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
