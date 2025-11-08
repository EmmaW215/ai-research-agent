"""Application configuration management"""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = "AI Research Agent"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database
    database_url: str = "sqlite:///./research_agent.db"
    
    # JWT Authentication
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Vector Store
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "research_documents"
    
    # File Upload Settings
    max_upload_size: int = 52428800  # 50MB
    allowed_file_types: str = "application/pdf"
    upload_directory: str = "./uploads"
    
    # PDF Processing
    pdf_extract_images: bool = False
    pdf_extract_tables: bool = True
    
    # Chunking Settings
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50
    chunking_strategy: str = "recursive"
    
    # Embedding Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:8000"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    @property
    def origins_list(self) -> List[str]:
        """Parse CORS allowed origins into a list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
