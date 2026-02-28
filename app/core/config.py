"""
Enhanced Configuration System
Provides validated, type-safe configuration with environment variable support
"""
from pydantic import (
    BaseModel,
    Field,
    validator,
    field_validator,
    model_validator
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from pathlib import Path
import os
from enum import Enum

from .exceptions import MissingConfigException, InvalidConfigException


class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings with validation
    Automatically loads from environment variables and .env file
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ========================================================================
    # Application Settings
    # ========================================================================
    
    APP_NAME: str = Field(
        default="AI Educational Document Reasoning System",
        description="Application name"
    )
    
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    
    DEBUG: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    
    PORT: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Server port"
    )
    
    # ========================================================================
    # Security Settings
    # ========================================================================
    
    SECRET_KEY: str = Field(
        default="change-this-secret-key-in-production",
        min_length=32,
        description="Secret key for JWT token generation"
    )
    
    ALGORITHM: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Access token expiration time in minutes"
    )
    
    # ========================================================================
    # Database Settings
    # ========================================================================
    
    DATABASE_URL: str = Field(
        default="sqlite:///./data/sqlite.db",
        description="Database connection URL"
    )
    
    DATABASE_POOL_SIZE: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Database connection pool size"
    )
    
    DATABASE_MAX_OVERFLOW: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum overflow connections"
    )
    
    # ========================================================================
    # Ollama/LLM Settings
    # ========================================================================
    
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    
    OLLAMA_MODEL: str = Field(
        default="qwen2.5:14b",
        description="Ollama model name"
    )
    
    OLLAMA_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation"
    )
    
    OLLAMA_TOP_P: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="LLM top_p sampling parameter"
    )
    
    OLLAMA_MAX_TOKENS: int = Field(
        default=2048,
        ge=100,
        le=8192,
        description="Maximum tokens for LLM response"
    )
    
    OLLAMA_TIMEOUT: int = Field(
        default=120,
        ge=30,
        le=600,
        description="LLM request timeout in seconds"
    )
    
    OLLAMA_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for LLM requests"
    )
    
    # ========================================================================
    # Embedding Settings
    # ========================================================================
    
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    EMBEDDING_DEVICE: str = Field(
        default="cuda",
        description="Device for embedding model (cuda or cpu)"
    )
    
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding generation"
    )
    
    # ========================================================================
    # ChromaDB Settings
    # ========================================================================
    
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="./data/chroma_db",
        description="ChromaDB persistence directory"
    )
    
    CHROMA_COLLECTION_NAME: str = Field(
        default="educational_documents",
        description="ChromaDB collection name"
    )
    
    CHROMA_DISTANCE_METRIC: str = Field(
        default="cosine",
        description="Distance metric for similarity (cosine, l2, ip)"
    )
    
    # ========================================================================
    # Document Processing Settings
    # ========================================================================
    
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=500,
        ge=1,
        le=1024,
        description="Maximum document upload size in MB"
    )
    
    CHUNK_SIZE: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Document chunk size in characters"
    )
    
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between document chunks"
    )
    
    SUPPORTED_FORMATS: str = Field(
        default="pdf,docx,txt,pptx",
        description="Comma-separated list of supported file formats"
    )
    
    UPLOAD_DIRECTORY: str = Field(
        default="./data/uploads",
        description="Directory for uploaded documents"
    )
    
    # ========================================================================
    # Retrieval Settings
    # ========================================================================
    
    TOP_K_RETRIEVAL: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top chunks to retrieve"
    )
    
    SIMILARITY_THRESHOLD: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval"
    )
    
    # ========================================================================
    # CORS Settings
    # ========================================================================
    
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    
    # ========================================================================
    # Logging Settings
    # ========================================================================
    
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    
    LOG_FILE: str = Field(
        default="./logs/app.log",
        description="Log file path"
    )
    
    LOG_JSON_FORMAT: bool = Field(
        default=False,
        description="Use JSON format for logs"
    )
    
    LOG_ROTATION_SIZE_MB: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Log file rotation size in MB"
    )
    
    LOG_BACKUP_COUNT: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of log backup files to keep"
    )
    
    # ========================================================================
    # Rate Limiting
    # ========================================================================
    
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of requests allowed per window"
    )
    
    RATE_LIMIT_WINDOW_SECONDS: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Rate limit time window in seconds"
    )
    
    # ========================================================================
    # Performance Settings
    # ========================================================================
    
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent requests"
    )
    
    CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable response caching"
    )
    
    CACHE_TTL_SECONDS: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds"
    )
    
    # ========================================================================
    # Validators
    # ========================================================================
    
    @field_validator('SECRET_KEY')
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key is strong enough"""
        if v == "change-this-secret-key-in-production" and os.getenv('ENVIRONMENT') == 'production':
            raise InvalidConfigException(
                "SECRET_KEY",
                v,
                "Default secret key cannot be used in production"
            )
        return v
    
    @field_validator('EMBEDDING_DEVICE')
    @classmethod
    def validate_embedding_device(cls, v: str) -> str:
        """Validate embedding device"""
        if v not in ['cuda', 'cpu', 'mps']:
            raise InvalidConfigException(
                "EMBEDDING_DEVICE",
                v,
                "Must be one of: cuda, cpu, mps"
            )
        return v
    
    @field_validator('CHROMA_DISTANCE_METRIC')
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate ChromaDB distance metric"""
        if v not in ['cosine', 'l2', 'ip']:
            raise InvalidConfigException(
                "CHROMA_DISTANCE_METRIC",
                v,
                "Must be one of: cosine, l2, ip"
            )
        return v
    
    @field_validator('CHUNK_OVERLAP')
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Validate chunk overlap is less than chunk size"""
        chunk_size = info.data.get('CHUNK_SIZE', 1000)
        if v >= chunk_size:
            raise InvalidConfigException(
                "CHUNK_OVERLAP",
                v,
                f"Overlap ({v}) must be less than chunk size ({chunk_size})"
            )
        return v
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return [fmt.strip() for fmt in self.SUPPORTED_FORMATS.split(',')]
    
    def get_max_upload_size_bytes(self) -> int:
        """Get maximum upload size in bytes"""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == Environment.DEVELOPMENT


# Global settings instance
settings = Settings()


def ensure_directories():
    """
    Ensure all required directories exist
    Creates directories if they don't exist
    """
    directories = [
        "data",
        "data/uploads",
        "data/chroma_db",
        "logs"
    ]
    
    # Add custom directories from settings
    if settings.UPLOAD_DIRECTORY and settings.UPLOAD_DIRECTORY != "./data/uploads":
        directories.append(settings.UPLOAD_DIRECTORY)
    
    if settings.CHROMA_PERSIST_DIRECTORY and settings.CHROMA_PERSIST_DIRECTORY != "./data/chroma_db":
        directories.append(settings.CHROMA_PERSIST_DIRECTORY)
    
    log_dir = Path(settings.LOG_FILE).parent
    if str(log_dir) not in directories:
        directories.append(str(log_dir))
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
    
    return directories


def validate_configuration():
    """
    Validate configuration and check dependencies
    Raises exceptions if configuration is invalid
    """
    # Check if Ollama URL is reachable
    import httpx
    try:
        response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            raise InvalidConfigException(
                "OLLAMA_BASE_URL",
                settings.OLLAMA_BASE_URL,
                "Ollama service not responding correctly"
            )
    except Exception as e:
        raise InvalidConfigException(
            "OLLAMA_BASE_URL",
            settings.OLLAMA_BASE_URL,
            f"Cannot connect to Ollama: {str(e)}"
        )
    
    # Validate embedding device
    if settings.EMBEDDING_DEVICE == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print(" WARNING: CUDA not available, falling back to CPU")
                settings.EMBEDDING_DEVICE = "cpu"
        except ImportError:
            print(" WARNING: PyTorch not installed, using CPU")
            settings.EMBEDDING_DEVICE = "cpu"
    
    print("Configuration validated successfully")


if __name__ == "__main__":
    print("🔧 Configuration Settings:")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug: {settings.DEBUG}")
    print(f"Ollama Model: {settings.OLLAMA_MODEL}")
    print(f"Embedding Device: {settings.EMBEDDING_DEVICE}")
    print(f"Database: {settings.DATABASE_URL}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    
    print("\n📁 Creating directories...")
    dirs = ensure_directories()
    for d in dirs:
        print(f"  ✓ {d}")
    
    print("\nValidating configuration...")
    try:
        validate_configuration()
    except Exception as e:
        print(f"Validation failed: {e}")
