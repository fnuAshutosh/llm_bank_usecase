"""Configuration management using Pydantic Settings"""

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "banking-llm"
    APP_ENV: str = "development"
    ENVIRONMENT: str = "development"  # Alias
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = True
    
    # Database (Supabase PostgreSQL)
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/banking_llm"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 0
    
    # Supabase
    SUPABASE_URL: str = ""  # https://your-project.supabase.co  
    SUPABASE_KEY: str = ""  # Your anon/public key
    SUPABASE_SERVICE_KEY: str = ""  # Service role key (admin access)
    SUPABASE_ACCESS_TOKEN: str = ""  # Personal Access Token for Management API
    SUPABASE_DB_PASSWORD: str = ""  # Direct database password
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: str = ""
    CACHE_TTL: int = 3600
    
    # Pinecone Vector Database
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "banking-chat-embeddings"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    
    # Ollama (Local)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2:7b"
    OLLAMA_TIMEOUT: int = 300
    
    # Together.ai (Cloud)
    TOGETHER_API_KEY: str = ""
    TOGETHER_MODEL: str = "meta-llama/Llama-2-34b-chat-hf"
    
    # OpenAI (Alternative)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4"
    
    # LLM Provider Selection
    LLM_PROVIDER: str = "ollama"  # Options: ollama, together, openai
    
    # RunPod (Training)
    RUNPOD_API_KEY: str = ""
    RUNPOD_TEMPLATE_ID: str = ""
    
    # Model Configuration
    MODEL_MAX_LENGTH: int = 4096
    MODEL_TEMPERATURE: float = 0.7
    MODEL_TOP_P: float = 0.9
    MODEL_TOP_K: int = 50
    
    # Security
    SECRET_KEY: str = "your_secret_key_here_change_in_production"
    JWT_SECRET: str = "your_jwt_secret_here"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: int = 3600
    ENCRYPTION_KEY: str = "your_encryption_key_32_bytes_here"
    
    # CORS - Allow all origins for development (including Codespaces)
    CORS_ORIGINS: List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # PII Detection
    PII_DETECTION_ENABLED: bool = True
    PII_DETECTION_THRESHOLD: float = 0.85
    PII_MASK_CHAR: str = "*"
    
    # Monitoring
    ENABLE_MONITORING: bool = True
    ENABLE_TRACING: bool = True
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    JAEGER_ENABLED: bool = True
    JAEGER_AGENT_HOST: str = "localhost"
    JAEGER_AGENT_PORT: int = 6831
    
    # Logging
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "logs/app.log"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"
    
    # Compliance
    AUDIT_LOG_ENABLED: bool = True
    AUDIT_LOG_PATH: str = "logs/audit.log"
    DATA_RETENTION_DAYS: int = 2555  # 7 years
    
    # Feature Flags
    ENABLE_FRAUD_DETECTION: bool = True
    ENABLE_KYC_SCREENING: bool = True
    ENABLE_MULTI_LANGUAGE: bool = False
    ENABLE_HALLUCINATION_DETECTION: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
