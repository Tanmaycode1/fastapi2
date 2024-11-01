from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Settings(BaseSettings):
    # API Configuration
    APP_NAME: str = "Medical Text Parser API"
    API_V1_STR: str = "/api/v1"

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = "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"
    MODEL_NAME: str = "gpt-4-1106-preview"

    # Processing Configuration
    MAX_CHUNK_SIZE: int = 15000
    OVERLAP_SIZE: int = 500
    TEMPERATURE: float = 0.3
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1

    # API Rate Limiting
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_PERIOD: int = 60

    # Cache Configuration
    CACHE_TTL: int = 3600

    class Config:
        case_sensitive = True
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()

    # Check if OPENAI_API_KEY is set
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "\nOPENAI_API_KEY is not set! Please set it using one of these methods:"
            "\n1. Create a .env file with: OPENAI_API_KEY=your_api_key_here"
            "\n2. Set environment variable: "
            "\n   - Windows (CMD): set OPENAI_API_KEY=your_api_key_here"
            "\n   - Windows (PowerShell): $env:OPENAI_API_KEY='your_api_key_here'"
            "\n   - Linux/MacOS: export OPENAI_API_KEY=your_api_key_here"
        )

    return settings