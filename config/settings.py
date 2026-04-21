"""
Configuration settings for the AI Insights Dashboard.
"""
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama Configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # File Settings
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    allowed_extensions: str = os.getenv(
        "ALLOWED_EXTENSIONS",
        "csv,xlsx,xls,json,pdf,txt,doc,docx,xml,parquet"
    )

    # Directory Settings
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")

    class Config:
        env_file = ".env"

    @property
    def allowed_extensions_list(self) -> list[str]:
        """Return normalized allowed extensions without blanks."""
        return [item.strip().lower() for item in self.allowed_extensions.split(",") if item.strip()]


# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.upload_dir, exist_ok=True)
