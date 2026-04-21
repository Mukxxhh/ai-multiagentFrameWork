"""
ClarityLens Analytics - Main Entry Point
Multi-agent analytics framework for data understanding and storytelling.
"""
import uvicorn
from config.settings import settings


def main():
    """Run the FastAPI server."""
    print(f"""
    ================================================================
    |          ClarityLens Analytics - Starting Server             |
    ================================================================
    |  Ollama Model: {settings.ollama_model:<46} |
    |  Ollama URL:   {settings.ollama_base_url:<46} |
    |  Server:       http://localhost:{settings.api_port:<37} |
    |  Dashboard:    http://localhost:{settings.api_port}/dashboard{' ' * 28} |
    |  API Docs:     http://localhost:{settings.api_port}/docs{' ' * 32} |
    ================================================================

    Supported File Formats: {settings.allowed_extensions}

    Make sure Ollama is running with the model installed:
    - Run: ollama pull {settings.ollama_model}
    - Run: ollama serve
    """)

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


if __name__ == "__main__":
    main()
