# AI Insights Dashboard

A multi-agent AI system for automated data analysis, visualization, and PDF report generation powered by CrewAI and Ollama (local LLM).

## Features

- **Multi-File Support**: CSV, Excel, JSON, PDF, TXT, DOCX, XML, Parquet
- **AI-Powered Analysis**: Uses CrewAI agents with Ollama for local inference
- **Automatic Visualizations**: Generates charts and graphs for numerical data
- **PDF Reports**: Professional downloadable reports with insights and charts
- **REST API**: FastAPI backend for easy integration
- **Web Dashboard**: Simple drag-and-drop interface

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent Orchestration | CrewAI |
| LLM | Ollama (llama3.2, mistral, etc.) |
| Data Processing | Pandas |
| Visualization | Plotly |
| PDF Generation | WeasyPrint + Jinja2 |
| API | FastAPI |
| Frontend | HTML + Tailwind CSS |

## Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
3. **Pull a model**:
   ```bash
   ollama pull llama3.2
   # or
   ollama pull mistral
   ```

## Installation

```bash
# Clone or navigate to the project
cd ai-insights-dashboard

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env to change model or settings
```

## Running the Application

### 1. Start Ollama
```bash
ollama serve
```

### 2. Start the API Server
```bash
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Dashboard**: Open `static/dashboard.html` in browser

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/upload` | POST | Upload file for analysis |
| `/status/{job_id}` | GET | Check job status |
| `/result/{job_id}` | GET | Get analysis result (JSON) |
| `/download/{job_id}` | GET | Download PDF report |
| `/jobs` | GET | List all jobs |
| `/job/{job_id}` | DELETE | Delete a job |

## Usage Examples

### cURL
```bash
# Upload a file
curl -X POST -F "file=@data.csv" http://localhost:8000/upload

# Check status
curl http://localhost:8000/status/{job_id}

# Download PDF
curl -O http://localhost:8000/download/{job_id}
```

### Python
```python
import requests

# Upload file
with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})
    job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:8000/status/{job_id}').json()
print(f"Progress: {status['progress']}%")

# Download PDF when complete
if status['status'] == 'completed':
    pdf_content = requests.get(f'http://localhost:8000/download/{job_id}').content
    with open('report.pdf', 'wb') as f:
        f.write(pdf_content)
```

## Project Structure

```
ai-insights-dashboard/
├── agents/
│   ├── __init__.py
│   └── insights_agents.py    # CrewAI multi-agent system
├── api/
│   ├── __init__.py
│   └── main.py               # FastAPI endpoints
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration management
├── utils/
│   ├── __init__.py
│   ├── file_parser.py        # Multi-format file parsing
│   ├── visualizer.py         # Chart generation
│   └── report_generator.py   # PDF report creation
├── templates/
│   └── report_template.html  # Jinja2 PDF template
├── static/
│   └── dashboard.html        # Web UI
├── uploads/                  # Uploaded files
├── output/                   # Generated PDFs and charts
├── main.py                   # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

Edit `.env` file to customize:

```env
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# File settings
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=csv,xlsx,xls,json,pdf,txt,docx,xml,parquet
```

## Available Ollama Models

Popular models for this use case:
- `llama3.2` - Good balance of speed and quality
- `mistral` - Fast and capable
- `codellama` - Better for code analysis
- `gemma` - Lightweight alternative

## Troubleshooting

### Ollama Connection Error
```bash
# Ensure Ollama is running
ollama serve

# Check available models
ollama list

# Pull required model
ollama pull llama3.2
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### PDF Generation Issues (Windows)
```bash
# Install GTK for WeasyPrint
# Download from: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Release
```

## License

MIT License