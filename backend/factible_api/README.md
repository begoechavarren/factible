# factible API

FastAPI backend for the factible YouTube fact-checking service with real-time streaming.

## Features

- **Server-Sent Events (SSE)**: Real-time progress updates during fact-checking
- **Async Processing**: Non-blocking pipeline execution
- **Structured Logging**: Comprehensive logging throughout the pipeline
- **CORS Support**: Pre-configured for frontend integration
- **OpenAPI Docs**: Auto-generated API documentation
- **Type Safety**: Full Pydantic validation for requests/responses

## Project Structure

```
factible_api/
├── api/
│   └── v1/
│       ├── endpoints/
│       │   ├── fact_check.py    # SSE streaming endpoint
│       │   └── health.py        # Health check endpoint
│       └── router.py            # API router configuration
├── core/
│   └── config.py                # Application settings
├── schemas/
│   └── v1/
│       ├── requests.py          # Request models
│       └── responses.py         # Response models
└── main.py                      # Application entry point
```

## Quick Start

### Installation

```bash
cd backend/factible_api
pip install -e .
```

### Run Development Server

```bash
# From backend/factible_api directory
python -m factible_api.main

# Or with uvicorn directly
uvicorn factible_api.main:app --reload --port 8000
```

### API Documentation

Once running, visit:
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## API Endpoints

### Health Check
```bash
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### Fact-Check Stream (SSE)
```bash
POST /api/v1/fact-check/stream
Content-Type: application/json

{
  "video_url": "https://www.youtube.com/watch?v=iGkLcqLWxMA",
  "max_claims": 5,
  "max_queries_per_claim": 2,
  "max_results_per_query": 3
}
```

**Response (Server-Sent Events):**
```
data: {"step":"transcript_extraction","message":"Extracting transcript...","progress":5,"data":null}

data: {"step":"transcript_complete","message":"Transcript extracted (1234 characters)","progress":15,"data":{"transcript_length":1234}}

data: {"step":"claim_extraction","message":"Extracting factual claims...","progress":20,"data":null}

data: {"step":"claims_extracted","message":"Extracted 5 claims","progress":35,"data":{"total_claims":5,...}}

data: {"step":"processing_claim_1","message":"Processing claim 1/5...","progress":40,"data":{...}}

data: {"step":"complete","message":"Fact-checking complete!","progress":100,"data":{"result":{...}}}
```

## Testing with cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Fact-check streaming
curl -X POST http://localhost:8000/api/v1/fact-check/stream \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=iGkLcqLWxMA",
    "max_claims": 3
  }' \
  --no-buffer
```

## Testing with Python

```python
import requests
import json

def stream_fact_check(video_url: str):
    url = "http://localhost:8000/api/v1/fact-check/stream"
    data = {
        "video_url": video_url,
        "max_claims": 3
    }

    with requests.post(url, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    update = json.loads(line[6:])
                    print(f"[{update['progress']}%] {update['message']}")
                    if update['step'] == 'complete':
                        print("Result:", update['data']['result'])

stream_fact_check("https://www.youtube.com/watch?v=iGkLcqLWxMA")
```

## Configuration

Environment variables (`.env`):

```bash
# API Configuration
DEBUG=false
LOG_LEVEL=info

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]

# Factible Pipeline
MAX_CLAIMS=5
MAX_QUERIES_PER_CLAIM=2
MAX_RESULTS_PER_QUERY=3
HEADLESS_SEARCH=true

# Server
HOST=0.0.0.0
PORT=8000
```

## Frontend Integration

### JavaScript/React Example

```javascript
const eventSource = new EventSource(
  'http://localhost:8000/api/v1/fact-check/stream?' +
  new URLSearchParams({
    video_url: 'https://www.youtube.com/watch?v=iGkLcqLWxMA',
    max_claims: 5
  })
);

eventSource.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log(`[${update.progress}%] ${update.message}`);

  if (update.step === 'complete') {
    console.log('Result:', update.data.result);
    eventSource.close();
  }
};

eventSource.onerror = () => {
  console.error('Connection error');
  eventSource.close();
};
```

## Development

### Project Dependencies

The API requires the `factible` package to be installed:

```bash
cd backend
pip install -e ./factible
pip install -e ./factible_api
```

### Running Tests

```bash
pytest
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn factible_api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY backend/factible ./factible
COPY backend/factible_api ./factible_api

RUN pip install -e ./factible -e ./factible_api

CMD ["uvicorn", "factible_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Architecture

### SSE Streaming Flow

1. Client sends POST request with video URL
2. Server starts async processing pipeline
3. Server streams progress updates via SSE
4. Each pipeline step yields progress update
5. Final update includes complete results
6. Client receives all updates in real-time

### Error Handling

- Invalid URLs return 400 Bad Request
- Pipeline failures stream error updates
- All errors logged with full context
- Graceful degradation for partial failures

## License

[To be determined]
