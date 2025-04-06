# Recipe Recommendation API

This Flask API provides recipe recommendations using RAG (Retrieval-Augmented Generation) with ChromaDB and Ollama.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running on your system

3. Start the Flask server:
```bash
python app.py
```

The server will start on http://localhost:5000

## API Endpoints

### 1. Search Recipes
**Endpoint:** `POST /api/search`

**Request Body:**
```json
{
    "query": "Give me a healthy dinner recipe",
    "embedding_model": "minilm",
    "top_k": 5,
    "ollama_model": "mistral"
}
```

**Response:**
```json
{
    "response": "Recipe details and instructions...",
    "context": [...],
    "timing": {
        "embedding_time": 0.1,
        "search_time": 0.2,
        "response_time": 1.5,
        "total_time": 1.8
    }
}
```

### 2. Health Check
**Endpoint:** `GET /api/health`

**Response:**
```json
{
    "status": "healthy"
}
```

## Parameters

- `query`: Your recipe search query (required)
- `embedding_model`: Model for text embeddings (optional, default: "minilm")
- `top_k`: Number of similar recipes to retrieve (optional, default: 5)
- `ollama_model`: LLM model for response generation (optional, default: "mistral")
