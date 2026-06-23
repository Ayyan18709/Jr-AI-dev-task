# 🤖 Production RAG Chatbot – Qwen2.5-1.5B-Instruct

A fully containerised, production-grade **Retrieval-Augmented Generation** chatbot
built on **Qwen2.5-1.5B-Instruct** with hybrid BM25 + FAISS retrieval,
cross-encoder reranking, conversation memory, Redis caching, and Prometheus monitoring.

---

### 🎥 **[Watch the Project Demo Video Here (Loom) - (https://www.loom.com/share/e41bd5fee88b4c1fa7200a459ee0c127)**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Streamlit Frontend (port 8501)                      │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ HTTP
┌───────────────────────────────────▼──────────────────────────────────┐
│                          FastAPI (port 8000)                         │
│  ┌──────────┐  ┌───────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │POST /chat│  │POST /upload_cv│  │GET/health│  │GET/metrics    │  │
│  └────┬─────┘  └───────┬───────┘  └──────────┘  └───────────────┘  │
│       │                │                                              │
│  ┌────▼────────┐  ┌────▼──────────────────────────────────────────┐ │
│  │ Redis Cache │  │           Ingestion Pipeline                   │ │
│  │  (6379)     │  │  PDF/TXT → Chunker → FAISS + BM25 index       │ │
│  └────┬────────┘  └───────────────────────────────────────────────┘ │
│       │                                                               │
│  ┌────▼──────────────────────────────────────────┐                  │
│  │            Hybrid Retriever                    │                  │
│  │  ┌──────────────────┐  ┌─────────────────────┐│                  │
│  │  │ Dense (FAISS)    │  │ Sparse (BM25)        ││                  │
│  │  │ all-MiniLM-L6-v2 │  │ rank-bm25            ││                  │
│  │  └────────┬─────────┘  └──────────┬──────────┘│                  │
│  │           └──────── Fusion ────────┘           │                  │
│  │                60% + 40% weighted               │                  │
│  │           ┌──── Cross-Encoder Reranker ─────┐  │                  │
│  │           │  ms-marco-MiniLM-L-6-v2          │  │                  │
│  │           └──────────────────────────────────┘  │                  │
│  └───────────────────────────────────────────────┘                  │
│                                                                       │
│  ┌──────────────────────────────────────────────┐                   │
│  │        Qwen2.5-1.5B-Instruct (HF)            │                   │
│  │   General Chat  │  RAG Mode (ctx injected)    │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                       │
│  ┌──────────────────────────────────────────────┐                   │
│  │  ConversationSummaryMemory (per session)     │                   │
│  │  True Summary Buffer + Disk Persistence      │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                       │
│  ┌──────────────────────────────────────────────┐                   │
│  │  Prometheus metrics → /metrics/prometheus     │                   │
│  └──────────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## How RAG Works

1. **Upload CV** → `POST /upload_cv`  
   PDF / TXT → cleaned → word-level chunks (400 words, 50 overlap) →  
   embedded with `all-MiniLM-L6-v2` → stored in FAISS & BM25.
   > **Why 400 chunks and 50 overlap?** 400 words captures meaningful context (like a full project) without diluting semantic meaning. The 50-word sliding overlap ensures sentence boundaries aren't lost across chunks.

2. **Query** → `POST /chat` (mode = `"rag"`)  
   - **Dense retrieval**: FAISS cosine search (top 10)  
   - **Sparse retrieval**: BM25 keyword search (top 10)  
   - **Score fusion**: `score = 0.6 × semantic + 0.4 × BM25`  
     *(FAISS handles conceptual understanding, BM25 prevents missing exact technical keywords).*
   - **Reranking**: cross-encoder (`ms-marco-MiniLM-L-6-v2`) narrows to top 5 chunks.
   - Top 5 chunks injected into prompt → Qwen2.5 generates grounded answer.
     *(Limiting to Top 5 prevents the LLM "Lost in the Middle" phenomenon).*

---

## How Memory Works

Each session maintains a **`ConversationSummaryMemory`** acting as a true Conversation Summary Buffer:

- Strictly preserves the exact text of the last 10 Q&A pairs (defined by `max_interactions`).
- When the 11th interaction is added, the oldest interaction is popped and a background thread is spawned.
- The background thread asynchronously uses the LLM to update a continuous, rolling summary of the conversation with the pruned interaction, ensuring zero API latency delay.
- Context injected into prompts: `[Summary of Previous Conversation] + [Exact Last 10 Interactions]`.
- All session states are persistently saved to `data/sessions.json` so memory survives server crashes and Docker restarts.

---

## Project Structure

```
project/
├── app/
│   ├── main.py                  ← FastAPI app + startup
│   ├── api/
│   │   ├── routes.py            ← All endpoints
│   │   ├── schemas.py           ← Pydantic models
│   │   └── dependencies.py      ← Singleton DI providers
│   ├── core/
│   │   ├── config.py            ← Pydantic Settings
│   │   ├── logger.py            ← Structured logging
│   │   ├── memory.py            ← ConversationSummaryMemory
│   │   └── cache.py             ← Redis cache layer
│   ├── llm/
│   │   ├── model.py             ← Qwen2.5-1.5B-Instruct wrapper
│   │   └── prompt_engine.py     ← Prompt templates
│   ├── rag/
│   │   ├── embedder.py          ← SentenceTransformer + batch cache
│   │   ├── vectorstore.py       ← FAISS index
│   │   ├── bm25.py              ← BM25Okapi wrapper
│   │   └── retriever.py         ← Hybrid retrieval + reranking
│   ├── ingestion/
│   │   ├── cv_loader.py         ← PDF + TXT loader
│   │   └── chunker.py           ← Sliding-window chunker
│   ├── evaluation/
│   │   ├── metrics.py           ← Faithfulness, precision, recall, relevance
│   │   └── evaluator.py         ← Batch evaluation pipeline
│   ├── monitoring/
│   │   └── metrics.py           ← Prometheus counters / histograms
│   └── utils/
│       ├── text_processing.py   ← Cleaning utilities
│       └── helpers.py           ← Timer, JSON I/O, source formatting
├── data/
│   ├── eval_dataset.json        ← Evaluation Q&A dataset
│   └── uploads/                 ← Uploaded CV files
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
├── run.sh
└── README.md
```

---

## Running Locally

### Prerequisites
- Python 3.10+
- Redis (optional – cache disabled automatically if unavailable)

```bash
# 1. Clone / enter project
cd "Jr AI dev task"

# 2. Copy env file
cp .env.example .env

# 3. Create venv & install
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Create required directories
mkdir -p data/uploads data/faiss_index logs

# 5. Start the API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the helper script:
```bash
bash run.sh           # or: sh run.sh
```

API available at **http://localhost:8000** | Docs at **http://localhost:8000/docs**

---

## Running via Docker

The application is heavily containerized using Docker-Compose.

```bash
# Build and start all 3 services (Redis Cache, FastAPI Backend, Streamlit Frontend)
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f api
```

> **First run**: The Qwen2.5-1.5B-Instruct model (~3 GB) is downloaded from
> HuggingFace on first startup and cached in a Docker volume.
> Subsequent restarts are instantaneous.

---

## 🎨 Streamlit Frontend GUI
Once running via Docker or Python directly, access the user interface at:
**[http://localhost:8501](http://localhost:8501)**

The frontend automatically tracks system health, allows drag-and-drop document uploads, and seamlessly integrates the conversation buffer memory via session IDs.

---

## API Usage Examples

### Upload CV
```bash
curl -X POST http://localhost:8000/upload_cv \
  -F "file=@/path/to/your_cv.pdf"
```

### Chat (RAG mode)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the candidate's educational background?",
    "session_id": "user-001",
    "mode": "rag",
    "top_k": 5
  }'
```

### Chat (general mode)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain RAG in one sentence.", "mode": "chat"}'
```

### Health check
```bash
curl http://localhost:8000/health
```

### Metrics (JSON)
```bash
curl http://localhost:8000/metrics
```

### Prometheus scrape
```bash
curl http://localhost:8000/metrics/prometheus
```

---

## Evaluation Instructions

### 1. Prepare dataset
Edit `data/eval_dataset.json` – add questions and reference answers specific
to the uploaded CV:
```json
[
  {"question": "What degree does the candidate hold?", "reference_answer": "BSc Computer Science"},
  ...
]
```

### 2. Upload the CV first
```bash
curl -X POST http://localhost:8000/upload_cv -F "file=@data/cv.pdf"
```

### 3. Run evaluation via API
```bash
curl -X POST http://localhost:8000/evaluate
```

### 4. Run evaluation directly (CLI)
```python
from app.api.dependencies import get_llm, get_retriever, get_embedder
from app.evaluation.evaluator import run_evaluation

llm = get_llm(); llm.load()
embedder = get_embedder(); embedder.load()
retriever = get_retriever()

report = run_evaluation(retriever, llm, embedder)
print(report["aggregate_metrics"])
```

Results are saved to `data/eval_report.json`.

### Metrics explained

| Metric | Description | Range |
|---|---|---|
| **Faithfulness** | Answer grounded in retrieved context | 0–1 |
| **Context Precision** | Retrieved chunks relevant to the answer | 0–1 |
| **Context Recall** | Answer covered by retrieved context | 0–1 |
| **Answer Relevance** | Cosine sim between question and answer | 0–1 |
| **Composite Score** | Average of all four metrics | 0–1 |

---

## Configuration

All parameters are configurable via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | HF model ID |
| `LLM_DEVICE` | `auto` | cpu / cuda / mps / auto |
| `LLM_LOAD_IN_4BIT` | `false` | 4-bit quant (CUDA only) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer |
| `USE_RERANKER` | `true` | Enable cross-encoder reranking |
| `CHUNK_SIZE` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per query |
| `SEMANTIC_WEIGHT` | `0.6` | Dense retrieval weight |
| `BM25_WEIGHT` | `0.4` | Sparse retrieval weight |
| `MEMORY_MAX_INTERACTIONS` | `10` | Max stored Q&A pairs |
| `CACHE_TTL` | `3600` | Redis TTL in seconds |
| `REDIS_HOST` | `localhost` | Redis hostname |

---

## Monitoring

Prometheus metrics are exposed at `/metrics/prometheus` and include:

- `rag_requests_total` – request count by endpoint + status
- `rag_request_latency_seconds` – end-to-end latency histogram
- `rag_retrieval_latency_ms` – retrieval latency
- `rag_inference_latency_ms` – LLM inference latency
- `rag_cache_hits_total` / `rag_cache_misses_total`
- `rag_cache_hit_ratio` – rolling hit rate
- `rag_chunks_indexed` – indexed chunk count

Point Prometheus at `http://localhost:8000/metrics/prometheus` to scrape.
