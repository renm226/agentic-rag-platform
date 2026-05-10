# Agentic RAG 

A production-grade document intelligence platform — upload documents, ask questions, get cited answers powered by a multi-agent AI pipeline.

**Stack:** FastAPI · PostgreSQL/pgvector · LangGraph · CrewAI · Celery · Redis · Prometheus/Grafana · xAI Grok · sentence-transformers

---

## What it does

1. **Ingest** — upload PDFs, HTML pages, or plain text; a Celery worker chunks and embeds them locally (no OpenAI embeddings needed)
2. **Retrieve** — hybrid BM25 + vector similarity search against pgvector, with deduplication and token-budget enforcement
3. **Answer** — three query modes:
   - `/query` — standard RAG answer with confidence score
   - `/query/intelligent` — LangGraph agent that reformulates low-confidence queries and retries
   - `/query/crew` — CrewAI pipeline with four sequential agents (Query Planner → Retrieval Specialist → Answer Synthesizer → Fact Checker)
4. **Monitor** — structured JSON logs + Prometheus metrics + Grafana dashboard out of the box

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │            FastAPI (async)            │
                    │  /ingest  /query  /query/crew  /docs  │
                    └────────────┬──────────────────────────┘
                                 │
              ┌──────────────────┼────────────────────┐
              ▼                  ▼                     ▼
    ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
    │  Celery Worker   │  │  LangGraph   │  │  CrewAI (4-agent)│
    │  (doc chunking + │  │  QA Agent    │  │  multi-step RAG  │
    │   embeddings)    │  │  (reformulate│  │  pipeline        │
    └────────┬─────────┘  │   & retry)   │  └──────────────────┘
             │            └──────┬───────┘
             ▼                   ▼
    ┌──────────────────────────────────────┐
    │   PostgreSQL + pgvector extension    │
    │   (documents · chunks · embeddings)  │
    └──────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │      Redis      │   ← Celery broker & result backend
    └─────────────────┘
```

---

## Run locally (Docker — one command)

```bash
git clone <repo-url>
cd agentic-rag-platform

cp env.example .env
# Add your XAI_API_KEY to .env can use free tire

docker compose up -d
```

Open **http://localhost:8000/docs** for the interactive API.  
Grafana dashboard: **http://localhost:3000** (admin / admin)

---

## Run without Docker

```bash
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt

cp env.example .env  # fill in DATABASE_URL, REDIS_URL, XAI_API_KEY

python migrate.py                              # run DB migrations

# Terminal 1 — Celery worker
celery -A app.celery_app worker --loglevel=info

# Terminal 2 — API server
uvicorn app.main:app --reload --port 8000
```

---

## Key API endpoints

| Method | Endpoint | What it does |
|--------|----------|--------------|
| `POST` | `/ingest` | Upload a document (URL or file path) |
| `POST` | `/query` | Standard RAG — returns answer + sources + confidence |
| `POST` | `/query/intelligent` | LangGraph agent — reformulates & retries if confidence is low |
| `POST` | `/query/crew` | 4-agent CrewAI pipeline for complex multi-part questions |
| `POST` | `/retrieve` | Raw hybrid retrieval (no answer generation) |
| `GET`  | `/documents` | List ingested documents |
| `GET`  | `/stats/{org_id}` | Usage stats per organisation |
| `GET`  | `/metrics` | Prometheus metrics |
| `GET`  | `/health` | Health check |

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `XAI_API_KEY` | xAI Grok API key — free tier at console.x.ai |
| `XAI_MODEL` | Model name (default: `grok-beta`) |
| `EMBEDDING_MODEL` | Local embedding model (default: `BAAI/bge-base-en-v1.5`) |
| `DATABASE_URL` | PostgreSQL + asyncpg connection string |
| `REDIS_URL` | Redis connection string |
| `SECRET_KEY` | JWT secret |
| `API_BASE_URL` | Self-call URL for CrewAI re-retrieval tool (Docker: `http://app:8000`) |

---

## Technical highlights

- **Zero-cost LLM** — xAI Grok via OpenAI-compatible API; local sentence-transformers for embeddings (no per-query embedding cost)
- **Multi-agent reasoning** — CrewAI orchestrates four specialised agents; each agent verifies the previous step's output before proceeding
- **Adaptive retrieval** — LangGraph agent detects low-confidence answers and reformulates the query using keywords extracted from the initial results
- **Async throughout** — FastAPI + asyncpg + SQLAlchemy 2.0; Celery offloads heavy document processing so the API stays non-blocking
- **Observable** — every query records latency, token counts, confidence scores, and agent attempts to Prometheus; pre-built Grafana dashboard included
