RAG-Based AI Chatbot (Ongoing Project)

In my previous role, I spearheaded the design and end-to-end development of a custom RAG-Chatbot platform to solve critical inefficiencies in customer support and data search. I architected the core Retrieval-Augmented Generation pipeline, utilizing vector search to perform semantic retrieval across historical documents and ensure the AI provided factually grounded answers. Beyond the backend logic, I led the system design to improve user engagement by automating complex, data-dependent inquiries that previously required manual support intervention. While the code in this repository is a personal sample implementation I built for interview demonstrations, it reflects the architectural principles and coding standards I established for the production-grade system that successfully reduced support ticket volume and streamlined knowledge access for my previous employer.

## High‑Level Architecture

```
Client / UI
     │
     ▼
Chat Orchestrator (API Gateway)
     │
     ├── Embedding Service
     ├── Vector Store (FAISS + Redis)
     ├── LangGraph Service (Agentic Workflow)
     └── Ingestion / Indexing Workers
```

Key design goals:

* Loose coupling between services
* Horizontal scalability for I/O‑bound workloads
* Clear extension points for models, retrievers, and workflows

---

## Repository Structure (Code Overview)

```
RAG-Chatbot/
├── chat-orchestrator/     # Main FastAPI service; entry point for chat queries
├── embedding-service/     # Standalone service for embedding generation
├── ingestion-worker/      # Background worker for document processing
├── rag-indexer/           # Builds and updates vector indexes
├── faiss_redis/           # Vector store layer using FAISS with Redis support
├── langgraph-service/     # LangGraph-based agentic orchestration service
├── upload-service/        # Handles document uploads from UI or API
├── vite-project/          # Frontend UI (Vite + React)
├── public/                # Static assets
├── config.py              # Centralized configuration
├── docker-compose.yml     # Local multi-service orchestration
└── README.md
```

Each directory represents a **self-contained service** with its own runtime, dependencies, and Dockerfile.

---

## Service Responsibilities

### 1. chat‑orchestrator

**Role:** API gateway and request coordinator.

Responsibilities:

* Accepts user chat requests
* Generates query embeddings via `embedding-service`
* Retrieves relevant context from the vector store
* Delegates reasoning and response generation to `langgraph-service`
* Returns final responses to the client

This service is intentionally thin and stateless, making it suitable for horizontal scaling.

---

### 2. embedding‑service

**Role:** Centralized embedding generation.

Responsibilities:

* Converts documents and queries into vector embeddings
* Abstracts embedding model choice from other services

This isolation allows embedding models to be swapped or optimized (batching, quantization) without touching ingestion or query logic.

---

### 3. ingestion‑worker

**Role:** Document ingestion and preprocessing.

Responsibilities:

* Reads uploaded documents
* Extracts and normalizes text
* Splits text into chunks
* Requests embeddings from `embedding-service`
* Persists vectors into FAISS via `faiss_redis`

Designed as a worker so ingestion can be scaled independently from query traffic.

---

### 4. rag‑indexer

**Role:** Vector index construction and maintenance.

Responsibilities:

* Builds FAISS indexes from embedded documents
* Rebuilds or updates indexes as data changes

This separation keeps indexing logic out of request‑time services.

---

### 5. faiss_redis

**Role:** Vector storage and retrieval layer.

Responsibilities:

* Stores embeddings in FAISS
* Uses Redis for metadata, caching, or coordination
* Exposes similarity search APIs to the orchestrator

The abstraction allows future replacement with managed vector databases.

---

### 6. langgraph‑service

**Role:** Agentic reasoning and workflow orchestration.

Responsibilities:

* Defines LangGraph workflows (retrieve → reason → answer)
* Supports multi‑step and conditional execution paths
* Keeps orchestration logic separate from the API layer

This enables advanced reasoning without coupling it to request handling.

---

### 7. upload‑service

**Role:** Document intake.

Responsibilities:

* Accepts file uploads from UI or API clients
* Stores raw files and metadata
* Triggers ingestion workflows

---

### 8. Frontend (vite‑project)

**Role:** User interface.

Responsibilities:

* Upload documents
* Send chat queries
* Display model responses

The frontend communicates only with `chat‑orchestrator` and `upload‑service`.

---

## Configuration

Configuration is **environment‑driven** and centralized via `config.py`.

Typical configuration includes:

* Service ports
* Model and embedding parameters
* Vector store paths
* Redis connection details

Secrets and environment‑specific values should be supplied via `.env` and Docker Compose.

---

## Running Locally (Development)

Build all services:

```bash
docker compose build
```

Start the full stack:

```bash
docker compose up
```

Once running:

* API traffic flows through `chat‑orchestrator`
* Document ingestion occurs asynchronously
* Frontend UI connects to local services

---

## Design Principles Reflected in Code

* **Microservice isolation**: each capability is independently deployable
* **Replaceable components**: embeddings, vector store, and LLMs can be swapped
* **Scalability‑first**: ingestion, retrieval, and inference scale independently
* **Production orientation**: Dockerized services, stateless APIs, clear boundaries

---

## Project Status & Core Technologies

LangGraph – Workflow orchestration and stateful LLM execution (Completed)

LangChain – Prompting, document loading, and chaining (Completed)

OpenAI / LLM APIs – Language understanding and generation (Completed)

Vector Database – FAISS (current), Pinecone (planned)

FastAPI – Async API and WebSocket support (Completed)

PostgreSQL – Persistent chat and metadata storage (Completed)  

Async Python (asyncio) – High-throughput I/O handling (Completed)

Docker – Containerized deployment and environment consistency (Completed)

S3 - Store uploaded documents (Completed)
 

 

 
