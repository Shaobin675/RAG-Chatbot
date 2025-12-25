# config.py (public-safe, project root)
ORCHESTRATOR_URL = "http://orchestrator:8000"
RAG_SERVICE_URL = "http://rag-service:8001"
EMBEDDING_SERVICE_URL = "http://embedding-service:8002"
LANGGRAPH_SERVICE_URL = "http://langgraph-service:8003"
# Upload service (HTTP)
UPLOAD_SERVICE_URL= "http://upload-service:9000"
# Ingestion worker (NOT for embeddings; admin/health only)
INGESTION_SERVICE_URL="http://ingestion-worker:9001"

POSTGRES_HOST = "postgres"
POSTGRES_PORT = 5432
REDIS_HOST = "redis"
REDIS_PORT = 6379

# Idle timeout for frontend
IDLE_TIMEOUT_SECONDS = 180

# Pinecone
PINECONE_API_KEY=xxx
PINECONE_INDEX=documents