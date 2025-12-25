from pydantic import BaseSettings

class Settings(BaseSettings):
    kafka_bootstrap_servers: str
    kafka_topic: str = "document.uploaded"

    pinecone_api_key: str
    pinecone_index: str

    database_url: str
    embedding_model: str = "text-embedding-3-large"

    class Config:
        env_file = ".env"

settings = Settings()
