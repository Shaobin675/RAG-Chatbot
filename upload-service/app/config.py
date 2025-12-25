from pydantic import BaseSettings

class Settings(BaseSettings):
    service_name: str = "upload-service"

    # S3 / MinIO
    s3_endpoint: str
    s3_bucket: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str = "us-east-1"

    # Database
    database_url: str

    # Kafka
    kafka_bootstrap_servers: str
    kafka_topic: str = "document.uploaded"

    max_file_size_mb: int = 50

    class Config:
        env_file = ".env"

settings = Settings()
