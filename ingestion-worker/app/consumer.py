import json
from kafka import KafkaConsumer
from app.config import settings
from app.extractor import extract_text
from app.chunker import chunk_text
from app.embedder import embed_chunks
from app.vector_store import upsert_vectors
from app.db import update_document_status

consumer = KafkaConsumer(
    settings.kafka_topic,
    bootstrap_servers=settings.kafka_bootstrap_servers,
    group_id="ingestion-workers",
    enable_auto_commit=False,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)

def run():
    for message in consumer:
        payload = message.value
        document_id = payload["document_id"]
        storage_uri = payload["storage_uri"]
        namespace = payload["namespace"]

        try:
            update_document_status(document_id, "PROCESSING")

            text = extract_text(storage_uri)
            chunks = chunk_text(text)
            vectors = embed_chunks(chunks)

            upsert_vectors(
                document_id=document_id,
                namespace=namespace,
                vectors=vectors,
            )

            update_document_status(document_id, "INDEXED")
            consumer.commit()

        except Exception as e:
            update_document_status(document_id, "FAILED")
            # no commit -> retry
            raise e
