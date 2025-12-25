import json
from kafka import KafkaProducer
from app.config import settings

producer = KafkaProducer(
    bootstrap_servers=settings.kafka_bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def publish_document_uploaded(payload: dict):
    '''
    Docstring for publish_document_uploaded
    document.uploaded event payload example:{
    "document_id": "uuid",
    "namespace": "customer-a",
    "storage_uri": "s3://bucket/raw/uuid"
    }   
    '''

    producer.send(settings.kafka_topic, payload)
    producer.flush()
