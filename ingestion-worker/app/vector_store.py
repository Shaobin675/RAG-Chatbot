import pinecone
from app.config import settings
import uuid

pinecone.init(api_key=settings.pinecone_api_key)
index = pinecone.Index(settings.pinecone_index)

def upsert_vectors(document_id, namespace, vectors):
    items = [
        (str(uuid.uuid4()), v, {"document_id": document_id})
        for v in vectors
    ]
    index.upsert(items, namespace=namespace)
