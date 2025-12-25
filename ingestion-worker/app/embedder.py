from openai import OpenAI
from app.config import settings

client = OpenAI()

def embed_chunks(chunks):
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=chunks,
    )
    return [r.embedding for r in response.data]
