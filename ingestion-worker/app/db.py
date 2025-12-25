from sqlalchemy import create_engine, text
from app.config import settings

engine = create_engine(settings.database_url)

def update_document_status(document_id: str, status: str):
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE documents SET status=:status WHERE id=:id"
            ),
            {"status": status, "id": document_id},
        )
