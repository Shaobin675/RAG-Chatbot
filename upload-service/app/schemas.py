from pydantic import BaseModel
from uuid import UUID

class UploadResponse(BaseModel):
    document_id: UUID
    status: str
