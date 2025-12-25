from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID
from app.schemas import UploadResponse
from app.storage import upload_file
from app.producer import publish_document_uploaded
from app.models import Document
from app.config import settings

router = APIRouter(prefix="/v1/documents", tags=["documents"])

def validate_file(file: UploadFile):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Missing content type")

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    user_id: UUID,
    namespace: str,
    file: UploadFile = File(...),
    db: Session = Depends(),
):
    validate_file(file)

    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(status_code=413, detail="File too large")

    storage_uri = upload_file(
        file_obj=iter([data]),
        content_type=file.content_type,
    )

    doc = Document(
        user_id=user_id,
        namespace=namespace,
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=len(data),
        storage_uri=storage_uri,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    publish_document_uploaded({
        "document_id": str(doc.id),
        "namespace": namespace,
        "storage_uri": storage_uri,
    })

    return UploadResponse(document_id=doc.id, status=doc.status)
