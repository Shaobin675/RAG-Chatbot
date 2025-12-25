import boto3
import uuid
from app.config import settings

s3 = boto3.client(
    "s3",
    endpoint_url=settings.s3_endpoint,
    aws_access_key_id=settings.s3_access_key,
    aws_secret_access_key=settings.s3_secret_key,
    region_name=settings.s3_region,
)

def upload_file(file_obj, content_type: str) -> str:
    key = f"raw/{uuid.uuid4()}"
    s3.upload_fileobj(
        file_obj,
        settings.s3_bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"s3://{settings.s3_bucket}/{key}"
