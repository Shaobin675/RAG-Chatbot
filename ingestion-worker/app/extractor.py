import boto3
from urllib.parse import urlparse

def extract_text(storage_uri: str) -> str:
    parsed = urlparse(storage_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)

    return obj["Body"].read().decode("utf-8")
