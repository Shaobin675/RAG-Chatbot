"""
temporal-worker/trigger.py

Called by upload-service after a document lands in S3.
Starts a new IngestDocumentWorkflow execution via Temporal client.
Drop-in replacement for direct Kafka publish if Temporal handles orchestration.
"""

import asyncio
from temporalio.client import Client
from workflows import IngestDocumentWorkflow, IngestRequest

TEMPORAL_HOST = "temporal:7233"
TASK_QUEUE = "ingestion-queue"


async def trigger_ingestion(
    document_id: str,
    s3_key: str,
    source_type: str,
    uploaded_by: str,
) -> str:
    """
    Trigger a durable IngestDocumentWorkflow for a newly uploaded document.
    Returns the Temporal workflow run ID for tracking.
    """
    client = await Client.connect(TEMPORAL_HOST)

    request = IngestRequest(
        document_id=document_id,
        s3_key=s3_key,
        source_type=source_type,
        uploaded_by=uploaded_by,
    )

    # Workflow ID is deterministic — safe to retry on duplicate uploads
    workflow_id = f"ingest-{document_id}"

    handle = await client.start_workflow(
        IngestDocumentWorkflow.run,
        request,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    print(f"Started ingestion workflow: {workflow_id} | run_id={handle.result_run_id}")
    return handle.result_run_id


# Example: call from upload-service FastAPI endpoint
# run_id = asyncio.run(trigger_ingestion("doc-001", "uploads/guide.pdf", "guide", "admin"))
