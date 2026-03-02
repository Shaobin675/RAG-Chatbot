"""
temporal-worker/main.py

Starts the Temporal worker that listens on the ingestion task queue.
Registers all activities and workflows defined in workflows.py.
"""

import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from workflows import (
    IngestDocumentWorkflow,
    fetch_document_activity,
    chunk_document_activity,
    embed_and_index_activity,
    notify_observability_activity,
)

TEMPORAL_HOST = "temporal:7233"
TASK_QUEUE = "ingestion-queue"


async def main():
    client = await Client.connect(TEMPORAL_HOST)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[IngestDocumentWorkflow],
        activities=[
            fetch_document_activity,
            chunk_document_activity,
            embed_and_index_activity,
            notify_observability_activity,
        ],
    )

    print(f"Temporal worker listening on task queue: {TASK_QUEUE}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
