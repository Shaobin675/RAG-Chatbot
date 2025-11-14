import asyncio
from typing import Optional

# ============================================================
# Read-Write Lock for asyncio
# ============================================================
class RWLock:
    """
    Async Read-Write Lock.
    - Multiple readers allowed concurrently.
    - Writers are exclusive.
    """
    def __init__(self):
        self._readers = 0
        self._writer = asyncio.Lock()
        self._reader_lock = asyncio.Lock()
        self._readers_queue = asyncio.Lock()

    async def acquire_read(self):
        """Acquire read lock (non-exclusive)."""
        async with self._readers_queue:
            async with self._reader_lock:
                self._readers += 1
                if self._readers == 1:
                    await self._writer.acquire()

    async def release_read(self):
        """Release read lock."""
        async with self._reader_lock:
            self._readers -= 1
            if self._readers == 0:
                self._writer.release()

    async def acquire_write(self):
        """Acquire write lock (exclusive)."""
        await self._writer.acquire()

    def release_write(self):
        """Release write lock."""
        self._writer.release()


# ============================================================
# Helper to safely send WebSocket text
# ============================================================
async def safe_ws_send(ws, message: str):
    if ws:
        try:
            await ws.send_text(message)
        except Exception:
            pass
