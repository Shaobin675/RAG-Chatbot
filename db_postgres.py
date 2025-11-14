# db_postgres.py
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
import asyncpg

class AsyncPostgresDB:
    """Asynchronous PostgreSQL DB for chat history storage."""
    
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.pool.Pool] = None

    async def init_db(self):
        """Initialize connection pool and create table with index if not exists."""
        self.pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=10)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL
                )
            """)
            # Index for faster retrieval by session and timestamp
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp
                ON chat_history(session_id, timestamp)
            """)
        print("✅ PostgreSQL DB initialized.")

    async def insert_chat(self, session_id: str, message: str, role: str, timestamp: Optional[datetime] = None):
        """Insert a new chat message."""
        timestamp = timestamp or datetime.now(timezone.utc)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO chat_history(session_id, role, message, timestamp)
                VALUES($1, $2, $3, $4)
            """, session_id, role, message, timestamp)

    async def get_history(self, session_id: str, limit: int = 100) -> List[str]:
        """Retrieve the last `limit` messages of a session, ordered oldest to newest."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT role, message
                FROM chat_history
                WHERE session_id = $1
                ORDER BY timestamp ASC
                LIMIT $2
            """, session_id, limit)
            return [f"{r['role']}: {r['message']}" for r in rows]

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            print("🧹 PostgreSQL DB connection pool closed.")
