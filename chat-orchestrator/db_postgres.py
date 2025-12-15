# chat-orchestrator/db_postgres.py
from typing import Optional, List

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import select, update

from model import Base, ChatSession, ChatMessage, UploadedFile


class AsyncPostgresDB:
    """
    Async PostgreSQL database wrapper.
    Used ONLY by chat-orchestrator.
    """

    def __init__(self, dsn: str):
        self._engine = create_async_engine(
            dsn,
            echo=False,
            pool_pre_ping=True,
        )
        self._sessionmaker = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self):
        """
        Create database tables if they do not already exist.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self):
        """
        Dispose the async database engine.
        """
        await self._engine.dispose()

    async def get_or_create_session(self, session_key: str) -> ChatSession:
        """
        Retrieve a ChatSession by session_id, or create one if it does not exist.
        """
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(ChatSession).where(ChatSession.session_id == session_key)
            )
            chat_session = result.scalar_one_or_none()

            if chat_session:
                return chat_session

            chat_session = ChatSession(session_id=session_key)
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            return chat_session

    async def save_message(
        self,
        session_key: str,
        role: str,
        content: str,
        meta: Optional[dict] = None,
    ):
        """
        Persist a single chat message associated with a session.
        """
        async with self._sessionmaker() as session:
            chat_session = await self.get_or_create_session(session_key)

            msg = ChatMessage(
                session_id=chat_session.id,
                role=role,
                content=content,
                meta=meta,
            )
            session.add(msg)
            await session.commit()

    async def get_history(
        self,
        session_key: str,
        limit: int = 50,
    ) -> List[dict]:
        """
        Retrieve ordered chat history for a session.
        """
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(ChatMessage)
                .join(ChatSession)
                .where(ChatSession.session_id == session_key)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
            )

            rows = result.scalars().all()

            return [
                {
                    "role": r.role,
                    "content": r.content,
                    "meta": r.meta,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def save_uploaded_file(
        self,
        session_id: str,
        filename: str,
        content_type: Optional[str],
        meta: Optional[dict] = None,
    ):
        """
        Persist metadata for an uploaded file.
        """
        async with self._sessionmaker() as session:
            file = UploadedFile(
                session_id=session_id,
                filename=filename,
                content_type=content_type,
                meta=meta,
            )
            session.add(file)
            await session.commit()

    async def update_file_status(
        self,
        session_id: str,
        filename: str,
        status: str,
    ):
        """
        Update processing status of an uploaded file.
        """
        async with self._sessionmaker() as session:
            await session.execute(
                update(UploadedFile)
                .where(
                    UploadedFile.session_id == session_id,
                    UploadedFile.filename == filename,
                )
                .values(status=status)
            )
            await session.commit()
