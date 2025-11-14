import os
import json
import base64
import tempfile
import shutil
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from langgraph_nodes import LangGraphNodes
from utils import RWLock, safe_ws_send

# ============================================================
# WebSocket manager for multi-user chat sessions
# ============================================================
class WebSocketManager:
    IDLE_TIMEOUT = timedelta(minutes=3)
    WARNING_SECONDS = 30
    WARNING_INTERVAL = 5

    def __init__(self, db, rag, llm=None):
        """
        db: AsyncPostgresDB instance
        rag: RAGEngine instance
        llm: Optional ChatOpenAI instance
        """
        self.db = db
        self.rag = rag
        self.llm = llm or rag.llm
        self.active_connections: Dict[str, WebSocket] = {}
        self.last_active: Dict[str, datetime] = {}
        self._last_warning_sent: Dict[str, int] = {}
        self._rag_lock = RWLock()

    # ============================================================
    # Setup FastAPI WebSocket route
    # ============================================================
    def setup_routes(self, app):
        @app.websocket("/ws/{session_id}")
        async def ws_endpoint(ws: WebSocket, session_id: str):
            await ws.accept()
            self.active_connections[session_id] = ws
            self.last_active[session_id] = datetime.now(timezone.utc)

            try:
                while True:
                    data = await ws.receive_text()
                    self.last_active[session_id] = datetime.now(timezone.utc)
                    self._last_warning_sent.pop(session_id, None)

                    # Handle JSON messages
                    try:
                        msg = json.loads(data)
                        if msg.get("type") == "file_upload":
                            await self._handle_file_upload(ws, session_id, msg)
                        else:
                            resp = await self._handle_user_message(session_id, str(msg), ws)
                            await safe_ws_send(ws, resp)
                    except json.JSONDecodeError:
                        resp = await self._handle_user_message(session_id, data, ws)
                        await safe_ws_send(ws, resp)

            except WebSocketDisconnect:
                self._cleanup_session(session_id)
            except Exception as e:
                print("[WebSocket Error]", e)
                self._cleanup_session(session_id)

    # ============================================================
    # Handle uploaded file
    # ============================================================
    async def _handle_file_upload(self, ws, session_id: str, msg: dict):
        filename, filedata = msg["filename"], base64.b64decode(msg["data"])
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)

        try:
            with open(temp_path, "wb") as f:
                f.write(filedata)
            await safe_ws_send(ws, f"📁 Received {filename}, processing...")

            # Acquire write lock for RAG index update
            await self._rag_lock.acquire_write()
            try:
                if hasattr(self.rag, "build_index_from_folder"):
                    await self.rag.build_index_from_folder(temp_dir)
                    await safe_ws_send(ws, f"✅ Knowledge base updated with {filename}")
            finally:
                self._rag_lock.release_write()

            # Summarize file with LangGraph
            state = {
                "session_id": session_id,
                "user_message": f"Summarize {filename}",
                "db": self.db,
                "llm": self.llm,
                "rag_lock": self._rag_lock,
                "ws": ws,
                "rag": self.rag,
            }
            state = await LangGraphNodes.retrieve_node(state, ws=ws)
            state = await LangGraphNodes.decide_node(state)
            summary = state.get("summary", "<no summary>")
            await self.db.insert_chat(session_id, f"[Uploaded File Summary]: {summary}", "Bot")
            await safe_ws_send(ws, f"📝 Final summary of {filename}:\n{summary}")

        except Exception as e:
            print("[File Upload Error]", e)
            await safe_ws_send(ws, f"⚠️ Failed to process {filename}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ============================================================
    # Handle normal user message
    # ============================================================
    async def _handle_user_message(self, session_id: str, message: str, ws=None):
        self.last_active[session_id] = datetime.now(timezone.utc)
        self._last_warning_sent.pop(session_id, None)

        await self.db.insert_chat(session_id, message, "User", timestamp=datetime.now(timezone.utc))

        state = {
            "session_id": session_id,
            "user_message": message,
            "db": self.db,
            "llm": self.llm,
            "rag_lock": self._rag_lock,
            "ws": ws,
            "rag": self.rag,
        }
        try:
            result = await LangGraphNodes.run_graph(state)
            return result.get("llm_output", "<no output>")
        except Exception as e:
            print("[LangGraph Error]", e)
            return "⚠️ Internal error occurred."

    # ============================================================
    # Monitor idle sessions
    # ============================================================
    async def monitor_idle_sessions(self):
        while True:
            now = datetime.now(timezone.utc)
            for sid, ws in list(self.active_connections.items()):
                idle = now - self.last_active.get(sid, now)
                secs = int((self.IDLE_TIMEOUT - idle).total_seconds())

                if secs <= 0:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    self._cleanup_session(sid)
                elif secs <= self.WARNING_SECONDS:
                    last_warned = self._last_warning_sent.get(sid, -999)
                    if secs % self.WARNING_INTERVAL == 0 and secs != last_warned:
                        await safe_ws_send(ws, f"⚠️ Idle timeout in {secs} seconds")
                        self._last_warning_sent[sid] = secs
            await asyncio.sleep(1)

    # ============================================================
    # Cleanup disconnected session
    # ============================================================
    def _cleanup_session(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.last_active.pop(session_id, None)
        self._last_warning_sent.pop(session_id, None)
