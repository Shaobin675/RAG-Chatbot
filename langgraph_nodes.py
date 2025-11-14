import asyncio
from datetime import datetime, timezone
from typing import Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from utils import RWLock

class LangGraphNodes:

    @staticmethod
    async def retrieve_node(state: dict, ws: Optional = None) -> dict:
        """
        Retrieve relevant documents from RAG engine and summarize.
        """
        rag = state.get("rag")
        query = state.get("user_message")
        state.update({"retrieved_docs": [], "rag_answer": "", "confidence": 0.0, "summary": ""})

        try:
            rwlock: RWLock = state.get("rag_lock")
            await rwlock.acquire_read()
            try:
                docs_with_scores = await rag.similarity_search_with_score(query, k=10)
            finally:
                await rwlock.release_read()

            if not docs_with_scores:
                return state

            docs, scores = zip(*docs_with_scores) if docs_with_scores else ([], [])
            state["retrieved_docs"] = list(docs)
            state["rag_answer"] = "\n\n".join([d.page_content for d in docs])
            state["confidence"] = round(sum(scores) / len(scores), 3) if scores else 0.5
            print(f"[RAG] Retrieved {len(docs)} docs, confidence: {state['confidence']:.3f}")

            # Chunk summaries
            summaries: List[str] = []
            all_text = state["rag_answer"]
            max_chunk_size = 2000
            total_chunks = (len(all_text) + max_chunk_size - 1) // max_chunk_size if all_text else 0

            for idx, start in enumerate(range(0, len(all_text), max_chunk_size), start=1):
                chunk_text = all_text[start:start + max_chunk_size]
                summary_prompt = f"Summarize the following document excerpt briefly:\n{chunk_text}\nChunk summary:"
                try:
                    resp = await state["llm"].ainvoke([HumanMessage(content=summary_prompt)])
                    chunk_summary = getattr(resp, "content", "").strip()
                except Exception:
                    chunk_summary = ""
                summaries.append(chunk_summary)
                if ws:
                    try:
                        await ws.send_text(f"⏳ Summarizing chunk {idx}/{total_chunks}...")
                    except Exception:
                        pass
                await asyncio.sleep(0.01)

            if summaries:
                combined_text = "\n\n".join([s for s in summaries if s])
                final_prompt = f"Combine the following summaries (<=300 words):\n{combined_text}\nFinal summary:"
                try:
                    final_resp = await state["llm"].ainvoke([HumanMessage(content=final_prompt)])
                    state["summary"] = getattr(final_resp, "content", "").strip()
                except Exception:
                    state["summary"] = combined_text

        except Exception as e:
            print("[Retrieve Node Error]", e)
        return state

    @staticmethod
    async def decide_node(state: dict) -> dict:
        state["use_rag"] = bool(state.get("rag_answer")) and float(state.get("confidence", 0)) >= 0.4
        print(f"[RAG Decision] Using {'RAG' if state['use_rag'] else 'Fallback'}")
        ws = state.get("ws")
        if ws:
            try:
                await ws.send_text(f"🛠️ Using {'RAG' if state['use_rag'] else 'Fallback'}")
            except Exception:
                pass
        return state

    @staticmethod
    async def rag_generate_node(state: dict) -> dict:
        llm, db, sid = state["llm"], state["db"], state["session_id"]
        msg, summary = state["user_message"], state.get("summary", "")
        hist = await db.get_history(sid, limit=100)
        conversation = "\n".join(hist)
        prompt = f"RAG Summary:\n{summary}\nConversation:\n{conversation}\nUser: {msg}"
        try:
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            state["llm_output"] = getattr(resp, "content", "").strip()
        except Exception:
            state["llm_output"] = "⚠️ Failed to generate RAG response."
        return state

    @staticmethod
    async def fallback_node(state: dict) -> dict:
        llm, db, sid = state["llm"], state["db"], state["session_id"]
        msg = state["user_message"]
        hist = await db.get_history(sid, limit=100)
        conv = "\n".join(hist)
        summary = state.get("summary", "")
        if summary:
            conv += f"\n\n[Uploaded File Summary]:\n{summary}"
        prompt = f"{conv}\nUser: {msg}"
        try:
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            state["llm_output"] = getattr(resp, "content", "").strip()
        except Exception:
            state["llm_output"] = "⚠️ Fallback LLM failed."
        return state

    @staticmethod
    async def memory_node(state: dict) -> dict:
        db, sid, output = state["db"], state["session_id"], state.get("llm_output", "<no output>")
        try:
            await db.insert_chat(sid, output, "Bot", timestamp=datetime.now(timezone.utc))
            state["history"] = await db.get_history(sid, limit=100)
        except Exception as e:
            print("[Memory Node Error]", e)
        return state

    @staticmethod
    async def run_graph(state: dict) -> dict:
        graph = StateGraph(dict)
        graph.add_node("retrieve", LangGraphNodes.retrieve_node)
        graph.add_node("decide", LangGraphNodes.decide_node)
        graph.add_node("rag_generate", LangGraphNodes.rag_generate_node)
        graph.add_node("fallback", LangGraphNodes.fallback_node)
        graph.add_node("memory", LangGraphNodes.memory_node)
        graph.add_edge("retrieve", "decide")
        graph.add_conditional_edges("decide", lambda s: "rag_generate" if s.get("use_rag") else "fallback")
        graph.add_edge("rag_generate", "memory")
        graph.add_edge("fallback", "memory")
        graph.add_edge("memory", END)
        graph.set_entry_point("retrieve")
        compiled_graph = graph.compile()
        return await compiled_graph.ainvoke(state)
