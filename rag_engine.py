import os
import asyncio
import shutil
from typing import List, Optional
import numpy as np
import redis.asyncio as aioredis
from config import REDIS_URL

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

try:
    from langchain_community.document_loaders import (
        UnstructuredWordDocumentLoader,
        UnstructuredExcelLoader,
        UnstructuredPowerPointLoader,
    )
    OFFICE_LOADER_AVAILABLE = True
except ImportError:
    OFFICE_LOADER_AVAILABLE = False
    print("⚠️ Office loaders not available. .doc/.xls/.ppt files skipped.")

FAISS_TEMP_DIR = os.path.join(os.getcwd(), "faiss_redis")
os.makedirs(FAISS_TEMP_DIR, exist_ok=True)


class RAGEngine:
    """
    Async RAG Engine using FAISS in local folder (faiss_redis).
    Supports incremental updates on file uploads and auto-loading local index.
    """

    def __init__(self, api_key=None, redis_url=REDIS_URL, redis_key="faiss_index"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set")

        self.redis_url = redis_url
        self.redis_key = redis_key
        self.redis: Optional[aioredis.Redis] = None

        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key)
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.rag_chain = None

        # Index paths
        self.index_dir = FAISS_TEMP_DIR
        self.redis_index_name = "faiss_index"

        # Start async task safely
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.load_local_index_if_available())
        except RuntimeError:
            asyncio.run(self.load_local_index_if_available())

    async def init_redis(self):
        self.redis = aioredis.from_url(self.redis_url)
        print(f"✅ Connected to Redis at {self.redis_url}")

    async def load_local_index_if_available(self):
        """
        Load FAISS index from local folder if it exists.
        """
        if self.redis:
            try:
                exists = await self.redis.exists(self.redis_index_name)
                if exists:
                    print("[RAG] Redis already has an index. Skipping local load.")
                    return
            except Exception:
                pass

        faiss_file = os.path.join(self.index_dir, "faiss_index.faiss")
        pkl_file = os.path.join(self.index_dir, "faiss_index.pkl")
        if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
            print("[RAG] No local FAISS index found. You may upload files to build it.")
            return

        try:
            print("[RAG] Loading local FAISS index from disk...")
            self.vectorstore = FAISS.load_local(
                folder_path=self.index_dir,
                embeddings=self.embeddings,
                index_name="faiss_index",
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            self._build_chain()
            print("[RAG] Local FAISS index successfully loaded.")
        except Exception as e:
            print(f"[RAG] Failed to load local FAISS index: {e}")

    async def build_index_from_folder(self, folder_path: str, incremental: bool = True):
        if incremental:
            try:
                await self.load_index()
            except FileNotFoundError:
                print("⚠️ No existing FAISS index. Creating new index.")
                self.vectorstore = None

        all_docs = []
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            loader = None
            lower_name = filename.lower()
            try:
                if lower_name.endswith(".pdf"):
                    loader = PyMuPDFLoader(path)
                elif lower_name.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                elif OFFICE_LOADER_AVAILABLE and lower_name.endswith((".doc", ".docx")):
                    loader = UnstructuredWordDocumentLoader(path)
                elif OFFICE_LOADER_AVAILABLE and lower_name.endswith((".xls", ".xlsx")):
                    try:
                        loader = UnstructuredExcelLoader(path)
                    except Exception as e:
                        print(f"⚠️ Excel loader failed for {filename}: {e}")
                        continue
                elif OFFICE_LOADER_AVAILABLE and lower_name.endswith((".ppt", ".pptx")):
                    try:
                        loader = UnstructuredPowerPointLoader(path)
                    except Exception as e:
                        print(f"⚠️ PowerPoint loader failed for {filename}: {e}")
                        continue
                else:
                    print(f"Skipping unsupported file: {filename}")
                    continue

                docs = await asyncio.to_thread(loader.load)
                print(f"📄 Loaded {len(docs)} document(s) from {filename}")
                all_docs.extend(docs)
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
                continue

        if not all_docs:
            print("⚠️ No new documents found. Skipping index update.")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = await asyncio.to_thread(splitter.split_documents, all_docs)
        print(f"Processing {len(chunks)} new chunks...")

        if self.vectorstore:
            await asyncio.to_thread(self.vectorstore.add_documents, chunks)
        else:
            self.vectorstore = await asyncio.to_thread(FAISS.from_documents, chunks, self.embeddings)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self._build_chain()
        self.persist_index_local()
        if not self.redis:
            await self.init_redis()
        await self.redis.set(self.redis_key, "1")
        print(f"✅ FAISS index updated and saved to '{FAISS_TEMP_DIR}'")

    async def load_index(self):
        faiss_file = os.path.join(FAISS_TEMP_DIR, "faiss_index.faiss")
        if not os.path.exists(faiss_file):
            raise FileNotFoundError("FAISS index not found. Build it first.")

        self.vectorstore = FAISS.load_local(
            FAISS_TEMP_DIR,
            self.embeddings,
            index_name="faiss_index",
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self._build_chain()
        print(f"✅ FAISS index loaded successfully from '{FAISS_TEMP_DIR}'")

    def _build_chain(self):
        system_prompt = (
            "You are a precise and helpful assistant. "
            "Answer the question using only the provided context. "
            "If the answer is not in the context, respond with: 'Calling OpenAI'\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        qa_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, qa_chain)

    def persist_index_local(self):
        if not self.vectorstore:
            raise RuntimeError("No vectorstore to persist.")
        self.vectorstore.save_local(FAISS_TEMP_DIR, index_name="faiss_index")
        print(f"💾 FAISS index persisted to '{FAISS_TEMP_DIR}'")

    async def delete_index(self):
        if not self.redis:
            await self.init_redis()
        await self.redis.delete(self.redis_key)
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        print(f"🧹 FAISS index deleted from Redis key '{self.redis_key}'")
        if os.path.exists(FAISS_TEMP_DIR):
            shutil.rmtree(FAISS_TEMP_DIR)
            os.makedirs(FAISS_TEMP_DIR, exist_ok=True)

    async def query(self, question: str):
        if self.retriever is None:
            return "I don't know.", 0.0

        try:
            docs_with_scores = await self.similarity_search_with_score(question, k=4)
        except Exception as e:
            print("[RAG Error]", e)
            return "I don't know.", 0.0

        if not docs_with_scores:
            return "I don't know.", 0.0

        scores = [score for _, score in docs_with_scores]
        confidence = round(sum(scores) / len(scores), 3) if scores else 0.5
        docs = [doc for doc, _ in docs_with_scores]
        print(f"[RAG] Confidence: {confidence}, using {len(docs)} retrieved documents")

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            answer = getattr(response, "content", "").strip()
            return answer, confidence
        except Exception as e:
            print("[RAG LLM Error]", e)
            return "I don't know.", confidence

    async def similarity_search_with_score(self, query: str, k: int = 4):
        """
        Correct FAISS similarity search returning docs and normalized confidence scores.
        """
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not loaded. Build or load index first.")

        def _search():
            query_emb = np.array(self.embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
            D, I = self.vectorstore.index.search(query_emb, k)  # distances & indices
            docs = []
            distances = D[0]
            for idx in I[0]:
                key = self.vectorstore.index_to_docstore_id[idx]
                docs.append(self.vectorstore.docstore[key])
            return docs, distances

        docs, distances = await asyncio.to_thread(_search)

        # Normalize distances -> confidence (higher = more similar)
        max_dist = max(distances) if distances.size > 0 else 1.0
        results = [(doc, 1.0 - dist / max_dist) for doc, dist in zip(docs, distances)]
        return results
