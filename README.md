RAG-Based AI Chatbot (Ongoing Project)

In my previous role, I spearheaded the design and end-to-end development of a custom RAG-Chatbot platform to solve critical inefficiencies in customer support and data search. I architected the core Retrieval-Augmented Generation pipeline, utilizing vector search to perform semantic retrieval across historical documents and ensure the AI provided factually grounded answers. Beyond the backend logic, I led the system design to improve user engagement by automating complex, data-dependent inquiries that previously required manual support intervention. While the code in this repository is a personal sample implementation I built for interview demonstrations, it reflects the architectural principles and coding standards I established for the production-grade system that successfully reduced support ticket volume and streamlined knowledge access for my previous employer.

This repository contains an ongoing AI chatbot project built on Retrieval-Augmented Generation (RAG) principles.
The system is designed using a microservice-friendly architecture, optimized for I/O-bound workloads, and intended to scale to high-concurrency, multi-user environments.


Core Technologies

LangGraph – Workflow orchestration and stateful LLM execution (Completed)

LangChain – Prompting, document loading, and chaining (Completed)

OpenAI / LLM APIs – Language understanding and generation (Completed)

Vector Database – FAISS (current), Pinecone (planned)

FastAPI – Async API and WebSocket support (Completed)

PostgreSQL – Persistent chat and metadata storage (Completed)  

Async Python (asyncio) – High-throughput I/O handling (Completed)

Docker – Containerized deployment and environment consistency (Completed)

S3 - Store uploaded documents (Completed)
 

 

 
