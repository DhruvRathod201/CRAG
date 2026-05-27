from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from typing import List, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_DIR = "./chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.1-8b-instant",   # free, fast, great for RAG
    api_key=os.getenv("GROQ_API_KEY"),
)


def load_vectorstore() -> Chroma:
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def retrieve_chunks(query: str, k: int = 4) -> List[Document]:
    vs = load_vectorstore()
    return vs.similarity_search(query, k=k)


def check_relevance(query: str, chunks: List[Document]) -> Tuple[bool, str]:
    context = "\n\n".join([c.page_content for c in chunks])
    prompt = f"""You are a relevance checker. Given the user question and retrieved context,
decide if the context is relevant enough to answer the question.

Question: {query}

Retrieved context:
{context}

Reply with ONLY 'RELEVANT' or 'NOT RELEVANT', then a one-sentence reason.
"""
    response = llm.invoke(prompt)
    result = response.content.strip()
    is_relevant = result.upper().startswith("RELEVANT")
    return is_relevant, result


def rewrite_query(original_query: str) -> str:
    prompt = f"""Rewrite the following question to make it clearer and more specific
for a document search system. Return ONLY the rewritten question, nothing else.

Original question: {original_query}
Rewritten question:"""
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_answer(query: str, chunks: List[Document]) -> str:
    context = "\n\n".join([c.page_content for c in chunks])
    prompt = f"""You are a helpful assistant. Answer the question based only on the context below.
If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content.strip()


def run_crag(query: str) -> dict:
    steps = []
    corrected = False

    steps.append("🔍 Retrieving chunks from ChromaDB...")
    chunks = retrieve_chunks(query)

    steps.append("🧠 Checking relevance with llama3-8b (Groq)...")
    is_relevant, relevance_reason = check_relevance(query, chunks)
    steps.append(f"  → {relevance_reason}")

    if not is_relevant:
        corrected = True
        steps.append("⚠️ Low relevance — rewriting query...")
        new_query = rewrite_query(query)
        steps.append(f"  → Rewritten: \"{new_query}\"")
        steps.append("🔄 Re-retrieving with improved query...")
        chunks = retrieve_chunks(new_query, k=6)

    steps.append("✍️ Generating final answer with llama3-8b (Groq)...")
    answer = generate_answer(query, chunks)

    return {
        "answer": answer,
        "steps": steps,
        "retrieved_chunks": chunks,
        "corrected": corrected,
    }