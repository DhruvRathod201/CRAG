import streamlit as st
from crag_pipeline import run_crag
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Corrective RAG", page_icon="🔍", layout="wide")

st.title("🔍 Corrective RAG (CRAG)")
st.caption("Powered by LangChain · llama3-8b via Groq · ChromaDB")

with st.sidebar:
    st.header("About CRAG")
    st.markdown("""
**Corrective RAG** improves on standard RAG by:
1. Retrieving relevant chunks
2. Checking if they're actually relevant
3. If not — rewriting the query and re-retrieving
4. Generating a grounded final answer

**Stack**
- 🦜 LangChain
- ☁️ llama3-8b via Groq
- 🗃️ ChromaDB
- 🌐 Streamlit
""")
    st.divider()

    if os.path.exists("./chroma_db"):
        st.success("✅ ChromaDB ready")
    else:
        st.error("❌ No ChromaDB — run `python ingest.py` first")

    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API key loaded")
    else:
        st.error("❌ No API key — add GROQ_API_KEY to .env")

    st.divider()
    st.subheader("⚙️ Settings")
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_trace = st.toggle("Show pipeline trace", value=True)

query = st.text_input(
    "Ask a question about your documents:",
    placeholder="e.g. What is Corrective RAG? How does it reduce hallucination?"
)

if st.button("Ask", type="primary") and query:
    if not os.path.exists("./chroma_db"):
        st.error("Please run `python ingest.py` first to build the vector store.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("Please add your GROQ_API_KEY to the .env file.")
    else:
        with st.spinner("Running CRAG pipeline..."):
            result = run_crag(query)

        if result["corrected"]:
            st.warning("⚠️ Query was corrected — low relevance detected in first retrieval.")
        else:
            st.success("✅ First retrieval was relevant — no correction needed.")

        if show_trace:
            with st.expander("🔎 Pipeline trace", expanded=True):
                for step in result["steps"]:
                    st.write(step)

        st.subheader("💬 Answer")
        st.write(result["answer"])

        if show_chunks:
            with st.expander("📄 Retrieved chunks"):
                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk {i}** — `{chunk.metadata.get('source', 'unknown')}`")
                    st.text(chunk.page_content[:400] + ("..." if len(chunk.page_content) > 400 else ""))
                    st.divider()