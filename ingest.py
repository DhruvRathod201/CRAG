from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"

def ingest_documents():
    print("Loading documents...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()

    txt_loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    try:
        documents += txt_loader.load()
    except Exception:
        pass

    if not documents:
        print("No documents found in ./docs — add PDFs or .txt files and re-run.")
        return

    print(f"Loaded {len(documents)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding and storing in ChromaDB...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Done! ChromaDB saved to {CHROMA_DIR}")

if __name__ == "__main__":
    ingest_documents()