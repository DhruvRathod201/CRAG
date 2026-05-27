# CRAG Project

This project is a simple Corrective Retrieval-Augmented Generation (CRAG) app built with Streamlit, LangChain, Groq, and ChromaDB.

## Requirements

- Python 3.10+
- A `GROQ_API_KEY` in `.env`

## Install

```bash
pip install -r requirements.txt
```

## Add Documents

Place your PDF or text files inside the `docs/` folder.

## Build the Vector Store

```bash
python ingest.py
```

## Run the App

```bash
streamlit run app.py
```

## Notes

- The vector database is stored locally in `chroma_db/`.
- Local secrets in `.env` are excluded from Git.
