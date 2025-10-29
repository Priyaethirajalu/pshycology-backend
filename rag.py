# rag.py
# Responsible for ingestion, embeddings, and building/returning a retriever
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Optional

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERSIST_DIR = "chroma_db"
USE_LOCAL = os.environ.get("USE_LOCAL_EMBEDDINGS", "0") == "1"

def ingest_data(data_dir="./data"):
    # Resolve data directory relative to this file and ensure it exists
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), data_dir))
    os.makedirs(base_dir, exist_ok=True)

    loader = DirectoryLoader(base_dir, glob="**/*.*")
    docs = loader.load()
    if not docs:
        print(f"No documents found in {base_dir}. Add files (pdf, txt, docx) and rerun.")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"Ingested {len(chunks)} chunks into {PERSIST_DIR}")

def get_retriever(k=4):
    embeddings = _get_embeddings()
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever

def _get_embeddings():
    if USE_LOCAL or not OPENAI_API_KEY:
        # Lazy import to avoid heavy load if not used
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = os.environ.get("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)


