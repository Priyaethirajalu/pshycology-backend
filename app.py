# app.py - FastAPI backend

import os
from fastapi import FastAPI, UploadFile, File, Query as FastAPIQuery
from pydantic import BaseModel
import uvicorn
from rag import get_retriever
from rag import ingest_data
from langchain_openai import ChatOpenAI
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
USE_LOCAL_EMBEDDINGS = os.environ.get("USE_LOCAL_EMBEDDINGS") == "1"

app = FastAPI()

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# prepare retriever and QA chain (lazy load)
_retriever = None
_llm = None

def get_qa_chain():
    global _retriever, _llm
    if _retriever is None or _llm is None:
        _retriever = get_retriever(k=4)
        _llm = _get_llm()

    def qa_run(question: str) -> str:
        # For LangChain v0.2+, retrievers are Runnables; use invoke()
        docs = _retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            "You are a supportive, careful psychology assistant. Use the provided context to answer the user's question.\n" 
            "If the context is insufficient, say so briefly and provide general guidance. Be concise and kind.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        result = _llm.invoke(prompt)
        return getattr(result, "content", str(result))

    return qa_run

def _get_llm():
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return ChatGroq(model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"), api_key=groq_key)
    # Prefer local Ollama next (no API key required)
    try:
        return ChatOllama(model=OLLAMA_MODEL)
    except Exception:
        pass
    # Finally, use OpenAI only if a key is explicitly provided
    if OPENAI_API_KEY:
        return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    raise RuntimeError("No LLM provider configured. Set GROQ_API_KEY or run Ollama locally, or provide OPENAI_API_KEY.")

class Query(BaseModel):
    text: str
    session_id: str = None

@app.get("/")
async def root():
    return {"message": "Psychology backend running on Render!"}

@app.post("/chat")
async def chat(q: Query):
    qa = get_qa_chain()
    # safety wrapper: check for critical phrases
    text_lower = q.text.lower()
    if any(word in text_lower for word in ["suicide", "kill myself", "self-harm"]):
        return {"text": "If you are in immediate danger or thinking about harming yourself, please contact local emergency services or a crisis hotline immediately. This assistant is not a substitute for emergency care."}

    try:
        resp = qa(q.text)
        return {"text": resp}
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "RateLimitError" in msg:
            # Try switching to local LLM via Ollama once
            try:
                global _llm
                # Prefer Groq if GROQ_API_KEY is available
                if os.environ.get("GROQ_API_KEY"):
                    _llm = ChatGroq(model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"), api_key=os.environ.get("GROQ_API_KEY"))
                else:
                    _llm = ChatOllama(model=OLLAMA_MODEL)
                resp = qa(q.text)
                return {"text": resp}
            except Exception:
                return {"text": "The language model API quota has been exceeded. I've switched to local embeddings; if a local LLM isn't available (Ollama), please try again later or update your API plan."}
        raise

@app.get("/debug/llm")
async def debug_llm():
    # Report which LLM provider/model is active and whether local embeddings are on
    provider = "unknown"
    model_name = None
    if os.environ.get("GROQ_API_KEY"):
        provider = "groq"
        model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    elif OPENAI_API_KEY:
        provider = "openai"
        model_name = "gpt-4o-mini"
    else:
        provider = "ollama"
        model_name = OLLAMA_MODEL
    return {
        "provider": provider,
        "model": model_name,
        "use_local_embeddings": os.environ.get("USE_LOCAL_EMBEDDINGS") == "1",
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_groq_key": bool(os.environ.get("GROQ_API_KEY")),
    }

# Upload one or more documents to backend data folder, optional reindex
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), reindex: bool = FastAPIQuery(True)):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    os.makedirs(data_dir, exist_ok=True)
    saved_files = []
    for f in files:
        contents = await f.read()
        out_path = os.path.join(data_dir, f.filename)
        with open(out_path, "wb") as out:
            out.write(contents)
        saved_files.append(f.filename)
    if reindex:
        # Allow reindexing if either OpenAI embeddings are available or local embeddings are enabled
        if not OPENAI_API_KEY and not USE_LOCAL_EMBEDDINGS:
            return {
                "saved": saved_files,
                "reindexed": False,
                "error": "Embeddings not configured. Set OPENAI_API_KEY or USE_LOCAL_EMBEDDINGS=1",
            }
        try:
            ingest_data(data_dir="./data")
            # force retriever refresh on next request
            global _retriever
            _retriever = None
        except Exception as e:
            return {"saved": saved_files, "reindexed": False, "error": str(e)}
    return {"saved": saved_files, "reindexed": bool(reindex)}

# Trigger reindex without uploading
@app.post("/reindex")
async def reindex():
    # Allow reindexing if either OpenAI embeddings are available or local embeddings are enabled
    if not OPENAI_API_KEY and not USE_LOCAL_EMBEDDINGS:
        return {"status": "error", "message": "Embeddings not configured. Set OPENAI_API_KEY or USE_LOCAL_EMBEDDINGS=1"}
    try:
        ingest_data(data_dir="./data")
        global _retriever
        _retriever = None
        return {"status": "ok", "message": "reindexed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Optional: accept audio file and return transcription using OpenAI Whisper
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    # Save temp file
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tf.write(contents)
    tf.flush()
    tf.close()
    # Server-side whisper example (if you have local whisper or call OpenAI's speech-to-text API)
    # For brevity, we return a placeholder
    return {"text": "transcription-placeholder"}

if __name__ == "__main__":
import os
port = int(os.environ.get("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)


