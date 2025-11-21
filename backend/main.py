"""
Robust FastAPI backend for an intents-based ML assistant.

Features:
- Loads intents from backend/intents.csv (columns: 'intent', 'text') or via /upload_intents (CSV multipart).
- Builds embeddings using SentenceTransformer.
- Uses FAISS if available (fast). If faiss is missing, falls back to scikit-learn NearestNeighbors (works on Windows).
- Endpoints:
    GET  /health
    GET  /metadata
    POST /predict          -> JSON {"query":"...", "top_k":3}
    POST /upload_intents   -> multipart file (CSV)
- Persists index/embeddings & metadata to joblib files for faster restarts.
"""
# backend/main.py
"""
Full main.py for Office Assistant MVP
- Intent prediction using embeddings + faiss
- Upload CSV to build index on the fly
- /chat endpoint that uses OpenAI Chat API (OPENAI_API_KEY)
- Serves frontend from ../frontend_build
Notes:
 - Dockerfile should run: uvicorn main:app --host 0.0.0.0 --port $PORT
 - Do NOT include uvicorn.run() here (deployment uses Docker/uvicorn)
"""

import os
import io
import json
import csv
from typing import List, Optional, Dict, Any

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Optional external libraries
try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# OpenAI
try:
    import openai
except Exception:
    openai = None


# ----------------------------
# App initialization
# ----------------------------
app = FastAPI(title="Office Assistant MVP")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # backend/
INTENTS_CSV = os.path.join(APP_ROOT, "intents.csv")
EMB_PATH = os.path.join(APP_ROOT, "embeddings.npy")
META_PATH = os.path.join(APP_ROOT, "metadata.joblib")

FRONTEND_DIR = os.path.join(os.path.dirname(APP_ROOT), "frontend_build")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ----------------------------
# Index Helpers
# ----------------------------

def load_index():
    """Load embeddings + metadata."""
    if not os.path.exists(EMB_PATH) or not os.path.exists(META_PATH):
        return None, None
    return np.load(EMB_PATH), joblib.load(META_PATH)


def save_index(embeddings: np.ndarray, metadata: dict):
    np.save(EMB_PATH, embeddings)
    joblib.dump(metadata, META_PATH)


def build_index_from_df(df: pd.DataFrame):
    """Build embeddings + metadata from dataframe."""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers missing.")

    # find text column
    text_col = None
    for c in ["text", "utterance", "query", "sentence", "content", "prompt"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]

    texts = df[text_col].astype(str).tolist()
    intents = df["intent"].astype(str).tolist() if "intent" in df.columns else None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False)

    metadata = {"texts": texts, "columns": list(df.columns)}
    if intents:
        metadata["intents"] = intents

    return embeddings, metadata


def knn_search(embeddings: np.ndarray, query_emb: np.ndarray, top_k: int = 3):
    """Run KNN search with FAISS or fallback brute force cosine."""
    if faiss is not None:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        emb = embeddings.astype("float32")
        faiss.normalize_L2(emb)

        index.add(emb)

        q = query_emb.astype("float32")
        faiss.normalize_L2(q)

        distances, indices = index.search(np.atleast_2d(q), top_k)
        return indices[0].tolist(), distances[0].tolist()

    # fallback brute force cosine
    from numpy.linalg import norm
    emb = embeddings
    q = np.asarray(query_emb).reshape(-1)

    emb_norm = emb / (norm(emb, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (norm(q) + 1e-12)

    sims = np.dot(emb_norm, q_norm)
    idx = np.argsort(-sims)[:top_k]

    return idx.tolist(), sims[idx].tolist()


# ----------------------------
# Predict Endpoint
# ----------------------------

class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3


@app.post("/predict")
async def predict(req: PredictRequest):
    # ensure index exists
    embeddings, metadata = load_index()
    if embeddings is None:
        # try auto-build from backend/intents.csv or /mnt/data/intent.csv
        source = None
        if os.path.exists(INTENTS_CSV):
            source = INTENTS_CSV
        elif os.path.exists("/mnt/data/intent.csv"):
            source = "/mnt/data/intent.csv"

        if not source:
            raise HTTPException(status_code=404, detail="No index available. Upload intents.")

        df = pd.read_csv(source)
        embeddings, metadata = build_index_from_df(df)
        save_index(embeddings, metadata)

    if SentenceTransformer is None:
        raise HTTPException(status_code=500, detail="sentence-transformers not installed.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([req.text])[0]

    idx, scores = knn_search(embeddings, q_emb, req.top_k)

    results = []
    for i, s in zip(idx, scores):
        entry = {
            "index": int(i),
            "text": metadata["texts"][i],
            "score": float(s),
            "intent": metadata["intents"][i] if "intents" in metadata else None,
        }
        results.append(entry)

    return {"query": req.text, "results": results}


# ----------------------------
# Upload Intents (FILE or JSON URL)
# ----------------------------

@app.post("/upload_intents")
async def upload_intents(
    file: UploadFile = File(None),
    url: Optional[str] = Body(None)
):
    """
    Accepts:
    - multipart file upload: file=<CSV>
    - JSON: {"url": "/mnt/data/intent.csv"}
    """
    # Case A: file upload
    if file is not None:
        try:
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(400, f"Failed to read uploaded CSV: {e}")

    # Case B: read CSV from server path
    elif url:
        if not os.path.exists(url):
            raise HTTPException(400, f"File not found: {url}")
        try:
            df = pd.read_csv(url)
        except Exception as e:
            raise HTTPException(400, f"Failed to read CSV at {url}: {e}")

    else:
        raise HTTPException(400, "Send file or JSON {'url': '/path/to/file.csv'}")

    try:
        embeddings, metadata = build_index_from_df(df)
        save_index(embeddings, metadata)
    except Exception as e:
        raise HTTPException(500, f"Failed to build index: {e}")

    return {"status": "ok", "entries": len(metadata["texts"])}


# ----------------------------
# Chat Endpoint (OpenAI)
# ----------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "gpt-4o-mini"
    max_tokens: Optional[int] = 250


OPENAI_API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("OPENAI_KEY")
)

if openai and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY


def openai_chat_reply(messages, model="gpt-4o-mini", max_tokens=250):
    if openai is None:
        return "OpenAI Python library missing."

    if not OPENAI_API_KEY:
        return "OpenAI API key not configured."

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"OpenAI error: {e}")


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    reply = openai_chat_reply(msgs, req.model, req.max_tokens)
    return {"reply": reply}


# ----------------------------
# Health Check
# ----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "embeddings": os.path.exists(EMB_PATH),
        "metadata": os.path.exists(META_PATH),
        "faiss": faiss is not None,
        "sentence_transformers": SentenceTransformer is not None,
        "frontend": os.path.isdir(FRONTEND_DIR),
        "openai": openai is not None
    }
