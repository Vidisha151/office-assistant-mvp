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
import csv
import json
from typing import List, Optional, Dict, Any

import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Optional libraries
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

# App
app = FastAPI(title="Office Assistant MVP")

# Paths (adjust if needed)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # backend/
INTENTS_CSV = os.path.join(APP_ROOT, "intents.csv")
EMB_PATH = os.path.join(APP_ROOT, "embeddings.npy")
META_PATH = os.path.join(APP_ROOT, "metadata.joblib")
FRONTEND_DIR = os.path.join(os.path.dirname(APP_ROOT), "frontend_build")  # ../frontend_build

# Mount frontend (serves index.html)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ---------------------------
# Utilities: load / save index
# ---------------------------
def load_index():
    """
    Load embeddings and metadata from disk. Returns (embeddings_np, metadata_dict)
    metadata expected format: {'texts': [...], 'intents': [...]} or similar
    """
    if not os.path.exists(EMB_PATH) or not os.path.exists(META_PATH):
        return None, None

    embeddings = np.load(EMB_PATH)
    metadata = joblib.load(META_PATH)
    return embeddings, metadata


def save_index(embeddings: np.ndarray, metadata: dict):
    np.save(EMB_PATH, embeddings)
    joblib.dump(metadata, META_PATH)


def build_index_from_csv(csv_path: str, model_name: str = "all-MiniLM-L6-v2"):
    """
    Build embeddings + metadata from CSV file.
    CSV must contain a text column (tries common column names otherwise uses first column)
    Returns (embeddings, metadata)
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed in the environment. Install it or upload index files.")

    # load CSV
    import pandas as pd
    df = pd.read_csv(csv_path)
    # find best text column
    text_col = None
    for c in ["text", "utterance", "query", "sentence", "content", "prompt"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]

    texts = df[text_col].astype(str).tolist()
    # optional: get intents column if present
    intents = df["intent"].astype(str).tolist() if "intent" in df.columns else None

    # create embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)
    metadata = {"texts": texts}
    if intents is not None:
        metadata["intents"] = intents
    metadata["columns"] = list(df.columns)
    return embeddings, metadata


# ---------------------------
# Search helper (faiss or brute)
# ---------------------------
def knn_search(embeddings: np.ndarray, query_emb: np.ndarray, top_k: int = 3):
    # embeddings: N x D, query_emb: D or 1 x D
    if faiss is not None:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product (we'll normalize)
        # normalize if necessary
        # convert to float32
        emb32 = embeddings.astype("float32")
        # normalize vectors for cosine similarity
        faiss.normalize_L2(emb32)
        index.add(emb32)
        q = query_emb.astype("float32")
        faiss.normalize_L2(q)
        distances, indices = index.search(np.atleast_2d(q), top_k)
        # distances are similarity scores (since normalized + IP)
        return indices[0].tolist(), distances[0].tolist()
    else:
        # fallback: brute-force cosine
        from numpy.linalg import norm
        # ensure 2D
        q = np.asarray(query_emb).reshape(-1)
        emb = np.asarray(embeddings)
        # normalize
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        sims = np.dot(emb_norm, q_norm)
        idx = np.argsort(-sims)[:top_k]
        return idx.tolist(), sims[idx].tolist()


# ---------------------------
# Intent prediction endpoint
# ---------------------------
class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3


@app.post("/predict")
async def predict(req: PredictRequest):
    # load index
    embeddings, metadata = load_index()
    if embeddings is None or metadata is None:
        # try to build from backend/intents.csv if present
        # fallback path /mnt/data/intent.csv (uploaded to chat) is also tested
        fallback_csv = None
        if os.path.exists(INTENTS_CSV):
            fallback_csv = INTENTS_CSV
        elif os.path.exists("/mnt/data/intent.csv"):
            fallback_csv = "/mnt/data/intent.csv"
        if fallback_csv:
            try:
                embeddings, metadata = build_index_from_csv(fallback_csv)
                save_index(embeddings, metadata)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"No index available and failed to build: {e}")
        else:
            raise HTTPException(status_code=404, detail="No index available. Upload intents or place intents.csv in backend/")

    # vectorize incoming text
    if SentenceTransformer is None:
        raise HTTPException(status_code=500, detail="sentence-transformers not installed on server for encoding. Provide precomputed embeddings.")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([req.text])[0]

    # search
    idxs, scores = knn_search(embeddings, q_emb, top_k=req.top_k or 3)

    results = []
    for i, s in zip(idxs, scores):
        text = metadata.get("texts", [])[i] if "texts" in metadata else ""
        intent = metadata.get("intents", [])[i] if "intents" in metadata else None
        results.append({"index": int(i), "text": text, "intent": intent, "score": float(s)})
    return {"query": req.text, "results": results}


# ---------------------------
# Upload CSV endpoint (build index server-side)
# ---------------------------
@app.post("/upload_intents")
async def upload_intents(file: UploadFile = File(...)):
    """
    Accept a CSV file with at least one text column. Builds embeddings & saves embeddings.npy + metadata.joblib
    """
    contents = await file.read()
    # write to temp path and build
    tmp_path = os.path.join(APP_ROOT, "uploaded_intents.csv")
    with open(tmp_path, "wb") as f:
        f.write(contents)

    try:
        embeddings, metadata = build_index_from_csv(tmp_path)
        save_index(embeddings, metadata)
        return {"status": "ok", "message": "Index built and saved", "entries": len(metadata.get("texts", []))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {e}")


# ---------------------------
# Chat endpoint using OpenAI
# ---------------------------
# Pydantic models for chat
class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    model: Optional[str] = "gpt-4o-mini"  # use appropriate available model


# configure OpenAI key from env
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY") or None
if openai is not None and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY


def openai_chat_reply(messages: List[Dict[str, str]], model: str = "gpt-4o-mini", max_tokens: int = 256):
    if openai is None:
        return "OpenAI library not installed on server."

    if not OPENAI_API_KEY:
        return "OpenAI API key not configured. Set OPENAI_API_KEY in environment."

    # Use ChatCompletion (legacy compat) or chat endpoint depending on openai version
    try:
        # Newer openai versions: openai.ChatCompletion.create(...)
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        if resp and "choices" in resp and len(resp["choices"]) > 0:
            return resp["choices"][0]["message"]["content"]
        return ""
    except Exception as e:
        # return error text for debugging
        raise RuntimeError(f"OpenAI request failed: {e}")


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # prepare messages for OpenAI
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        reply_text = openai_chat_reply(msgs, model=req.model or "gpt-4o-mini", max_tokens=req.max_tokens or 256)
        return {"reply": reply_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Health / simple endpoints
# ---------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/info")
async def info():
    emb_exists = os.path.exists(EMB_PATH)
    meta_exists = os.path.exists(META_PATH)
    return {
        "embeddings": "present" if emb_exists else "missing",
        "metadata": "present" if meta_exists else "missing",
        "frontend": FRONTEND_DIR if os.path.isdir(FRONTEND_DIR) else "missing",
        "faiss": faiss is not None,
        "sentence_transformers": SentenceTransformer is not None,
        "openai": openai is not None,
    }

# Note: Do not add uvicorn.run() here. Start via CLI or Dockerfile as intended.
